"""HiLS-SWA: sliding-window attention that skips periodic CLS tokens.

Layout: every ``period`` positions hold ``period - 1`` content tokens followed by one
CLS token at ``kv_idx % period == cls_offset`` (default: period 64, CLS at index 63).
The sliding window must not attend to those CLS keys. CLS tokens are still ordinary
queries — only the key side is masked.

Two ways to express this in FA4, with very different cost:

* ``score_mod`` (this module's recommendation) — keeps FA4's *native* local-window
  path, so the kernel still analytically skips out-of-window KV blocks. The CLS
  predicate is 2 ALU ops per score, vectorized 2-wide.
* ``mask_mod`` — ``interface._resolve_causal_local_window`` returns
  ``causal=False, local=False`` whenever ``mask_mod is not None``, so the window must
  be re-expressed *inside* the mask and the block skipping has to be rebuilt from a
  precomputed block-sparse mask. Also defaults to scalar (``__vec_size__ = 1``)
  evaluation. Implemented here only as a comparison baseline.
"""

import math
import operator

import cutlass
import cutlass.cute as cute
import torch

NEG_INF = float("-inf")


# ── score_mod route (recommended) ─────────────────────────────────────────────


def make_cls_score_mod(period: int = 64, cls_offset: int | None = None):
    """Build (score_mod, score_mod_bwd) that mask CLS keys out of the attention.

    ``period`` and ``cls_offset`` are closure constants, so they are baked into the
    kernel at compile time and are part of the JIT cache key (``utils.hash_callable``
    hashes closure values).
    """
    if cls_offset is None:
        cls_offset = period - 1
    is_pow2 = period & (period - 1) == 0

    @cute.jit
    def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # Position of kv_idx within its CLS group.
        if cutlass.const_expr(is_pow2):
            lane = kv_idx & cute.full_like(kv_idx, period - 1)
        else:
            lane = kv_idx % cute.full_like(kv_idx, period)
        is_cls = operator.eq(lane, cute.full_like(lane, cls_offset))
        return cute.where(is_cls, cute.full_like(tSrS_ssa, NEG_INF), tSrS_ssa)

    @cute.jit
    def score_mod_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # d/ds where(is_cls, -inf, s) = where(is_cls, 0, 1); the kernel already zeroes
        # grad at masked positions because P = 0 there, so pass through.
        return grad

    return score_mod, score_mod_bwd


def hils_swa_func(q, k, v, window, period=64, cls_offset=None, softmax_scale=None, **kwargs):
    """Sliding-window attention over content tokens only, skipping CLS keys."""
    from flash_attn.cute.interface import flash_attn_func

    score_mod, score_mod_bwd = make_cls_score_mod(period, cls_offset)
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(window, 0),
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        **kwargs,
    )


# ── chunk-aligned variant ─────────────────────────────────────────────────────


def make_hils_chunk_score_mod(window: int, period: int = 64, cls_offset: int | None = None):
    """Chunk-aligned HiLS-SWA: whole KV chunks are visible or invisible, never partial.

    With ``C = window // period`` chunks of budget, query q sees chunks
    ``[q_chunk - C + 1 .. q_chunk]`` (its own chunk causally truncated at q), minus
    landmark keys at ``kv % period == cls_offset``. All queries in one chunk share the
    same left edge, which jumps by ``period`` at chunk boundaries. Convention: the
    query's own chunk counts toward the budget, so the maximum reach is exactly
    ``window`` tokens (at q % period == period-1); to exclude the own chunk from the
    budget, pass ``window + period`` instead.

    Must be combined with the native ``window_size=(window - 1, 0)``: the widest
    per-query reach is q - window + 1, so the native window is a superset — it keeps
    block pruning and causality, and this mod only trims the ragged left edge (< period
    columns) and the landmark keys.

    ``(q & ~(period-1)) - (k & ~(period-1)) >= window`` is the "too old" test — chunk
    starts compared directly, no division.
    """
    assert window % period == 0, "window must be a multiple of the chunk period"
    assert period & (period - 1) == 0, "period must be a power of two"
    if cls_offset is None:
        cls_offset = period - 1
    neg_period = -period  # two's-complement ~(period-1): rounds an index down to its chunk start

    @cute.jit
    def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # too_old ⇔ k < q_base - window + period ⇔ q_base - k > window - period,
        # so k never needs rounding to its own chunk start. q_base is constant per
        # thread on the SM100 forward (constant_q_idx), so it hoists out of the loop.
        q_base = q_idx & cute.full_like(q_idx, neg_period)
        too_old = operator.gt(q_base - kv_idx, cute.full_like(q_idx, window - period))
        lane = kv_idx & cute.full_like(kv_idx, period - 1)
        is_lmk = operator.eq(lane, cute.full_like(lane, cls_offset))
        drop = too_old | is_lmk
        return cute.where(drop, cute.full_like(tSrS_ssa, NEG_INF), tSrS_ssa)

    @cute.jit
    def score_mod_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad  # pure masking: P=0 at dropped positions already zeroes the grad

    return score_mod, score_mod_bwd


def hils_chunk_mask(seqlen_q, seqlen_k, window, period, cls_offset, device):
    """(seqlen_q, seqlen_k) bool mask: causal, chunk-aligned window, minus lmk columns."""
    row = torch.arange(seqlen_q, device=device).view(-1, 1) + (seqlen_k - seqlen_q)
    col = torch.arange(seqlen_k, device=device).view(1, -1)
    causal = col <= row
    in_chunks = (row // period - col // period) < (window // period)
    not_lmk = (col % period) != cls_offset
    return causal & in_chunks & not_lmk


def hils_chunk_ref(q, k, v, window, period=64, cls_offset=None, upcast=True):
    """Reference attention with the chunk-aligned HiLS mask."""
    if cls_offset is None:
        cls_offset = period - 1
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)
    mask = hils_chunk_mask(sq, sk, window, period, cls_offset, q.device)
    if upcast:
        scores = (qt.float() @ kt.float().transpose(-1, -2)) / math.sqrt(d)
        scores = scores.masked_fill(~mask, NEG_INF)
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = attn @ vt.float()
    else:
        out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
        out = torch.nan_to_num(out, nan=0.0)
    return out.transpose(1, 2)


# ── kernel-native chunk alignment (forward) ───────────────────────────────────


class HilsChunkAttnFunc(torch.autograd.Function):
    """Chunk-aligned HiLS-SWA with the chunk edge folded into the SM100 native mask.

    Forward: ``_flash_attn_fwd(window_left_chunk=period)`` makes the kernel's own
    left-window comparison chunk-aligned (one extra AND per mask call, boundary blocks
    only), so the score_mod only has to drop landmark keys.

    Backward: same split. In the SM100 backward the left edge is a per-thread row
    threshold, so chunk alignment is a one-time rounding of that constant.
    """

    @staticmethod
    def forward(ctx, q, k, v, window, period, cls_offset, softmax_scale, deterministic):
        from flash_attn.cute.interface import _flash_attn_fwd

        lmk_mod, _ = make_cls_score_mod(period, cls_offset)
        out, lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            window_size_left=window - 1,
            window_size_right=0,
            score_mod=lmk_mod,
            window_left_chunk=period,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window, ctx.period, ctx.cls_offset = window, period, cls_offset
        ctx.softmax_scale, ctx.deterministic = softmax_scale, deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        from flash_attn.cute.interface import _flash_attn_bwd

        q, k, v, out, lse = ctx.saved_tensors
        lmk_mod, lmk_mod_bwd = make_cls_score_mod(ctx.period, ctx.cls_offset)
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            False,
            0.0,
            window_size_left=ctx.window - 1,
            window_size_right=0,
            deterministic=ctx.deterministic,
            score_mod=lmk_mod,
            score_mod_bwd=lmk_mod_bwd,
            window_left_chunk=ctx.period,
            dlse=dlse,
        )
        return dq, dk, dv, None, None, None, None, None


def hils_chunk_attn(q, k, v, window, period=64, cls_offset=None, softmax_scale=None,
                    deterministic=False):
    """Chunk-aligned HiLS-SWA; forward uses the kernel-native chunk edge."""
    if cls_offset is None:
        cls_offset = period - 1
    return HilsChunkAttnFunc.apply(q, k, v, window, period, cls_offset, softmax_scale, deterministic)


# ── mask_mod route (comparison only) ──────────────────────────────────────────


def make_cls_mask_mod(window: int, period: int = 64, cls_offset: int | None = None):
    """mask_mod expressing the window *and* the CLS exclusion (window is not native here)."""
    if cls_offset is None:
        cls_offset = period - 1
    is_pow2 = period & (period - 1) == 0

    @cute.jit
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        in_window = (q_idx >= kv_idx) & (q_idx - kv_idx <= cutlass.Int32(window))
        if cutlass.const_expr(is_pow2):
            lane = kv_idx & cutlass.Int32(period - 1)
        else:
            lane = kv_idx % cutlass.Int32(period)
        return in_window & (lane != cutlass.Int32(cls_offset))

    return mask_mod


# ── reference ─────────────────────────────────────────────────────────────────


def hils_swa_mask(seqlen_q, seqlen_k, window, period, cls_offset, device):
    """(seqlen_q, seqlen_k) bool mask: causal window minus CLS key columns."""
    row = torch.arange(seqlen_q, device=device).view(-1, 1) + (seqlen_k - seqlen_q)
    col = torch.arange(seqlen_k, device=device).view(1, -1)
    in_window = (col <= row) & (col >= row - window)
    not_cls = (col % period) != cls_offset
    return in_window & not_cls


def hils_swa_ref(q, k, v, window, period=64, cls_offset=None, upcast=True):
    """Reference attention with the HiLS-SWA mask; fp32 when upcast, else input dtype."""
    if cls_offset is None:
        cls_offset = period - 1
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # (b, h, s, d)
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)
    mask = hils_swa_mask(sq, sk, window, period, cls_offset, q.device)
    if upcast:
        qt, kt, vt = qt.float(), kt.float(), vt.float()
        scores = (qt @ kt.transpose(-1, -2)) / math.sqrt(d)
        scores = scores.masked_fill(~mask, NEG_INF)
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = attn @ vt
    else:
        out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
        out = torch.nan_to_num(out, nan=0.0)
    return out.transpose(1, 2)
