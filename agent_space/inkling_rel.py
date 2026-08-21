"""Inkling-style relative attention bias expressed as an FA4 score_mod.

Inkling (Thinking Machines, 2026) replaces RoPE with a learned, input-dependent
relative-position bias added to the pre-softmax logits:

    rel_logits[b, h, q, :] = r[b, h, q, :] @ P          # P: (d_rel, rel_extent)
    bias[b, h, q, k]       = rel_logits[b, h, q, q - k] # 0 outside [0, rel_extent)
    logits                 = QK^T / sqrt(d) + bias + mask

For the sliding-window layers ``rel_extent == sliding_window_size`` (both 512), so
inside the window the distance ``q - k`` is always in range and the bounds check that
``modeling_inkling.py`` applies via ``masked_fill`` is dead code. FA4's
``window_size=(rel_extent - 1, 0)`` reproduces exactly that key range.

This module materializes ``rel_logits`` (one GEMM per layer) and gathers one element
per score inside the score_mod. That is the only route FA4 supports today; note it
forces ``__vec_size__`` to 1 (scalar score_mod evaluation) because aux_tensors are used.

FORWARD ONLY: FA4's autograd returns ``None`` for aux_tensors, so no gradient flows
back to ``rel_logits`` (and hence to r_proj / P). Training needs kernel work.
"""

import math

import cutlass
import cutlass.cute as cute
import torch

D_REL = 16
REL_EXTENT = 512  # sliding-window layers: rel_extent == sliding_window_size


def make_inkling_score_mod(num_heads: int, seqlen: int, rel_extent: int = REL_EXTENT):
    """score_mod reading a flat rel_logits buffer laid out as (B, H, T, rel_extent).

    Call with ``window_size=(rel_extent - 1, 0)``: every *unmasked* score then has
    ``0 <= q_idx - kv_idx < rel_extent``.

    IMPORTANT: score_mod runs *pre-mask* over the whole MMA tile, so it is also
    evaluated on elements the window mask will discard, where ``q_idx - kv_idx`` falls
    outside ``[0, rel_extent)`` (roughly [-tile_n, rel_extent + tile_n)). Feeding that
    straight into the address computation reads out of bounds and traps. Since
    rel_extent is a power of two in Inkling (512 local / 1024 global), ``& (R - 1)``
    folds any distance back in range in one instruction; the value loaded for those
    lanes is irrelevant because the window mask overwrites them with -inf afterwards.
    """
    assert rel_extent & (rel_extent - 1) == 0, "rel_extent must be a power of two"

    @cute.jit
    def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        rel = aux_tensors[0]
        dtype = rel.element_type
        b0, h0, q0, k0 = b_idx[0], h_idx[0], q_idx[0], kv_idx[0]
        # flat offset of rel_logits[b, h, q, q - k]; distance folded to stay in bounds.
        dist = (q0 - k0) & cutlass.Int32(rel_extent - 1)
        row = (b0 * cutlass.Int32(num_heads) + h0) * cutlass.Int32(seqlen) + q0
        offset = row * cutlass.Int32(rel_extent) + dist
        frag = cute.make_rmem_tensor(1, dtype)
        frag[0] = rel[offset]
        return tSrS_ssa + (frag.load()).to(cutlass.Float32)

    @cute.jit
    def score_mod_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # d(score + bias)/d(score) = 1. NOTE: this gives correct dq/dk/dv but NO
        # gradient for the bias itself -- aux_tensors receive None from autograd.
        return grad

    return score_mod, score_mod_bwd


def build_rel_logits(relative_states, P):
    """relative_states (B, T, H, d_rel) @ P (d_rel, rel_extent) -> flat (B*H*T*R,)."""
    rel = torch.einsum("bthc,cr->bhtr", relative_states, P)
    return rel.contiguous().view(-1)


def inkling_rel_ref(q, k, v, relative_states, P, rel_extent=REL_EXTENT, upcast=True):
    """Reference following modeling_inkling.InklingRelativeLogits + sliding window."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # (B, H, T, D)
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)

    rel_logits = torch.einsum("bthc,cr->bhtr", relative_states, P)  # (B, H, T, R)
    qpos = torch.arange(sq, device=q.device)
    kpos = torch.arange(sk, device=q.device)
    distance = (qpos[:, None] - kpos[None, :])[None, None, :, :]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    bias = rel_logits.gather(-1, gather_index)
    bias = bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)

    window_mask = (distance >= 0) & (distance < rel_extent)
    if upcast:
        scores = (qt.float() @ kt.float().transpose(-1, -2)) / math.sqrt(d) + bias.float()
        scores = scores.masked_fill(~window_mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = attn @ vt.float()
    else:
        out = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=torch.where(window_mask, bias, float("-inf")).to(q.dtype)
        )
        out = torch.nan_to_num(out, nan=0.0)
    return out.transpose(1, 2)
