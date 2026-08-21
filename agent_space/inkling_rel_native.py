"""Kernel-native Inkling relative bias: host-side operand preparation + wrapper.

bias(q, k) = r[b, q, h, :] . P[:, q - k]    for 0 <= q - k < rel_extent, else 0

The SM100 forward kernel (``rel_bias=True``) has its correction warps compute each
(128 x 128) bias tile with warp-level MMA from two small operands:

* ``r``      (B, T, H, 16)           per-token relative features, as produced by r_proj
* ``P_rev``  (window + 640, 16)      P^T, zero-padded outside [0, rel_extent), stored in
                                     *reversed* distance order and pre-divided by
                                     softmax_scale (S is unscaled inside the kernel)

Both are handed to the kernel as float16 (the bias MMAs accumulate in f16, which is
more precise than bf16 for these O(1) values; the tile is stored as f16 anyway).

Row ``i`` of P_rev holds distance ``d = D_OFF - i`` with ``D_OFF = window + 256``:
the kernel's MMA B-fragment walks distances downwards while its columns walk upwards,
so reversing on the host turns that into a plain contiguous load.
"""

import math

import torch

D_REL = 16
D_OFF_PAD = 256


def pad_rel_r(r):
    """(B, T, H, 16) -> (B, T + 256, H, 16): the kernel reads whole 128-row q-tiles."""
    B, T, H, d = r.shape
    out = torch.zeros(B, T + 256, H, d, device=r.device, dtype=torch.float16)
    out[:, :T] = r.to(torch.float16)
    return out


def prepare_rel_bias_operands(P, window, softmax_scale=None, head_dim=None):
    """P: (d_rel, rel_extent). Returns P_rev (window + 640, d_rel) in P.dtype."""
    d_rel, rel_extent = P.shape
    assert d_rel == D_REL
    if softmax_scale is None:
        assert head_dim is not None
        softmax_scale = 1.0 / math.sqrt(head_dim)
    n_rows = window + 640
    d_off = window + D_OFF_PAD
    i = torch.arange(n_rows, device=P.device)
    d = d_off - i
    valid = (d >= 0) & (d < rel_extent)
    P_rev = torch.zeros(n_rows, d_rel, device=P.device, dtype=torch.float32)
    P_rev[valid] = (P.float().t() / softmax_scale)[d[valid]]
    return P_rev.to(torch.float16).contiguous()


def rel_bias_attn(q, k, v, r, P, window, rel_extent=None, softmax_scale=None, lmk_score_mod=None):
    """Forward-only sliding-window attention with the Inkling relative bias.

    window: number of keys each query may see, including itself (Inkling SWA: 512).
    The bias extent is taken from P.shape[1]; distances beyond it get bias 0.
    """
    from flash_attn.cute.interface import _flash_attn_fwd

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    P_rev = prepare_rel_bias_operands(P, window, softmax_scale)
    kwargs = {}
    if lmk_score_mod is not None:
        kwargs["score_mod"] = lmk_score_mod
    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        window_size_left=window - 1,
        window_size_right=0,
        rel_bias_r=pad_rel_r(r),
        rel_bias_p=P_rev,
        return_lse=True,
        **kwargs,
    )
    return out, lse


def rel_bias_ref(q, k, v, r, P, window, upcast=True):
    """Reference: modeling_inkling.InklingRelativeLogits semantics + causal window."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    rel_extent = P.shape[1]
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # (B, H, T, D)
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)
    rel_logits = torch.einsum("bthc,cr->bhtr", r.float(), P.float())  # (B, H, T, R)
    qpos = torch.arange(sq, device=q.device)
    kpos = torch.arange(sk, device=q.device)
    distance = (qpos[:, None] - kpos[None, :])[None, None]
    gather_index = distance.clamp(0, rel_extent - 1).expand(b, hq, -1, -1)
    bias = rel_logits.gather(-1, gather_index)
    bias = bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)
    mask = (distance >= 0) & (distance < window)
    if upcast:
        scores = (qt.float() @ kt.float().transpose(-1, -2)) / math.sqrt(d) + bias
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out = attn @ vt.float()
    else:
        attn_mask = torch.where(mask, bias, float("-inf")).to(q.dtype)
        out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask)
        out = torch.nan_to_num(out, nan=0.0)
    return out.transpose(1, 2)
