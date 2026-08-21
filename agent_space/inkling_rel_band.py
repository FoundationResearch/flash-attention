"""Coalesced ("band") layout for the Inkling relative-position bias.

Why a second layout exists
--------------------------
On SM100 the QK accumulator lives in tmem with the M (query) dimension mapped to tmem
lanes, so ``flash_fwd_sm100.apply_score_mod`` passes ``constant_q_idx``: within one
thread ``q`` is fixed and ``k`` varies, and across the 32 lanes of a warp ``q`` varies
while ``k`` is the same.

With the natural ``rel_logits[b, h, q, d]`` layout (d = q - k) the address is
``q*R + (q-k)``, which advances by ``R+1`` elements per lane -> 32 distinct cache lines
per warp-wide load, i.e. a fully uncoalesced gather (measured: 6.95x slowdown).

Storing the same values in a k-major band, ``band[b, h, k, j] = rel_logits[b, h, k+j, j]``
with ``j = q - k``, makes the address ``k*R + (q-k)``: ``k`` is shared across the warp and
``q`` advances by one, so the 32 lanes read 32 consecutive elements. Same bytes, same
arithmetic, coalesced.

The band is a pure shear of rel_logits, so it is an ``as_strided`` view (stride R+1 along
the j axis) plus one contiguous copy -- no gather, no index tensor.
"""

import cutlass
import cutlass.cute as cute
import torch

D_REL = 16
REL_EXTENT = 512


def build_rel_band(relative_states, P, rel_extent=REL_EXTENT):
    """(B,T,H,d_rel) x (d_rel,R) -> flat band buffer of shape (B*H*T*R,).

    band[b, h, k, j] = rel_logits[b, h, k+j, j]
    """
    B, T, H, _ = relative_states.shape
    R = rel_extent
    # Pad the token axis by R so the sheared view never runs off the end; the padding
    # rows correspond to q >= T, which the causal window masks out anyway.
    rel = torch.einsum("bthc,cr->bhtr", relative_states, P)  # (B, H, T, R)
    rel_pad = torch.zeros(B, H, T + R, R, device=rel.device, dtype=rel.dtype)
    rel_pad[:, :, :T] = rel
    rel_pad = rel_pad.contiguous()
    sB, sH = H * (T + R) * R, (T + R) * R
    band = torch.as_strided(rel_pad, (B, H, T, R), (sB, sH, R, R + 1))
    return band.contiguous().view(-1)


def make_inkling_band_score_mod(num_heads: int, seqlen: int, rel_extent: int = REL_EXTENT):
    """score_mod reading the k-major band buffer. Use window_size=(rel_extent-1, 0)."""
    assert rel_extent & (rel_extent - 1) == 0, "rel_extent must be a power of two"

    @cute.jit
    def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        band = aux_tensors[0]
        b0, h0, q0, k0 = b_idx[0], h_idx[0], q_idx[0], kv_idx[0]
        # band[b, h, k, q - k]; fold the distance because score_mod runs pre-mask and
        # sees tile elements outside the window (see inkling_rel.py).
        j = (q0 - k0) & cutlass.Int32(rel_extent - 1)
        row = (b0 * cutlass.Int32(num_heads) + h0) * cutlass.Int32(seqlen) + k0
        frag = cute.make_rmem_tensor(1, band.element_type)
        frag[0] = band[row * cutlass.Int32(rel_extent) + j]
        return tSrS_ssa + (frag.load()).to(cutlass.Float32)

    @cute.jit
    def score_mod_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad  # forward-only: aux_tensors get no gradient from FA4 autograd

    return score_mod, score_mod_bwd
