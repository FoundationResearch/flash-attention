"""Inkling relative bias computed inside score_mod from r and P -- no materialization.

Rationale
---------
Materializing ``rel_logits = r @ P`` (B*H*T*rel_extent, 2.1 GB at the Inkling SWA shape)
and gathering one element per score costs ~7x, and re-laying it out for coalescing only
buys ~20% (see inkling_rel_band.py). The cost is not bandwidth and not coalescing: it is
a *dependent global load per score element* sitting in the softmax epilogue, between the
QK and PV MMAs, where nothing can hide its latency.

So don't read the bias from HBM at all -- recompute it from the two small operands:

    bias(q, k) = r[b, h, q, :] . P[:, q - k]          d_rel = 16

* ``P`` is 16 x rel_extent, shared by every head and every token: 16 KB in bf16, so it
  stays resident in L1 across the whole kernel.
* ``r[b, h, q, :]`` is loop-invariant per thread, because SM100 maps M to tmem lanes and
  ``apply_score_mod`` passes ``constant_q_idx`` -- one thread only ever sees a single q.
  Its 16 loads hoist out of the fragment loop into registers.

Per score element this leaves ~16 L1 reads of ``P_T[d, :]`` (consecutive lanes read
consecutive rows of P_T, so they coalesce) plus 16 FMAs -- all register/L1 traffic.

``P`` is passed transposed (rel_extent, d_rel) so the 16 values for one distance are
contiguous.
"""

import cutlass
import cutlass.cute as cute
import torch

D_REL = 16
REL_EXTENT = 512


def prepare_operands(relative_states, P):
    """(B,T,H,d_rel), (d_rel,R) -> (flat r as (B,H,T,d_rel), flat P_T as (R,d_rel))."""
    r_bht = relative_states.permute(0, 2, 1, 3).contiguous()  # (B, H, T, d_rel)
    return r_bht.view(-1), P.t().contiguous().view(-1)


def make_inkling_fused_score_mod(
    num_heads: int, seqlen: int, rel_extent: int = REL_EXTENT, d_rel: int = D_REL
):
    """score_mod computing r . P[:, q-k] on the fly. Use window_size=(rel_extent-1, 0)."""
    assert rel_extent & (rel_extent - 1) == 0, "rel_extent must be a power of two"

    @cute.jit
    def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        r_t, p_t = aux_tensors[0], aux_tensors[1]
        dtype = r_t.element_type
        b0, h0, q0, k0 = b_idx[0], h_idx[0], q_idx[0], kv_idx[0]
        # Loop-invariant per thread (q0 is constant for the whole tile on SM100).
        r_base = ((b0 * cutlass.Int32(num_heads) + h0) * cutlass.Int32(seqlen) + q0) * (
            cutlass.Int32(d_rel)
        )
        # Fold the distance: score_mod runs pre-mask over the whole tile.
        p_base = ((q0 - k0) & cutlass.Int32(rel_extent - 1)) * cutlass.Int32(d_rel)

        rfrag = cute.make_rmem_tensor(d_rel, dtype)
        pfrag = cute.make_rmem_tensor(d_rel, dtype)
        for c in cutlass.range_constexpr(d_rel):
            rfrag[c] = r_t[r_base + cutlass.Int32(c)]
            pfrag[c] = p_t[p_base + cutlass.Int32(c)]

        acc = cute.make_rmem_tensor(1, cutlass.Float32)
        acc[0] = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(d_rel):
            acc[0] = acc[0] + rfrag[c].to(cutlass.Float32) * pfrag[c].to(cutlass.Float32)
        return tSrS_ssa + acc.load()

    @cute.jit
    def score_mod_bwd(grad, score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        return grad  # forward-only; FA4 autograd gives aux_tensors no gradient

    return score_mod, score_mod_bwd
