"""Correctness + speed comparison for HiLS-SWA implementation routes.

Routes compared:
  swa        plain sliding window, no CLS masking      -> cost ceiling
  score_mod  native window + score_mod CLS mask        -> recommended
  mask_mod   mask_mod (window inside mask) + block sparsity
  causal     full causal, for scale reference

Usage:
    python agent_space/bench_hils_swa.py --check
    python agent_space/bench_hils_swa.py --seqlen 8k,32k --window 1024,2048
"""

import argparse
import os
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch  # noqa: E402
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity  # noqa: E402
from flash_attn.cute.interface import flash_attn_func  # noqa: E402
from hils_swa import (  # noqa: E402
    hils_swa_ref,
    make_cls_mask_mod,
    make_cls_score_mod,
)

PERIOD, CLS_OFFSET = 64, 63


def parse_ints(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        out.append(int(float(tok[:-1]) * 1024) if tok.endswith("k") else int(tok))
    return out


def useful_flops(batch, nheads, seqlen, window, headdim, period=PERIOD, off=CLS_OFFSET):
    """FLOPs for the pairs the HiLS-SWA mask actually keeps (CLS columns excluded)."""
    i = torch.arange(seqlen, dtype=torch.int64)
    lo = torch.clamp(i - window, min=0)
    total = i - lo + 1
    fdiv = lambda x: torch.div(x, period, rounding_mode="floor")  # noqa: E731
    n_cls = fdiv(i - off) - fdiv(lo - 1 - off)
    avg = (total - n_cls).double().mean().item()
    return batch * nheads * 2 * seqlen * avg * (2 * headdim)


def make_inputs(batch, seqlen, nheads, headdim, dtype, requires_grad=False):
    return tuple(
        torch.randn(
            batch, seqlen, nheads, headdim, device="cuda", dtype=dtype,
            requires_grad=requires_grad,
        )
        for _ in range(3)
    )


def build_block_sparse(mask_mod, batch, nheads, seqlen, tile=(256, 256)):
    # tile_m must be a multiple of q_stage * m_block_size (256 for the 2CTA hd128 kernel)
    # and tile_n a multiple of n_block_size.
    return compute_block_sparsity(
        tile_m=tile[0],
        tile_n=tile[1],
        batch_size=batch,
        num_heads=nheads,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        mask_mod=mask_mod,
        aux_tensors=None,
        device="cuda",
    )


def run_fns(q, k, v, window, batch, nheads, seqlen, want):
    """Build {name: callable} for the requested routes."""
    score_mod, score_mod_bwd = make_cls_score_mod(PERIOD, CLS_OFFSET)

    def build(name):
        if name == "causal":
            return lambda: flash_attn_func(q, k, v, causal=True)
        if name == "swa":
            return lambda: flash_attn_func(q, k, v, window_size=(window, 0))
        if name == "score_mod":
            return lambda: flash_attn_func(
                q, k, v, window_size=(window, 0),
                score_mod=score_mod, score_mod_bwd=score_mod_bwd,
            )
        if name == "mask_mod":
            mm = make_cls_mask_mod(window, PERIOD, CLS_OFFSET)
            bs = BlockSparseTensorsTorch(*build_block_sparse(mm, batch, nheads, seqlen))
            return lambda: flash_attn_func(q, k, v, mask_mod=mm, block_sparse_tensors=bs)
        raise ValueError(f"unknown route {name}")

    return {name: build(name) for name in want}


def check(args):
    dtype = torch.bfloat16
    fails = 0
    for seqlen in (512, 1024, 2048):
        for window in (64, 128, 1024):
            if window > seqlen:
                continue
            torch.manual_seed(0)
            q, k, v = make_inputs(2, seqlen, 8, args.headdim, dtype, requires_grad=True)
            score_mod, score_mod_bwd = make_cls_score_mod(PERIOD, CLS_OFFSET)
            out, _ = flash_attn_func(
                q, k, v, window_size=(window, 0),
                score_mod=score_mod, score_mod_bwd=score_mod_bwd,
            )
            ref = hils_swa_ref(q, k, v, window, PERIOD, CLS_OFFSET)
            ref_lp = hils_swa_ref(q, k, v, window, PERIOD, CLS_OFFSET, upcast=False)
            err = (out.float() - ref).abs().max().item()
            lp = (ref_lp.float() - ref).abs().max().item()
            ok_f = err <= 2 * lp + 1e-4 and torch.isfinite(out).all().item()

            # Backward: compare grads against the fp32 reference graph.
            g = torch.randn_like(out)
            (dq, dk, dv) = torch.autograd.grad(out, (q, k, v), g, retain_graph=False)
            qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
            ref2 = hils_swa_ref(qr, kr, vr, window, PERIOD, CLS_OFFSET)
            (rq, rk, rv) = torch.autograd.grad(ref2, (qr, kr, vr), g.float())
            gerrs = [
                (a.float() - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
                for a, b in ((dq, rq), (dk, rk), (dv, rv))
            ]
            ok_b = max(gerrs) < 2e-2 and all(torch.isfinite(x).all().item() for x in (dq, dk, dv))
            fails += not (ok_f and ok_b)
            print(
                f"  seqlen={seqlen:<5} window={window:<5} fwd_err={err:.2e} (sdpa {lp:.2e}) "
                f"grad_rel={max(gerrs):.2e}  {'OK' if ok_f and ok_b else 'FAIL'}"
            )
    return fails


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqlen", type=parse_ints, default=[8192, 32768])
    p.add_argument("--window", type=parse_ints, default=[1024, 2048])
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--nheads", type=int, default=16)
    p.add_argument("--total-tokens", type=int, default=32768)
    p.add_argument("--routes", type=str, default="causal,swa,score_mod,mask_mod")
    p.add_argument("--repeat", type=int, default=1, help="Repeat the route sweep N times")
    p.add_argument("--check", action="store_true")
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=50)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"HiLS layout: period={PERIOD}, CLS at idx%{PERIOD}=={CLS_OFFSET}\n")

    if args.check:
        print("HiLS-SWA correctness (score_mod route), fwd + bwd:")
        n = check(args)
        print("\nALL PASS" if n == 0 else f"\n{n} FAILURES")
        return

    want = args.routes.split(",")
    hdr = f"{'seqlen':>7} {'window':>7} {'route':>10} {'fwd ms':>9} {'TFLOP/s':>9} {'vs swa':>8}"
    if args.bwd:
        hdr += f" {'bwd ms':>9} {'vs swa':>8}"
    print(hdr)
    print("-" * len(hdr))

    for seqlen in args.seqlen:
        batch = max(1, args.total_tokens // seqlen)
        for window in args.window:
            torch.manual_seed(0)
            q, k, v = make_inputs(batch, seqlen, args.nheads, args.headdim,
                                  torch.bfloat16, requires_grad=args.bwd)
            g = torch.randn_like(q) if args.bwd else None
            fns = run_fns(q, k, v, window, batch, args.nheads, seqlen, want)
            f = useful_flops(batch, args.nheads, seqlen, window, args.headdim)
            outs = {name: fn()[0] for name, fn in fns.items()} if args.bwd else {}

            def bwd_fn(name):
                out = outs[name]

                def bwd():
                    q.grad = k.grad = v.grad = None
                    out.backward(g, retain_graph=True)

                return bwd

            # Interleave routes across rounds so clock drift hits every route equally.
            samples = {name: [] for name in fns}
            bsamples = {name: [] for name in fns}
            for _ in range(args.repeat):
                for name, fn in fns.items():
                    samples[name].append(do_bench(fn, warmup=args.warmup, rep=args.rep))
                    if args.bwd and name != "mask_mod":
                        bsamples[name].append(
                            do_bench(bwd_fn(name), warmup=args.warmup, rep=args.rep)
                        )

            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            base_ms = med(samples["swa"]) if "swa" in samples else None
            base_bms = med(bsamples["swa"]) if bsamples.get("swa") else None
            for name in fns:
                ms = med(samples[name])
                rel = f"{ms / base_ms:.3f}x" if base_ms else "-"
                row = (f"{seqlen:>7} {window:>7} {name:>10} {ms:>9.3f} "
                       f"{f / (ms * 1e-3) / 1e12:>9.1f} {rel:>8}")
                if args.bwd and bsamples[name]:
                    bms = med(bsamples[name])
                    brel = f"{bms / base_bms:.3f}x" if base_bms else "-"
                    row += f" {bms:>9.3f} {brel:>8}"
                print(row)
            print()


if __name__ == "__main__":
    main()
