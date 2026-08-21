"""Correctness + speed for chunk-aligned HiLS-SWA (whole-chunk visibility + lmk mask).

Usage:
    python agent_space/bench_hils_chunk.py --check
    python agent_space/bench_hils_chunk.py --seqlen 8k,32k --window 1024,2048
"""

import argparse
import os
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attn.cute.interface import flash_attn_func  # noqa: E402
from hils_swa import (  # noqa: E402
    hils_chunk_attn,
    hils_chunk_mask,
    hils_chunk_ref,
    make_cls_score_mod,
    make_hils_chunk_score_mod,
)

PERIOD, CLS_OFFSET = 64, 63


def parse_ints(s):
    return [
        int(float(t[:-1]) * 1024) if t.strip().lower().endswith("k") else int(t)
        for t in s.split(",")
    ]


def make_inputs(batch, seqlen, nheads, nheads_kv, headdim, dtype, requires_grad=False):
    def t(h):
        return torch.randn(
            batch, seqlen, h, headdim, device="cuda", dtype=dtype, requires_grad=requires_grad
        )

    return t(nheads), t(nheads_kv), t(nheads_kv)


def mask_properties(seqlen, window):
    """Assert the mask really is all-or-nothing per fully-past chunk."""
    m = hils_chunk_mask(seqlen, seqlen, window, PERIOD, CLS_OFFSET, "cuda")
    C = window // PERIOD
    for q in torch.randint(0, seqlen, (32,)).tolist():
        q_chunk = q // PERIOD
        for c in range(seqlen // PERIOD):
            cols = m[q, c * PERIOD : (c + 1) * PERIOD]
            content = torch.cat([cols[:CLS_OFFSET], cols[CLS_OFFSET + 1 :]])  # drop lmk col
            assert not cols[CLS_OFFSET], f"lmk visible at q={q} chunk={c}"
            if c > q_chunk or c <= q_chunk - C:
                assert not content.any(), f"chunk {c} partially visible for q={q}"
            elif c < q_chunk:
                assert content.all(), f"past in-window chunk {c} not fully visible for q={q}"
            else:  # own chunk: causal prefix
                expect = torch.arange(c * PERIOD, (c + 1) * PERIOD, device="cuda") <= q
                expect[CLS_OFFSET] = False
                assert torch.equal(cols, expect), f"own chunk wrong for q={q}"


def check(args):
    dtype = torch.bfloat16
    fails = 0
    for seqlen in (512, 1024, 2048):
        for window in (64, 128, 1024):
            if window > seqlen:
                continue
            mask_properties(seqlen, window)
            torch.manual_seed(0)
            q, k, v = make_inputs(2, seqlen, 8, 8 // args.gqa, args.headdim, dtype, True)
            sm, smb = make_hils_chunk_score_mod(window, PERIOD, CLS_OFFSET)
            out, _ = flash_attn_func(
                q, k, v, window_size=(window - 1, 0), score_mod=sm, score_mod_bwd=smb
            )
            ref = hils_chunk_ref(q, k, v, window, PERIOD, CLS_OFFSET)
            ref_lp = hils_chunk_ref(q, k, v, window, PERIOD, CLS_OFFSET, upcast=False)
            err = (out.float() - ref).abs().max().item()
            lp = (ref_lp.float() - ref).abs().max().item()
            ok_f = err <= 2 * lp + 1e-4 and torch.isfinite(out).all().item()

            g = torch.randn_like(out)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
            rq, rk, rv = torch.autograd.grad(
                hils_chunk_ref(qr, kr, vr, window, PERIOD, CLS_OFFSET), (qr, kr, vr), g.float()
            )
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
    p.add_argument("--gqa", type=int, default=1)
    p.add_argument("--total-tokens", type=int, default=32768)
    p.add_argument("--check", action="store_true")
    p.add_argument("--bwd", action="store_true")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--rep", type=int, default=200)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"chunk period={PERIOD}, lmk at idx%{PERIOD}=={CLS_OFFSET}, own chunk in budget\n")

    if args.check:
        print("Chunk-aligned HiLS-SWA correctness (fwd + bwd + mask properties):")
        n = check(args)
        print("\nALL PASS" if n == 0 else f"\n{n} FAILURES")
        return

    hdr = f"{'seqlen':>7} {'window':>7} {'route':>16} {'fwd ms':>9} {'vs swa':>8}"
    if args.bwd:
        hdr += f" {'bwd ms':>9} {'vs swa':>8}"
    print(hdr)
    print("-" * len(hdr))

    for seqlen in args.seqlen:
        batch = max(1, args.total_tokens // seqlen)
        for window in args.window:
            torch.manual_seed(0)
            q, k, v = make_inputs(
                batch, seqlen, args.nheads, args.nheads // args.gqa, args.headdim,
                torch.bfloat16, requires_grad=args.bwd,
            )
            g = torch.randn_like(q) if args.bwd else None
            cls_sm, cls_bwd = make_cls_score_mod(PERIOD, CLS_OFFSET)
            chunk_sm, chunk_bwd = make_hils_chunk_score_mod(window, PERIOD, CLS_OFFSET)
            fns = {
                "swa (no mods)": lambda: flash_attn_func(q, k, v, window_size=(window - 1, 0)),
                "lmk only": lambda: flash_attn_func(
                    q, k, v, window_size=(window - 1, 0),
                    score_mod=cls_sm, score_mod_bwd=cls_bwd,
                ),
                "chunk + lmk": lambda: flash_attn_func(
                    q, k, v, window_size=(window - 1, 0),
                    score_mod=chunk_sm, score_mod_bwd=chunk_bwd,
                ),
                "native chunk": lambda: hils_chunk_attn(q, k, v, window, PERIOD, CLS_OFFSET),
            }
            outs = {n_: f()[0] for n_, f in fns.items()} if args.bwd else {}

            def bwd_fn(name):
                out = outs[name]

                def bwd():
                    q.grad = k.grad = v.grad = None
                    out.backward(g, retain_graph=True)

                return bwd

            samples = {n_: [] for n_ in fns}
            bsamples = {n_: [] for n_ in fns}
            for _ in range(args.repeat):
                for name, fn in fns.items():
                    samples[name].append(do_bench(fn, warmup=args.warmup, rep=args.rep))
                    if args.bwd:
                        bsamples[name].append(
                            do_bench(bwd_fn(name), warmup=args.warmup, rep=args.rep)
                        )

            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            base = med(samples["swa (no mods)"])
            base_b = med(bsamples["swa (no mods)"]) if args.bwd else None
            for name in fns:
                ms = med(samples[name])
                row = f"{seqlen:>7} {window:>7} {name:>16} {ms:>9.3f} {ms / base:>7.3f}x"
                if args.bwd:
                    bms = med(bsamples[name])
                    row += f" {bms:>9.3f} {bms / base_b:>7.3f}x"
                print(row)
            print()


if __name__ == "__main__":
    main()
