"""Benchmark FA4 sliding-window (local) attention on SM100.

Sliding-window causal attention is expressed as ``window_size=(W, 0)``: each query
attends to itself and the ``W`` keys before it. ``causal=True`` alone is the same as
``window_size=(None, 0)``.

Effective FLOPs count only the (q, k) pairs inside the window, so TFLOP/s stays
comparable across window sizes. Full-causal FA4 at the same shape is timed as the
baseline that sliding window is supposed to beat.

Usage:
    python agent_space/bench_sliding_window.py --seqlen 4k,8k,16k,32k --window 512,1024,4096
    python agent_space/bench_sliding_window.py --check          # correctness vs reference
    python agent_space/bench_sliding_window.py --bwd            # include backward
"""

import argparse
import math

import torch
from triton.testing import do_bench

from flash_attn.cute.bench_utils import flops
from flash_attn.cute.interface import flash_attn_func


def parse_ints(s):
    """Parse '4k,8k,16384' into [4096, 8192, 16384]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("none", "full", "-1"):
            out.append(None)
        elif tok.endswith("k"):
            out.append(int(float(tok[:-1]) * 1024))
        else:
            out.append(int(tok))
    return out


def make_inputs(batch, seqlen, nheads, nheads_kv, headdim, dtype, requires_grad):
    def t(h):
        return torch.randn(
            batch, seqlen, h, headdim, device="cuda", dtype=dtype, requires_grad=requires_grad
        )

    return t(nheads), t(nheads_kv), t(nheads_kv)


def attention_ref(q, k, v, window_size):
    """fp32 reference with an explicit sliding-window causal mask."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    q, k, v = q.float(), k.float(), v.float()
    if hq != hk:  # GQA: repeat KV heads
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    # (b, h, sq, sk)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(d)
    row = torch.arange(sq, device=q.device).view(-1, 1) + (sk - sq)
    col = torch.arange(sk, device=q.device).view(1, -1)
    left, right = window_size
    mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
    if left is not None:
        mask &= col >= row - left
    if right is not None:
        mask &= col <= row + right
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    # Rows with an entirely masked-out window produce NaN in softmax; FA4 emits 0.
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v)


def sdpa_ref(q, k, v, window_size):
    """Low-precision (bf16/fp16) baseline via torch SDPA with an explicit window mask."""
    b, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # (b, h, s, d)
    if hq != hk:
        kt = kt.repeat_interleave(hq // hk, dim=1)
        vt = vt.repeat_interleave(hq // hk, dim=1)
    row = torch.arange(sq, device=q.device).view(-1, 1) + (sk - sq)
    col = torch.arange(sk, device=q.device).view(1, -1)
    left, right = window_size
    mask = torch.ones(sq, sk, dtype=torch.bool, device=q.device)
    if left is not None:
        mask &= col >= row - left
    if right is not None:
        mask &= col <= row + right
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return torch.nan_to_num(out, nan=0.0).transpose(1, 2)


def check_correctness(args):
    torch.manual_seed(0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    failures = 0
    for seqlen in (256, 1024):
        for window in (0, 63, 256, 1023):
            q, k, v = make_inputs(2, seqlen, 8, 8 // args.gqa, args.headdim, dtype, False)
            ws = (window, 0)
            out, _ = flash_attn_func(q, k, v, causal=False, window_size=ws)
            ref = attention_ref(q, k, v, ws)  # fp32
            ref_lp = sdpa_ref(q, k, v, ws)  # same low precision as FA4
            err = (out.float() - ref).abs().max().item()
            # Standard FA criterion: no worse than 2x a same-precision baseline.
            tol = 2 * (ref_lp.float() - ref).abs().max().item() + 1e-4
            ok = err <= tol
            failures += not ok
            print(
                f"  seqlen={seqlen:<5} window={window:<5} max_err={err:.3e} "
                f"sdpa_err={(ref_lp.float() - ref).abs().max().item():.3e} "
                f"tol={tol:.3e}  {'OK' if ok else 'FAIL'}"
            )
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seqlen", type=parse_ints, default=[4096, 8192, 16384, 32768])
    p.add_argument(
        "--window",
        type=parse_ints,
        default=[512, 2048, 8192, None],
        help="Left window size(s); 'none' = full causal",
    )
    p.add_argument("--headdim", type=int, default=128)
    p.add_argument("--nheads", type=int, default=16)
    p.add_argument("--gqa", type=int, default=1, help="nheads // nheads_kv")
    p.add_argument("--total-tokens", type=parse_ints, default=[32768])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--bwd", action="store_true", help="Also benchmark backward")
    p.add_argument("--check", action="store_true", help="Run correctness check and exit")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rep", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    nheads_kv = args.nheads // args.gqa
    total_tokens = args.total_tokens[0]

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(
        f"dtype={args.dtype} headdim={args.headdim} nheads={args.nheads} "
        f"nheads_kv={nheads_kv} total_tokens={total_tokens}\n"
    )

    if args.check:
        print("Correctness vs fp32 reference (sliding-window causal):")
        n_fail = check_correctness(args)
        print("\nALL PASS" if n_fail == 0 else f"\n{n_fail} FAILURES")
        return

    hdr = f"{'seqlen':>7} {'window':>7} {'batch':>5} {'fwd ms':>9} {'TFLOP/s':>9} {'vs causal':>10}"
    if args.bwd:
        hdr += f" {'bwd ms':>9} {'TFLOP/s':>9}"
    print(hdr)
    print("-" * len(hdr))

    for seqlen in args.seqlen:
        batch = args.batch_size or max(1, total_tokens // seqlen)
        q, k, v = make_inputs(
            batch, seqlen, args.nheads, nheads_kv, args.headdim, dtype, args.bwd
        )
        g = torch.randn_like(q) if args.bwd else None
        causal_ms = None
        # Time full causal first so later rows can report a speedup against it.
        windows = sorted(args.window, key=lambda w: (w is not None, w))

        for window in windows:
            # window=None -> full causal; else sliding-window causal (W, 0).
            ws = (None, None) if window is None else (window, 0)
            causal = window is None
            fwd = lambda: flash_attn_func(q, k, v, causal=causal, window_size=ws)  # noqa: E731
            ms = do_bench(fwd, warmup=args.warmup, rep=args.rep)
            if causal:
                causal_ms = ms

            f = flops(
                batch,
                args.nheads,
                seqlen,
                seqlen,
                args.headdim,
                args.headdim,
                causal=causal,
                window_size=ws,
            )
            tflops = f / (ms * 1e-3) / 1e12
            speedup = f"{causal_ms / ms:.2f}x" if causal_ms else "-"
            wname = "causal" if window is None else str(window)
            row = f"{seqlen:>7} {wname:>7} {batch:>5} {ms:>9.3f} {tflops:>9.1f} {speedup:>10}"

            if args.bwd:
                out, _ = flash_attn_func(q, k, v, causal=causal, window_size=ws)

                def bwd():
                    q.grad = k.grad = v.grad = None
                    out.backward(g, retain_graph=True)

                ms_b = do_bench(bwd, warmup=args.warmup, rep=args.rep)
                # Backward is ~2.5x the forward GEMM work.
                row += f" {ms_b:>9.3f} {2.5 * f / (ms_b * 1e-3) / 1e12:>9.1f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
