"""Inkling rel-bias (kernel-native): MODE=check correctness vs reference, MODE=bench timing; REL_BIAS_DIAG=zero tests the zero-tile path bit-for-bit.

Run from outside the repo root (see memory: repo-root cwd shadows flash_attn).
"""
import os, sys, torch
from triton.testing import do_bench
sys.path.insert(0, "/home/hal-alex/workspace/flash-attention/agent_space")
from flash_attn.cute.interface import _flash_attn_fwd
from inkling_rel_native import prepare_rel_bias_operands, pad_rel_r, rel_bias_ref, D_REL
mode = os.environ.get("MODE", "check")
torch.manual_seed(0); dt = torch.bfloat16
W = 512
if mode == "check":
    for (B, T, HQ, HKV) in [(2, 1024, 8, 8), (1, 2048, 8, 2), (2, 1000, 8, 8)]:
        q = torch.randn(B, T, HQ, 128, device="cuda", dtype=dt)
        k = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
        v = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
        r = torch.randn(B, T, HQ, D_REL, device="cuda", dtype=dt) * 0.1
        P = torch.randn(D_REL, W, device="cuda", dtype=dt) * 0.1
        P_rev = prepare_rel_bias_operands(P, W, head_dim=128)
        out, *_ = _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0,
                                  rel_bias_r=pad_rel_r(r), rel_bias_p=P_rev, pack_gqa=False)
        if os.environ.get("REL_BIAS_DIAG") == "zero":
            base, *_ = _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0, pack_gqa=False)
            print(f"B={B} T={T} HQ={HQ} HKV={HKV}: max|zero_bias - swa| = {(out.float()-base.float()).abs().max().item():.3e} finite={torch.isfinite(out).all().item()}")
        else:
            ref = rel_bias_ref(q, k, v, r, P, W); lp = rel_bias_ref(q, k, v, r, P, W, upcast=False)
            e = (out.float()-ref).abs().max().item(); l = (lp.float()-ref).abs().max().item()
            print(f"B={B} T={T} HQ={HQ} HKV={HKV}: err={e:.3e} sdpa_err={l:.3e} finite={torch.isfinite(out).all().item()} -> {'OK' if e <= 2*l + 2e-3 else 'FAIL'}")
else:
    B, T, HQ, HKV = 4, 8192, 64, 16
    q = torch.randn(B, T, HQ, 128, device="cuda", dtype=dt)
    k = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
    v = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
    r = torch.randn(B, T, HQ, D_REL, device="cuda", dtype=dt) * 0.1
    P_rev = prepare_rel_bias_operands(torch.randn(D_REL, W, device="cuda", dtype=dt) * 0.1, W, head_dim=128)
    rp = pad_rel_r(r)
    med = lambda f: sorted(do_bench(f, warmup=50, rep=200) for _ in range(5))[2]
    b = med(lambda: _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0, pack_gqa=False))
    bp = med(lambda: _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0))
    rb = med(lambda: _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0, rel_bias_r=rp, rel_bias_p=P_rev, pack_gqa=False))
    print(f"[{os.environ.get('REL_BIAS_DIAG','real')}] swa(pack_gqa=False) {b:.3f} ms | swa(pack_gqa) {bp:.3f} | rel_bias {rb:.3f} -> {rb/b:.3f}x / {rb/bp:.3f}x")
