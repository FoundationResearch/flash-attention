import os, sys, torch
sys.path.insert(0, "/home/hal-alex/workspace/flash-attention/agent_space")
from flash_attn.cute.interface import _flash_attn_fwd
from inkling_rel_native import prepare_rel_bias_operands, pad_rel_r, rel_bias_ref, D_REL
torch.manual_seed(1); dt = torch.bfloat16
fails = 0
for (B, T, HQ, HKV, W, R) in [(1, 4096, 16, 4, 1024, 1024), (1, 4096, 8, 8, 2048, 512), (2, 3000, 8, 2, 1024, 1024), (1, 8192, 4, 4, 512, 512)]:
    q = torch.randn(B, T, HQ, 128, device="cuda", dtype=dt)
    k = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
    v = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
    r = torch.randn(B, T, HQ, D_REL, device="cuda", dtype=dt) * 0.2
    P = torch.randn(D_REL, R, device="cuda", dtype=dt) * 0.2
    P_rev = prepare_rel_bias_operands(P, W, head_dim=128)
    out, *_ = _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0,
                              rel_bias_r=pad_rel_r(r), rel_bias_p=P_rev, pack_gqa=False)
    ref = rel_bias_ref(q, k, v, r, P, W); lp = rel_bias_ref(q, k, v, r, P, W, upcast=False)
    e = (out.float()-ref).abs().max().item(); l = (lp.float()-ref).abs().max().item()
    ok = e <= 2*l + 2e-3 and torch.isfinite(out).all().item(); fails += not ok
    print(f"B={B} T={T} HQ={HQ} HKV={HKV} W={W} R={R}: err={e:.3e} sdpa_err={l:.3e} -> {'OK' if ok else 'FAIL'}")
print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
