import os, sys, torch
sys.path.insert(0, "/home/hal-alex/workspace/flash-attention/agent_space")
from flash_attn.cute.interface import _flash_attn_fwd
from inkling_rel_native import prepare_rel_bias_operands, pad_rel_r, D_REL
torch.manual_seed(0); dt = torch.bfloat16
W = 512; B, T, HQ, HKV = 4, 8192, 64, 16
q = torch.randn(B, T, HQ, 128, device="cuda", dtype=dt)
k = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
v = torch.randn(B, T, HKV, 128, device="cuda", dtype=dt)
r = torch.randn(B, T, HQ, D_REL, device="cuda", dtype=dt) * 0.1
rp = pad_rel_r(r)
P_rev = prepare_rel_bias_operands(torch.randn(D_REL, W, device="cuda", dtype=dt) * 0.1, W, head_dim=128)
for _ in range(2):
    rp[:, T:].zero_()
    _flash_attn_fwd(q, k, v, window_size_left=W-1, window_size_right=0, rel_bias_r=rp, rel_bias_p=P_rev, pack_gqa=False)
torch.cuda.synchronize()
t = rp[0, T:].contiguous().view(torch.int64).flatten()[:24*96].view(24, 96).cpu()
names = {0:"COR before wait_ops", 2:"COR ops ready", 4:"COR compute done", 6:"COR lo stored", 8:"COR P consumed", 10:"COR hi+arrive", 12:"COR prefetch issued", 14:"COR stats_done", 16:"COR spo_released"}
nz = t[t != 0]; t0 = nz.min().item()
cols = sorted([c for c in range(96) if (t[:, c] != 0).any()], reverse=True)
print(f"{'event':<22}" + "".join(f"{'nb'+str(c):>10}" for c in cols))
for s in (0, 1):
    print(f"--- stage {s} ---")
    for e in sorted(names):
        row = t[e + s]
        print(f"{names[e]:<22}" + "".join(f"{(row[c].item()-t0)/1900:>10.2f}" if row[c] != 0 else f"{'-':>10}" for c in cols))
print("(us @1.9GHz; stats/spo columns are the stats block, the others the produced block)")
