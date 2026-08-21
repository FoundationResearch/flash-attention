"""Standalone mma.sync m16n8k16 (f16 acc) throughput per warp on this GPU."""
import torch, cutlass, cutlass.cute as cute
from cutlass import Float16, Float32, Int32, Int64
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

NITER, NIND = 400, 8

@cute.kernel
def kern(out: cute.Tensor, nwarps_active: Int32):
    tidx = cute.arch.thread_idx()[0]
    widx = tidx // 32
    lane = tidx % 32
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(Float16, Float16, (16, 8, 16)))
    thr = mma.get_slice(lane)
    a = thr.make_fragment_A(thr.partition_A(cute.make_tensor(cute.make_ptr(Float16, 0, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((16, 16), stride=(16, 1)))))
    b = thr.make_fragment_B(thr.partition_B(cute.make_tensor(cute.make_ptr(Float16, 0, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((8, 16), stride=(16, 1)))))
    a.store(cute.full_like(a.load(), 0.5))
    b.store(cute.full_like(b.load(), 0.25))
    accs = [thr.make_fragment_C(thr.partition_shape_C((16, 8))) for _ in range(NIND)]
    for c in accs:
        c.store(cute.full_like(c.load(), 0.0))
    if widx < nwarps_active:
        t0 = cute.arch.clock64()
        for it in cutlass.range(NITER, unroll=1):
            for c in accs:
                cute.gemm(mma, c, a[None, None, 0], b[None, None, 0], c)
        t1 = cute.arch.clock64()
        s = Float32(0.0)
        for c in accs:
            s = s + c[0].to(Float32)
        if lane == 0:
            out[cute.arch.block_idx()[0], widx, 0] = (t1 - t0).to(Float32)
            out[cute.arch.block_idx()[0], widx, 1] = s

@cute.jit
def launch(out: cute.Tensor, nwarps_active: Int32):
    kern(out, nwarps_active).launch(grid=[148, 1, 1], block=[128, 1, 1])

for nw in (1, 2, 4):
    out = torch.zeros(148, 4, 2, device="cuda", dtype=torch.float32)
    f = cute.compile(launch, from_dlpack(out), Int32(nw))
    f(from_dlpack(out), Int32(nw)); torch.cuda.synchronize()
    f(from_dlpack(out), Int32(nw)); torch.cuda.synchronize()
    cyc = out[:, :nw, 0].mean().item()
    print(f"{nw} warp(s)/CTA issuing: {cyc / (NITER * NIND):.1f} cycles per HMMA.16816.F16 per warp "
          f"(aggregate {NITER*NIND*nw/cyc:.2f} HMMA/cycle/SM)")
