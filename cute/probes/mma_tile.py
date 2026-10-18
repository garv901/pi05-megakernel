"""Phase-0 probe: does idiomatic CuTe (TiledMma + permuted-K views + copy atoms) reproduce the tensor-core tiles of pi05_mk.cu bit-exactly?

Cases (CUDA reference in mma_ref.cu), each as `cute` (TiledMma, copy atoms) and `raw` (explicit per-lane fragments + mma PTX):
  gemm   gemm_phase tile, PI05_GEMM_FRAG=2: 8 warps x (64 x 32 acc), one K chunk (KC=128) at k0
  wsplit gemm_phase_wsplit<2> tile: 2 warps split the 4 k slabs, partials through smem, fixed-order sum
  fa     FA unit: S = Q K^T by ldmatrix, P (input) as A fragments by a C->A repack, O = P V by ldmatrix.trans

K permutation of gemm/wsplit (lane (g, t4) of 32-wide slab sl holds physical k = 32 sl + 8 t4 + j): the logical k of the
tiled mma (k16 step ks = 2 sl + m, position 8 h + 2 t4 + e) is physical 32 sl + 8 t4 + 4 m + 2 h + e, i.e. the layout
KPERM = (2,4,2,2,4):(1,8,2,4,32) applied to the K mode of A and B.
Run on a GPU: python -m megabench.sota.pi05.cute.probes.mma_tile --out DIR   (writes mma.json)
CPU check:    CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a python -m megabench.sota.pi05.cute.probes.mma_tile --compile-only"""

import argparse
import json
import os
import time

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint16, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp, warp

from ..config import ATT_ROWS, AS_STRIDE, KC, KS_STRIDE, T, VS_STRIDE, WS_STRIDE

HERE = os.path.dirname(os.path.abspath(__file__))
NB, K = 16, 512
HDIM, KEYS = 256, 64
GEMM_SMEM = 64 * AS_STRIDE * 2
WSPLIT_SMEM = GEMM_SMEM + 8 * 64 * WS_STRIDE * 4
FA_SMEM = (16 + 2 * KEYS) * KS_STRIDE * 2
LD = {"gemm": 256, "wsplit": 128}


def kperm():
    return cute.make_layout((2, 4, 2, 2, 4), stride=(1, 8, 2, 4, 32))


def tiled_mma():
    return cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))


def uni(dtype, bits=None):
    if bits is None:
        return cute.make_copy_atom(CopyUniversalOp(), dtype)
    return cute.make_copy_atom(CopyUniversalOp(), dtype, num_bits_per_copy=bits)


def vec_copy(rows: int, cols: int, nthreads: int):
    """128-bit gmem<->smem tiled copy over `nthreads` threads; one pass covers (nthreads / (cols/8)) rows."""
    tpr = cols // 8
    return cute.make_tiled_copy_tv(uni(BFloat16, 128), cute.make_layout((nthreads // tpr, tpr), stride=(tpr, 1)),
                                   cute.make_layout((1, 8)))


@cute.jit
def zero_reg():
    return cute.arch.inline_ptx("mov.f32 {$w0}, 0f00000000;", write_only_types=[Float32])


@cute.jit
def mma_raw(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[Float32] * 4, read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3])


@cute.jit
def raw_mma_into(acc, i: cutlass.Constexpr, a0, a1, a2, a3, b0, b1):
    r = mma_raw(acc[i], acc[i + 1], acc[i + 2], acc[i + 3], a0, a1, a2, a3, b0, b1)
    for e in cutlass.range_constexpr(4):
        acc[i + e] = r[e]


@cute.jit
def raw_acc():
    acc = cute.make_rmem_tensor((64,), Float32)
    for i in cutlass.range_constexpr(64):
        acc[i] = zero_reg()
    return acc


@cute.jit
def raw_slabs(A32, B32, bid, arow, brow, kw, sl0, nsl: cutlass.Constexpr, acc, g, t4):
    """acc[(mt*4+nt)*4+e] += slabs sl0 .. sl0+nsl-1 of the chunk starting at u32 column kw (explicit CUDA fragments)."""
    for j in cutlass.range_constexpr(nsl):
        base = kw + (sl0 + j) * 16 + t4 * 4
        for mt in cutlass.range_constexpr(4):
            for nt in cutlass.range_constexpr(4):
                for m in cutlass.range_constexpr(2):
                    a0 = A32[bid, arow + mt * 16 + g, base + m * 2]
                    a2 = A32[bid, arow + mt * 16 + g, base + m * 2 + 1]
                    a1 = A32[bid, arow + mt * 16 + g + 8, base + m * 2]
                    a3 = A32[bid, arow + mt * 16 + g + 8, base + m * 2 + 1]
                    b0 = B32[bid, brow + nt * 8 + g, base + m * 2]
                    b1 = B32[bid, brow + nt * 8 + g, base + m * 2 + 1]
                    raw_mma_into(acc, (mt * 4 + nt) * 4, a0, a1, a2, a3, b0, b1)


@cute.jit
def raw_store(acc, dst, g, t4):
    """dst: (64, ncols) view of the warp's 64 x 32 tile."""
    for mt in cutlass.range_constexpr(4):
        for nt in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                dst[mt * 16 + g + (e >> 1) * 8, nt * 8 + 2 * t4 + (e & 1)] = acc[(mt * 4 + nt) * 4 + e]


# ---------------------------------------------------------------- gemm
@cute.kernel
def gemm_cute(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    wp = tid // 32
    lane = tid % 32
    gA = cute.local_tile(A[bid, None, None], (64, KC), (0, k0 // KC))
    gB = cute.local_tile(Wt[bid, None, None], (32, KC), (wp, k0 // KC))
    gC = cute.local_tile(C[bid, None, None], (64, 32), (0, wp))
    sA = cute.make_tensor(cute.arch.get_dyn_smem(BFloat16, alignment=16), cute.make_layout((64, KC), stride=(AS_STRIDE, 1)))
    cp = vec_copy(64, KC, 256)
    th = cp.get_slice(tid)
    cute.copy(cp, th.partition_S(gA), th.partition_D(sA))
    cute.arch.sync_threads()
    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    sAp = cute.composition(sA, (None, kperm()))
    gBp = cute.composition(gB, (None, kperm()))
    rA = tmm.make_fragment_A(tmm.partition_A(sAp))
    tB = tmm.partition_B(gBp)
    rB = tmm.make_fragment_B(tB)
    acc = cute.make_rmem_tensor(tmm.partition_shape_C((64, 32)), Float32)
    acc.fill(0.0)
    ca = cute.make_tiled_copy_A(uni(BFloat16, 32), tm)
    tha = ca.get_slice(lane)
    cute.copy(ca, tha.partition_S(sAp), tha.retile(rA))
    cute.copy(uni(BFloat16, 32), tB, rB)
    for kb in cutlass.range_constexpr(cute.size(rA.shape[2])):
        cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)
    cute.copy(uni(Float32), acc, tmm.partition_C(gC))


@cute.kernel
def gemm_raw(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    wp = tid // 32
    lane = tid % 32
    g = lane // 4
    t4 = lane % 4
    A32 = cute.recast_tensor(A, Uint32)
    B32 = cute.recast_tensor(Wt, Uint32)
    acc = raw_acc()
    raw_slabs(A32, B32, bid, 0, wp * 32, k0 // 2, 0, 4, acc, g, t4)
    raw_store(acc, cute.local_tile(C[bid, None, None], (64, 32), (0, wp)), g, t4)


@cute.jit
def host_gemm(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32, impl: cutlass.Constexpr):
    kern = gemm_cute if impl == "cute" else gemm_raw
    kern(A, Wt, C, k0).launch(grid=(A.shape[0], 1, 1), block=(256, 1, 1), smem=GEMM_SMEM)


# ---------------------------------------------------------------- wsplit
@cute.jit
def reduce_pairs(red, C, bid, wp, lane):
    """Fixed-order sum of the group's two partial tiles red[2 gi], red[2 gi + 1] (rows < T), float4-column mapping of the kernel."""
    gi = wp // 2
    wi = wp % 2
    rr = lane // 8
    c4 = (lane % 8) * 4
    for q in cutlass.range(wi, (T + 3) // 4, 2, unroll=1):
        r = q * 4 + rr
        if r < T:
            for i in cutlass.range_constexpr(4):
                C[bid, r, gi * 32 + c4 + i] = red[gi * 2, r, c4 + i] + red[gi * 2 + 1, r, c4 + i]


@cute.kernel
def wsplit_cute(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    wp = tid // 32
    lane = tid % 32
    gi = wp // 2
    wi = wp % 2
    gA = cute.local_tile(A[bid, None, None], (64, KC), (0, k0 // KC))
    gB = cute.local_tile(Wt[bid, None, None], (32, KC), (gi, k0 // KC))
    smem = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sA = cute.make_tensor(smem, cute.make_layout((64, KC), stride=(AS_STRIDE, 1)))
    red = cute.make_tensor(cute.arch.get_dyn_smem(Float32, alignment=16) + 64 * AS_STRIDE // 2,
                           cute.make_layout((8, 64, WS_STRIDE), stride=(64 * WS_STRIDE, WS_STRIDE, 1)))
    cp = vec_copy(64, KC, 256)
    th = cp.get_slice(tid)
    cute.copy(cp, th.partition_S(gA), th.partition_D(sA))
    cute.arch.sync_threads()
    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    sAh = cute.local_tile(cute.composition(sA, (None, kperm())), (64, KC // 2), (0, wi))
    gBh = cute.local_tile(cute.composition(gB, (None, kperm())), (32, KC // 2), (0, wi))
    rA = tmm.make_fragment_A(tmm.partition_A(sAh))
    tB = tmm.partition_B(gBh)
    rB = tmm.make_fragment_B(tB)
    acc = cute.make_rmem_tensor(tmm.partition_shape_C((64, 32)), Float32)
    acc.fill(0.0)
    ca = cute.make_tiled_copy_A(uni(BFloat16, 32), tm)
    tha = ca.get_slice(lane)
    cute.copy(ca, tha.partition_S(sAh), tha.retile(rA))
    cute.copy(uni(BFloat16, 32), tB, rB)
    for kb in cutlass.range_constexpr(cute.size(rA.shape[2])):
        cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)
    mine = cute.local_tile(red[wp, None, None], (64, 32), (0, 0))
    cute.copy(uni(Float32), acc, tmm.partition_C(mine))
    cute.arch.barrier(barrier_id=1 + gi, number_of_threads=64)
    reduce_pairs(red, C, bid, wp, lane)


@cute.kernel
def wsplit_raw(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    wp = tid // 32
    lane = tid % 32
    g = lane // 4
    t4 = lane % 4
    gi = wp // 2
    wi = wp % 2
    red = cute.make_tensor(cute.arch.get_dyn_smem(Float32, alignment=16) + 64 * AS_STRIDE // 2,
                           cute.make_layout((8, 64, WS_STRIDE), stride=(64 * WS_STRIDE, WS_STRIDE, 1)))
    A32 = cute.recast_tensor(A, Uint32)
    B32 = cute.recast_tensor(Wt, Uint32)
    acc = raw_acc()
    raw_slabs(A32, B32, bid, 0, gi * 32, k0 // 2, wi * 2, 2, acc, g, t4)
    raw_store(acc, cute.local_tile(red[wp, None, None], (64, 32), (0, 0)), g, t4)
    cute.arch.barrier(barrier_id=1 + gi, number_of_threads=64)
    reduce_pairs(red, C, bid, wp, lane)


@cute.jit
def host_wsplit(A: cute.Tensor, Wt: cute.Tensor, C: cute.Tensor, k0: Int32, impl: cutlass.Constexpr):
    kern = wsplit_cute if impl == "cute" else wsplit_raw
    kern(A, Wt, C, k0).launch(grid=(A.shape[0], 1, 1), block=(256, 1, 1), smem=WSPLIT_SMEM)


# ---------------------------------------------------------------- fa
@cute.jit
def fa_smem():
    base = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sQ = cute.make_tensor(base, cute.make_layout((16, HDIM), stride=(KS_STRIDE, 1)))
    sK = cute.make_tensor(base + 16 * KS_STRIDE, cute.make_layout((KEYS, HDIM), stride=(KS_STRIDE, 1)))
    sV = cute.make_tensor(base + (16 + KEYS) * KS_STRIDE, cute.make_layout((KEYS, HDIM), stride=(VS_STRIDE, 1)))
    return sQ, sK, sV


@cute.kernel
def fa_cute(Q: cute.Tensor, Kt: cute.Tensor, V: cute.Tensor, P: cute.Tensor, S: cute.Tensor, O: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    dh = tid // 32
    lane = tid % 32
    sQ, sK, sV = fa_smem()
    cp = vec_copy(16, HDIM, 64)
    th = cp.get_slice(tid)
    cute.copy(cp, th.partition_S(Q[bid, None, None]), th.partition_D(sQ))
    cute.copy(cp, th.partition_S(Kt[bid, None, None]), th.partition_D(sK))
    cute.copy(cp, th.partition_S(V[bid, None, None]), th.partition_D(sV))
    cute.arch.sync_threads()

    tm = cute.make_tiled_mma(warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16))
    tmm = tm.get_slice(lane)
    ld = lambda trans: cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=trans, num_matrices=4), BFloat16)
    tSrQ = tmm.make_fragment_A(tmm.partition_A(sQ))
    tSrK = tmm.make_fragment_B(tmm.partition_B(sK))
    cq = cute.make_tiled_copy_A(ld(False), tm)
    ck = cute.make_tiled_copy_B(ld(False), tm)
    thq = cq.get_slice(lane)
    thk = ck.get_slice(lane)
    cute.copy(cq, thq.partition_S(sQ), thq.retile(tSrQ))
    cute.copy(ck, thk.partition_S(sK), thk.retile(tSrK))
    accS = cute.make_rmem_tensor(tmm.partition_shape_C((16, KEYS)), Float32)
    accS.fill(0.0)
    for kk in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
        cute.gemm(tm, accS, tSrQ[None, None, kk], tSrK[None, None, kk], accS)
    if dh == 0:
        cute.copy(uni(Float32), accS, tmm.partition_C(S[bid, None, None]))

    rP = cute.make_fragment_like(accS, BFloat16)
    cute.copy(uni(BFloat16), tmm.partition_C(P[bid, None, None]), rP)
    div = cute.logical_divide(rP.layout, (None, None, 2))
    pview = cute.make_layout(((div.shape[0], div.shape[2][0]), div.shape[1], div.shape[2][1]),
                             stride=((div.stride[0], div.stride[2][0]), div.stride[1], div.stride[2][1]))
    tOrP = cute.make_tensor(rP.iterator, pview)

    sVt = cute.local_tile(cute.make_tensor(sV.iterator, cute.make_layout((HDIM, KEYS), stride=(1, VS_STRIDE))), (128, KEYS), (dh, 0))
    tOrV = tmm.make_fragment_B(tmm.partition_B(sVt))
    cv = cute.make_tiled_copy_B(ld(True), tm)
    thv = cv.get_slice(lane)
    cute.copy(cv, thv.partition_S(sVt), thv.retile(tOrV))
    accO = cute.make_rmem_tensor(tmm.partition_shape_C((16, 128)), Float32)
    accO.fill(0.0)
    for ks in cutlass.range_constexpr(cute.size(tOrP.shape[2])):
        cute.gemm(tm, accO, tOrP[None, None, ks], tOrV[None, None, ks], accO)
    cute.copy(uni(Float32), accO, tmm.partition_C(cute.local_tile(O[bid, None, None], (16, 128), (0, dh))))


@cute.kernel
def fa_raw(Q: cute.Tensor, Kt: cute.Tensor, V: cute.Tensor, P: cute.Tensor, S: cute.Tensor, O: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    dh = tid // 32
    lane = tid % 32
    g = lane // 4
    t4 = lane % 4
    Q32 = cute.recast_tensor(Q, Uint32)
    K32 = cute.recast_tensor(Kt, Uint32)
    P32 = cute.recast_tensor(P, Uint32)
    V16 = cute.recast_tensor(V, Uint16)
    sc = raw_acc()
    for kk in cutlass.range_constexpr(HDIM // 16):
        a0 = Q32[bid, g, kk * 8 + t4]
        a1 = Q32[bid, g + 8, kk * 8 + t4]
        a2 = Q32[bid, g, kk * 8 + t4 + 4]
        a3 = Q32[bid, g + 8, kk * 8 + t4 + 4]
        for nt in cutlass.range_constexpr(KEYS // 8):
            raw_mma_into(sc, nt * 4, a0, a1, a2, a3, K32[bid, nt * 8 + g, kk * 8 + t4], K32[bid, nt * 8 + g, kk * 8 + t4 + 4])
    if dh == 0:
        for nt in cutlass.range_constexpr(KEYS // 8):
            for e in cutlass.range_constexpr(4):
                S[bid, g + (e >> 1) * 8, nt * 8 + 2 * t4 + (e & 1)] = sc[nt * 4 + e]
    acc = raw_acc()
    for ks in cutlass.range_constexpr(KEYS // 16):
        p0 = P32[bid, g, (2 * ks) * 4 + t4]
        p1 = P32[bid, g + 8, (2 * ks) * 4 + t4]
        p2 = P32[bid, g, (2 * ks + 1) * 4 + t4]
        p3 = P32[bid, g + 8, (2 * ks + 1) * 4 + t4]
        for nt in cutlass.range_constexpr(16):
            d = dh * 128 + nt * 8 + g
            k0r = ks * 16 + 2 * t4
            b0 = V16[bid, k0r, d].to(Uint32) | (V16[bid, k0r + 1, d].to(Uint32) << Uint32(16))
            b1 = V16[bid, k0r + 8, d].to(Uint32) | (V16[bid, k0r + 9, d].to(Uint32) << Uint32(16))
            raw_mma_into(acc, nt * 4, p0, p1, p2, p3, b0, b1)
    for nt in cutlass.range_constexpr(16):
        for e in cutlass.range_constexpr(4):
            O[bid, g + (e >> 1) * 8, dh * 128 + nt * 8 + 2 * t4 + (e & 1)] = acc[nt * 4 + e]


@cute.jit
def host_fa(Q: cute.Tensor, Kt: cute.Tensor, V: cute.Tensor, P: cute.Tensor, S: cute.Tensor, O: cute.Tensor,
            impl: cutlass.Constexpr):
    kern = fa_cute if impl == "cute" else fa_raw
    kern(Q, Kt, V, P, S, O).launch(grid=(Q.shape[0], 1, 1), block=(64, 1, 1), smem=FA_SMEM)


# ---------------------------------------------------------------- driver
SHAPES = {
    "gemm": [((NB, 64, K), BFloat16), ((NB, 256, K), BFloat16), ((NB, 64, 256), Float32)],
    "wsplit": [((NB, 64, K), BFloat16), ((NB, 128, K), BFloat16), ((NB, 64, 128), Float32)],
    "fa": [((NB, 16, HDIM), BFloat16), ((NB, KEYS, HDIM), BFloat16), ((NB, KEYS, HDIM), BFloat16),
           ((NB, 16, KEYS), BFloat16), ((NB, 16, KEYS), Float32), ((NB, 16, HDIM), Float32)],
}
HOSTS = {"gemm": host_gemm, "wsplit": host_wsplit, "fa": host_fa}
K0S = (0, 128, 256, 384)


def fakes(case):
    return [cute.runtime.make_fake_compact_tensor(t, s, assumed_align=16, stride_order=(2, 1, 0)) for s, t in SHAPES[case]]


def extra(case, k0=0):
    return (k0,) if case != "fa" else ()


def compile_case(case, impl, tensors=None):
    return cute.compile(HOSTS[case], *(tensors or fakes(case)), *extra(case), impl)


def load_ref():
    from torch.utils.cpp_extension import load
    import torch
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(*torch.cuda.get_device_capability()) \
            if torch.cuda.is_available() else "10.0"
    build = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"), "pi05_mma_ref")
    os.makedirs(build, exist_ok=True)
    return load("pi05_mma_ref", [os.path.join(HERE, "mma_ref.cu")], extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
                build_directory=build, verbose=False)


def make_inputs(case, seed):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda").to(torch.bfloat16)
    if case in ("gemm", "wsplit"):
        a = r(NB, 64, K)
        a[:, T:] = 0
        return [a, (r(NB, 256 if case == "gemm" else 128, K) * 0.25).to(torch.bfloat16)]
    p = torch.softmax(torch.randn(NB, 16, KEYS, generator=g, device="cuda"), dim=-1).to(torch.bfloat16)
    return [r(NB, 16, HDIM), r(NB, KEYS, HDIM), r(NB, KEYS, HDIM), p]


def stat(a, b):
    import torch
    ne = (a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32))
    return int(ne.sum()), float((a - b).abs().max())


def run_case(ref, case, impl, fn, seed):
    import torch
    from cutlass.cute.runtime import from_dlpack
    ins = make_inputs(case, seed)
    rows = []
    for k0 in (K0S if case != "fa" else (0,)):
        if case == "fa":
            rs, ro = torch.zeros(NB, 16, KEYS, device="cuda"), torch.zeros(NB, 16, HDIM, device="cuda")
            ref.fa(*ins, rs, ro)
            cs, co = torch.zeros_like(rs), torch.zeros_like(ro)
            fn(*[from_dlpack(t, assumed_align=16) for t in ins + [cs, co]])
            torch.cuda.synchronize()
            sm, sd = stat(rs, cs)
            om, od = stat(ro, co)
            rows.append({"k0": 0, "mismatches": sm + om, "max_abs_diff": max(sd, od), "S_mismatches": sm, "O_mismatches": om})
            continue
        n = LD[case]
        rr = torch.zeros(NB, T, n, device="cuda")
        getattr(ref, case)(*ins, rr, k0)
        cc = torch.zeros(NB, 64, n, device="cuda")
        fn(*[from_dlpack(t, assumed_align=16) for t in ins + [cc]], k0)
        torch.cuda.synchronize()
        m, d = stat(rr, cc[:, :T].contiguous())
        rows.append({"k0": k0, "mismatches": m, "max_abs_diff": d})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.compile_only:
        for case in SHAPES:
            for impl in ("cute", "raw"):
                t = time.time()
                compile_case(case, impl)
                print(f"compiled {case}/{impl} in {time.time() - t:.1f}s")
        return

    import torch
    from cutlass.cute.runtime import from_dlpack
    ref = load_ref()
    res = {"cases": {}}
    for case in SHAPES:
        for impl in ("cute", "raw"):
            name = f"{case}/{impl}"
            try:
                tensors = []
                for s, t in SHAPES[case]:
                    dt = torch.bfloat16 if t is BFloat16 else torch.float32
                    tensors.append(from_dlpack(torch.zeros(*s, dtype=dt, device="cuda"), assumed_align=16))
                kern = compile_case(case, impl, tensors)
                rows = []
                for seed in range(args.seeds):
                    for r in run_case(ref, case, impl, lambda *a: kern(*a), seed):
                        rows.append({"seed": seed, **r})
                res["cases"][name] = {"bit_equal": all(r["mismatches"] == 0 for r in rows),
                                      "mismatches": sum(r["mismatches"] for r in rows),
                                      "max_abs_diff": max(r["max_abs_diff"] for r in rows), "runs": rows}
            except Exception as ex:
                res["cases"][name] = {"error": f"{type(ex).__name__}: {ex}"[:2000]}
            c = res["cases"][name]
            print(name, {k: v for k, v in c.items() if k != "runs"})
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "mma.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
