"""`gemm_phase` of the CuTe port (pi05_mk.cu:415-609, PI05_GEMM_FRAG=2): the split-K weight-streaming GEMM engine.

part[s][t][n] = sum_{k in chunk s} A[t][k] * Wt[n][k]   (t < 50; K chunk s = blockIdx % S, KC = 128)
Virtual N over up to three weight tensors (boundaries n1, n2); each warp takes 32-column tasks (nblk blocks share a chunk).
Pieces, all expressed as CuTe layouts over the validated phase-0 tiled MMA (m16n8k16, K permutation KPERM):
  B   straight from global into registers, one 128-bit .nc load per (n-tile, slab) per lane (a regrouped view of the
      partition_B / make_fragment_B layouts, which are 8 contiguous bf16 per slab for this K permutation)
  A   smem tile in the fragment-interleaved layout a_layout(): one 128-bit LDS is one mma A operand
  acc 64 x 32 fp32 per warp, stored as float2 for rows < 50 only
Modes: plain, norm (adaRMS staged A: ssp + nmod), counter (red.release per finished 32-column tile), WPRE (deferred grid
barrier: arrive, first weight loads, wait, then A staging)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp

from . import iket_marks as IM
from . import mathx as X
from . import rowops as R
from . import sync as S
from .attention import cp16
from .config import AS_STRIDE, CNT_STRIDE, GEMM_SCRATCH, KC, NSSP, NTHREADS, NWARPS, T, W, WS_STRIDE
from .probes.mma_tile import kperm, tiled_mma, uni

PIECES = 64 * (KC // 8)  # 16 B pieces of the A tile
PER_THREAD = PIECES // NTHREADS  # 4


def a_layout():
    """Logical (m, k) -> element offset of the fragment-interleaved A tile (64 x 128 bf16, no padding):
    m = (g, r8, mt), k = (e, t4, k8, (m16 step, slab)); offset = ((mt*4 + sl)*2 + m16)*256 + (g*4 + t4)*8 + 4*k8 + 2*r8 + e,
    i.e. uint4 [mt][sl][half][g*4+t4] holding {row g, row g+8} x 2 words, exactly one mma A operand."""
    return cute.make_layout(((8, 2, 4), (2, 4, 2, (2, 4))), stride=((32, 2, 2048), (1, 8, 4, (256, 512))))


@cute.jit
def stage_a(A, k0, ssp, nmod, K: cutlass.Constexpr, norm: cutlass.Constexpr, sA32):
    """Load A[0:50, k0:k0+128] (rows >= 50 zero) with .cg, optionally apply the adaRMS norm, store into the smem tile."""
    tid, _, _ = cute.arch.thread_idx()
    q = tid // (KC // 8)
    pc = tid % (KC // 8)
    cg = S.cg_atom(BFloat16, 128)
    cg4 = S.cg_atom(Float32, 128)
    av = R.rm_tensor((PER_THREAD, 8), BFloat16)
    a_flat = cute.make_tensor(A.iterator, cute.make_layout(T * K))
    for x in cutlass.range_constexpr(PER_THREAD):
        r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7)
        if r < T:
            cute.copy(cg, cute.local_tile(a_flat, (8,), ((r * K + k0) // 8 + pc,)), av[x, None])
        else:
            for i in cutlass.range_constexpr(8):
                av[x, i] = Float32(0.0).to(BFloat16)
    if cutlass.const_expr(norm):
        # ss part j of row r sits in lane j of the half-warp (j = pc); the 16 parts are summed by xor 8, 4, 2, 1 and
        # broadcast from the half-warp's first lane. scale/shift are the same 8 columns for every row of this thread.
        ssv = []
        for x in cutlass.range_constexpr(PER_THREAD):
            r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7)
            v = Float32(0.0)
            if r < T:
                v = S.ld_cg_f32(ssp.iterator + (r * NSSP + pc))
            ssv.append(v)
        nm = cute.make_tensor(nmod.iterator, cute.make_layout(2 * W))
        sc8 = R.rm_tensor((8,))
        sh8 = R.rm_tensor((8,))
        for i in cutlass.range_constexpr(2):
            cute.copy(cg4, cute.local_tile(nm, (4,), (k0 // 4 + pc * 2 + i,)), cute.local_tile(sc8, (4,), (i,)))
            cute.copy(cg4, cute.local_tile(nm, (4,), (W // 4 + k0 // 4 + pc * 2 + i,)), cute.local_tile(sh8, (4,), (i,)))
        for i in cutlass.range_constexpr(8):
            sc8[i] = X.fadd_rn(X.f32(1.0), sc8[i])
        hbase = tid & 16
        for x in cutlass.range_constexpr(PER_THREAD):
            r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7)
            v = ssv[x]
            for o in (8, 4, 2, 1):
                v = v + cute.arch.shuffle_sync_bfly(v, o)
            v = cute.arch.shuffle_sync(v, hbase)
            rs = X.ada_rs(v)
            if r < T:
                for i in cutlass.range_constexpr(8):
                    o = X.fadd_rn(X.fmul_rn(X.fmul_rn(av[x, i].to(Float32), rs), sc8[i]), sh8[i])
                    av[x, i] = o.to(BFloat16)
    # repack the pieces (rows r, r + 8 of one thread) into mma A operands and store each as one 128-bit STS
    av32 = cute.recast_tensor(av, Uint32)
    sm_atom = cute.make_copy_atom(CopyUniversalOp(), Uint32, num_bits_per_copy=128)
    ov = R.rm_tensor((2, 4), Uint32)
    for j in cutlass.range_constexpr(PER_THREAD // 2):
        for m in cutlass.range_constexpr(2):
            for k8 in cutlass.range_constexpr(2):
                for r8 in cutlass.range_constexpr(2):
                    ov[m, 2 * k8 + r8] = av32[2 * j + r8, 2 * m + k8]
        mt = j * 2 + (q >> 3)
        lane8 = (q & 7) * 4 + (pc & 3)
        for m in cutlass.range_constexpr(2):
            idx = ((mt * 4 + (pc >> 2)) * 2 + m) * 32 + lane8
            cute.copy(sm_atom, ov[m, None], cute.local_tile(sA32, (1, 4), (idx, 0))[0, None])


@cute.jit
def load_w(w0, w1, w2, n1: cutlass.Constexpr, n2: cutlass.Constexpr, K: cutlass.Constexpr, ng, k0, tmm, rB):
    """This lane's B operands of 32-column task ng: 16 x 128-bit .nc loads into rB (16 regs x 4)."""
    n0 = ng * 32
    base = w0.iterator.toint()
    nl = n0
    if n0 >= n2:
        base = w2.iterator.toint()
        nl = n0 - n2
    elif n0 >= n1:
        base = w1.iterator.toint()
        nl = n0 - n1
    ptr = cute.make_ptr(BFloat16, base + ((nl * K + k0) * 2).to(Int64), cute.AddressSpace.gmem, assumed_align=16)
    gB = cute.make_tensor(ptr, cute.make_layout((32, KC), stride=(K, 1)))
    tB = tmm.partition_B(cute.composition(gB, (None, kperm())))  # ((2,2),4,(2,4)) : ((1,2),8*K,(4,32))
    # 8 contiguous bf16 per (n-tile, slab): (e, h, m16) -> one 128-bit copy
    src = cute.make_tensor(cute.make_ptr(BFloat16, tB.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
                           cute.make_layout(((2, 2, 2), 4, 4), stride=((1, 2, 4), 8 * K, 32)))
    dst = cute.make_tensor(rB.iterator, cute.make_layout(((2, 2, 2), 4, 4), stride=((1, 2, 4), 32, 8)))
    cute.copy(S.nc_atom(BFloat16, 128), src, dst)


@cute.jit
def gemm_phase(A, w0, w1, w2, part, bar, pend, target, cnt, ssp, nmod,
               K: cutlass.Constexpr, n1: cutlass.Constexpr, n2: cutlass.Constexpr, N: cutlass.Constexpr,
               cdiv: cutlass.Constexpr, cmod: cutlass.Constexpr, norm: cutlass.Constexpr, counted: cutlass.Constexpr,
               wpre: cutlass.Constexpr, load_hoist: cutlass.Constexpr):
    """A: (T, K) bf16 tensor; w0/w1/w2: (rows, K) bf16 weight tensors (virtual N = rows of w0 up to n1, w1 up to n2, w2 up to N);
    part: f32 workspace tensor; bar: u32 workspace `bar` tensor, pend/target: deferred barrier (pend runtime Boolean, wpre
    Constexpr says the call site may defer one); cnt: u32 workspace `cnt` tensor (counted); ssp (T, NSSP) f32 and nmod (f32,
    scale at nmod[k], shift at nmod[W + k]) for norm mode. load_hoist (Knobs.load_hoist / PI05_LOAD_HOIST): 1 = the next tile's
    weight load goes right after the tile's last MMA, before the partial stores (rB is dead there; same math and store order),
    0 = after the stores."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    warp = tid // 32
    g = lane // 4
    t4 = lane % 4
    NG = N // 32
    NCH = K // KC
    s = bid % NCH
    slot = bid // NCH
    nblk = (nb - s + NCH - 1) // NCH
    k0 = s * KC
    ng = slot * NWARPS + warp
    ng_step = nblk * NWARPS

    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    smem = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sA = cute.make_tensor(smem, a_layout())
    sA32 = cute.make_tensor(cute.recast_ptr(smem, dtype=Uint32), cute.make_layout((1024, 4), stride=(4, 1)))
    gB0 = cute.make_tensor(w0.iterator, cute.make_layout((32, KC), stride=(K, 1)))
    rB = tmm.make_fragment_B(tmm.partition_B(cute.composition(gB0, (None, kperm()))))
    rA = tmm.make_fragment_A(tmm.partition_A(sA))
    ca = cute.make_tiled_copy_A(uni(BFloat16, 128), tm)
    tha = ca.get_slice(lane)
    sAp = tha.partition_S(sA)
    rAv = tha.retile(rA)
    if cutlass.const_expr(IM.GEMM):  # PI05_IKET_GEMM: this warp's scratch word (gmark_dep) sits at the end of the dynamic smem
        gscr = cute.recast_ptr(smem, dtype=Uint32) + (cute.arch.get_dyn_smem_size() - GEMM_SCRATCH) // 4 + warp
    IM.gmark("g_0")

    if cutlass.const_expr(wpre):
        if pend:
            S.wpre_arrive(bar.iterator)
            if ng < NG:
                load_w(w0, w1, w2, n1, n2, K, ng, k0, tmm, rB)
            IM.gmark("g_wiss")
            S.wpre_wait(bar.iterator, target)
        else:
            if ng < NG:
                load_w(w0, w1, w2, n1, n2, K, ng, k0, tmm, rB)
            IM.gmark("g_wiss")
    else:
        if ng < NG:
            load_w(w0, w1, w2, n1, n2, K, ng, k0, tmm, rB)
        IM.gmark("g_wiss")
    stage_a(A, k0, ssp, nmod, K, norm, sA32)
    IM.gmark("g_adone")
    cute.arch.sync_threads()
    IM.gmark("g_sync")

    part_base = part.iterator.toint()
    for ngi in cutlass.range(ng, NG, ng_step, unroll=1):
        acc = cute.make_rmem_tensor(tmm.partition_shape_C((64, 32)), Float32)
        acc.fill(0.0)
        if cutlass.const_expr(IM.GEMM):
            IM.gmark_dep("g_wrdy", rB, gscr)
        for kb in cutlass.range_constexpr(8):
            cute.copy(ca, sAp[None, None, kb], rAv[None, None, kb])
            cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)
        IM.gmark_loop("g_mma")
        n0 = ngi * 32
        if cutlass.const_expr(load_hoist):
            if ngi + ng_step < NG:
                load_w(w0, w1, w2, n1, n2, K, ngi + ng_step, k0, tmm, rB)
                IM.gmark_loop("g_wnext")
        for mt in cutlass.range_constexpr(4):
            for r8 in cutlass.range_constexpr(2):
                row = mt * 16 + g + 8 * r8
                if row < T:
                    for nt in cutlass.range_constexpr(4):
                        addr = part_base + (((s * T + row) * N + n0 + nt * 8 + 2 * t4) * 4).to(Int64)
                        dst = cute.make_tensor(cute.make_ptr(Float32, addr, cute.AddressSpace.gmem, assumed_align=8),
                                               cute.make_layout(2))
                        cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst)
        IM.gmark_loop("g_st")
        if cutlass.const_expr(not load_hoist):
            if ngi + ng_step < NG:
                load_w(w0, w1, w2, n1, n2, K, ngi + ng_step, k0, tmm, rB)
                IM.gmark_loop("g_wnext")
        if cutlass.const_expr(counted):
            cute.arch.sync_warp()
            if lane == 0:
                S.counter_publish(cnt.iterator + ((n0 // cdiv) % cmod) * CNT_STRIDE)


# ------------------------------------------------------------------------------------------- gemm_phase_wsplit (D, o_proj)
@cute.jit
def ws_stage_a(A, k0, K: cutlass.Constexpr, sAi):
    """A[0:50, k0:k0+128] (rows >= 50 zero) into the fragment-interleaved tile with the PLAIN mapping of the CUDA wsplit
    (piece i = tid + 256 x: row i / 16, 16 B column piece i % 16), four 32-bit STS per piece (sAi: flat u32 view of the tile)."""
    tid, _, _ = cute.arch.thread_idx()
    cg = S.cg_atom(BFloat16, 128)
    a_flat = cute.make_tensor(A.iterator, cute.make_layout(T * K))
    av = R.rm_tensor((PER_THREAD, 8), BFloat16)
    for x in cutlass.range_constexpr(PER_THREAD):
        i = tid + x * NTHREADS
        r = i // (KC // 8)
        pc = i % (KC // 8)
        if r < T:
            cute.copy(cg, cute.local_tile(a_flat, (8,), ((r * K + k0) // 8 + pc,)), av[x, None])
        else:
            for e in cutlass.range_constexpr(8):
                av[x, e] = Float32(0.0).to(BFloat16)
    av32 = cute.recast_tensor(av, Uint32)
    for x in cutlass.range_constexpr(PER_THREAD):
        i = tid + x * NTHREADS
        r = i // (KC // 8)
        pc = i % (KC // 8)
        base = (((r >> 4) * 4 + (pc >> 2)) * 2 * 32 + (r & 7) * 4 + (pc & 3)) * 4 + ((r >> 3) & 1)
        sAi[base] = av32[x, 0]
        sAi[base + 2] = av32[x, 1]
        sAi[base + 32 * 4] = av32[x, 2]
        sAi[base + 32 * 4 + 2] = av32[x, 3]


@cute.jit
def ws_load_w(wt, wbuf, ngi, k0, wi, g, t4, lane, K: cutlass.Constexpr, NSL: cutlass.Constexpr):
    """cp.async the weights of 32-column task ngi into this warp's smem buffer [j][nt][lane] (16 B each), then commit."""
    n0 = ngi * 32
    base = wt.iterator.toint()
    for j in cutlass.range_constexpr(NSL):
        for nt in cutlass.range_constexpr(4):
            addr = base + (((n0 + nt * 8 + g) * K + k0 + (wi * NSL + j) * 32 + t4 * 8) * 2).to(Int64)
            cp16(wbuf.iterator + ((j * 4 + nt) * 32 + lane) * 8, addr, 16)
    cute.arch.cp_async_commit_group()


@cute.jit
def gemm_phase_wsplit(A, wt, part, bar, target, K: cutlass.Constexpr, N: cutlass.Constexpr, WPT: cutlass.Constexpr,
                      wpre: cutlass.Constexpr):
    """pi05_mk.cu gemm_phase_wsplit<WPT>: plain-mode gemm_phase whose 128-k chunk is split over WPT warps of a group (warp wi
    runs k slabs [wi NSL, (wi+1) NSL), NSL = 4 / WPT). Weights arrive by cp.async into per-warp smem buffers, the A tile is
    staged with the plain mapping, the WPT fp32 partial tiles (50 rows, stride WS_STRIDE) are summed in warp order through smem
    (named barrier 1 + group, 32 WPT threads) and the group splits the row-quad stores. wpre: the grid barrier of target
    `target` is deferred into this call (arrive, first weight loads, wait)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    warp = tid // 32
    g = lane // 4
    t4 = lane % 4
    NGRP = NWARPS // WPT
    NSL = 4 // WPT
    gi = warp // WPT
    wi = warp % WPT
    NG = N // 32
    NCH = K // KC
    s = bid % NCH
    slot = bid // NCH
    nblk = (nb - s + NCH - 1) // NCH
    k0 = s * KC
    ng = slot * NGRP + gi
    ng_step = nblk * NGRP

    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    smem = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sA = cute.make_tensor(smem, a_layout())
    sAi = cute.make_tensor(cute.recast_ptr(smem, dtype=Uint32), cute.make_layout(64 * KC // 2))
    red = cute.recast_ptr(smem + 64 * AS_STRIDE, dtype=Float32)  # [warp][T][WS_STRIDE] partials, then the weight buffers
    sMine = cute.make_tensor(red, cute.make_layout((NWARPS, T, WS_STRIDE), stride=(T * WS_STRIDE, WS_STRIDE, 1)))
    mine = sMine[warp, None, None]
    grp = cute.make_tensor(red, cute.make_layout((NWARPS * T, WS_STRIDE), stride=(WS_STRIDE, 1)))  # row gi WPT T + w T + r
    wsm = cute.make_tensor(cute.recast_ptr(red + NWARPS * T * WS_STRIDE, dtype=BFloat16),
                           cute.make_layout((NWARPS, 32, (2, 2, 2), 4, NSL),
                                            stride=(NSL * 4 * 32 * 8, 8, (1, 2, 4), 32 * 8, 4 * 32 * 8)))
    wbuf = cute.make_tensor(wsm.iterator + warp * (NSL * 4 * 32 * 8), cute.make_layout(NSL * 4 * 32 * 8))
    wsrc = wsm[warp, lane, None, None, None]  # this lane's ((e, h, m16), nt, j) B operands in the weight buffer

    gB0 = cute.make_tensor(wt.iterator, cute.make_layout((32, KC), stride=(K, 1)))
    rB = tmm.make_fragment_B(tmm.partition_B(cute.composition(gB0, (None, kperm()))))
    rBd = cute.make_tensor(rB.iterator, cute.make_layout(((2, 2, 2), 4, NSL), stride=((1, 2, 4), 32, 8)))
    rA = tmm.make_fragment_A(tmm.partition_A(sA))
    ca = cute.make_tiled_copy_A(uni(BFloat16, 128), tm)
    tha = ca.get_slice(lane)
    sAp = tha.partition_S(sA)
    rAv = tha.retile(rA)

    if cutlass.const_expr(wpre):
        S.wpre_arrive(bar.iterator)  # the weight buffers are free after the CTA sync
        if ng < NG:
            ws_load_w(wt, wbuf, ng, k0, wi, g, t4, lane, K, NSL)
        S.wpre_wait(bar.iterator, target)
    else:
        if ng < NG:
            ws_load_w(wt, wbuf, ng, k0, wi, g, t4, lane, K, NSL)
    ws_stage_a(A, k0, K, sAi)
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    part_base = part.iterator.toint()
    rr = lane // 8
    c4 = (lane % 8) * 4
    for ngi in cutlass.range(ng, NG, ng_step, unroll=1):  # uniform within a group
        acc = cute.make_rmem_tensor(tmm.partition_shape_C((64, 32)), Float32)
        acc.fill(0.0)
        cute.copy(uni(BFloat16, 128), wsrc, rBd)
        for j in cutlass.range_constexpr(NSL):
            for m in cutlass.range_constexpr(2):
                kbd = 2 * (wi * NSL + j) + m  # logical k16 step of this warp's slab (runtime), register slot 2 j + m
                cute.copy(ca, sAp[None, None, kbd], rAv[None, None, 2 * j + m])
                cute.gemm(tm, acc, rA[None, None, 2 * j + m], rB[None, None, 2 * j + m], acc)
        n0 = ngi * 32
        cute.arch.sync_warp()
        if ngi + ng_step < NG:
            ws_load_w(wt, wbuf, ngi + ng_step, k0, wi, g, t4, lane, K, NSL)  # own buffer, already read into registers
        for mt in cutlass.range_constexpr(4):
            for r8 in cutlass.range_constexpr(2):
                row = mt * 16 + g + 8 * r8
                if row < T:
                    for nt in cutlass.range_constexpr(4):
                        dst = cute.local_tile(mine, (1, 2), (row, nt * 4 + t4))
                        cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst[0, None])
        cute.arch.barrier(barrier_id=1 + gi, number_of_threads=32 * WPT)
        for q in cutlass.range(wi, (T + 3) // 4, WPT, unroll=1):  # row quad q: rows 4q .. 4q + 3
            r = q * 4 + rr
            if r < T:
                v4 = R.rm_tensor((4,))
                u4 = R.rm_tensor((4,))
                cute.copy(uni(Float32, 128), cute.local_tile(grp, (1, 4), (gi * WPT * T + r, c4 // 4))[0, None], v4)
                for w in cutlass.range_constexpr(1, WPT):
                    cute.copy(uni(Float32, 128), cute.local_tile(grp, (1, 4), (gi * WPT * T + w * T + r, c4 // 4))[0, None], u4)
                    for e in cutlass.range_constexpr(4):
                        v4[e] = v4[e] + u4[e]
                addr = part_base + (((s * T + r) * N + n0 + c4) * 4).to(Int64)
                out = cute.make_tensor(cute.make_ptr(Float32, addr, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(4))
                cute.copy(uni(Float32, 128), v4, out)
        if ngi + ng_step < NG:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier(barrier_id=1 + gi, number_of_threads=32 * WPT)
