"""FUSE_ABL (fuse bit 32): the QKV GEMM (A), the grid barrier and the reduce / RoPE / store phase (B) as ONE phase without a barrier.

80 units; unit u: head hh = u // 8 (0..7 Q, 8 K, 9 V), pair p = u % 8. Its 32 columns of the virtual N = 2560 space are
hb + 16 p .. +15 (MMA tile columns 0..15) and hb + 128 + 16 p .. +15 (columns 16..31), hb = 256 hh: a RoPE pair (j, j + 128) never
leaves the unit. Warp w of the unit's CTA owns K chunk w (same 32-column MMA tile and register fragment as gemm_phase), all 8 A
chunks are staged (adaRMS-normalised exactly like stage_a) into 8 private smem tiles, the 8 fp32 partials go through smem, and
the sum (s = 0 .. 7 from +0.0), rbf, RoPE and the bf16 stores are the arithmetic of rowops.reduce_part4 / kernel.phase_b.
Unit u runs on CTA nb - 1 - u (the CTAs phase_b uses); CTAs without a unit only take the deferred WPRE barrier step."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp

from . import mathx as X
from . import rowops as R
from . import sync as S
from .config import HD, KC, KVD, NSSP, NTHREADS, NWARPS, QD, T, W
from .gemm import PER_THREAD, a_layout
from .probes.mma_tile import kperm, tiled_mma, uni

NUNITS = 80  # 8 Q heads x 8 + K 8 + V 8
TILE_ELEMS = 64 * KC  # bf16 elements of one A tile (16 KB)
PB_STRIDE = 40  # fp32 partial buffer row stride (conflict-free float2 stores; [warp][T][PB_STRIDE])
ABL_SMEM = 8 * TILE_ELEMS * 2  # 8 A tiles; the partial buffer (8 x T x PB_STRIDE x 4 = 64000 B) reuses them


def unit_columns(u):
    """Pure python: the 32 virtual columns of unit u in MMA tile order (tile column n -> virtual column)."""
    hh, p = u // 8, u % 8
    hb = hh * HD
    return [hb + 16 * p + n for n in range(16)] + [hb + HD // 2 + 16 * p + n for n in range(16)]


@cute.jit
def stage_a8(A, ssp, nmod, K: cutlass.Constexpr, smem):
    """stage_a(A, k0 = 128 c, ssp, nmod, K, True, tile c) for all 8 chunks c, bit-identical per element: per-row rs once (it does not
    depend on k0), global loads one chunk ahead. Tile c = bf16 elements [c TILE_ELEMS, (c + 1) TILE_ELEMS) of smem."""
    tid, _, _ = cute.arch.thread_idx()
    q = tid // (KC // 8)
    pc = tid % (KC // 8)
    cg = S.cg_atom(BFloat16, 128)
    cg4 = S.cg_atom(Float32, 128)
    a_flat = cute.make_tensor(A.iterator, cute.make_layout(T * K))
    nm = cute.make_tensor(nmod.iterator, cute.make_layout(2 * W))
    NCH = K // KC

    def rows(x):
        return (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7)

    def load_chunk(c):
        k0 = c * KC
        av = R.rm_tensor((PER_THREAD, 8), BFloat16)
        for x in cutlass.range_constexpr(PER_THREAD):
            r = rows(x)
            if r < T:
                cute.copy(cg, cute.local_tile(a_flat, (8,), ((r * K + k0) // 8 + pc,)), av[x, None])
            else:
                for i in cutlass.range_constexpr(8):
                    av[x, i] = Float32(0.0).to(BFloat16)
        sc8 = R.rm_tensor((8,))
        sh8 = R.rm_tensor((8,))
        for i in cutlass.range_constexpr(2):
            cute.copy(cg4, cute.local_tile(nm, (4,), (k0 // 4 + pc * 2 + i,)), cute.local_tile(sc8, (4,), (i,)))
            cute.copy(cg4, cute.local_tile(nm, (4,), (W // 4 + k0 // 4 + pc * 2 + i,)), cute.local_tile(sh8, (4,), (i,)))
        return av, sc8, sh8

    cur = load_chunk(0)
    ssv = []
    for x in cutlass.range_constexpr(PER_THREAD):
        r = rows(x)
        v = Float32(0.0)
        if r < T:
            v = S.ld_cg_f32(ssp.iterator + (r * NSSP + pc))
        ssv.append(v)
    nxt = load_chunk(1)
    hbase = tid & 16
    rss = []
    for x in cutlass.range_constexpr(PER_THREAD):
        v = ssv[x]
        for o in (8, 4, 2, 1):
            v = v + cute.arch.shuffle_sync_bfly(v, o)
        v = cute.arch.shuffle_sync(v, hbase)
        rss.append(X.ada_rs(v))
    sm_atom = cute.make_copy_atom(CopyUniversalOp(), Uint32, num_bits_per_copy=128)
    for c in cutlass.range_constexpr(NCH):
        av, sc8, sh8 = cur
        cur = nxt
        for i in cutlass.range_constexpr(8):
            sc8[i] = X.fadd_rn(X.f32(1.0), sc8[i])
        for x in cutlass.range_constexpr(PER_THREAD):
            r = rows(x)
            if r < T:
                for i in cutlass.range_constexpr(8):
                    o = X.fadd_rn(X.fmul_rn(X.fmul_rn(av[x, i].to(Float32), rss[x]), sc8[i]), sh8[i])
                    av[x, i] = o.to(BFloat16)
        sA32 = cute.make_tensor(cute.recast_ptr(smem + c * TILE_ELEMS, dtype=Uint32), cute.make_layout((1024, 4), stride=(4, 1)))
        av32 = cute.recast_tensor(av, Uint32)
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
        if cutlass.const_expr(c + 2 < NCH):  # chunk c + 1 stays in flight while c + 2 is requested (registers of 2 chunks)
            nxt = load_chunk(c + 2)


@cute.jit
def load_w_unit(w0, w1, w2, n1: cutlass.Constexpr, n2: cutlass.Constexpr, K: cutlass.Constexpr, u, k0, tmm, rB):
    """load_w for the 32 gathered rows of unit u: tile columns 0..15 = rows nl .. nl + 15, 16..31 = rows nl + 128 .. + 143
    (nl = row of column 256 hh + 16 p in its weight tensor). Same fragment rB, same kperm, same 16 x 128-bit .nc loads."""
    n0 = (u // 8) * HD + (u % 8) * 16
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
    tB = tmm.partition_B(cute.composition(gB, (None, kperm())))  # lane offset g K + ...; row of nt: 8 b K + 128 a K, nt = b + 2 a
    src = cute.make_tensor(cute.make_ptr(BFloat16, tB.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16),
                           cute.make_layout(((2, 2, 2), (2, 2), 4), stride=((1, 2, 4), (8 * K, (HD // 2) * K), 32)))
    dst = cute.make_tensor(rB.iterator, cute.make_layout(((2, 2, 2), (2, 2), 4), stride=((1, 2, 4), (32, 64), 8)))
    cute.copy(S.nc_atom(BFloat16, 128), src, dst)


@cute.jit
def abl_phase(A, w0, w1, w2, bar, pend, target, ssp, nmod, rope, q_out, k_out, v_out, K: cutlass.Constexpr,
              n1: cutlass.Constexpr, n2: cutlass.Constexpr, wpre: cutlass.Constexpr):
    """The fused A + B phase (see the module docstring). A: (T, K) bf16 (v["h"]); rope: the (T * 128 * 2) f32 table; q_out /
    k_out / v_out: the bf16 workspace tensors q (T, QD), ksuf (T, KVD), vsuf (T, KVD)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    warp = tid // 32
    g = lane // 4
    t4 = lane % 4
    k0 = warp * KC
    u0 = nb - 1 - bid

    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    smem = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sA = cute.make_tensor(smem, a_layout())
    sAw = cute.make_tensor(cute.make_ptr(BFloat16, smem.toint() + (warp * (TILE_ELEMS * 2)).to(Int64), cute.AddressSpace.smem, assumed_align=16),
                           a_layout())  # this warp's chunk tile
    gB0 = cute.make_tensor(w0.iterator, cute.make_layout((32, KC), stride=(K, 1)))
    rB = tmm.make_fragment_B(tmm.partition_B(cute.composition(gB0, (None, kperm()))))
    rA = tmm.make_fragment_A(tmm.partition_A(sA))
    ca = cute.make_tiled_copy_A(uni(BFloat16, 128), tm)
    tha = ca.get_slice(lane)
    sAp = tha.partition_S(sAw)
    rAv = tha.retile(rA)
    pbuf = cute.make_tensor(cute.recast_ptr(smem, dtype=Float32), cute.make_layout((NWARPS, T, PB_STRIDE), stride=(T * PB_STRIDE, PB_STRIDE, 1)))
    pgrp = cute.make_tensor(cute.recast_ptr(smem, dtype=Float32), cute.make_layout((NWARPS * T, PB_STRIDE), stride=(PB_STRIDE, 1)))
    cg4 = S.cg_atom(Float32, 128)
    qa = q_out.iterator.toint()
    ka = k_out.iterator.toint()
    va = v_out.iterator.toint()

    if cutlass.const_expr(wpre):
        if pend:
            S.wpre_arrive(bar.iterator)
            if u0 < NUNITS:
                load_w_unit(w0, w1, w2, n1, n2, K, u0, k0, tmm, rB)
            S.wpre_wait(bar.iterator, target)
        else:
            if u0 < NUNITS:
                load_w_unit(w0, w1, w2, n1, n2, K, u0, k0, tmm, rB)
    else:
        if u0 < NUNITS:
            load_w_unit(w0, w1, w2, n1, n2, K, u0, k0, tmm, rB)

    for u in cutlass.range(u0, NUNITS, nb, unroll=1):
        hh = u // 8
        pp = u % 8
        stage_a8(A, ssp, nmod, K, smem)
        cute.arch.sync_threads()
        acc = cute.make_rmem_tensor(tmm.partition_shape_C((64, 32)), Float32)
        acc.fill(0.0)
        for kb in cutlass.range_constexpr(8):
            cute.copy(ca, sAp[None, None, kb], rAv[None, None, kb])
            cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], acc)
        cute.arch.sync_threads()  # every warp is done reading the A tiles (the partial buffer reuses them)
        mine = pbuf[warp, None, None]
        for mt in cutlass.range_constexpr(4):
            for r8 in cutlass.range_constexpr(2):
                row = mt * 16 + g + 8 * r8
                if row < T:
                    for nt in cutlass.range_constexpr(4):
                        pdst = cute.local_tile(mine, (1, 2), (row, nt * 4 + t4))
                        cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], pdst[0, None])
        cute.arch.sync_threads()
        # reduce + RoPE + store: task = (row t, quad of 4 first-half columns); 200 tasks on 256 threads
        t = tid // 4
        qd = tid % 4
        j = pp * 16 + qd * 4  # column within the head (0..127); the partner is j + 128
        if tid < T * 4:
            x1r = R.rm_tensor((8, 4))
            x2r = R.rm_tensor((8, 4))
            for s in cutlass.range_constexpr(NWARPS):
                cute.copy(uni(Float32, 128), cute.local_tile(pgrp, (1, 4), (s * T + t, qd))[0, None], x1r[s, None])
                cute.copy(uni(Float32, 128), cute.local_tile(pgrp, (1, 4), (s * T + t, 4 + qd))[0, None], x2r[s, None])
            x1 = R.rm_tensor((4,))
            x2 = R.rm_tensor((4,))
            o1 = R.rm_tensor((4,))
            o2 = R.rm_tensor((4,))
            for e in cutlass.range_constexpr(4):
                a1 = Float32(0.0)
                a2 = Float32(0.0)
                for s in cutlass.range_constexpr(NWARPS):
                    a1 = a1 + x1r[s, e]
                    a2 = a2 + x2r[s, e]
                x1[e] = X.rbf(a1)
                x2[e] = X.rbf(a2)
            if hh < 9:
                rp = R.rm_tensor((8,))
                for i in cutlass.range_constexpr(2):
                    cute.copy(cg4, cute.local_tile(rope, (4,), ((t * (HD // 2) + j) // 2 + i,)), cute.local_tile(rp, (4,), (i,)))
                for e in cutlass.range_constexpr(4):
                    o1[e], o2[e] = X.rope_apply(x1[e], x2[e], rp[2 * e], rp[2 * e + 1])
            else:
                for e in cutlass.range_constexpr(4):
                    o1[e] = x1[e]
                    o2[e] = x2[e]
            dst = qa + ((t * QD + hh * HD) * 2).to(Int64)
            if hh == 8:
                dst = ka + ((t * KVD) * 2).to(Int64)
            elif hh == 9:
                dst = va + ((t * KVD) * 2).to(Int64)
            ob = R.rm_tensor((2, 4), BFloat16)
            for e in cutlass.range_constexpr(4):
                ob[0, e] = o1[e].to(BFloat16)
                ob[1, e] = o2[e].to(BFloat16)
            for hf in cutlass.range_constexpr(2):
                out = cute.make_tensor(cute.make_ptr(BFloat16, dst + ((j + hf * (HD // 2)) * 2).to(Int64), cute.AddressSpace.gmem,
                                                     assumed_align=8), cute.make_layout(4))
                cute.copy(cute.make_copy_atom(CopyUniversalOp(), BFloat16, num_bits_per_copy=64), ob[hf, None], out)
        if u + nb < NUNITS:  # another unit on this CTA (nb < 80): the partial buffer is read, then the next weights
            cute.arch.sync_threads()
            load_w_unit(w0, w1, w2, n1, n2, K, u + nb, k0, tmm, rB)
