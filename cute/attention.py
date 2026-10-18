"""Attention of the CuTe port: K/V prefill (ATT_PREFILL 2), phase C (FA-style split-key attention, PI05_ATT_FA=1) and phase C3
(chunk combine) of pi05_mk.cu:1262-1287, 1356-1731.

Unit = (ATT_ROWS = 64 query rows, ATT_KEYS keys). Smem carve (same bytes as CUDA), bf16 tensors with padded row strides:
Ks [ATT_KEYS][KS_STRIDE] at 0, Qs [64][KS_STRIDE], Vs [ATT_KEYS][VS_STRIDE], Mk uint8 [ATT_KEYS]. Warp (mt = warp & 3, dh = warp >> 2):
S = Q K^T (ldmatrix A / B), mask + scale + quad row max, p = bf16(exp(s - m)) kept as A fragments (C -> A regroup of the phase-0 `fa`
probe), l = fp32 sum, O_c = P V (ldmatrix.trans B)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp, warp

from . import mathx as X
from . import rowops as R
from . import sync as S
from .config import ATT_NRG, ATT_ROWS, CNT_ATT, CNT_STRIDE, HD, KS_STRIDE, NTHREADS, QD, QROWS, T, VS_STRIDE, Knobs
from .probes.mma_tile import tiled_mma, uni

NEG_INF = float("-inf")


@cute.jit
def att_smem(keys: cutlass.Constexpr):
    """Ks, Qs, Vs (bf16, padded strides) and Mk (uint8) over the dynamic smem, as the CUDA carve."""
    base = cute.arch.get_dyn_smem(BFloat16, alignment=16)
    sK = cute.make_tensor(base, cute.make_layout((keys, HD), stride=(KS_STRIDE, 1)))
    sQ = cute.make_tensor(base + keys * KS_STRIDE, cute.make_layout((ATT_ROWS, HD), stride=(KS_STRIDE, 1)))
    vptr = base + (keys + ATT_ROWS) * KS_STRIDE
    sV = cute.make_tensor(vptr, cute.make_layout((keys, HD), stride=(VS_STRIDE, 1)))
    mk = cute.make_tensor(cute.recast_ptr(vptr + keys * VS_STRIDE, dtype=Uint8), cute.make_layout(keys))
    return sK, sQ, sV, mk


@cute.jit
def mask_bytes(mk, prefix_mask, j0, P: cutlass.Constexpr, NK: cutlass.Constexpr, keys: cutlass.Constexpr):
    """Mk[tid] = j < P ? mask[j] != 0 : j < NK, for tid < ATT_KEYS."""
    tid, _, _ = cute.arch.thread_idx()
    if tid < keys:
        j = j0 + tid
        v = Uint8(0)
        if j < P:
            if prefix_mask[j] != Uint8(0):
                v = Uint8(1)
        elif j < NK:
            v = Uint8(1)
        mk[tid] = v


@cute.jit
def cp16(dst, addr, nbytes):
    """cp.async.cg 16 B global -> shared with src-size `nbytes` (0 = zero fill)."""
    cute.arch.cp_async_shared_global(dst, cute.make_ptr(BFloat16, addr, cute.AddressSpace.gmem, assumed_align=16), 16, "cg",
                                     cp_size=nbytes)


@cute.jit
def load_kv(sK, sV, pk, pv, ksuf, vsuf, j0, jlo, jhi, P: cutlass.Constexpr, NK: cutlass.Constexpr, prefill: cutlass.Constexpr):
    """K / V rows jlo..jhi-1 of the chunk starting at key j0: prefix row j < P, ksuf / vsuf row j - P for P <= j < NK, zero fill
    for j >= NK. prefill: rows P <= j < NK are skipped (B writes them later), others as above. pk / pv / ksuf / vsuf: int addresses."""
    tid, _, _ = cute.arch.thread_idx()
    for i in cutlass.range(jlo * 32 + tid, jhi * 32, NTHREADS, unroll=1):
        jj = i >> 5
        pc = i & 31
        j = j0 + jj
        ka = ksuf
        va = vsuf
        nb = Int32(0)
        row = Int32(0)
        if j < P:
            ka = pk
            va = pv
            row = j
            nb = Int32(16)
        elif j < NK:
            row = j - P
            if cutlass.const_expr(not prefill):
                nb = Int32(16)
        if cutlass.const_expr(prefill):
            if (j < P) | (j >= NK):
                cp16(sK.iterator + (jj * KS_STRIDE + pc * 8), ka + ((row * HD + pc * 8) * 2).to(Int64), nb)
                cp16(sV.iterator + (jj * VS_STRIDE + pc * 8), va + ((row * HD + pc * 8) * 2).to(Int64), nb)
        else:
            cp16(sK.iterator + (jj * KS_STRIDE + pc * 8), ka + ((row * HD + pc * 8) * 2).to(Int64), nb)
            cp16(sV.iterator + (jj * VS_STRIDE + pc * 8), va + ((row * HD + pc * 8) * 2).to(Int64), nb)


@cute.jit
def prefill(prefix_k, prefix_v, prefix_mask, v, l, knobs: cutlass.Constexpr):
    """ATT_PREFILL 2: blocks of the first wave load their first unit's prefix K/V rows, zero rows and mask before the B tasks."""
    bid, _, _ = cute.arch.block_idx()
    P = knobs.prefix
    NK = knobs.nk
    keys = knobs.att_keys
    if bid < ATT_NRG * knobs.att_nchunk:
        cute.arch.sync_threads()  # the A tile's smem is reused
        sK, sQ, sV, mk = att_smem(keys)
        j0 = (bid // ATT_NRG) * keys
        mask_bytes(mk, prefix_mask, j0, P, NK, keys)
        load_kv(sK, sV, prefix_k[l, None, None, None].iterator.toint(), prefix_v[l, None, None, None].iterator.toint(),
                v["ksuf"].iterator.toint(), v["vsuf"].iterator.toint(), j0, Int32(0), Int32(keys), P, NK, True)
        cute.arch.cp_async_commit_group()


@cute.jit
def fa_unit(sK, sQ, sV, mk, stats, opart, kc, r0, mt, dh, keys: cutlass.Constexpr, nchunk: cutlass.Constexpr):
    """One warp's (16 rows, ATT_KEYS keys) S / softmax-stats / P V of unit (r0, kc); the caller guarantees r0 + mt*16 < QROWS."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    g = lane // 4
    t4 = lane % 4
    nt_n = keys // 8
    tm = tiled_mma()
    tmm = tm.get_slice(lane)
    ld = lambda trans: cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=trans, num_matrices=4), BFloat16)
    sQw = cute.local_tile(sQ, (16, HD), (mt, 0))
    tSrQ = tmm.make_fragment_A(tmm.partition_A(sQw))
    tSrK = tmm.make_fragment_B(tmm.partition_B(sK))
    cq = cute.make_tiled_copy_A(ld(False), tm)
    ck = cute.make_tiled_copy_B(ld(False), tm)
    thq = cq.get_slice(lane)
    thk = ck.get_slice(lane)
    sq_s = thq.partition_S(sQw)
    sk_s = thk.partition_S(sK)
    rq = thq.retile(tSrQ)
    rk = thk.retile(tSrK)
    acc = cute.make_rmem_tensor(tmm.partition_shape_C((16, keys)), Float32)
    acc.fill(0.0)
    for kk in cutlass.range_constexpr(HD // 16):
        cute.copy(cq, sq_s[None, None, kk], rq[None, None, kk])
        cute.copy(ck, sk_s[None, None, kk], rk[None, None, kk])
        cute.gemm(tm, acc, tSrQ[None, None, kk], tSrK[None, None, kk], acc)

    # scale, mask, row max over the quad (rows g, g + 8)
    rmax = [Float32(NEG_INF), Float32(NEG_INF)]
    for nt in cutlass.range_constexpr(nt_n):
        for r8 in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(2):
                cc = nt * 8 + 2 * t4 + e
                sv = X.attn_score(acc[(e, r8), 0, nt], mk[cc] != Uint8(0))
                acc[(e, r8), 0, nt] = sv
                rmax[r8] = cute.arch.fmax(rmax[r8], sv)
    for r8 in cutlass.range_constexpr(2):
        for o in (1, 2):
            rmax[r8] = cute.arch.fmax(rmax[r8], cute.arch.shuffle_sync_bfly(rmax[r8], o))
    # p = bf16(exp(s - m)); l = fp32 sum in n-tile order
    rp = cute.make_fragment_like(acc, BFloat16)
    rsum = [Float32(0.0), Float32(0.0)]
    for r8 in cutlass.range_constexpr(2):
        msafe = Float32(cutlass.select_(rmax[r8] == X.f32(NEG_INF), X.f32(0.0), rmax[r8]))
        for nt in cutlass.range_constexpr(nt_n):
            for e in cutlass.range_constexpr(2):
                pr = X.rbf(X.expf(acc[(e, r8), 0, nt] - msafe))
                rsum[r8] = rsum[r8] + pr
                rp[(e, r8), 0, nt] = pr.to(BFloat16)
    for r8 in cutlass.range_constexpr(2):
        for o in (1, 2):
            rsum[r8] = rsum[r8] + cute.arch.shuffle_sync_bfly(rsum[r8], o)
    if (dh == 0) & (t4 == 0):
        st = R.rm_tensor((2, 2))
        for r8 in cutlass.range_constexpr(2):
            st[r8, 0] = rmax[r8]
            st[r8, 1] = rsum[r8]
        for r8 in cutlass.range_constexpr(2):
            row = r0 + mt * 16 + g + 8 * r8
            if row < QROWS:
                dst = cute.make_tensor(cute.make_ptr(Float32, stats.iterator.toint() + ((row * nchunk + kc) * 8).to(Int64),
                                                     cute.AddressSpace.gmem, assumed_align=8), cute.make_layout(2))
                cute.copy(uni(Float32, 64), st[r8, None], dst)

    # O_c = P V: P (C layout) regrouped as A fragments, V by ldmatrix.trans
    div = cute.logical_divide(rp.layout, (None, None, 2))
    pview = cute.make_layout(((div.shape[0], div.shape[2][0]), div.shape[1], div.shape[2][1]),
                             stride=((div.stride[0], div.stride[2][0]), div.stride[1], div.stride[2][1]))
    tOrP = cute.make_tensor(rp.iterator, pview)
    sVt = cute.local_tile(cute.make_tensor(sV.iterator, cute.make_layout((HD, keys), stride=(1, VS_STRIDE))), (128, keys), (dh, 0))
    tOrV = tmm.make_fragment_B(tmm.partition_B(sVt))
    cv = cute.make_tiled_copy_B(ld(True), tm)
    thv = cv.get_slice(lane)
    sv_s = thv.partition_S(sVt)
    rv = thv.retile(tOrV)
    accO = cute.make_rmem_tensor(tmm.partition_shape_C((16, 128)), Float32)
    accO.fill(0.0)
    for ks in cutlass.range_constexpr(keys // 16):
        cute.copy(cv, sv_s[None, None, ks], rv[None, None, ks])
        cute.gemm(tm, accO, tOrP[None, None, ks], tOrV[None, None, ks], accO)
    for r8 in cutlass.range_constexpr(2):
        row = r0 + mt * 16 + g + 8 * r8
        if row < QROWS:
            for nt in cutlass.range_constexpr(16):
                c = dh * 128 + nt * 8 + 2 * t4
                dst = cute.make_tensor(cute.make_ptr(Float32, opart.iterator.toint() + (((kc * QROWS + row) * HD + c) * 4).to(Int64),
                                                     cute.AddressSpace.gmem, assumed_align=8), cute.make_layout(2))
                cute.copy(uni(Float32, 64), accO[(None, r8), 0, nt], dst)


@cute.jit
def phase_c(v, prefix_k, prefix_v, prefix_mask, l, knobs: cutlass.Constexpr):
    """Phase C unit loop (pi05_mk.cu:1407-1684, FA branch); publishes each finished unit on its row group's counter (FUSE_CC3)."""
    from .config import FUSE_CC3
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    P = knobs.prefix
    NK = knobs.nk
    keys = knobs.att_keys
    nchunk = knobs.att_nchunk
    sK, sQ, sV, mk = att_smem(keys)
    pk = prefix_k[l, None, None, None].iterator.toint()
    pv = prefix_v[l, None, None, None].iterator.toint()
    ksuf = v["ksuf"].iterator.toint()
    vsuf = v["vsuf"].iterator.toint()
    qa = v["q"].iterator.toint()
    warp_id = tid // 32
    mt = warp_id & 3
    dh = warp_id >> 2
    for u in cutlass.range(bid, ATT_NRG * nchunk, nb, unroll=1):
        rg = u % ATT_NRG
        kc = u // ATT_NRG
        r0 = rg * ATT_ROWS
        j0 = kc * keys
        # the first unit of a block was prefilled: only rows P <= j < NK remain (and no mask write)
        pre = u == bid
        jlo = Int32(0)
        jhi = Int32(keys)
        if pre:
            jlo = Int32(P) - j0
            if jlo < 0:
                jlo = Int32(0)
            if jlo > keys:
                jlo = Int32(keys)
            jhi = Int32(NK) - j0
            if jhi < 0:
                jhi = Int32(0)
            if jhi > keys:
                jhi = Int32(keys)
        if not pre:
            mask_bytes(mk, prefix_mask, j0, P, NK, keys)
        load_kv(sK, sV, pk, pv, ksuf, vsuf, j0, jlo, jhi, P, NK, False)
        for i in cutlass.range(tid, ATT_ROWS * 32, NTHREADS, unroll=1):
            rr = i >> 5
            pc = i & 31
            r = r0 + rr
            rc = r
            nbq = Int32(16)
            if r >= QROWS:
                rc = Int32(QROWS - 1)
                nbq = Int32(0)
            cp16(sQ.iterator + (rr * KS_STRIDE + pc * 8), qa + ((rc * HD + pc * 8) * 2).to(Int64), nbq)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if r0 + mt * 16 < QROWS:
            fa_unit(sK, sQ, sV, mk, v["stats"], v["opart"], kc, r0, mt, dh, keys, nchunk)
        cute.arch.sync_threads()  # publish stats / opart of this unit
        if cutlass.const_expr(bool(knobs.fuse & FUSE_CC3)):
            if tid == 0:
                S.counter_publish(v["cnt"].iterator + (CNT_ATT + rg) * CNT_STRIDE)


@cute.jit
def phase_c3(v, it1, knobs: cutlass.Constexpr):
    """Phase C3 (pi05_mk.cu:1695-1731): per 4-row task, combine the key chunks: out = sum_c e^(m_c - M) o_c / sum_c e^(m_c - M) l_c."""
    from .config import FUSE_BACKOFF, FUSE_CC3
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    nchunk = knobs.att_nchunk
    cr = NTHREADS * 4 // HD
    cg2 = S.cg_atom(Float32, 64)
    cg4 = S.cg_atom(Float32, 128)
    stats = cute.make_tensor(v["stats"].iterator, cute.make_layout(QROWS * nchunk * 2))
    opart = cute.make_tensor(v["opart"].iterator, cute.make_layout(nchunk * QROWS * HD))
    attn_a = v["attn"].iterator.toint()
    for task in cutlass.range(nb - 1 - bid, QROWS // cr, nb, unroll=1):
        if cutlass.const_expr(bool(knobs.fuse & FUSE_CC3)):
            S.block_wait(v["cnt"].iterator + (CNT_ATT + task * cr // ATT_ROWS) * CNT_STRIDE, (it1 * nchunk).to(Uint32),
                         bool(knobs.fuse & FUSE_BACKOFF))
        i = (task * NTHREADS + tid) * 4
        r = i // HD
        wgt = R.rm_tensor((nchunk,))
        ls = R.rm_tensor((nchunk,))
        for c in cutlass.range_constexpr(nchunk):
            st = R.rm_tensor((2,))
            cute.copy(cg2, cute.local_tile(stats, (2,), (r * nchunk + c,)), st)
            wgt[c] = st[0]
            ls[c] = st[1]
        mx = wgt[0]
        for c in cutlass.range_constexpr(1, nchunk):
            mx = cute.arch.fmax(mx, wgt[c])
        for c in cutlass.range_constexpr(nchunk):
            wgt[c] = X.comb_w(wgt[c], mx)
        num = R.rm_tensor((4,))
        num.fill(0.0)
        den = Float32(0.0)
        for c0 in cutlass.range_constexpr(0, nchunk, 8):
            n_in = min(8, nchunk - c0)
            o = R.rm_tensor((8, 4))
            for c in cutlass.range_constexpr(n_in):
                cute.copy(cg4, cute.local_tile(opart, (4,), (((c0 + c) * QROWS * HD + i) // 4,)), o[c, None])
            for c in cutlass.range_constexpr(n_in):
                w = wgt[c0 + c]
                for e in cutlass.range_constexpr(4):
                    num[e] = X.fma(w, o[c, e], num[e])
                den = X.fma(w, ls[c0 + c], den)
        ob = R.rm_tensor((4,), BFloat16)
        for e in cutlass.range_constexpr(4):
            ob[e] = X.div(num[e], den).to(BFloat16)
        dst = cute.make_tensor(cute.make_ptr(BFloat16, attn_a + (i * 2).to(Int64), cute.AddressSpace.gmem, assumed_align=8),
                               cute.make_layout(4))
        cute.copy(uni(BFloat16, 64), ob, dst)
