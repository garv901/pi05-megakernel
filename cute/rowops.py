"""Reusable device ops of the CuTe port (warp / block reductions, FP32 condition matvec, action_in task).
Summation orders and fma chains follow pi05_mk.cu exactly; loads of read-only weights are `.nc`, loads of data produced
inside the launch are `.cg`."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp

from . import mathx as X
from . import sync as S
from .config import ADIM, NSSP, NSTEP, NTHREADS, NWARPS, T, W


def warp_sum(v):
    """xor butterfly 16, 8, 4, 2, 1 (every lane gets the sum)."""
    for o in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, o)
    return v


@cute.jit
def block_sum(v, red):
    """Block-wide sum, all threads receive it; red: smem Float32 tensor of NWARPS."""
    v = warp_sum(v)
    tid, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if tid % 32 == 0:
        red[tid // 32] = v
    cute.arch.sync_threads()
    s = Float32(0.0)
    for i in cutlass.range_constexpr(NWARPS):
        s = s + red[i]
    return s


@cute.jit
def ld4(row, idx, atom):
    """4 consecutive floats of the 1-D tensor `row` (vector index idx) as an rmem tensor."""
    r = cute.make_rmem_tensor((4,), Float32)
    cute.copy(atom, cute.local_tile(row, (4,), (idx,)), r)
    return r


def rm_tensor(shape, dtype=Float32):
    """rmem tensor with the last mode contiguous (rows can be vector-copy targets)."""
    st, acc = [], 1
    for n in reversed(shape):
        st.append(acc)
        acc *= n
    return cute.make_rmem_tensor(cute.make_layout(shape, stride=tuple(reversed(st))), dtype)


def f32_row(ptr, n):
    return cute.make_tensor(ptr, cute.make_layout(n))


@cute.jit
def cond_matvec(inp, sm, rows: cutlass.Constexpr, rowfn: cutlass.Constexpr, biasfn: cutlass.Constexpr,
                outfn: cutlass.Constexpr):
    """out[s][r] = outfn(sum_k Wrow(r)[k] * inp[s][k] + bias(r)) for the NSTEP condition vectors, rows >= 4 per warp task.
    inp: (NSTEP, W) f32 gmem written in this launch; sm: (NSTEP, W) f32 smem; rowfn(r) -> 1-D (W,) weight row tensor,
    biasfn(r) -> 1-D (1,) tensor, outfn(s, r, v) stores."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    cg = S.cg_atom(Float32, 128)
    nc = S.nc_atom(Float32, 128)
    nc1 = S.nc_atom(Float32, 32)
    sm_atom = cute.make_copy_atom(CopyUniversalOp(), Float32, num_bits_per_copy=128)
    gin = f32_row(inp.iterator, NSTEP * W)
    sm1 = f32_row(sm.iterator, NSTEP * W)
    for i in cutlass.range(tid, NSTEP * W // 4, NTHREADS, unroll=1):
        r = cute.make_rmem_tensor((4,), Float32)
        cute.copy(cg, cute.local_tile(gin, (4,), (i,)), r)
        cute.copy(sm_atom, r, cute.local_tile(sm1, (4,), (i,)))
    cute.arch.sync_threads()
    gw = bid * NWARPS + tid // 32
    nw = nb * NWARPS
    for task in cutlass.range(gw, rows // 4, nw, unroll=1):
        # Side-effecting asm: NVVM may not hoist the 80 loop-invariant smem loads of `c` above it (they were hoisted out of the
        # task loop and then spilled, ~600 B stack per matvec). Instruction order / arithmetic unchanged.
        cute.arch.inline_ptx("// cond_matvec task")
        r0 = task * 4
        bias = cute.make_rmem_tensor((4,), Float32)
        wv = rm_tensor((4, 8, 4))
        for j in cutlass.range_constexpr(4):
            b1 = cute.make_rmem_tensor((1,), Float32)
            cute.copy(nc1, biasfn(r0 + j), b1)
            bias[j] = b1[0]
        acc = cute.make_rmem_tensor((4, NSTEP), Float32)
        acc.fill(0.0)
        for j in cutlass.range_constexpr(4):
            row = rowfn(r0 + j)
            for i in cutlass.range_constexpr(8):
                cute.copy(nc, cute.local_tile(row, (4,), (i * 32 + lane,)), wv[j, i, None])
        for i in cutlass.range_constexpr(8):
            for s in cutlass.range_constexpr(NSTEP):
                c = cute.make_rmem_tensor((4,), Float32)
                cute.copy(sm_atom, cute.local_tile(sm1, (4,), (s * (W // 4) + i * 32 + lane,)), c)
                for j in cutlass.range_constexpr(4):
                    for k in cutlass.range_constexpr(4):
                        acc[j, s] = X.fma(wv[j, i, k], c[k], acc[j, s])
        mine = cute.make_rmem_tensor((4,), Float32)
        for j in cutlass.range_constexpr(4):
            for s in cutlass.range_constexpr(NSTEP):
                v = warp_sum(acc[j, s])
                if lane == s:
                    mine[j] = v
        if lane < NSTEP:
            for j in cutlass.range_constexpr(4):
                outfn(lane, r0 + j, mine[j] + bias[j])
    cute.arch.sync_threads()


@cute.jit
def action_in_task(xrow, smem_x: cutlass.Constexpr, aw, ab, h2, ssp, t, gi):
    """h[t][gi*64 + 2 lane .. +1] = bf16(action_in(x)) and ssp[t][gi] = the group's sum of squares; xrow: the (ADIM,) x row,
    gmem (.cg, written in this launch) or, smem_x, the smem row of the fused tail."""
    lane = cute.arch.thread_idx()[0] % 32
    n = gi * 64 + lane * 2
    cg = S.cg_atom(Float32, 128)
    nc = S.nc_atom(Float32, 128)
    xatom = cute.make_copy_atom(CopyUniversalOp(), Float32, num_bits_per_copy=128) if smem_x else cg
    w0row = aw[n, None]
    w1row = aw[n + 1, None]
    xv = rm_tensor((ADIM // 4, 4))
    w0 = rm_tensor((ADIM // 4, 4))
    w1 = rm_tensor((ADIM // 4, 4))
    for i in cutlass.range_constexpr(ADIM // 4):
        cute.copy(xatom, cute.local_tile(xrow, (4,), (i,)), xv[i, None])
        cute.copy(nc, cute.local_tile(w0row, (4,), (i,)), w0[i, None])
        cute.copy(nc, cute.local_tile(w1row, (4,), (i,)), w1[i, None])
    bv = cute.make_rmem_tensor((2,), Float32)
    cute.copy(S.nc_atom(Float32, 64), cute.local_tile(ab, (2,), (n // 2,)), bv)
    v0 = Float32(0.0)
    v1 = Float32(0.0)
    for i in cutlass.range_constexpr(ADIM // 4):
        for k in cutlass.range_constexpr(4):
            v0 = X.fma(xv[i, k], w0[i, k], v0)
        for k in cutlass.range_constexpr(4):
            v1 = X.fma(xv[i, k], w1[i, k], v1)
    h0 = X.rbf(v0 + bv[0])
    h1 = X.rbf(v1 + bv[1])
    h2[t, n // 2] = X.pack_bf16x2(h0, h1)
    ss = warp_sum(X.fma(h1, h1, h0 * h0))
    if lane == 0:
        ssp[t, gi] = ss


@cute.jit
def action_in_tasks(noise, aw, ab, h2, ssp, xbuf, copy_x: cutlass.Constexpr):
    """(row, 64-column group) warp tasks over the whole grid, as action_in_tasks of the CUDA kernel."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    for task in cutlass.range(tid // 32 * nb + (nb - 1 - bid), T * NSSP, nb * NWARPS, unroll=1):
        t = task // NSSP
        gi = task % NSSP
        action_in_task(noise[t, None], False, aw, ab, h2, ssp, t, gi)
        if cutlass.const_expr(copy_x):
            if gi == 0:
                xbuf[t, lane] = S.ld_cg_f32(noise.iterator + t * ADIM + lane)


@cute.jit
def part4_load(part, S_: cutlass.Constexpr, N: cutlass.Constexpr, t, c):
    """the S .cg float4 loads of reduce_part4 (all in flight), returns the (S, 4) register tensor"""
    cg4 = S.cg_atom(Float32, 128)
    flat = cute.make_tensor(part.iterator, cute.make_layout(S_ * T * N))
    x = rm_tensor((S_, 4))
    for s in cutlass.range_constexpr(S_):
        cute.copy(cg4, cute.local_tile(flat, (4,), (((s * T + t) * N + c) // 4,)), x[s, None])
    return x


@cute.jit
def part4_sum(x, S_: cutlass.Constexpr):
    """sum over s = 0..S-1 in order from +0.0, per element; returns the (4,) register tensor"""
    v = rm_tensor((4,))
    for e in cutlass.range_constexpr(4):
        a = Float32(0.0)
        for s in cutlass.range_constexpr(S_):
            a = a + x[s, e]
        v[e] = a
    return v


@cute.jit
def reduce_part4(part, S_: cutlass.Constexpr, N: cutlass.Constexpr, t, c):
    """reduce_part4<S>: sum over s = 0..S-1 (in order, from +0.0) of part[s][t][c .. c+3], all S .cg float4 loads in flight."""
    return part4_sum(part4_load(part, S_, N, t, c), S_)


@cute.jit
def final_norm_row(v, t, fmod, hn):
    """final_norm_row (pi05_mk.cu:981): adaRMS of row t from h and the 16 ssp parts (xor 8..1, broadcast of lane 0), thread tid
    handles columns 4 tid .. 4 tid + 3; hn (smem, W f32) receives the bf16-rounded result as f32. fmod: scale at [0, W), shift [W, 2W)."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    ss = S.ld_cg_f32(v["ssp"].iterator + (t * NSSP + (lane & 15)))
    for o in (8, 4, 2, 1):
        ss = ss + cute.arch.shuffle_sync_bfly(ss, o)
    ss = cute.arch.shuffle_sync(ss, 0)
    rs = X.ada_rs(ss)
    cg4 = S.cg_atom(Float32, 128)
    hv = rm_tensor((4,), BFloat16)
    sc = rm_tensor((4,))
    sh = rm_tensor((4,))
    h = cute.make_tensor(v["h"].iterator, cute.make_layout(T * W))
    cute.copy(S.cg_atom(BFloat16, 64), cute.local_tile(h, (4,), (t * (W // 4) + tid,)), hv)
    cute.copy(cg4, cute.local_tile(fmod, (4,), (tid,)), sc)
    cute.copy(cg4, cute.local_tile(fmod, (4,), (W // 4 + tid,)), sh)
    o4 = rm_tensor((4,))
    for e in cutlass.range_constexpr(4):
        o4[e] = X.rbf(X.ada_apply(hv[e].to(Float32), rs, sc[e], sh[e]))
    cute.copy(cute.make_copy_atom(CopyUniversalOp(), Float32, num_bits_per_copy=128), o4, cute.local_tile(hn, (4,), (tid,)))


@cute.jit
def action_out_pair(v, aow, aob, actions, t, a0, hn, xs, last_step):
    """action_out_pair (pi05_mk.cu:1004): dims a0, a0 + 1 of row t (weights in registers before the CTA sync that publishes hn,
    lane-strided dot with fma in j order, warp_sum) and the Euler update of xbuf; xs (smem) gets the new x, actions on the last step."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32
    nc1 = S.nc_atom(Float32, 32)
    xbuf = v["xbuf"]
    w0 = rm_tensor((W // 32, 1))
    w1 = rm_tensor((W // 32, 1))
    r0 = aow[a0, None]
    r1 = aow[a0 + 1, None]
    for j in cutlass.range_constexpr(W // 32):
        cute.copy(nc1, cute.local_tile(r0, (1,), (lane + 32 * j,)), w0[j, None])
        cute.copy(nc1, cute.local_tile(r1, (1,), (lane + 32 * j,)), w1[j, None])
    cute.arch.sync_threads()  # hn staged
    v0 = Float32(0.0)
    v1 = Float32(0.0)
    for j in cutlass.range_constexpr(W // 32):
        hk = hn[lane + 32 * j]
        v0 = X.fma(hk, w0[j, 0], v0)
        v1 = X.fma(hk, w1[j, 0], v1)
    v0 = warp_sum(v0)
    v1 = warp_sum(v1)
    if lane < 2:
        a = a0 + lane
        acc = v0
        if lane == 1:
            acc = v1
        b1 = rm_tensor((1,))
        cute.copy(nc1, cute.local_tile(aob, (1,), (a,)), b1)
        vel = acc + b1[0]
        xnew = X.fadd_rn(S.ld_cg_f32(v["xbuf"].iterator + (t * ADIM + a)), X.fmul_rn(X.f32(-0.1), vel))
        xbuf[t, a] = xnew
        xs[a] = xnew
        if last_step:
            actions[t, a] = xnew


@cute.jit
def action_tail_rows(v, aow, aob, aiw, aib, actions, fmod, hn, xs, last_step, more):
    """action_tail_rows (pi05_mk.cu:1046): per row t of this block: final norm, all 32 action_out dims (4 per warp), Euler, then
    action_in of the row (warp w -> 64-column groups 2w, 2w + 1) and its ssp; last_step / more: runtime Booleans (not more)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    warp = tid // 32
    h2 = cute.make_tensor(cute.recast_ptr(v["h"].iterator, dtype=Uint32), cute.make_layout((T, W // 2), stride=(W // 2, 1)))
    for t in cutlass.range(bid, T, nb, unroll=1):
        final_norm_row(v, t, fmod, hn)
        for pr in cutlass.range(2, unroll=1):
            action_out_pair(v, aow, aob, actions, t, warp * 4 + pr * 2, hn, xs, last_step)
        cute.arch.sync_threads()  # xs complete
        if more:
            for gi in cutlass.range(warp * 2, warp * 2 + 2, unroll=1):
                action_in_task(xs, True, aiw, aib, h2, v["ssp"], t, gi)
        cute.arch.sync_threads()  # hn / xs reuse
