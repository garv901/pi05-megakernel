"""Stages D..I of a layer (pi05_mk.cu:1766-1895): o_proj (D), gated residual + FFN norm (E), gate/up GEMM (F), GeGLU (G),
down GEMM (H) and the gated residual of the next norm (I). Barriers sit in the stage that follows the work they close, so a
debug stop (which returns before the trailing barrier) is a plain skip of the later stage functions.

Modulation rows are addressed by byte address (`modstep` = mod + step * MODROW, Int64): the layer index is a runtime value."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp

from . import mathx as X
from . import rowops as R
from . import sync as S
from .config import ADIM, L, ARR_GU, CNT_GU, CNT_STRIDE, FF, FUSE_BACKOFF, FUSE_FG, KC, NSSP, NWARPS, QD, T, W
from .gemm import gemm_phase, gemm_phase_wsplit

GR = (T + 1) // 2  # G tasks: (32-column group, half of the rows), one column quad per thread


def gptr(dtype, addr, align):
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def mod_tensor(addr):
    """Float32 view of one norm's [scale | shift | gate] rows starting at byte address `addr`."""
    return cute.make_tensor(gptr(Float32, addr, 16), cute.make_layout(3 * W))


def row_addr(modstep, norm):
    """Byte address of norm `norm` (0 .. 2L) of the step: modstep + norm * 3W floats."""
    return modstep + Int64(norm * (3 * W * 4)) if isinstance(norm, int) else modstep + (norm * (3 * W * 4)).to(Int64)


@cute.jit
def residual_tasks(v, gate_addr, S_: cutlass.Constexpr):
    """residual_tasks<S_>: one warp task per (row t, 64-column group), highest block indices first. h = bf16(h + bf16(bf16(sum_s
    part[s]) * bf16(gate))) per column pair, ssp[task] = the group's sum of squares."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tid % 32
    warp = tid // 32
    cg2 = S.cg_atom(Float32, 64)
    part = cute.make_tensor(v["part"].iterator, cute.make_layout(S_ * T * W))
    h2 = cute.make_tensor(cute.recast_ptr(v["h"].iterator, dtype=Uint32), cute.make_layout((T, W // 2), stride=(W // 2, 1)))
    gate = cute.make_tensor(gptr(Float32, gate_addr, 16), cute.make_layout(W))
    ssp = v["ssp"]  # (a subscript store rooted at the dict `v` makes the DSL treat v as written and rebuild it per region)
    for task in cutlass.range(warp * nb + (nb - 1 - bid), T * NSSP, nb * NWARPS, unroll=1):
        t = task // NSSP
        gi = task % NSSP
        c = gi * 64 + lane * 2
        x = R.rm_tensor((S_, 2))
        gv = R.rm_tensor((2,))
        for s in cutlass.range_constexpr(S_):
            cute.copy(cg2, cute.local_tile(part, (2,), (((s * T + t) * W + c) // 2,)), x[s, None])
        cute.copy(cg2, cute.local_tile(gate, (2,), (c // 2,)), gv)
        hraw = S.ld_cg_u32(h2.iterator + (t * (W // 2) + c // 2))
        h_lo = X.from_bits((hraw << Uint32(16)))
        h_hi = X.from_bits(hraw & Uint32(0xFFFF0000))
        v0 = Float32(0.0)
        v1 = Float32(0.0)
        for s in cutlass.range_constexpr(S_):
            v0 = v0 + x[s, 0]
            v1 = v1 + x[s, 1]
        o0 = X.residual(h_lo, v0, gv[0])
        o1 = X.residual(h_hi, v1, gv[1])
        h2[t, c // 2] = X.pack_bf16x2(o0, o1)
        ss = R.warp_sum(X.fma(o1, o1, o0 * o0))
        if lane == 0:
            ssp[t, gi] = ss


@cute.jit
def stage_d(v, wo, wd, l, modstep, epoch, nb, knobs: cutlass.Constexpr):
    """D: [barrier unless deferred into the wsplit GEMM], L2 prefetch of wd and the modulation rows of norms 2l, 2l+1, o_proj."""
    defer = bool(knobs.wpre & 8) and knobs.d_wpt > 1
    target = Uint32(0)
    if cutlass.const_expr(not defer):
        epoch = epoch + 1
        S.grid_barrier(v["bar"].iterator, epoch, nb)
    S.prefetch_l2(cute.recast_ptr(wd[l, None, None].iterator, dtype=Uint8), W * FF * 2)
    S.prefetch_l2(gptr(Uint8, row_addr(modstep, 2 * l), 16), 6 * W * 4)
    wol = wo[l, None, None]
    if cutlass.const_expr(knobs.d_wpt > 1):
        if cutlass.const_expr(defer):
            epoch = epoch + 1
            target = (epoch * nb).to(Uint32)
        gemm_phase_wsplit(v["attn"], wol, v["part"], v["bar"], target, K=QD, N=W, WPT=knobs.d_wpt, wpre=defer)
    else:
        gemm_phase(v["attn"], wol, wol, wol, v["part"], v["bar"], Boolean(False), target, v["cnt"], v["ssp"], v["mod"],
                   K=QD, n1=W, n2=W, N=W, cdiv=1, cmod=1, norm=False, counted=False, wpre=False, load_hoist=knobs.load_hoist)
    return epoch


@cute.jit
def stage_e(v, l, modstep, epoch, nb):
    """E: barrier after D, gated residual of the attention output (gate of norm 2l), ssp for F's staging."""
    epoch = epoch + 1
    S.grid_barrier(v["bar"].iterator, epoch, nb)
    residual_tasks(v, row_addr(modstep, 2 * l) + (2 * W * 4), QD // KC)
    return epoch


@cute.jit
def stage_f(v, wg, wu, l, modstep, epoch, nb, knobs: cutlass.Constexpr):
    """F: [barrier unless WPRE bit 2], gate / up GEMM with the FFN adaRMS applied while staging A; counted if FUSE_FG."""
    defer = bool(knobs.wpre & 2)
    target = Uint32(0)
    epoch = epoch + 1
    if cutlass.const_expr(defer):
        target = (epoch * nb).to(Uint32)
    else:
        S.grid_barrier(v["bar"].iterator, epoch, nb)
    wgl = wg[l, None, None]
    wul = wu[l, None, None]
    cnt = cute.make_tensor(v["cnt"].iterator + CNT_GU * CNT_STRIDE, cute.make_layout(1))
    nmod = mod_tensor(row_addr(modstep, 2 * l + 1))
    gemm_phase(v["h"], wgl, wul, wul, v["part"], v["bar"], Boolean(True), target, cnt, v["ssp"], nmod,
               K=W, n1=FF, n2=2 * FF, N=2 * FF, cdiv=32, cmod=FF // 32, norm=True, counted=bool(knobs.fuse & FUSE_FG),
               wpre=defer, load_hoist=knobs.load_hoist)
    return epoch


@cute.jit
def stage_g(v, wq, wk, wv, l, it1, n_layers, epoch, nb, knobs: cutlass.Constexpr):
    """G: [barrier unless FUSE_FG], prefetch the next layer's QKV weights, GeGLU per 32-column group once its 16 tiles arrived."""
    fuse_fg = bool(knobs.fuse & FUSE_FG)
    backoff = bool(knobs.fuse & FUSE_BACKOFF)
    if cutlass.const_expr(not fuse_fg):
        epoch = epoch + 1
        S.grid_barrier(v["bar"].iterator, epoch, nb)
    ln = l + 1
    if ln >= n_layers:
        ln = Int32(0)
    S.prefetch_l2(cute.recast_ptr(wq[ln, None, None].iterator, dtype=Uint8), QD * W * 2)
    S.prefetch_l2(cute.recast_ptr(wk[ln, None, None].iterator, dtype=Uint8), 256 * W * 2)
    S.prefetch_l2(cute.recast_ptr(wv[ln, None, None].iterator, dtype=Uint8), 256 * W * 2)
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nbk, _, _ = cute.arch.grid_dim()
    act = v["act"].iterator.toint()
    for task in cutlass.range(nbk - 1 - bid, 2 * (FF // 32), nbk, unroll=1):
        jg = task >> 1
        t = (task & 1) * GR + tid // 8
        c = jg * 32 + (tid % 8) * 4
        if cutlass.const_expr(fuse_fg):
            S.block_wait(v["cnt"].iterator + (CNT_GU + jg) * CNT_STRIDE, (it1 * ARR_GU).to(Uint32), backoff)
        if (tid // 8 < GR) & (t < T):
            gv = R.reduce_part4(v["part"], W // KC, 2 * FF, t, c)
            uv = R.reduce_part4(v["part"], W // KC, 2 * FF, t, FF + c)
            ob = R.rm_tensor((4,), BFloat16)
            for e in cutlass.range_constexpr(4):
                ob[e] = X.gelu_mul(gv[e], uv[e]).to(BFloat16)
            out = cute.make_tensor(gptr(BFloat16, act + ((t * FF + c) * 2).to(Int64), 8), cute.make_layout(4))
            cute.copy(cute.make_copy_atom(CopyUniversalOp(), BFloat16, num_bits_per_copy=64), ob, out)
    return epoch


@cute.jit
def stage_h(v, wd, l, modstep, epoch, nb, knobs: cutlass.Constexpr):
    """H: [barrier unless WPRE bit 4], prefetch the next norm's modulation row, down projection."""
    defer = bool(knobs.wpre & 4)
    target = Uint32(0)
    if cutlass.const_expr(not defer):
        epoch = epoch + 1
        S.grid_barrier(v["bar"].iterator, epoch, nb)
    S.prefetch_l2(gptr(Uint8, row_addr(modstep, 2 * (l + 1)), 16), 3 * W * 4)
    if cutlass.const_expr(defer):
        epoch = epoch + 1
        target = (epoch * nb).to(Uint32)
    wdl = wd[l, None, None]
    gemm_phase(v["act"], wdl, wdl, wdl, v["part"], v["bar"], Boolean(True), target, v["cnt"], v["ssp"], v["mod"],
               K=FF, n1=W, n2=W, N=W, cdiv=1, cmod=1, norm=False, counted=False, wpre=defer,
               load_hoist=knobs.load_hoist)
    return epoch


@cute.jit
def stage_i(v, l, modstep, step, n_steps, epoch, nb, aow, aob, aiw, aib, actions):
    """I: barrier after H, then the gated residual of the FFN output (gate of norm 2l+1). Layers before the last feed the next
    layer's A (hand-off in kernel._rest). At l == L - 1 (PI05_TAIL 2): prefetch the action weights, residual, barrier, then
    action_tail_rows (final norm, action_out, Euler update, and unless it is the last step action_in for the next step)."""
    epoch = epoch + 1
    S.grid_barrier(v["bar"].iterator, epoch, nb)
    if l < L - 1:
        residual_tasks(v, row_addr(modstep, 2 * l + 1) + (2 * W * 4), FF // KC)
    else:
        S.prefetch_l2(cute.recast_ptr(aow.iterator, dtype=Uint8), ADIM * W * 4)
        S.prefetch_l2(cute.recast_ptr(aiw.iterator, dtype=Uint8), W * ADIM * 4)
        residual_tasks(v, row_addr(modstep, 2 * l + 1) + (2 * W * 4), FF // KC)
        epoch = epoch + 1
        S.grid_barrier(v["bar"].iterator, epoch, nb)
        more = step + 1 < n_steps
        last_step = step + 1 >= n_steps
        sm = cute.make_tensor(cute.arch.get_dyn_smem(Float32, alignment=16), cute.make_layout(W + ADIM))
        hn = cute.make_tensor(sm.iterator, cute.make_layout(W))
        xs = cute.make_tensor(sm.iterator + W, cute.make_layout(ADIM))
        fmod = mod_tensor(row_addr(modstep, 2 * L))
        R.action_tail_rows(v, aow, aob, aiw, aib, actions, fmod, hn, xs, last_step, more)
    return epoch
