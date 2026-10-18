"""The CuTe pi0.5 megakernel (port of mk/pi05_mk.cu). Phase 2: prologue (`pro`), then per layer A (QKV GEMM) and B (reduce + RoPE).

Grid = one CTA per SM, 256 threads, dynamic smem = knobs.smem_bytes, cooperative launch. Workspace buffers are typed views of
one uint8 tensor at `config.workspace_layout` offsets. The host zeroes `cnt` and `bar` before each launch; the grid barrier
counts epochs from 1 (SYNC)."""

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import CopyUniversalOp

from . import mathx as X
from . import rowops as R
from . import sync as S
from . import iket_marks as IM
from .config import (ADIM, ARR_QKV, CNT_QKV, CNT_STRIDE, FF, FUSE_AB, FUSE_ABL, FUSE_BACKOFF, FUSE_DEFNORM, HD, KVD, L, MODROW, NQKV,
                     NSSP, NSTEP, NTHREADS, QD, T, W, Knobs, workspace_layout)
from . import attention as ATT
from .config import FUSE_CC3
from . import ffn
from .gemm import gemm_phase
from .abl import abl_phase

STAGES = ("pro", "A", "B", "C3", "D", "E", "F", "G", "H", "I")  # stop points (same order as parity.STAGES)


@dataclass(frozen=True)
class Stop:
    """Debug stop (parity builds): every block returns right after `stage` of (step, layer). None = production."""
    stage: Optional[str] = None
    step: int = 0
    layer: int = 0

    def __post_init__(self):
        if self.stage is not None and self.stage not in STAGES:
            raise ValueError(f"stage {self.stage!r} is not implemented yet (have {STAGES})")


F32, BF16, U32, U8 = Float32, BFloat16, Uint32, Uint8


def input_specs(k: Knobs):
    """(name, dtype, shape) of the 25 kernel inputs in megakernel.ORDER; prefix_mask is passed as uint8."""
    P = k.prefix
    return [
        ("action_in_weight", F32, (W, ADIM)), ("action_in_bias", F32, (W,)),
        ("time_in_weight", F32, (W, W)), ("time_in_bias", F32, (W,)),
        ("time_out_weight", F32, (W, W)), ("time_out_bias", F32, (W,)),
        ("action_out_weight", F32, (ADIM, W)), ("action_out_bias", F32, (ADIM,)),
        ("wq", BF16, (L, QD, W)), ("wk", BF16, (L, KVD, W)), ("wv", BF16, (L, KVD, W)), ("wo", BF16, (L, W, QD)),
        ("wg", BF16, (L, FF, W)), ("wu", BF16, (L, FF, W)), ("wd", BF16, (L, W, FF)),
        ("attn_norm_weight", F32, (L, 3 * W, W)), ("attn_norm_bias", F32, (L, 3 * W)),
        ("ffn_norm_weight", F32, (L, 3 * W, W)), ("ffn_norm_bias", F32, (L, 3 * W)),
        ("final_norm_weight", F32, (3 * W, W)), ("final_norm_bias", F32, (3 * W,)),
        ("noise", F32, (T, ADIM)), ("prefix_k", BF16, (L, P, 1, HD)), ("prefix_v", BF16, (L, P, 1, HD)),
        ("prefix_mask", U8, (P,)),
    ]


def ws_specs(k: Knobs):
    """(name, dtype, shape) of the workspace buffers (config.LAYOUT_ORDER)."""
    nc = k.att_nchunk
    return {
        "temb": (F32, (NSTEP, W)), "tmid": (F32, (NSTEP, W)), "cond": (F32, (NSTEP, W)), "mod": (F32, (NSTEP, MODROW)),
        "xbuf": (F32, (T, ADIM)), "rope": (F32, (T * (HD // 2) * 2,)), "h": (BF16, (T, W)), "xn": (BF16, (T, W)),
        "part": (F32, (workspace_layout(k)["part"][1] // 4,)), "q": (BF16, (T, QD)), "ksuf": (BF16, (T, KVD)),
        "vsuf": (BF16, (T, KVD)), "scores": (F32, (workspace_layout(k)["scores"][1] // 4,)),
        "stats": (F32, (workspace_layout(k)["stats"][1] // 4,)), "opart": (F32, (workspace_layout(k)["opart"][1] // 4,)),
        "attn": (BF16, (T, QD)), "act": (BF16, (T, FF)), "cnt": (U32, (workspace_layout(k)["cnt"][1] // 4,)),
        "ssp": (F32, (T, NSSP)), "bar": (U32, (workspace_layout(k)["bar"][1] // 4,)),
    }


@cute.jit
def ws_views(ws, k: cutlass.Constexpr):
    """Typed tensors over the uint8 workspace at the config offsets (same bytes as the CUDA kernel's pointers)."""
    lay = workspace_layout(k)
    out = {}
    for name, (dt, shape) in ws_specs(k).items():
        ptr = cute.recast_ptr(ws.iterator + lay[name][0], dtype=dt)
        out[name] = cute.make_tensor(ptr, cute.make_layout(shape, stride=_row_major(shape)))
    return out


def _row_major(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= s
    return tuple(reversed(st))


@cute.jit
def SYNC(bar, epoch, nblocks):
    """The CUDA SYNC() macro: next grid barrier; returns the new epoch."""
    epoch = epoch + 1
    S.grid_barrier(bar.iterator, epoch, nblocks)
    return epoch


@cute.jit
def wpre(bar, epoch, nblocks):
    """The CUDA wpre() lambda: consumes the next epoch and returns (epoch, target) for wpre_wait."""
    epoch = epoch + 1
    return epoch, (epoch * nblocks).to(Uint32)


def _view(t, rows, cols):
    return cute.make_tensor(t.iterator, cute.make_layout((rows, cols), stride=(cols, 1)))


# Blocks of a layer after A: 0 after_a (barrier, prefetch, prefill, B), 1 after_b (SYNC, C, SYNC, C3), 2..7 D..I, 8 hand-off.
NBLOCKS = 9


def _legacy(stop):
    """Stops A, B, C3 of step 0 layer 0 (phases 2-3 builds): emit only what the stop needs (nothing of D..I)."""
    return stop.stage in ("A", "B", "C3") and stop.step == 0 and stop.layer == 0


def _n_run(stop):
    """Blocks that run unguarded before the stop returns (stop A -> 0, B -> 1, C3 -> 2, D -> 3 ... I -> 8; no stop -> all)."""
    return NBLOCKS if stop.stage is None else STAGES.index(stop.stage) - 1


@cute.jit
def _rest(v, wq, wk, wv, wo, wg, wu, wd, prefix_k, prefix_v, prefix_mask, aow, aob, aiw, aib, actions, l, it1, step, n_steps,
          n_layers, modstep, epoch, nb, pend, a_nmod, knobs: cutlass.Constexpr, lo: cutlass.Constexpr, hi: cutlass.Constexpr):
    """Blocks lo .. hi - 1 of the layer after A (see NBLOCKS); each block starts with the barrier that closes the previous one."""
    wpre_on = bool(knobs.wpre & 1)
    if cutlass.const_expr(lo <= 0 < hi):
        IM.push("B")
        epoch = _after_a(v, wo, prefix_k, prefix_v, prefix_mask, l, it1, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 1 < hi):
        IM.push("C")
        epoch = _after_b(v, wg, wu, prefix_k, prefix_v, prefix_mask, l, it1, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 2 < hi):
        IM.push("D")
        epoch = ffn.stage_d(v, wo, wd, l, modstep, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 3 < hi):
        IM.push("E")
        epoch = ffn.stage_e(v, l, modstep, epoch, nb)
        IM.pop()
    if cutlass.const_expr(lo <= 4 < hi):
        IM.push("F")
        epoch = ffn.stage_f(v, wg, wu, l, modstep, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 5 < hi):
        IM.push("G")
        epoch = ffn.stage_g(v, wq, wk, wv, l, it1, n_layers, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 6 < hi):
        IM.push("H")
        epoch = ffn.stage_h(v, wd, l, modstep, epoch, nb, knobs)
        IM.pop()
    if cutlass.const_expr(lo <= 7 < hi):
        IM.push("I")
        epoch = ffn.stage_i(v, l, modstep, step, n_steps, epoch, nb, aow, aob, aiw, aib, actions)
        IM.pop()
    if cutlass.const_expr(lo <= 8 < hi):
        # hand-off to the next layer's A: its barrier is deferred into A (WPRE bit 1, `pend`) or taken here. The last layer of
        # the last step ends the kernel (nothing more); the last layer of an earlier step feeds step + 1 (norm 0 of its row).
        if (l < L - 1) | (step + 1 < n_steps):
            if l < L - 1:
                a_nmod = ffn.row_addr(modstep, 2 * (l + 1))
            else:
                a_nmod = modstep + (MODROW * 4)
            if cutlass.const_expr(wpre_on):
                pend = Boolean(True)
            else:
                epoch = SYNC(v["bar"], epoch, nb)
    return epoch, pend, a_nmod


def _stops_at(stop, stage):
    """Trace-time: does this build stop at or before `stage` (so what follows `stage` must be skipped in the last iteration)?"""
    return stop.stage is not None and STAGES.index(stop.stage) <= STAGES.index(stage)


@cute.jit
def phase_b(v, it1, fuse_ab: cutlass.Constexpr, backoff: cutlass.Constexpr):
    """B (pi05_mk.cu:1288-1320): per (head, 8-row slice) task, once the head's counters are complete: sum the 8 K-chunk
    partials, bf16 round, RoPE for heads < 9, 64-bit stores into q / ksuf / vsuf. Tasks go to the highest block indices."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    rs = NTHREADS // (HD // 8)  # 8 rows x 32 column quads per task
    ntask = (T + rs - 1) // rs
    cg4 = S.cg_atom(Float32, 128)
    rope = cute.make_tensor(v["rope"].iterator, cute.make_layout(T * (HD // 2) * 2))
    qa = v["q"].iterator.toint()
    ka = v["ksuf"].iterator.toint()
    va = v["vsuf"].iterator.toint()
    for task in cutlass.range(nb - 1 - bid, 10 * ntask, nb, unroll=1):
        hh = task // ntask
        t = (task % ntask) * rs + tid // (HD // 8)
        j = (tid % (HD // 8)) * 4
        if cutlass.const_expr(fuse_ab):
            S.block_wait(v["cnt"].iterator + (CNT_QKV + hh) * CNT_STRIDE, (it1 * ARR_QKV).to(Uint32), backoff)
        if t < T:
            c = hh * HD + j
            v1 = R.reduce_part4(v["part"], W // 128, NQKV, t, c)
            v2 = R.reduce_part4(v["part"], W // 128, NQKV, t, c + HD // 2)
            x1 = R.rm_tensor((4,))
            x2 = R.rm_tensor((4,))
            o1 = R.rm_tensor((4,))
            o2 = R.rm_tensor((4,))
            for e in cutlass.range_constexpr(4):
                x1[e] = X.rbf(v1[e])
                x2[e] = X.rbf(v2[e])
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


@cute.jit
def _after_a(v, wo, prefix_k, prefix_v, prefix_mask, l, it1, epoch, nb, knobs: cutlass.Constexpr):
    """Barrier after A (unless FUSE_AB), L2 prefetch of the weights of the next phases, K/V prefill, then B."""
    fuse_ab = bool(knobs.fuse & FUSE_AB)
    fuse_abl = bool(knobs.fuse & FUSE_ABL)  # A and B already ran as one phase (abl.py): no barrier, no phase_b
    backoff = bool(knobs.fuse & FUSE_BACKOFF)
    if cutlass.const_expr(not fuse_ab and not fuse_abl):
        epoch = SYNC(v["bar"], epoch, nb)
    S.prefetch_l2(cute.recast_ptr(wo[l, None, None].iterator, dtype=Uint8), W * QD * 2)
    S.prefetch_l2(cute.recast_ptr(prefix_k[l, None, None, None].iterator, dtype=Uint8), prefix_k.shape[1] * HD * 2)
    S.prefetch_l2(cute.recast_ptr(prefix_v[l, None, None, None].iterator, dtype=Uint8), prefix_v.shape[1] * HD * 2)
    ATT.prefill(prefix_k, prefix_v, prefix_mask, v, l, knobs)  # ATT_PREFILL=2 (pi05_mk.cu:1262-1287)
    if cutlass.const_expr(not fuse_abl):
        phase_b(v, it1, fuse_ab, backoff)
    return epoch


@cute.jit
def _after_b(v, wg, wu, prefix_k, prefix_v, prefix_mask, l, it1, epoch, nb, knobs: cutlass.Constexpr):
    """Barrier, phase C (split-key attention), barrier unless FUSE_CC3, L2 prefetch of wg / wu, phase C3 (chunk combine)."""
    epoch = SYNC(v["bar"], epoch, nb)
    ATT.phase_c(v, prefix_k, prefix_v, prefix_mask, l, knobs)
    if cutlass.const_expr(not (knobs.fuse & FUSE_CC3)):
        epoch = SYNC(v["bar"], epoch, nb)
    S.prefetch_l2(cute.recast_ptr(wg[l, None, None].iterator, dtype=Uint8), FF * W * 2)
    S.prefetch_l2(cute.recast_ptr(wu[l, None, None].iterator, dtype=Uint8), FF * W * 2)
    ATT.phase_c3(v, it1, knobs)
    return epoch


@cute.kernel
def pi05_megakernel(action_in_weight: cute.Tensor, action_in_bias: cute.Tensor, time_in_weight: cute.Tensor,
                    time_in_bias: cute.Tensor, time_out_weight: cute.Tensor, time_out_bias: cute.Tensor,
                    action_out_weight: cute.Tensor, action_out_bias: cute.Tensor, wq: cute.Tensor, wk: cute.Tensor,
                    wv: cute.Tensor, wo: cute.Tensor, wg: cute.Tensor, wu: cute.Tensor, wd: cute.Tensor,
                    attn_norm_weight: cute.Tensor, attn_norm_bias: cute.Tensor, ffn_norm_weight: cute.Tensor,
                    ffn_norm_bias: cute.Tensor, final_norm_weight: cute.Tensor, final_norm_bias: cute.Tensor,
                    noise: cute.Tensor, prefix_k: cute.Tensor, prefix_v: cute.Tensor, prefix_mask: cute.Tensor,
                    actions: cute.Tensor, ws: cute.Tensor, n_steps: Int32, n_layers: Int32,
                    knobs: cutlass.Constexpr, stop: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    gtid = bid * NTHREADS + tid
    gthreads = nb * NTHREADS
    P = knobs.prefix
    v = ws_views(ws, knobs)
    alloc = cutlass.utils.SmemAllocator()
    red = cute.make_tensor(alloc.allocate_array(Float32, 8), cute.make_layout(8))
    s_valid = cute.make_tensor(alloc.allocate_array(Int32, 1), cute.make_layout(1))
    smf = cute.make_tensor(cute.arch.get_dyn_smem(Float32, alignment=16), cute.make_layout((NSTEP, W), stride=(W, 1)))
    epoch = Int32(0)

    IM.push("pro_tab")
    # valid prefix count (every block, same block_sum as CUDA)
    c = Int32(0)
    for i in cutlass.range(tid, P, NTHREADS, unroll=1):
        c = c + (prefix_mask[i] != Uint8(0)).to(Int32)
    sv = R.block_sum(c.to(Float32), red)
    if tid == 0:
        s_valid[0] = (sv + Float32(0.5)).to(Int32)
    cute.arch.sync_threads()
    nvalid = s_valid[0]

    # P0: time embedding (FP64) and RoPE tables
    for i in cutlass.range(gtid, NSTEP * (W // 2), gthreads, unroll=1):
        s = i // (W // 2)
        j = i % (W // 2)
        sn, cs = X.temb(s, j)
        v["temb"][s, j] = sn
        v["temb"][s, W // 2 + j] = cs
    for i in cutlass.range(gtid, T * (HD // 2), gthreads, unroll=1):
        t = i // (HD // 2)
        j = i % (HD // 2)
        cs, sn = X.rope_cs(nvalid + t, X.rope_inv(j))
        v["rope"][i * 2] = cs
        v["rope"][i * 2 + 1] = sn
    IM.pop()
    epoch = SYNC(v["bar"], epoch, nb)

    # P1 / P2: time MLP for all 10 times
    IM.push("pro_t1")
    tiw = _view(time_in_weight, W, W)
    tow = _view(time_out_weight, W, W)
    R.cond_matvec(v["temb"], smf, W, lambda r: tiw[r, None], lambda r: cute.local_tile(time_in_bias, (1,), (r,)),
                  lambda s, r, x: v["tmid"].__setitem__((s, r), X.silu(x)))
    IM.pop()
    epoch = SYNC(v["bar"], epoch, nb)
    IM.push("pro_t2")
    R.cond_matvec(v["tmid"], smf, W, lambda r: tow[r, None], lambda r: cute.local_tile(time_out_bias, (1,), (r,)),
                  lambda s, r, x: v["cond"].__setitem__((s, r), X.silu(x)))
    IM.pop()
    epoch = SYNC(v["bar"], epoch, nb)

    # P3: every adaRMS modulation for every step: mod[s][norm][3W]
    IM.push("pro_mod")
    fw, fb = final_norm_weight.iterator.toint(), final_norm_bias.iterator.toint()
    ffw, ffb = ffn_norm_weight.iterator.toint(), ffn_norm_bias.iterator.toint()
    aw, ab = attn_norm_weight.iterator.toint(), attn_norm_bias.iterator.toint()

    def p3_w(r):
        n = r // (3 * W)
        j = r % (3 * W)
        lw = ((n // 2) * (3 * W * W) + j * W) * 4
        base = aw + lw.to(Int64)
        if n == 2 * L:
            base = fw + (j * W * 4).to(Int64)
        elif n % 2 == 1:
            base = ffw + lw.to(Int64)
        return cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(W))

    def p3_b(r):
        n = r // (3 * W)
        j = r % (3 * W)
        lb = ((n // 2) * (3 * W) + j) * 4
        base = ab + lb.to(Int64)
        if n == 2 * L:
            base = fb + (j * 4).to(Int64)
        elif n % 2 == 1:
            base = ffb + lb.to(Int64)
        return cute.make_tensor(cute.make_ptr(Float32, base, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1))

    R.cond_matvec(v["cond"], smf, MODROW, p3_w, p3_b, lambda s, r, x: v["mod"].__setitem__((s, r), x))
    IM.pop()
    epoch = SYNC(v["bar"], epoch, nb)

    # step-0 input: h = action_in(noise), ssp, xbuf
    IM.push("pro_act")
    S.prefetch_l2(cute.recast_ptr(wq.iterator, dtype=Uint8), QD * W * 2)
    S.prefetch_l2(cute.recast_ptr(wk.iterator, dtype=Uint8), KVD * W * 2)
    S.prefetch_l2(cute.recast_ptr(wv.iterator, dtype=Uint8), KVD * W * 2)
    h2 = cute.make_tensor(cute.recast_ptr(v["h"].iterator, dtype=Uint32), cute.make_layout((T, W // 2), stride=(W // 2, 1)))
    R.action_in_tasks(noise, _view(action_in_weight, W, ADIM), action_in_bias, h2, v["ssp"], v["xbuf"], True)
    IM.pop()
    # stop point `pro`: nothing left in flight (no cp.async yet)
    if cutlass.const_expr(stop.stage != "pro"):
        wpre_on = bool(knobs.wpre & 1)
        fuse_ab = bool(knobs.fuse & FUSE_AB)
        fuse_abl = bool(knobs.fuse & FUSE_ABL)
        backoff = bool(knobs.fuse & FUSE_BACKOFF)
        pend = Boolean(wpre_on)  # layer 0 of step 0: its barrier is deferred into A (WPRE bit 1)
        target = Uint32(0)
        mod_base = v["mod"].iterator.toint()
        a_nmod = mod_base  # modulation row of A's norm: norm 0 of step 0 (afterwards set by I / the step tail)
        if cutlass.const_expr(not wpre_on):
            epoch = SYNC(v["bar"], epoch, nb)
        n_s = n_steps
        if cutlass.const_expr(stop.stage is not None):
            n_s = stop.step + 1
        for step in cutlass.range(n_s, unroll=1):
            l_end = n_layers
            if cutlass.const_expr(stop.stage is not None):
                if step == stop.step:
                    l_end = stop.layer + 1
            for l in cutlass.range(l_end, unroll=1):
                it1 = (step * n_layers + l + 1).to(Uint32)
                keep = Boolean(True)  # False in the iteration that stops (stages after the stop stage are skipped)
                if cutlass.const_expr(stop.stage is not None):
                    keep = (step != stop.step) | (l != stop.layer)
                # ---- A: QKV projections; every 32-column partial tile arrives on its head's counter
                if cutlass.const_expr(wpre_on):
                    if pend:
                        epoch = epoch + 1
                        target = (epoch * nb).to(Uint32)
                modstep = mod_base + (step * (MODROW * 4)).to(Int64)
                if cutlass.const_expr(fuse_abl):
                    IM.push("AB")
                    abl_phase(v["h"], wq[l, None, None], wk[l, None, None], wv[l, None, None], v["bar"], pend, target, v["ssp"],
                              ffn.mod_tensor(a_nmod), cute.make_tensor(v["rope"].iterator, cute.make_layout(T * (HD // 2) * 2)),
                              v["q"], v["ksuf"], v["vsuf"], K=W, n1=QD, n2=QD + KVD, wpre=wpre_on)
                    IM.pop()
                else:
                    IM.push("A")
                    gemm_phase(v["h"], wq[l, None, None], wk[l, None, None], wv[l, None, None], v["part"], v["bar"], pend, target,
                               v["cnt"], v["ssp"], ffn.mod_tensor(a_nmod), K=W, n1=QD, n2=QD + KVD, N=NQKV, cdiv=HD, cmod=10,
                               norm=True, counted=fuse_ab, wpre=wpre_on)
                    IM.pop()
                pend = Boolean(False)
                # Blocks up to the stop run always, the rest only when `keep` (a stop returns before the trailing barrier, which
                # belongs to the next block). Stops A, B, C3 of step 0 layer 0 keep the phase 2-3 builds (nothing after them).
                n_run = _n_run(stop)
                if cutlass.const_expr(n_run > 0):
                    epoch, pend, a_nmod = _rest(v, wq, wk, wv, wo, wg, wu, wd, prefix_k, prefix_v, prefix_mask, action_out_weight,
                                                action_out_bias, _view(action_in_weight, W, ADIM), action_in_bias, actions, l, it1,
                                                step, n_steps, n_layers, modstep, epoch, nb, pend, a_nmod, knobs, 0, n_run)
                if cutlass.const_expr(n_run < NBLOCKS and not (_legacy(stop) and n_run >= 2)):
                    if keep:
                        epoch, pend, a_nmod = _rest(v, wq, wk, wv, wo, wg, wu, wd, prefix_k, prefix_v, prefix_mask,
                                                    action_out_weight, action_out_bias, _view(action_in_weight, W, ADIM),
                                                    action_in_bias, actions, l, it1, step, n_steps, n_layers, modstep, epoch, nb,
                                                    pend, a_nmod, knobs, n_run, n_run + 1 if _legacy(stop) else NBLOCKS)
        # a run truncated by n_layers / n_steps can end on a deferred barrier: the CUDA kernel completes it (stops return earlier)
        if cutlass.const_expr(stop.stage is None):
            if pend:
                epoch = SYNC(v["bar"], epoch, nb)
        # a truncated run that ends on a deferred barrier is not completed (the CUDA stop macros return before it)
    # every CTA of every build (stops included) falls through to here: the last one out zeroes the counters for the next launch
    if cutlass.const_expr(not knobs.host_zero):
        cnt_t = v["cnt"]
        bar_t = v["bar"]
        IM.push("exit")
        S.exit_reset(cnt_t, bar_t, s_valid, nb)
        IM.pop()


@cute.jit
def pi05_host(action_in_weight: cute.Tensor, action_in_bias: cute.Tensor, time_in_weight: cute.Tensor,
              time_in_bias: cute.Tensor, time_out_weight: cute.Tensor, time_out_bias: cute.Tensor,
              action_out_weight: cute.Tensor, action_out_bias: cute.Tensor, wq: cute.Tensor, wk: cute.Tensor,
              wv: cute.Tensor, wo: cute.Tensor, wg: cute.Tensor, wu: cute.Tensor, wd: cute.Tensor,
              attn_norm_weight: cute.Tensor, attn_norm_bias: cute.Tensor, ffn_norm_weight: cute.Tensor,
              ffn_norm_bias: cute.Tensor, final_norm_weight: cute.Tensor, final_norm_bias: cute.Tensor,
              noise: cute.Tensor, prefix_k: cute.Tensor, prefix_v: cute.Tensor, prefix_mask: cute.Tensor,
              actions: cute.Tensor, ws: cute.Tensor, n_steps: Int32, n_layers: Int32, nblocks: Int32,
              knobs: cutlass.Constexpr, stop: cutlass.Constexpr):
    pi05_megakernel(action_in_weight, action_in_bias, time_in_weight, time_in_bias, time_out_weight, time_out_bias,
                    action_out_weight, action_out_bias, wq, wk, wv, wo, wg, wu, wd, attn_norm_weight, attn_norm_bias,
                    ffn_norm_weight, ffn_norm_bias, final_norm_weight, final_norm_bias, noise, prefix_k, prefix_v,
                    prefix_mask, actions, ws, n_steps, n_layers, knobs, stop).launch(
        grid=(nblocks, 1, 1), block=(NTHREADS, 1, 1), smem=knobs.smem_bytes, min_blocks_per_mp=1, cooperative=True)


def compile_kernel(k: Knobs, stop: Stop, args):
    """args: the 27 cute tensors (25 inputs in ORDER, actions, ws) exactly as CuteArm.bind builds them (no fake tensors:
    a fake compile does not exercise the real shapes / dtypes / strides)."""
    if k.fuse & FUSE_ABL and stop.stage == "A":
        raise ValueError("Stop('A') with FUSE_ABL: A and B are one phase (use stop B)")
    return cute.compile(pi05_host, *args, NSTEP, L, 148, k, stop)
