"""Device sync helpers of the CuTe port: counter grid barrier, WPRE split barrier, arrival counters, L2 prefetch, .cg loads.

Sequences follow mk/pi05_mk.cu (PI05_BARRIER=3). Pointers are global-memory cute pointers (`tensor.iterator`).
The first barrier is epoch 1: the `bar` and `cnt` regions are zero at launch. Either the host zeroes them before each launch
(Knobs.host_zero) or the last CTA to leave the previous launch did (`exit_reset`, the default; CUDA zeroes them in-kernel behind
one cooperative-groups grid.sync, which the DSL does not have)."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Float32
from cutlass.cute.nvgpu import CopyG2ROp, LoadCacheMode
from cutlass.cute.arch import nvvm_wrappers as W

from . import iket_marks as IM
from .config import BAR_CNT_WORD, BAR_EXIT_WORD, CNT_STRIDE, NCNT, BAR_WORDS, NTHREADS


@cute.jit
def _spin_acquire(ptr, target):
    v = W.load(ptr, Uint32, sem="acquire", scope="gpu")
    while (v - target).to(Int32) < Int32(0):
        v = W.load(ptr, Uint32, sem="acquire", scope="gpu")


@cute.jit
def grid_barrier(bar, epoch: Int32, nblocks: Int32):
    """Grid barrier number `epoch` (1, 2, ...): CTA sync, thread 0 release-add, acquire poll, CTA sync."""
    IM.sync_push("sync")
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        cnt = bar + BAR_CNT_WORD
        W.atomic_add(cnt, Uint32(1), sem="release", scope="gpu")
        _spin_acquire(cnt, (epoch * nblocks).to(Uint32))
    cute.arch.sync_threads()
    IM.sync_pop()


@cute.jit
def wpre_arrive(bar):
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        W.atomic_add(bar + BAR_CNT_WORD, Uint32(1), sem="release", scope="gpu")


@cute.jit
def wpre_wait(bar, target: Uint32):
    IM.sync_push("wpre")
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        _spin_acquire(bar + BAR_CNT_WORD, target)
    cute.arch.sync_threads()
    IM.sync_pop()


@cute.jit
def counter_publish(ptr):
    """One thread, after a CTA sync over the published stores."""
    W.red(ptr, Uint32(1), op="add", dtype="u32", sem="release", scope="gpu")


@cute.jit
def block_wait(ptr, target: Uint32, backoff: cutlass.Constexpr):
    IM.sync_push("cwait")
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        v = W.load(ptr, Uint32, sem="relaxed", scope="gpu")
        while v < target:
            if cutlass.const_expr(backoff):
                cute.arch.inline_ptx("nanosleep.u32 64;")
            v = W.load(ptr, Uint32, sem="relaxed", scope="gpu")
        cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()
    IM.sync_pop()


def cnt_ptr(cnt, index):
    """Counter `index` of the workspace counter region (one 128 B line each)."""
    return cnt + index * CNT_STRIDE


@cute.jit
def prefetch_l2(base, nbytes: Int32):
    """Grid-strided prefetch.global.L2 of [base, base + nbytes) per 128 B (base: pointer to bytes)."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nblk, _, _ = cute.arch.grid_dim()
    ntid, _, _ = cute.arch.block_dim()
    n = nbytes // 128
    for i in cutlass.range(bid * ntid + tid, n, nblk * ntid, unroll=1):
        cute.arch.inline_ptx("prefetch.global.L2 [{$r0}];", read_only_args=[base + i * 128])


def cg_atom(dtype, bits: int):
    """gmem->rmem copy atom that lowers to ld.global.cg (L2; the data was produced by another CTA of this launch)."""
    return cute.make_copy_atom(CopyG2ROp(), dtype, num_bits_per_copy=bits, load_cache_mode=LoadCacheMode.GLOBAL)


def nc_atom(dtype, bits: int):
    """gmem->rmem copy atom for read-only inputs (ld.global.nc)."""
    return cute.make_copy_atom(CopyG2ROp(), dtype, num_bits_per_copy=bits, invariant=True)


@cute.jit
def ld_cg_u32(ptr):
    return W.load(ptr, Uint32, cop="cg")


@cute.jit
def ld_cg_f32(ptr):
    return W.load(ptr, Float32, cop="cg")


@cute.jit
def exit_reset(cnt, bar, flag, nblocks: Int32):
    """Last-CTA-out reset of every counter / barrier word (called by all CTAs as the very last thing of the kernel).

    CTA sync; thread 0 does atom.add.acq_rel.gpu on the exit word and publishes `old == nblocks - 1` through the smem word `flag`;
    the one CTA that sees it zeroes `cnt` and `bar` (the exit word included) with plain stores and a gpu fence. `cnt` / `bar`
    are the u32 workspace tensors; stores go through these arguments (not through a dict of views). See NOTES.md for the argument.
    """
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        old = W.atomic_add(bar.iterator + BAR_EXIT_WORD, Uint32(1), sem="acq_rel", scope="gpu")
        flag[0] = (old == (nblocks - 1).to(Uint32)).to(Int32)
    cute.arch.sync_threads()
    if flag[0] != Int32(0):
        for i in cutlass.range(tid, NCNT * CNT_STRIDE, NTHREADS, unroll=1):
            cnt[i] = Uint32(0)
        for i in cutlass.range(tid, BAR_WORDS, NTHREADS, unroll=1):
            bar[i] = Uint32(0)
        cute.arch.fence_acq_rel_gpu()
