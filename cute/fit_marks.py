"""FlashInfer-profiler-style point markers (port of include/flashinfer/profiler.cuh) for the gemm_phase sites of iket_marks
(g_0 .. g_wnext). Probe only; everything here is a trace-time no-op unless PI05_FIT is set (read once at import), and then
the PTX of a plain build is unchanged.

  PI05_FIT=1   lane 0 of each warp appends a 64-bit entry {tag, globaltimer_lo} per marker
  PI05_FIT=2   same, with FlashInfer's __threadfence_block() (membar.cta) before and after every entry
  PI05_FIT=3   null marker (control): an empty side-effecting inline asm at every site, no timer read, no store

Entry k of warp (block, w) sits at fit[1 + k * 148 * NWARPS + block * NWARPS + w] (u64), fit[0] = {nblocks, NWARPS}; the
`fit` region is appended to the workspace (config.workspace_layout). Tag: bits 0-1 type (2 = instant), 2-11 event
(1 + index in iket_marks.GEMM_NAMES), 12-23 block * NWARPS + warp, 24-31 smid. Unlike FlashInfer (cursor in registers,
passed to every function), the cursor lives in shared memory (one u64 per warp at the end of the dynamic smem,
config.FIT_SCRATCH) and each marker is one predicated inline-PTX block: no DSL control flow and no register live across
markers. Not combinable with iket (PI05_IKET_GEMM)."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Int64

MODE = int(os.environ.get("PI05_FIT", "0"))
CAP = 32768        # entries per warp the region holds (no bound check in the kernel, as FlashInfer)
NBLOCKS = 148      # config.NUM_SMS (the grid the host launches)
NWARPS_ = 8
NTHREADS_ = 256
STRIDE = NBLOCKS * NWARPS_ * 8  # bytes between consecutive entries of one warp
REGION_BYTES = (1 + CAP * NBLOCKS * NWARPS_) * 8 if MODE else 0
SCRATCH_BYTES = NWARPS_ * 8 if MODE else 0

if MODE not in (0, 1, 2, 3):
    raise ValueError("PI05_FIT must be 0, 1, 2 or 3")

_FENCE = "membar.cta;" if MODE == 2 else ""


@cute.jit
def _slot():
    smem = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=16)
    tid, _, _ = cute.arch.thread_idx()
    return cute.recast_ptr(smem + (cute.arch.get_dyn_smem_size() - SCRATCH_BYTES), dtype=Int64) + tid // 32


@cute.jit
def init(fit_base: Int64):
    """Kernel start: every warp's cursor (lane 0 stores it) and the header (block 0, thread 0). fit_base: byte address.
    (inline_ptx rewrites every percent sign to $, so special registers come in as operands; $-names are local PTX regs.)"""
    if cutlass.const_expr(MODE):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        cur = fit_base + ((bid * NWARPS_ + tid // 32 + 1) * 8).to(Int64)
        cute.arch.inline_ptx(
            "{ .reg .pred %%fq, %%fz; .reg .u64 %%fh; setp.eq.u32 %%fq, {$r2}, 0; @%%fq st.volatile.shared.u64 [{$r0}], {$r1};"
            " setp.eq.u32 %%fz, {$r3}, 0; mov.u64 %%fh, " + str((NWARPS_ << 32) | NBLOCKS) + "; @%%fz st.global.u64 [{$r4}], %%fh; }",
            read_only_args=[_slot(), cur, cute.arch.lane_idx(), bid * NTHREADS_ + tid, fit_base])


@cute.jit
def mark(eid: cutlass.Constexpr):
    """One entry for event id eid (1..1023) by lane 0 of the calling warp: the timer and the tag are DSL values (two
    integer ops), the cursor update is one straight-line inline-PTX block predicated on lane 0 and a bound buffer."""
    if cutlass.const_expr(MODE == 3):
        cute.arch.inline_ptx("// fit null marker")
    if cutlass.const_expr(MODE in (1, 2)):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        tag = (cute.arch.smid() << 24) | ((bid * NWARPS_ + tid // 32) << 12) | ((eid << 2) | 2)
        cute.arch.inline_ptx(
            "{ .reg .u64 %%fc, %%fv, %%fx; .reg .pred %%fp, %%fq; " + _FENCE
            + " setp.eq.u32 %%fq, {$r1}, 0; ld.volatile.shared.u64 %%fc, [{$r0}]; setp.ne.and.u64 %%fp, %%fc, 0, %%fq;"
            " cvt.u64.u32 %%fv, {$r2}; shl.b64 %%fv, %%fv, 32; cvt.u64.u32 %%fx, {$r3}; or.b64 %%fv, %%fv, %%fx;"
            " @%%fp st.global.u64 [%%fc], %%fv; add.u64 %%fc, %%fc, " + str(STRIDE) + ";"
            " @%%fp st.volatile.shared.u64 [{$r0}], %%fc; " + _FENCE + " }",
            read_only_args=[_slot(), cute.arch.lane_idx(), cute.arch.globaltimer_lo(), tag])
