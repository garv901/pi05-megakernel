"""IKET range markers for the profiling builds (see NOTES.md "Profiling"). Everything here is a trace-time no-op unless the compile
option `iket` is on (run-iket sets CUTE_DSL_COMPILER_OPT=iket in the profiled process); then the generated PTX of a plain build is
unchanged byte for byte.

Switches (read once at import):
  CUTE_DSL_COMPILER_OPT contains "iket"   ON
  PI05_IKET_CTA_STRIDE=N                  record only CTAs with blockIdx.x % N == 0 (keeps the event count small)
  PI05_IKET_SYNC=0                        drop the nested barrier / wait ranges (sync, wpre, cwait)
  PI05_IKET_GEMM=1                        (with iket) point markers g_* inside gemm_phase (gmark / gmark_dep) and a 32 B smem scratch
                                          at the end of the dynamic smem (config.GEMM_SCRATCH); unset: no code, PTX unchanged

PI05_IKET_GEMM_DEP=0 (g_wrdy without the XOR fold / STS) and PI05_IKET_GEMM_LOOP=0 (no tile-loop markers) are spill-perturbation
switches, only read with PI05_IKET_GEMM=1; defaults keep every marker.
Every push / pop must be CTA-uniform (all 256 threads); the gmark point markers may be warp-uniform. Each push / pop is preceded by a "settle": one volatile ld.shared whose result is
unused. BAR.SYNC.DEFER_BLOCKING lets a warp issue past the barrier, and a %globaltimer read right after it can run before the
barrier releases (megabench/mkprobe/markers/defer-blocking.md); a memory instruction after the barrier and before the timer read
fixes that."""

import os

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

ON = "iket" in os.environ.get("CUTE_DSL_COMPILER_OPT", "")
CTA_STRIDE = int(os.environ.get("PI05_IKET_CTA_STRIDE", "1"))
SYNC_RANGES = os.environ.get("PI05_IKET_SYNC", "1") != "0"
GEMM = ON and os.environ.get("PI05_IKET_GEMM", "0") == "1"
GEMM_DEP = os.environ.get("PI05_IKET_GEMM_DEP", "1") != "0"  # only read with GEMM: g_wrdy folds rB (gmark_dep) or is a plain gmark
GEMM_LOOP = os.environ.get("PI05_IKET_GEMM_LOOP", "1") != "0"  # only read with GEMM: markers inside the tile loop

STAGE_NAMES = ("pro_tab", "pro_t1", "pro_t2", "pro_mod", "pro_act", "A", "AB", "B", "C", "D", "E", "F", "G", "H", "I", "exit")
SYNC_NAMES = ("sync", "wpre", "cwait")
GEMM_NAMES = ("g_0", "g_wiss", "g_adone", "g_sync", "g_wrdy", "g_mma", "g_st", "g_wnext")  # gemm_phase point markers
NAMES = STAGE_NAMES + SYNC_NAMES + GEMM_NAMES  # 27 names (the compact id form holds up to 30)

if CTA_STRIDE < 1:
    raise ValueError("PI05_IKET_CTA_STRIDE must be >= 1")


@cute.jit
def _settle():
    cute.arch.inline_ptx("{ .reg .b32 %%ika, %%ikz; mov.u32 %%ika, 0; ld.volatile.shared.b32 %%ikz, [%%ika]; }")


@cute.jit
def _push(name: cutlass.Constexpr):
    from cutlass.cute.experimental import iket
    _settle()
    iket.range_push(name)


@cute.jit
def _pop():
    from cutlass.cute.experimental import iket
    _settle()
    iket.range_pop()


@cute.jit
def push(name: cutlass.Constexpr):
    if cutlass.const_expr(ON):
        if cutlass.const_expr(CTA_STRIDE > 1):
            bid, _, _ = cute.arch.block_idx()
            if bid % CTA_STRIDE == 0:
                _push(name)
        else:
            _push(name)


@cute.jit
def pop():
    if cutlass.const_expr(ON):
        if cutlass.const_expr(CTA_STRIDE > 1):
            bid, _, _ = cute.arch.block_idx()
            if bid % CTA_STRIDE == 0:
                _pop()
        else:
            _pop()


@cute.jit
def sync_push(name: cutlass.Constexpr):
    if cutlass.const_expr(ON and SYNC_RANGES):
        push(name)


@cute.jit
def sync_pop():
    if cutlass.const_expr(ON and SYNC_RANGES):
        pop()


@cute.jit
def gmark(name: cutlass.Constexpr):
    """Point marker (no range) inside gemm_phase; warp-uniform use is fine. Trace-time no-op unless PI05_IKET_GEMM=1."""
    if cutlass.const_expr(GEMM):
        from cutlass.cute.experimental import iket
        if cutlass.const_expr(CTA_STRIDE > 1):
            bid, _, _ = cute.arch.block_idx()
            if bid % CTA_STRIDE == 0:
                _settle()
                iket.mark(name)
        else:
            _settle()
            iket.mark(name)


@cute.jit
def gmark_dep(name: cutlass.Constexpr, rB, scratch_ptr):
    """gmark that waits for the weight registers rB (the 64 u32 of one warp's B fragment). An LDG into registers does not stall;
    the warp stalls at the first reader, and warps issue in order, so a timer read placed after an STS that reads every word of
    rB runs only after all of rB arrived. The words are XOR-folded into one u32 that is stored (volatile) to this warp's scratch
    word, then gmark(name). Upper bound of the weight latency: the real loop first waits only for the kb = 0 registers."""
    if cutlass.const_expr(GEMM and GEMM_LOOP and not GEMM_DEP):
        gmark(name)
    if cutlass.const_expr(GEMM and GEMM_LOOP and GEMM_DEP):
        r32 = cute.recast_tensor(rB, Uint32)
        assert cute.size(r32) == 64, cute.size(r32)
        x = r32[0]
        for i in cutlass.range_constexpr(1, 64):
            x = x ^ r32[i]
        cute.arch.inline_ptx("st.volatile.shared.b32 [{$r0}], {$r1};", read_only_args=[scratch_ptr, x])
        gmark(name)


@cute.jit
def gmark_loop(name: cutlass.Constexpr):
    """gmark inside the tile loop (g_mma, g_st, g_wnext); dropped with PI05_IKET_GEMM_LOOP=0."""
    if cutlass.const_expr(GEMM and GEMM_LOOP):
        gmark(name)
