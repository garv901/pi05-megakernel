# SPEC phase 6: performance parity (keep 0 ulp)

Read first: `../PLAN.md` (section 4, phase 6), `SPEC-phase5.md`, `../NOTES.md`, all of `cute/*.py`, `../../bench.py`,
`../../megakernel.py`. CUDA: `../../mk/pi05_mk.cu`. Same hard rules: **no GPU**, CPU compile only via `arm.cpu_arm`, no
commits, no slurm submit (`--time <= 00:30:00`), only files under `megabench/sota/pi05/` and `megabench/tests/`, idiomatic
CuTe with exceptions logged in `NOTES.md`. **Every change must keep byte parity**: do not change any arithmetic order,
rounding point, fma/rn choice or reduction order. Changes are only to register allocation, load scheduling, unrolling,
code structure and host overhead.

## GPU results so far — record in NOTES.md
- Phase 4 (job 7782, `runs/20261017T002522Z-p4/`): 17/17 compares byte-equal (D..I at layer 0, I at layers 1/16, seeds,
  mask-tail, wpre 0, d_wpt 1/4, fuse 16/22/23). I-stop layer 16: P128/k16 CUDA 0.877 / CuTe 0.930 ms; P968/k64 0.941 / 1.010.
- Phase 5 (job 7784, `runs/20261017T002538Z-p5/`): 15/15 byte-equal incl. `full` with `actions` (all 12 configs).
  Full chunk (L2 not flushed, median 20): P128/k16 CUDA 7.665 / CuTe 7.778 ms (+1.5%); P968/k64 8.338 / 8.422 (+1.0%).
- Bench smoke (job 7786, `../runs/20261017T005220Z-cute-smoke/`): both arms pass correctness with identical errors;
  steady megakernel 7.716 / cute 7.807 ms; isolated 7.722 / 7.807. cute shows 2 kernels per chunk (the host `zero_()` fill).

## Goal
Close the gap to CUDA (target: CuTe within 1% of CUDA on the bench at P=128 k16 and P=968 k64, ideally equal) by
(1) a per-stage time breakdown tool so the GPU job tells us where the gap is, (2) removing the host zero-fill kernel,
(3) removing the register spills (CUDA full kernel: REG 255, STACK 72; CuTe: STACK 952, ~1.9 KB spills).

## Deliverables

### 1. Stage ladder timing (`cute/perf.py`, GPU script; CPU-testable argument parsing)
For a given config, time stop launches for the cumulative ladder `pro, A, B, C3, D, E, F, G, H, I` at step 0 layer 0,
`I` at layer 1 (one full layer) and `full`, both CUDA (PI05_STOP builds, as parity) and CuTe, same inputs, interleaved
(CUDA, CuTe, CUDA, ...), median of N (default 20) after 3 warmups, CUDA events, L2 not flushed. Print one JSON line per
stop plus a table of per-stage deltas (stop[i] - stop[i-1]) for CUDA, CuTe and CuTe - CUDA. Reuse `parity.py` helpers
(do not duplicate the build / input code; refactor into shared functions if needed, without changing parity behavior).
Options: `--prefix --att-keys --seed --iters --stages --fuse --wpre --d-wpt --out`.

### 2. In-kernel counter reset (remove the host `zero_()` launch)
CuTe DSL has no grid.sync, which is why the host zeroes `cnt..bar` before each launch. Replace it with a
**last-CTA-out reset**: after a CTA's final use of any counter in the launch (after the final barrier / end-of-run code,
for every stop and for `full`), CTA sync, thread 0 does `atom.add.acq_rel.gpu` on a dedicated exit counter (new word in the
`bar` area, add it to `config.workspace_layout` without moving existing offsets if possible; otherwise update parity's
compare exclusions accordingly); the CTA that gets `gridDim - 1` zeroes all counter / barrier words (including the exit
counter) with plain stores followed by `fence` (release, gpu scope). The next launch is stream-ordered, so it sees zeros.
- The workspace must be zeroed once at allocation (`bind`); keep a `reset()` on the arm for recovery.
- Make it a knob (`Knobs.host_zero`, default False = in-kernel reset); `True` keeps the phase-5 path for A/B.
- Check carefully that no CTA can still be spinning on or adding to a counter when the last CTA resets it (in particular
  CTAs that leave a stop early, arrival counters whose consumers never wait in a truncated run, and the deferred final
  barrier). Explain the argument in NOTES.md.
- parity: compare must still exclude `bar`; add a check that after a CuTe launch all counter / barrier words are zero.
- `arm.CuteArm` per launch is then the compiled call only.

### 3. Spills
Find which code regions spill (ptxas `-v` per region is not available; use stop builds / isolated `@cute.jit` probes and
the SASS `STL/LDL` locations, as in phase 1 where one `cond_matvec` alone spilled 600 B). Known: `cond_matvec`
(4 x 8 float4 weights + 40 accumulators, prologue), the tail (`action_out_pair` 64 weight registers), some GEMM staging.
Candidate techniques, each must keep the CUDA arithmetic order:
- Shorter live ranges: load weights in smaller batches inside the dot loop (same fma order), keep fragments constexpr-indexed
  so they stay in registers, avoid tensors whose layouts force dynamic indexing (local memory).
- Look at what CUDA does differently in SASS for the same function (`cuobjdump -sass` of the CUDA ext build; the DSL cubin
  via `CUTE_DSL_KEEP`); compare instruction counts and register use per function.
- Check whether stack comes from real spills or from local arrays (`rm_tensor` indexed with runtime values); fix those first.
- If the DSL / ptxas just allocates worse at 255, try `maxrregcount`-like options or launch bounds the DSL exposes, and
  report.
Report STACK / spill bytes before and after for each build of the phase-5 test set.

### 4. Other overheads (look, report, fix if simple)
- Compare the per-launch host path of `MegakernelArm` and `CuteArm` (DSL call overhead, argument marshalling); if the
  compiled call has measurable host cost, consider `cute.compile` with the TVM-FFI / fast-call options available in
  4.7.1, or a captured CUDA graph only if `MegakernelArm` does the same (it must stay comparable).
- SASS diff of the hot loops (A/F/H gemm inner loop, wsplit, FA unit, residual tasks, reduce_part4): instruction count
  and memory op widths vs CUDA. List differences in NOTES.md; fix the ones that are clearly codegen-only.

### 5. Tests (CPU) `megabench/tests/test_pi05_cute_phase6.py`
`perf.py` argument parsing and ladder construction; compile `full` with `host_zero` True and False; the spill numbers
printed (and a non-strict assertion that STACK did not grow vs phase 5). Run ALL test modules (phase 0-6) in one final
command and report that run.

### 6. `cute/slurm/phase6.sbatch` (do not submit)
1. Parity first (exit nonzero on failure): `full` for P=128 k16 seeds 7301/7302, P=128 k32, P=128 k64, P=968 k64,
   P=968 k64 mask-tail 200, P=128 k16 wpre 0, P=128 k16 d_wpt 1, P=128 k16 fuse 23, and the same `full` with
   `host_zero` True for P=128 k16; plus the counters-are-zero check; plus 2 back-to-back `full` launches compared (the second
   launch must equal the first: tests the reset).
2. Ladder (`perf.py`) for P=128 k16 and P=968 k64.
3. Bench: `slurm/pi05.sbatch` is a separate job; instead run `bench.py` directly inside this job with `--arms megakernel,cute
   --mk-variants k16,k32,k64 --seeds 7301 --rounds 3` at P=128, and P=968 with k64 only, if it fits in the time budget
   (check how `slurm/pi05.sbatch` invokes it and its runtime; otherwise leave the bench to me and say so).
Under 30 minutes; most important first; elapsed per run.

### 7. NOTES.md
Phase 4/5/smoke GPU results, phase-6 changes with before/after spill numbers, the reset argument, SASS findings, idiom
exceptions, open guesses.

## Done when
All CPU tests (phase 0-6 modules) pass in one final run, parity code paths still byte-level unchanged in arithmetic, and a
short factual report (what changed, spills before / after, what to look for in the GPU job).
