# SPEC: IKET sub-stage point markers inside `gemm_phase` (pi05 CuTe megakernel)

Implementer: Sonnet subagent, CPU only. Do NOT run sbatch / srun / anything on a GPU. Do not commit.
Repo root: `/raid/garv901/vibe-megakernel` (paths below are relative to it). Python: `.venv/bin/python`.
Package: `megabench/sota/pi05/cute/` (below: `cute/`). Read `cute/NOTES.md` section "Profiling" first.

## Why
We want per-warp timestamps inside `gemm_phase` (`cute/gemm.py`, used by stages A, F, H and the D fallback) to split each
stage into: weight issue, wpre barrier wait, stage_a, CTA sync, exposed weight latency, MMA, stores, next-tile weight issue.
Hypotheses to test later on GPU (not by you): (1) H's weight-register spill (STL right after `load_w`) exposes the weight
latency; (3) F's 2nd tile pays its full weight latency; (A) where A's 3.6 us goes. Your job is only the instrumentation,
the CPU checks, the summary and the sbatch.

## Hard rules
- Do not change the math, the loop order, the load order or the store order of `gemm_phase`. Every new code path is behind
  `cutlass.const_expr(...)` so that it does not exist in builds without the switch.
- Byte identity (must be proven, see Acceptance): with `PI05_IKET_GEMM` unset, (a) the plain build PTX and (b) the iket build PTX
  (`CUTE_DSL_COMPILER_OPT=iket`) of the full P=128 k16 kernel are identical to the pre-edit tree.
- Before any edit, dump those two PTX files (step 0). Pre-edit copies of the files are also at
  `/raid/tmp/claude-1029/-raid-garv901-vibe-megakernel-megabench-sota-pi05/f13713f1-1870-483b-8df1-648789707484/scratchpad/pre-gemm-marks/`.
- Match the surrounding code style (dense docstrings, short comments, the existing idioms).
- Files you own: `cute/iket_marks.py`, `cute/gemm.py`, `cute/config.py` (scratch bytes only), `cute/iket_summary.py`,
  new `cute/gemm_marks_check.py`, new `cute/slurm/gemm_marks.sbatch`, `megabench/tests/test_pi05_cute_profile.py`
  (add tests; keep existing ones), `cute/NOTES.md` (one paragraph in "Profiling"). Touch `kernel.py` / `ffn.py` only if a
  call-site change is strictly needed (say why in the result).

## Step 0: pre-edit snapshots
Use `compile_build` from `megabench/tests/test_pi05_cute_profile.py` as the pattern (fresh process, `CUDA_VISIBLE_DEVICES=`,
`CUTE_DSL_ARCH=sm_100a`, `CUTE_DSL_KEEP=ptx,cubin`, `CUTE_DSL_DUMP_DIR`, no cache). Save `plain_full.ptx` and `iket_full.ptx`
to `cute/runs/gemm-marks-cpu/pre/`. Each compile takes minutes; run what you can in parallel (background shells).

## Design

### 1. `cute/iket_marks.py`
- New switch read at import: `GEMM = ON and os.environ.get("PI05_IKET_GEMM", "0") == "1"`. Document it in the module docstring.
- `GEMM_NAMES = ("g_0", "g_wiss", "g_adone", "g_sync", "g_wrdy", "g_mma", "g_st", "g_wnext")`, appended to `NAMES` (27 < 30;
  the existing `test_names` must still pass). Keep `NAMES` order: STAGE + SYNC + GEMM.
- `gmark(name: Constexpr)`: a point marker, `cutlass.cute.experimental.iket.mark(name)`, preceded by the settle (`_settle()`), with
  the same CTA-stride guard as `push`. Trace-time no-op unless `GEMM`. Unlike `push`, it may be called in warp-uniform (not
  CTA-uniform) code: it is a point marker, not a range. No payloads (they need NativeDump).
- `gmark_dep(name, rB, scratch_ptr)`: for `g_wrdy`. XOR-fold the 64 u32 words of the warp's `rB` fragment (recast rB to Uint32;
  check the count is 64) into one Uint32, `st.volatile.shared.b32` it to this warp's scratch word, then `gmark(name)`. Reason
  (put in the docstring): an LDG into registers does not stall; the warp stalls at the first reader; warps issue in order, so a
  timer read placed after this STS in SASS runs only after all of rB arrived. It is an upper bound (the real loop first waits
  only for the kb = 0 registers).

### 2. Scratch smem for `gmark_dep`
NWARPS x 4 bytes, placed beyond every other dynamic-smem use, present only when `GEMM` is on. Suggested: in `Knobs.smem_bytes`
(`cute/config.py:98`) add `+ GEMM_SCRATCH` (16-byte aligned, 0 when off) after the existing max, and give `gemm_phase` the
scratch byte offset (the old `smem_bytes` value) in a way that does not change the PTX when off (e.g. a module-level helper that
computes it from the same constants, or a Constexpr). Your choice; justify it in the result. Importing `iket_marks` in
`config.py` is fine if no cycle results.

### 3. Marker placement in `gemm_phase` (`cute/gemm.py:122`)
Order per warp (CTA-wide calls first, then warp-uniform ones):
- `g_0`: at entry, after the setup code and before the wpre / load_w block.
- `g_wiss`: right after the first `load_w`. In the `wpre and pend` branch it goes after the `if ng < NG: load_w(...)` and BEFORE
  `S.wpre_wait` (so the existing `wpre` sync range measures the barrier wait separately). In the other branches it goes right after
  the load. Every warp records it (warps without a tile too).
- `g_adone`: right after `stage_a(...)`.
- `g_sync`: right after `cute.arch.sync_threads()` (the settle covers BAR.SYNC.DEFER_BLOCKING).
- In the tile loop, each iteration: `g_wrdy` via `gmark_dep` before the first `cute.copy` / `cute.gemm` (the kb loop); `g_mma` after
  the kb loop; `g_st` after the store loops; `g_wnext` inside `if ngi + ng_step < NG:` right after `load_w(next)`.
  Leave the `counted` publish where it is.

### 4. CPU checker `cute/gemm_marks_check.py PLAIN.cubin MARK.cubin [--out FILE]`
Cubins come from builds with `CUTE_DSL_LINEINFO=1`. Use `/usr/local/cuda/bin/nvdisasm` with line info (e.g. `-g` / `-gi`,
check which gives file:line per instruction) and `cuobjdump -res-usage`. Report:
- (a) For each build: REG, STACK, and LDL / STL counts grouped by source file:line. Call out the `load_w` 128-bit copy line in
  gemm.py and `kernel.py:279` (the kernel signature). Show the plain-vs-marker diff per line.
- (b) For the marker build: each `SR_GLOBALTIMERLO` read, the source line of its marker call (gemm.py lines of the g_* calls).
  Checks: for every g_wrdy timer read, an STS that comes from the `gmark_dep` line comes before it in the same basic block.
  For every timer read right after a BAR.SYNC, an LDS lies between them (same logic as `sass_check` in the test file).
  Print violations and exit 1 if any.
Stdlib only + subprocess.

### 5. `cute/iket_summary.py --gemm`
Add an optional section (default output unchanged). Per warp (reuse `per_warp_ranges`, `warp_key`, `pct`, the marker parsing
pattern of `megabench/kernels/cute_fx1/parse_iket.py`, where `markers[]` = `{markerNameIdx, locIdx, timestamp}`):
- Assign each g_* marker to the enclosing stage range of the same warp (A, F, H, D) and to the `wpre` range inside it if any.
- Split into calls at `g_0`. Intervals: `g_0->g_wiss`, `wpre` (range length), `stage_a` = (wpre end, or g_wiss if no wpre)
  -> g_adone, `g_adone->g_sync`, `g_sync->g_wrdy` (tile 1), `g_wrdy->g_mma`, `g_mma->g_st` per tile, tile k >= 2:
  `g_wnext(k-1)->g_wrdy(k)`, and the call total `g_0->last marker`.
- Table: rows = intervals, columns = A, F t1, F t2, H, D (t = tile index); cells `mean / median / p90 (n)` in ns. Headline is
  the mean (32 ns ticks average out over many samples). Second table: the same grouped by tiles per warp (0, 1, 2).
- Use the last launch, like the existing summary. Unknown / unmatched markers are counted and reported, not dropped silently.

### 6. `cute/slurm/gemm_marks.sbatch`
Copy the structure of `cute/slurm/profile.sbatch` (same header, env, `run()`, `note()`), P=128 k16 only, 30 min:
1. plain, `CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all` dump to `$D/dsl_plain`, `--mode plain --n-steps 10 --launches 20`, and
   `--n-steps 1 --launches 5 --save-actions`.
2. iket stage-only (no PI05_IKET_GEMM), `PI05_IKET_CTA_STRIDE=2`, 1 step, `--warmups 1 --launches 1`, save actions.
3. iket + `PI05_IKET_GEMM=1`, same settings, `CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all` dump to `$D/dsl_gemm`, save actions.
   Print an ESTIMATE line for its timestamp count (extend the `estimate()` formula with the gemm markers; must be <= 2 M).
4. `--compare` plain_s1 vs each iket run, to `compare.log`.
5. Both iket runs print the `PROFILE {json}` CUDA-event times; print the difference (the gemm marker cost) in `status.txt`.
6. `gemm_marks_check.py` on the two cubins; `iket_summary.py --gemm` on run 3 (and plain summary on run 2).
`bash -n` must pass. Do not submit it.

### 7. Tests (`megabench/tests/test_pi05_cute_profile.py`)
Add: names test for GEMM_NAMES; PTX of iket+GEMM full build contains the 8 `__iket_evt_decl_g_*` names; Stop A, F and H
iket+GEMM builds compile; `gemm_marks_check.py` exit 0 on the full build; `iket_summary --gemm` on a synthetic trace (extend
the existing synthetic-trace test pattern) gives the expected intervals.

## Acceptance (write the results into the "Result" section at the end of this file)
1. `cmp` equal: plain_full.ptx and iket_full.ptx vs step-0 snapshots (GEMM unset).
2. iket+GEMM: full, Stop A, F, H compile; 8 names present; checker exit 0.
3. Checker LDL/STL table plain vs marker build (full P128 k16), verbatim. If H's spill lines change, say so clearly at the top.
4. All tests in `test_pi05_cute_profile.py` pass (give the command and the summary line).
5. `bash -n cute/slurm/gemm_marks.sbatch` ok.
6. List of changed / new files and a short note per file. Anything you could not do: say so; do not fake it.

## Result
**H spill CHANGES (top-line finding).** With `PI05_IKET_GEMM=1` the spills of H's `load_w` (gemm.py:119) go from 19 LDL / 36 STL to 0 LDL / 2 STL, and the whole kernel from STACK 392 (plain; 400 stage-only iket) to STACK 264; LDL / STL totals go 52/64 (plain) / 54/67 (stage-only iket) -> 85/75 but those are spread over other lines (F load_w 10 STL, A 3 STL, kernel.py:447 +12 STL). So a gemm-marker build is NOT a faithful sample of the unmarked kernel's spill behaviour: the tile-loop markers (the `gmark_dep` XOR-fold sits after `sync_threads`, before the kb loop, not right after `load_w`) change ptxas's allocation; the follow-up below shows the fold alone is not the whole cause, the tile-loop markers are. Any GPU timing of H with the marks must be read with that in mind (the spill hypothesis (1) cannot be tested by marks alone; compare the marker run's H time with the stage-only run's H time instead).
Stage split uses the last stage range pushed before each instruction (needs the PTX ids), and the stage-only iket build is the like-for-like baseline (the plain build has no iket code at all). Pure source-line rows of iket_marks.py are pseudo lines of the iket instrumentation (the CS2R / LOP3 / STS / PMTRIG of iket have invented line numbers 49-53, 56-60, 105-109), so they are not trustworthy as "which marker", the checker uses the PMTRIG id instead.

All CPU only; builds made with `compile_build` (fresh process, CUDA_VISIBLE_DEVICES=, CUTE_DSL_ARCH=sm_100a, no cache) via a small driver in the scratchpad; products in `cute/runs/gemm-marks-cpu/` (`pre/` = step 0 snapshots, `*.ptx`, `*.cubin`, `check_*.md`).

1. cmp (GEMM unset, post-edit vs step-0 snapshot): `cmp pre/plain_full.ptx plain_full.ptx` equal; `cmp pre/iket_full.ptx iket_full.ptx` equal. PASS.
2. iket+GEMM builds full, Stop A, Stop F, Stop H compile; the 8 `__iket_evt_decl_g_*` names are in all four PTX (test_names_in_ptx); `gemm_marks_check.py` on the full build exits 0 (`RESULT: PASS`: 87 timer reads, 32 right after a BAR.SYNC, 3 gmark_dep STS sites = 3 g_wrdy timer reads, 3 gemm_phase call sites A / F / H; D uses gemm_phase_wsplit and has no marks). PASS.
3. Checker table, plain build (no iket) vs gemm-marker build, both CUTE_DSL_LINEINFO=1 (verbatim, section (a)):

# gemm marks check: plain `runs/gemm-marks-cpu/plain_li.cubin`, marker `runs/gemm-marks-cpu/gemm_li.cubin`

## (a) registers, stack, LDL / STL

| build | REG | STACK | LDL | STL |
|---|---|---|---|---|
| plain | 255 | 392 | 52 | 64 |
| marker | 255 | 264 | 85 | 75 |

LDL / STL per source line (`*` = load_w 128-bit copy gemm.py:119, `K` = kernel signature kernel.py:279); marker minus plain in the last column:

| line | plain LDL | plain STL | marker LDL | marker STL | diff LDL | diff STL | source |
|---|---|---|---|---|---|---|---|
| attention.py:155 | 0 | 0 | 0 | 1 | +0 | +1 | `rp[(e, r8), 0, nt] = pr.to(BFloat16)` |
| attention.py:186 | 0 | 0 | 1 | 3 | +1 | +3 | `cute.gemm(tm, accO, tOrP[None, None, ks], tOrV[None, None, k` |
| attention.py:192 | 0 | 0 | 8 | 0 | +8 | +0 | `dst = cute.make_tensor(cute.make_ptr(Float32, opart.iterator` |
| attention.py:194 | 0 | 0 | 3 | 0 | +3 | +0 | `cute.copy(uni(Float32, 64), accO[(None, r8), 0, nt], dst)` |
| gemm.py:112 | 2 | 0 | 0 | 0 | -2 | +0 | `ptr = cute.make_ptr(BFloat16, base + ((nl * K + k0) * 2).to(` |
| gemm.py:119 * | 19 | 36 | 0 | 15 | -19 | -21 | `cute.copy(S.nc_atom(BFloat16, 128), src, dst)` |
| gemm.py:140 | 0 | 0 | 5 | 0 | +5 | +0 | `s = bid % NCH` |
| gemm.py:144 | 0 | 0 | 0 | 5 | +0 | +5 | `ng = slot * NWARPS + warp` |
| gemm.py:157 | 0 | 0 | 3 | 2 | +3 | +2 | `sAp = tha.partition_S(sA)` |
| gemm.py:184 | 0 | 0 | 2 | 0 | +2 | +0 | `for ngi in cutlass.range(ng, NG, ng_step, unroll=1):` |
| gemm.py:191 | 0 | 0 | 3 | 1 | +3 | +1 | `cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], a` |
| gemm.py:199 | 0 | 0 | 2 | 0 | +2 | +0 | `addr = part_base + (((s * T + row) * N + n0 + nt * 8 + 2 * t` |
| gemm.py:202 | 0 | 0 | 0 | 6 | +0 | +6 | `cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst)` |
| gemm.py:252 | 0 | 0 | 8 | 1 | +8 | +1 | `cute.arch.cp_async_commit_group()` |
| gemm.py:295 | 0 | 0 | 0 | 8 | +0 | +8 | `wbuf = cute.make_tensor(wsm.iterator + warp * (NSL * 4 * 32 ` |
| gemm.py:329 | 0 | 0 | 0 | 3 | +0 | +3 | `cute.copy(ca, sAp[None, None, kbd], rAv[None, None, 2 * j + ` |
| gemm.py:330 | 0 | 0 | 10 | 2 | +10 | +2 | `cute.gemm(tm, acc, rA[None, None, 2 * j + m], rB[None, None,` |
| gemm.py:341 | 0 | 0 | 1 | 0 | +1 | +0 | `cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst[0, ` |
| iket_marks.py:39 | 0 | 0 | 8 | 1 | +8 | +1 | `cute.arch.inline_ptx("{ .reg .b32 %%ika, %%ikz; mov.u32 %%ik` |
| iket_marks.py:53 | 0 | 0 | 2 | 0 | +2 | +0 | `iket.range_pop()` |
| iket_marks.py:56 | 0 | 0 | 3 | 0 | +3 | +0 | `@cute.jit` |
| iket_marks.py:58 | 0 | 0 | 0 | 1 | +0 | +1 | `if cutlass.const_expr(ON):` |
| iket_marks.py:105 | 0 | 0 | 3 | 0 | +3 | +0 | `@cute.jit` |
| iket_marks.py:107 | 0 | 0 | 0 | 1 | +0 | +1 | `"""gmark that waits for the weight registers rB (the 64 u32 ` |
| iket_marks.py:109 | 0 | 0 | 5 | 1 | +5 | +1 | `rB runs only after all of rB arrived. The words are XOR-fold` |
| iket_marks.py:116 | 0 | 0 | 1 | 0 | +1 | +0 | `x = x ^ r32[i]` |
| kernel.py:83 | 0 | 0 | 0 | 4 | +0 | +4 | `ptr = cute.recast_ptr(ws.iterator + lay[name][0], dtype=dt)` |
| kernel.py:266 | 0 | 0 | 1 | 0 | +1 | +0 | `S.prefetch_l2(cute.recast_ptr(wg[l, None, None].iterator, dt` |
| kernel.py:279 K | 0 | 0 | 0 | 0 | +0 | +0 | `noise: cute.Tensor, prefix_k: cute.Tensor, prefix_v: cute.Te` |
| kernel.py:282 | 18 | 21 | 1 | 0 | -17 | -21 | `tid, _, _ = cute.arch.thread_idx()` |
| kernel.py:447 | 4 | 0 | 6 | 12 | +2 | +12 | `@cute.jit` |
| mathx.py:155 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| mathx.py:162 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| rowops.py:18 | 3 | 2 | 3 | 2 | +0 | +0 | `v = v + cute.arch.shuffle_sync_bfly(v, o)` |
| sync.py:29 | 0 | 1 | 0 | 2 | +0 | +1 | `cute.arch.sync_threads()` |

lines whose LDL or STL count differs: 31

LDL / STL by stage (last stage range pushed before the instruction; None = before the first stage push):

| stage | marker LDL | marker STL | marker gemm.py:119 LDL | marker gemm.py:119 STL |
|---|---|---|---|---|
| pro_tab | 6 | 4 | 0 | 0 |
| A | 7 | 10 | 0 | 3 |
| B | 0 | 2 | 0 | 0 |
| C | 13 | 9 | 0 | 0 |
| D | 20 | 15 | 0 | 0 |
| E | 1 | 0 | 0 | 0 |
| F | 19 | 22 | 0 | 10 |
| G | 2 | 1 | 0 | 0 |
| H | 13 | 10 | 0 | 2 |
| I | 1 | 0 | 0 | 0 |
| exit | 3 | 2 | 0 | 0 |

Same, stage-only iket build vs gemm-marker build (verbatim, section (a)):

# gemm marks check: plain `runs/gemm-marks-cpu/iket_li.cubin`, marker `runs/gemm-marks-cpu/gemm_li.cubin`

## (a) registers, stack, LDL / STL

| build | REG | STACK | LDL | STL |
|---|---|---|---|---|
| plain | 255 | 400 | 54 | 67 |
| marker | 255 | 264 | 85 | 75 |

LDL / STL per source line (`*` = load_w 128-bit copy gemm.py:119, `K` = kernel signature kernel.py:279); marker minus plain in the last column:

| line | plain LDL | plain STL | marker LDL | marker STL | diff LDL | diff STL | source |
|---|---|---|---|---|---|---|---|
| attention.py:155 | 0 | 0 | 0 | 1 | +0 | +1 | `rp[(e, r8), 0, nt] = pr.to(BFloat16)` |
| attention.py:186 | 0 | 0 | 1 | 3 | +1 | +3 | `cute.gemm(tm, accO, tOrP[None, None, ks], tOrV[None, None, k` |
| attention.py:192 | 0 | 0 | 8 | 0 | +8 | +0 | `dst = cute.make_tensor(cute.make_ptr(Float32, opart.iterator` |
| attention.py:194 | 0 | 0 | 3 | 0 | +3 | +0 | `cute.copy(uni(Float32, 64), accO[(None, r8), 0, nt], dst)` |
| gemm.py:112 | 1 | 0 | 0 | 0 | -1 | +0 | `ptr = cute.make_ptr(BFloat16, base + ((nl * K + k0) * 2).to(` |
| gemm.py:114 | 1 | 0 | 0 | 0 | -1 | +0 | `tB = tmm.partition_B(cute.composition(gB, (None, kperm()))) ` |
| gemm.py:119 * | 19 | 36 | 0 | 15 | -19 | -21 | `cute.copy(S.nc_atom(BFloat16, 128), src, dst)` |
| gemm.py:140 | 0 | 0 | 5 | 0 | +5 | +0 | `s = bid % NCH` |
| gemm.py:144 | 0 | 0 | 0 | 5 | +0 | +5 | `ng = slot * NWARPS + warp` |
| gemm.py:157 | 20 | 23 | 3 | 2 | -17 | -21 | `sAp = tha.partition_S(sA)` |
| gemm.py:184 | 0 | 0 | 2 | 0 | +2 | +0 | `for ngi in cutlass.range(ng, NG, ng_step, unroll=1):` |
| gemm.py:191 | 0 | 1 | 3 | 1 | +3 | +0 | `cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], a` |
| gemm.py:199 | 0 | 0 | 2 | 0 | +2 | +0 | `addr = part_base + (((s * T + row) * N + n0 + nt * 8 + 2 * t` |
| gemm.py:202 | 0 | 0 | 0 | 6 | +0 | +6 | `cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst)` |
| gemm.py:252 | 0 | 0 | 8 | 1 | +8 | +1 | `cute.arch.cp_async_commit_group()` |
| gemm.py:295 | 0 | 0 | 0 | 8 | +0 | +8 | `wbuf = cute.make_tensor(wsm.iterator + warp * (NSL * 4 * 32 ` |
| gemm.py:329 | 0 | 0 | 0 | 3 | +0 | +3 | `cute.copy(ca, sAp[None, None, kbd], rAv[None, None, 2 * j + ` |
| gemm.py:330 | 0 | 0 | 10 | 2 | +10 | +2 | `cute.gemm(tm, acc, rA[None, None, 2 * j + m], rB[None, None,` |
| gemm.py:341 | 0 | 0 | 1 | 0 | +1 | +0 | `cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt], dst[0, ` |
| iket_marks.py:39 | 0 | 0 | 8 | 1 | +8 | +1 | `cute.arch.inline_ptx("{ .reg .b32 %%ika, %%ikz; mov.u32 %%ik` |
| iket_marks.py:53 | 0 | 0 | 2 | 0 | +2 | +0 | `iket.range_pop()` |
| iket_marks.py:56 | 0 | 0 | 3 | 0 | +3 | +0 | `@cute.jit` |
| iket_marks.py:58 | 0 | 0 | 0 | 1 | +0 | +1 | `if cutlass.const_expr(ON):` |
| iket_marks.py:105 | 0 | 0 | 3 | 0 | +3 | +0 | `@cute.jit` |
| iket_marks.py:107 | 0 | 0 | 0 | 1 | +0 | +1 | `"""gmark that waits for the weight registers rB (the 64 u32 ` |
| iket_marks.py:109 | 0 | 0 | 5 | 1 | +5 | +1 | `rB runs only after all of rB arrived. The words are XOR-fold` |
| iket_marks.py:116 | 0 | 0 | 1 | 0 | +1 | +0 | `x = x ^ r32[i]` |
| kernel.py:83 | 0 | 0 | 0 | 4 | +0 | +4 | `ptr = cute.recast_ptr(ws.iterator + lay[name][0], dtype=dt)` |
| kernel.py:266 | 0 | 0 | 1 | 0 | +1 | +0 | `S.prefetch_l2(cute.recast_ptr(wg[l, None, None].iterator, dt` |
| kernel.py:279 K | 0 | 0 | 0 | 0 | +0 | +0 | `noise: cute.Tensor, prefix_k: cute.Tensor, prefix_v: cute.Te` |
| kernel.py:282 | 0 | 0 | 1 | 0 | +1 | +0 | `tid, _, _ = cute.arch.thread_idx()` |
| kernel.py:447 | 4 | 0 | 6 | 12 | +2 | +12 | `@cute.jit` |
| mathx.py:155 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| mathx.py:162 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| rowops.py:18 | 3 | 2 | 3 | 2 | +0 | +0 | `v = v + cute.arch.shuffle_sync_bfly(v, o)` |
| sync.py:29 | 0 | 1 | 0 | 2 | +0 | +1 | `cute.arch.sync_threads()` |

lines whose LDL or STL count differs: 32

LDL / STL by stage (last stage range pushed before the instruction; None = before the first stage push):

| stage | plain LDL | plain STL | plain gemm.py:119 LDL | plain gemm.py:119 STL | marker LDL | marker STL | marker gemm.py:119 LDL | marker gemm.py:119 STL |
|---|---|---|---|---|---|---|---|---|
| pro_tab | 6 | 4 | 0 | 0 | 6 | 4 | 0 | 0 |
| A | 0 | 0 | 0 | 0 | 7 | 10 | 0 | 3 |
| B | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| C | 0 | 0 | 0 | 0 | 13 | 9 | 0 | 0 |
| D | 0 | 0 | 0 | 0 | 20 | 15 | 0 | 0 |
| E | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| F | 0 | 0 | 0 | 0 | 19 | 22 | 0 | 10 |
| G | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 |
| H | 41 | 60 | 19 | 36 | 13 | 10 | 0 | 2 |
| I | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| exit | 3 | 2 | 0 | 0 | 3 | 2 | 0 | 0 |

Section (b) of the checker (plain vs marker), verbatim:
 timer reads of the marker build

- SR_GLOBALTIMERLO reads 87, right after a BAR.SYNC 32
- gmark_dep STS sites (iket_marks.py:117) 3, g_wrdy timer reads 3

| PMTRIG id | name | sites | nearest preceding gemm.py line (first site) |
|---|---|---|---|
| 1 | pro_tab | 1 | - `` |
| 2 | sync | 11 | - `` |
| 3 | pro_t1 | 1 | - `` |
| 4 | pro_t2 | 1 | - `` |
| 5 | pro_mod | 1 | - `` |
| 6 | pro_act | 1 | - `` |
| 7 | A | 1 | - `` |
| 8 | g_0 | 3 | - `` |
| 9 | g_wiss | 4 | - `` |
| 10 | wpre | 4 | - `` |
| 11 | g_adone | 3 | gemm.py:97 `cute.copy(sm_atom, ov[m, None], cute.local_tile(sA` |
| 12 | g_sync | 3 | gemm.py:180 `cute.arch.sync_threads()` |
| 13 | g_wrdy | 3 | gemm.py:184 `for ngi in cutlass.range(ng, NG, ng_step, unroll=1` |
| 14 | g_mma | 3 | gemm.py:157 `sAp = tha.partition_S(sA)` |
| 15 | g_st | 3 | gemm.py:202 `cute.copy(uni(Float32, 64), acc[(None, r8), mt, nt` |
| 16 | g_wnext | 3 | gemm.py:119 `cute.copy(S.nc_atom(BFloat16, 128), src, dst)` |
| 17 | B | 1 | - `` |
| 18 | C | 1 | - `` |
| 19 | cwait | 1 | - `` |
| 20 | D | 1 | - `` |
| 21 | E | 1 | - `` |
| 22 | F | 1 | - `` |
| 23 | G | 1 | - `` |
| 24 | H | 1 | - `` |
| 25 | I | 1 | - `` |
| 26 | exit | 1 | - `` |
| 31 | ? | 31 | - `` |

- RESULT: PASS


4. `.venv/bin/python -m unittest megabench.tests.test_pi05_cute_profile -v` (CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a): `Ran 12 tests in 74.454s` / `OK`. PASS.
5. `bash -n cute/slurm/gemm_marks.sbatch`: ok. Not submitted.
6. Files:
- `cute/iket_marks.py`: `PI05_IKET_GEMM` switch (`GEMM`), `GEMM_NAMES`, `NAMES` = 27, `gmark`, `gmark_dep`; docstring.
- `cute/gemm.py`: the 8 markers in `gemm_phase` (all behind `IM.GEMM` const_expr; marker call at g_wiss placed after the load in each of the three branches, before `wpre_wait`), scratch pointer `gscr`.
- `cute/config.py`: `GEMM_SCRATCH` (32 B, 0 when off), added in `Knobs.smem_bytes`; imports `iket_marks` (no cycle: it imports only cutlass).
- `cute/iket_summary.py`: `--gemm` (functions `gemm_calls`, `gemm_intervals`, `gemm_section`); default output unchanged (test).
- `cute/gemm_marks_check.py` (new): checker; added an optional `--ptx` / `--plain-ptx` (id -> name, stage split of spills), not in the spec's CLI.
- `cute/slurm/gemm_marks.sbatch` (new).
- `megabench/tests/test_pi05_cute_profile.py`: new `Names.test_gemm_names`, class `GemmMarks` (3 tests); `run()` no longer passes `PI05_IKET_GEMM` from the parent env (only via extra_env); `test_full_markers` now also removes the g_* names from the expected set (they exist only with the switch).
- `cute/NOTES.md`: one paragraph in "Profiling".
- `kernel.py` / `ffn.py`: not touched (no call-site change needed: scratch offset comes from `get_dyn_smem_size() - GEMM_SCRATCH`, which needs no plumbing and emits nothing when off).
Scratch design justification: a knob-independent offset (end of dynamic smem), so `gemm_phase`'s signature and all call sites are unchanged; `smem_bytes` grows by 32 B only with the switch.
Not done / caveats: no GPU run (as ordered), so `iket_summary.py --gemm` was only tested on a synthetic trace, and the real marker order / cost is unmeasured; the sbatch ESTIMATE uses an upper bound of 36 gemm markers per layer (stride 2: about 0.9 M stage-only, 1.7 M with gemm markers, below 2 M) and was not run; the checker cannot name a marker by source line (see above), it names by PMTRIG id with `--ptx`; `profile.py` ENV_KEYS (not an owned file) does not record `PI05_IKET_GEMM` in the PROFILE json.

## Spill perturbation by marker set
New switches in `iket_marks.py` (read only with `PI05_IKET_GEMM=1`): `PI05_IKET_GEMM_DEP=0` (g_wrdy = plain gmark, scratch smem kept), `PI05_IKET_GEMM_LOOP=0` (no g_wrdy / g_mma / g_st / g_wnext; g_0, g_wiss, g_adone, g_sync stay). Defaults unchanged: GEMM=1 PTX is `cmp`-equal to the earlier GEMM=1 build, and GEMM-unset plain / iket full PTX are `cmp`-equal to the step-0 snapshots (re-checked after the edit). Full P128 k16 builds with LINEINFO; columns `l119` = LDL/STL at the gemm.py:119 `load_w` copy per stage (stage = last stage range pushed before the instruction); `k:279` / `k:447` = LDL/STL at kernel.py:279 / 447.

| build | REG | STACK | LDL | STL | A l119 | F l119 | H l119 | k:279 | k:447 |
|---|---|---|---|---|---|---|---|---|---|
| baseline (stage-only iket) | 255 | 400 | 54 | 67 | 0/0 | 0/0 | 19/36 | 0/0 | 4/0 |
| current GEMM=1 | 255 | 264 | 85 | 75 | 0/3 | 0/10 | 0/2 | 0/0 | 6/12 |
| V1 DEP=0 | 255 | 272 | 103 | 88 | 0/0 | 0/0 | 0/12 | 0/0 | 9/13 |
| V2 LOOP=0 | 255 | 400 | 54 | 67 | 0/0 | 0/0 | 19/36 | 0/0 | 4/0 |
| V3 DEP=0 LOOP=0 | 255 | 400 | 54 | 67 | 0/0 | 0/0 | 19/36 | 0/0 | 4/0 |

Conclusion: V2 (no tile-loop markers) reproduces the baseline spills exactly (REG / STACK / LDL / STL totals and every column above; only totals and the listed lines compared, not the full SASS); V1 (no fold) still perturbs (H 0/12, STACK 272), so the tile-loop markers cause the perturbation, not only the XOR fold. V3 was built for completeness and equals V2 (V3 was only conditional on V1 and V2 both perturbing, which did not hold).
