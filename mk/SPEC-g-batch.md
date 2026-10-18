# SPEC: stage G two-task load batching in the CUDA megakernel (`PI05_G_BATCH`)

Owner: Sonnet subagent, CPU ONLY. Do not run sbatch, srun, or anything that uses a GPU (set `CUDA_VISIBLE_DEVICES=` on every
build). The main agent reviews the diff and runs the GPU job. No git commits.

Port of the CuTe change `Knobs.g_batch` (`cute/specs/SPEC-g-batch.md`, `cute/ffn.py:stage_g`). Config: P968 k64.

## Problem (measured layout, not timing)

`mk/pi05_mk.cu:1824-1839`, stage G: 2 * (FF / 32) = 256 tasks, loop
`for (task = gridDim.x - 1 - blockIdx.x; task < 256; task += gridDim.x)`. On 148 CTAs, CTA b runs {147 - b, 295 - b}:
CTAs 0-39 run 1 task, CTAs 40-147 run 2. A task = two `reduce_part4<W / KC>` calls (16 float4 `__ldcg` per thread), a little
math, one 64-bit store. Task 2's loads issue only after task 1's math and store: 2 serial L2 round trips. The next grid
barrier (before H, default fuse 18 has no FUSE_FG) waits for the 2-task CTAs.

## Change

### 1. Macro
Next to the other knobs (near `PI05_D_WPT`, `mk/pi05_mk.cu:137`):
```c++
// Stage G (round g_batch): 0 = one task at a time; 1 = loads of both tasks of a CTA in flight before any math;
// 2 = task 1 (g, u) + task 2 g loads, task 1 sums, task 2 u loads, epilogues (fewer live registers).
#ifndef PI05_G_BATCH
#define PI05_G_BATCH 0
#endif
static_assert(PI05_G_BATCH >= 0 && PI05_G_BATCH <= 2, "PI05_G_BATCH");
```

### 2. Split `reduce_part4` (`mk/pi05_mk.cu:757-767`)
Add next to it (keep `reduce_part4` itself unchanged):
```c++
template <int S> __device__ __forceinline__ void part4_load(float4 (&x)[S], const float* part, int N, int t, int c);  // the S __ldcg
template <int S> __device__ __forceinline__ float4 part4_sum(const float4 (&x)[S]);  // s = 0..S-1 in order from 0.f, per component
```
Same load expression and same summation order as `reduce_part4`.

### 3. Stage G loop
Keep the current loop as it is under `#if PI05_G_BATCH == 0`. Add a G epilogue helper (or a lambda) used only by the new path:
`act[t, c..c+3] = pack_bf16x4(rbf(gelu_tanh(rbf(gv.x))) * rbf(uv.x), ...)`: the exact expression of today's store.

`#else` path:
```c++
constexpr int NT = 2 * (FF / 32);
// Loads of both tasks of this CTA in flight before any math: one L2 round trip per pass, not two.
for (int task = gridDim.x - 1 - blockIdx.x; task < NT; task += 2 * gridDim.x) {
  const int task2 = task + gridDim.x;
  const bool has2 = task2 < NT;
  const int tb = has2 ? task2 : task;                 // clamp: task 2's address stays valid, no branch around loads
  const int jg1 = task >> 1, jg2 = tb >> 1;
  const int t1 = (task & 1) * GR + tid / 8, c1 = jg1 * 32 + (tid % 8) * 4;
  const int t2 = (tb & 1) * GR + tid / 8,   c2 = jg2 * 32 + (tid % 8) * 4;
  if (p.fuse & FUSE_FG) { block_wait(<jg1 counter>, ...); block_wait(<jg2 counter>, ...); }  // same counter/target as today
  if (tid / 8 < GR && t1 < T) {   // t2 < T whenever this holds (tid/8 < GR = 25, row half <= 1)
    G_BATCH 1: part4_load gate1, up1, gate2, up2 (32 float4) -> epilogue(task 1) -> if (has2) epilogue(task 2)
    G_BATCH 2: load gate1, up1, gate2 -> sums of task 1 -> load up2 -> epilogue(task 1) -> if (has2) epilogue(task 2)
  }
}
```
Rules: all loads are inside the row guard and do not depend on `has2`; only task 2's store is under `has2`. Do not change the
math, the store width, the thread-to-element map, the prefetch block above it, or the barriers. Every element gets the same
arithmetic, so the result is bit-exact by construction. Check that the guard argument holds (t2 < T): if it does not, guard
task 2's store on `t2 < T` too.

### 4. Bench plumbing (`megakernel.py`, `bench.py` only if needed)
`MegakernelArm` variant may carry a suffix: `k64` (unchanged) or `k64-gb1` / `k64-gb2`. Parse `att_keys` from the part before
`-`; suffix `gbN` adds `-DPI05_G_BATCH=N` to the `extra` flags passed to `load` (so the build gets its own hashed extension
name). Put `g_batch` into `self.config`. Plain `k64` must give the same build name and flags as today. The CuTe arm is not
touched (do not use suffixed variants with `--arms cute`). `bench.py --mk-variants k64,k64-gb1,k64-gb2` must then work
(check how `bench.py` uses the variant string: labels, summary, `best megakernel` selection).

### 5. Bitwise check script
New `megabench/sota/pi05/gbatch_equal.py` (small, reuse `fixture.py` / the bench input builder; read how `bench.py` makes
inputs): builds `MegakernelArm` for k64, k64-gb1, k64-gb2 at P968, runs one chunk per seed (7301, 8917, 42811) with the same
inputs and prints `EQUAL` / `DIFF max_abs=...` per (variant, seed) against k64 (`torch.equal` on actions). Exit 1 on any DIFF.

### 6. CuTe arm suffix (addendum, approved)
The CuTe code already has `Knobs.g_batch`, but `cute/arm.py:CuteArm` builds `Knobs(prefix, att_keys=int(variant.lstrip("k")))`,
so the bench's cute arm always runs g_batch 0. Give `CuteArm` the same variant parsing as `MegakernelArm` (`k64-gb1` ->
`Knobs(..., g_batch=1)`), add `g_batch` to its `config`. Put the parser in one place (e.g. a helper in `megakernel.py` that
both arms import) so the two arms cannot drift. `bench.py --arms megakernel,cute --mk-variants k64,k64-gb1,k64-gb2` must
build 6 entries with distinct labels. `gbatch_equal.py` also compares cute k64 / k64-gb1 / k64-gb2 against CUDA k64
(`--arms` option, default both; the CuTe port is byte-equal to CUDA at P968 per earlier parity runs).

## Acceptance (CPU; write a "## Results" section at the end of this file)

Build the kernel to cubin with nvcc directly (same flags as `megakernel.load`: `-O3 -std=c++17 -lineinfo -DPI05_PREFIX=968
-DPI05_ATT_KEYS=64`, `-arch=sm_100a -cubin`; add the torch include paths if the file needs them, or use `-Xptxas -v` via the
torch extension build with `TORCH_CUDA_ARCH_LIST=10.0`, whichever works without a GPU). Products in
`megabench/sota/pi05/runs/g-batch-cpu/`.

1. **Byte identity.** Before any edit, save `pre/gb0.cubin` (and its `cuobjdump -sass`). After the edit, the default build
   (no `PI05_G_BATCH`) SASS equals the pre-edit SASS (compare `cuobjdump -sass` text with line-info comments stripped if
   `-lineinfo` shifts them; say which comparison you used).
2. **Compile.** `-DPI05_G_BATCH=1` and `=2` compile. Also with the FUSE_FG path reachable (fuse is a runtime arg, so one build
   covers it; just confirm both block_waits are in the code).
3. **Registers and spills (report).** For gb0 / gb1 / gb2: REG, STACK (`cuobjdump -res-usage`), ptxas spill bytes. Call out any
   change vs gb0 (the CUDA kernel is REG 255, STACK 72 today per `cute/NOTES.md`; rederive).
4. **SASS order (report).** In the G loop of gb1 / gb2: count `LDG.E.128` (the `.cg` / `.STRONG.GPU` form) and how many come
   before the first reduction FADD and before the first `STG.E.64`. Reuse `cute/g_batch_check.py` if it works on an nvcc cubin
   (it takes a cubin and uses nvdisasm line info; the source file names differ: pass/extend for `pi05_mk.cu` lines of
   stage G and `part4_load` / `part4_sum`). Report gb0 too (expect 16 loads per task).
5. **Python side.** `python -c` import of `megakernel` and a parse test of variant strings (`k64`, `k64-gb1`, `k64-gb2`,
   `k32`): att_keys, extra flags, config. The plain `k64` extension name is unchanged.
6. **NOTES.** Two or three lines in `megabench/sota/pi05/README.md` (or wherever the CUDA knobs are documented: grep
   `PI05_D_WPT`) on `PI05_G_BATCH`, default 0 until the GPU A/B.

## Files Sonnet may touch
`mk/pi05_mk.cu`, `megakernel.py`, `cute/arm.py` (variant parsing only), `bench.py` (only if the variant suffix needs it), new `gbatch_equal.py`, the README/notes
line, `cute/g_batch_check.py` (only to accept the CUDA source names, without changing its CuTe behavior), this spec (Results
only). Nothing else.

## GPU job (main agent)
One sbatch (`slurm/pi05.sbatch` env): `gbatch_equal.py`, then
`python -m megabench.sota.pi05 --arms megakernel --mk-variants k64,k64-gb1,k64-gb2 --rounds 5 --no-profile --tag gbatch`.
Compare steady / isolated ms against k64 in the same run (baseline: k64 8.278 ms steady, run 20261018T024140Z).

## Results

All CPU. Products in `megabench/sota/pi05/runs/g-batch-cpu/` (`pre/gb0.{cubin,sass,}`, `post/gb{0,1,2}.{cubin,sass,dis,ptxas,res}`, `inc.txt` = torch/python include paths).
Build command (every build, `CUDA_VISIBLE_DEVICES=` set): `nvcc -O3 -std=c++17 -lineinfo -DPI05_PREFIX=968 -DPI05_ATT_KEYS=64 [-DPI05_G_BATCH=N] -arch=sm_100a -cubin $(cat inc.txt) -DTORCH_EXTENSION_NAME=x -Xptxas -v mk/pi05_mk.cu`.
Note: this is the nvcc flag set of `megakernel.load` plus the torch includes, not the torch-extension build itself (the extension's extra `-D__CUDA_NO_*` flags are not in it); REG/STACK of the actual extension build are not verified.

### 1. Byte identity (default build)
Pre-edit snapshot taken before the first edit of `pi05_mk.cu`. After the edit, `PI05_G_BATCH` unset:
`diff <(cuobjdump -sass pre/gb0.cubin) <(cuobjdump -sass post/gb0.cubin)` -> no output, **SASS IDENTICAL** (38905 lines; `cuobjdump -sass` carries no line-info comments, so no stripping was needed). The cubin files themselves differ (`cmp`: byte 33), expected because `-lineinfo` debug sections hold shifted source line numbers; the code sections are the same.

### 2. Compile
`-DPI05_G_BATCH=1` and `=2` compile. Both `block_wait`s of the FUSE_FG path (`jg1`, `jg2` counters) are in the source of the `#else` path (fuse is a runtime arg, one build covers it); SASS presence of them not separately inspected.

### 3. REG / STACK / spills (`cuobjdump -res-usage`, ptxas `-v`)
| build | REG | STACK | ptxas spill stores / loads |
|---|---|---|---|
| gb0 (default; pre-edit same) | 254 | 72 | 0 B / 0 B |
| gb1 | 255 | 72 | 0 B / 0 B |
| gb2 | 255 | 80 | 8 B / 12 B |

(Spec said 255 / 72 for today's CUDA kernel; rederived here: 254 / 72 with this flag set.) gb1 +1 reg, no spills; gb2 spills 8 B stores / 12 B loads (stack 72 -> 80), more than gb1, opposite of the intent of gb2 (fewer live registers). Spill location not attributed to a source line.

### 4. SASS load order in the G loop (`cute/g_batch_check.py` extended to nvcc cubins: G stores = `STG.E.64` with `pi05_mk.cu` line in the stage-G region; reduction FADD = lines of `part4_sum` / `reduce_part4`)
```
gb0: G body bb 481: 16 LDG.E.128, 64 reduction FADD, 1 STG.E.64; LDG before first FADD: 16, before first STG.E.64: 16   (RESULT PASS)
gb1: G body bb 492: 32 LDG.E.128, 64 reduction FADD, 1 STG.E.64; LDG before first FADD: 30, before first STG.E.64: 32
     G body bb 493: 0 LDG.E.128, 64 reduction FADD, 1 STG.E.64; (has2 branch: task 2 epilogue, no loads)
gb2: G body bb 491: 32 LDG.E.128, 64 reduction FADD, 1 STG.E.64; LDG before first FADD: 31, before first STG.E.64: 32
     G body bb 492: 0 LDG.E.128, 64 reduction FADD, 1 STG.E.64; (has2 branch)
```
All 32 loads of both tasks come before the first STG in gb1 and gb2, and none are inside the has2 branch. The checker flags a "violation" for gb1 (2 loads after the first FADD: instrs 10532, 10537, first FADD at 10529) and gb2 (1 load at 10553 after FADD at 10551): ptxas starts task 1's sums while the last loads are still being issued (an unavoidable interleave at the tail; for gb2 the task-1-sum-before-u2-load order is by design). So the exit code of `g_batch_check.py` on the CUDA cubins is 1 for gb1/gb2; read the counts, not RESULT. The "violation" strictness was not relaxed (CuTe behavior unchanged).

### 5. Python side
```
k64      att_keys 64 extra ()                      config g_batch 0 ext name pi05_mk_p968_k64
k64-gb1  att_keys 64 extra ('-DPI05_G_BATCH=1',)   config g_batch 1 ext name pi05_mk_p968_k64_x3d4e167f
k64-gb2  att_keys 64 extra ('-DPI05_G_BATCH=2',)   config g_batch 2 ext name pi05_mk_p968_k64_x32af5e2a
k32      att_keys 32 extra ()                      config g_batch 0 ext name pi05_mk_p968_k32
k64-gb0  -> same as k64 (no flag, same name)
```
(`MegakernelArm.__init__` with `load` stubbed passes `(968, 64, ())`, `(968, 64, ('-DPI05_G_BATCH=1',))`, `(968, 64, ('-DPI05_G_BATCH=2',))`; names above computed with the same hash code as `load`.) Bad variants (`k64-gb3`, `64`, `k64-x`) raise ValueError. `CuteArm(geo, v, "cpu")` for the three variants gives `knobs.g_batch` 0/1/2 and `config["g_batch"]`. Parser: `megakernel.parse_variant`, imported by `cute/arm.py`.
`bench.py` unchanged: labels are `f"{arm}/{variant}"` (6 distinct labels for `--arms megakernel,cute`), "best megakernel" keys on the `megakernel/` prefix. A 6-entry bench run was not executed (needs GPU): not verified.

### 6. Bitwise check
`gbatch_equal.py` was written and only `py_compile`d: **not run, not verified** (needs a GPU). Method: CUDA k64 is the reference per seed (7301, 8917, 42811), same `make_inputs` tensors bound to every arm, `torch.equal` on actions. Bit-exactness of gb1/gb2 is by construction (same expressions and summation order; the SASS-level math was not diffed).

## GPU results (main agent, job 8374, `runs/20261018T225615Z-p968-gbatch/`, P968 k64, fuse 18, 5 rounds)

Bitwise (`gbatch_equal.py`, seeds 7301 / 8917 / 42811): CUDA gb1 / gb2 and CuTe k64 / gb1 / gb2 all EQUAL to CUDA k64. PASS.

| arm / variant | steady ms (min..max of 5) | isolated median ms (n=50) | vs own k64 |
|---|---|---|---|
| megakernel/k64 | 8.274 (8.273..8.274) | 8.283 | - |
| megakernel/k64-gb1 | 8.330 (8.330..8.330) | 8.338 | +56 us (+0.7%) |
| megakernel/k64-gb2 | 8.341 (8.340..8.432) | 8.349 | +67 us (+0.8%) |
| cute/k64 | 8.430 (8.428..8.431) | 8.436 | - |
| cute/k64-gb1 | 8.562 (8.561..8.565) | 8.568 | +133 us (+1.6%) |
| cute/k64-gb2 | 8.504 (8.503..8.505) | 8.510 | +75 us (+0.9%) |

Measured: G batching is slower on both arms, well outside the round spread. Per G stage (180 per chunk): CUDA gb1 +0.31 us.
Not measured (hypotheses): (a) the change reallocates registers for the whole 255-register kernel, so other stages get slower
(CuTe gb1 STACK 392 -> 416); (b) G is not limited by the second L2 round trip. A per-stage trace (IKET / PT markers) would
separate them. Default stays 0.

## Why it is slower (jobs 8380, 8383 + CPU SASS diff)

Plain-binary block-0 barrier stamps (`probe/mk_stamps_ab.py`, job 8383, `probe/runs/20261018T230438Z-gbstamps/`; the stamp store is
runtime-gated, stamped = unstamped launch time within 3 us). Delta vs k64 per layer instance (us, +- 2 SE over 30 launches):

| stage | gb1 | gb2 |
|---|---|---|
| A qkv | +0.064 +- 0.005 | -0.022 +- 0.006 |
| B rope | -0.026 +- 0.005 | -0.001 (n.s.) |
| C attn | **+0.192 +- 0.006** | +0.062 +- 0.004 |
| D o | -0.089 +- 0.003 | -0.039 +- 0.003 |
| E res1 | -0.026 +- 0.003 | +0.009 +- 0.002 |
| F gate/up | **+0.131 +- 0.006** | +0.071 +- 0.005 |
| G act | **-0.072 +- 0.003** | -0.034 +- 0.003 |
| H down | +0.002 (n.s.) | -0.122 +- 0.004 |
| I res2 | -0.048 +- 0.003 | **+0.329 +- 0.004** |
| sum x 180 | +23 us | +45 us |

Measured: G itself gets faster (gb1 -0.07 us/layer), but stages whose source did not change move by up to +0.19 (C) / +0.33 (gb2 I)
us/layer, and the sum is a net loss. SASS (CPU, `nvdisasm -gi` of `runs/g-batch-cpu/post/*.cubin`): the gb1 change reschedules
code outside G: in the kernel body C has 663 of 1038 opcodes changed, I 215 / 2019, pro 655; the `__noinline__` `gemm_phase`
(A, F, H) keeps its opcode mix but ~280 / 1298 instructions move; `gemm_phase_wsplit` (D) loses 74 R2UR. No spills in any build.
Conclusion: the cost is ptxas codegen perturbation of the single 255-register kernel, not G. Linking a given stage's slowdown to
a specific SASS change is not done.
Traced builds (PTRACE, job 8380) are not a sample of the plain binary: there gb1 is net faster (-38 us, G -0.30 us/layer).
