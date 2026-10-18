# SPEC: stage G as a `__noinline__` function (`PI05_G_NOINLINE`)

Owner: Sonnet subagent, CPU ONLY (`CUDA_VISIBLE_DEVICES=` on every command; no sbatch/srun/GPU). No git commits.
Follows `mk/SPEC-g-batch.md` (read its last two sections: G batching makes G faster, but ptxas reschedules the unchanged
stages of the single kernel function, net loss). Goal: isolate G's code so that changing G (PI05_G_BATCH) leaves the SASS of
everything else unchanged.

## Change

1. Macro next to `PI05_G_BATCH` (`mk/pi05_mk.cu` ~142): `PI05_G_NOINLINE` (default 0, allowed 0/1, static_assert), with a
   one-line comment (G loop in its own noinline function so G edits do not reschedule the rest of the kernel).
2. New `__device__ __noinline__ void g_phase(...)` placed next to `gemm_phase` / `gemm_phase_wsplit` (follow their style).
   Body = the current G task loop exactly (`mk/pi05_mk.cu` "G (no barrier)" block, the `{ constexpr int GR ... }` part only,
   all three `PI05_G_BATCH` paths, unchanged arithmetic). NOT the L2 prefetch block before it (stays in the kernel body).
   Pass only what the loop uses, as scalars / pointers (do not pass `Params` by value): `part`, `act`, `cnt`, `fuse`, `it1`,
   `backoff` (and anything else the loop reads; `tid` / blockIdx / gridDim can be read inside). Keep `__restrict__` and const
   as the surrounding code does.
3. Kernel body: `#if PI05_G_NOINLINE` -> `g_phase(...)` call, `#else` -> the existing inline block, untouched.
4. `megakernel.parse_variant`: accept suffixes in any combination after `k<N>`: `gb<0|1|2>` and `gn` (= G noinline), e.g.
   `k64-gn`, `k64-gb1-gn`. Return a dict or a small tuple; update both callers (`MegakernelArm`, `cute/arm.py`). `gn` adds
   `-DPI05_G_NOINLINE=1`. For the CuTe arm `gn` is not supported: raise ValueError. `config` gets `g_noinline`. Plain `k64`
   and `k64-gb1` keep their current flags and extension names (flag order: G_BATCH first, then G_NOINLINE).
5. `gbatch_equal.py`: add `k64-gn`, `k64-gb1-gn`, `k64-gb2-gn` to the CUDA variants.

## Acceptance (CPU; "## Results" at the end of this file)

Build cubins like `runs/g-batch-cpu/` (read `runs/g-batch-cpu/inc.txt` and the gb-batch spec Results for the exact nvcc
command used there; same flags). Products in `runs/g-noinline-cpu/`. Builds: gb0, gn0 (=G_NOINLINE 1, G_BATCH 0), gn1, gn2.

1. **Default unchanged.** gb0 (no new macros) SASS equals `runs/g-batch-cpu/post/gb0.sass` (cuobjdump -sass, diff empty).
2. **Isolation (the main result).** Use the per-function / per-stage tools in
   `/raid/tmp/claude-1029/-raid-garv901-vibe-megakernel-megabench-sota-pi05/ad83c231-fdaf-4a36-9531-a5b4f9a8adcd/scratchpad/`
   (`funcdiff.py A.gi.dis B.gi.dis`, `stagediff.py A.gi.dis B.gi.dis`, inputs from `nvdisasm -gi X.cubin`; copy them into
   `runs/g-noinline-cpu/` and adapt the stage line table in `stagediff.py` to the new line numbers). Report:
   - gn0 vs gn1 and gn0 vs gn2: is every function other than `g_phase` byte-identical (kernel body and `gemm_phase*`)? If
     not, per stage: instruction counts, opcode changes, operand-only changes.
   - gn0 vs gb0: same table (the price of the noinline call itself).
3. **Resources.** REG / STACK / spill bytes for gb0, gn0, gn1, gn2 (`cuobjdump -res-usage` per function incl. `g_phase`,
   `ptxas -v`).
4. **SASS order in g_phase** for gn1 / gn2: LDG.E.128 count, how many before the first reduction FADD and before the first
   STG.E.64 (as in the gb spec).
5. **Python.** Parse test of `k64`, `k64-gb1`, `k64-gn`, `k64-gb1-gn`, `k64-gn-gb2`, bad suffixes, CuTe `gn` -> ValueError;
   extension names of `k64` / `k64-gb1` unchanged.

If (2) shows the rest of the kernel still changes between gn0 and gn1, stop and report (no further workarounds).

## Files Sonnet may touch
`mk/pi05_mk.cu`, `megakernel.py`, `cute/arm.py` (caller update only), `gbatch_equal.py`, this spec (Results only), new
files under `runs/g-noinline-cpu/`.

## Results

All CPU (`CUDA_VISIBLE_DEVICES=`). Products in `runs/g-noinline-cpu/` (`gb0|gn0|gn1|gn2` `.cubin .ptxas .sass .gi.dis .res`, `stagediff.out`, `fd2.py` = per-function compare incl. the kernel body, `order.py`, copies of `funcdiff.py` / `stagediff.py` (line table adapted: stage starts pro 1176, A 1364, B 1379, C 1481, C3 1811, D 1864, E 1883, F 1897, G 1908, H 1984, I 1994)).
Build command = the g-batch one: `nvcc -O3 -std=c++17 -lineinfo -DPI05_PREFIX=968 -DPI05_ATT_KEYS=64 [-DPI05_G_BATCH=N] [-DPI05_G_NOINLINE=1] -arch=sm_100a -cubin $(cat inc.txt) -DTORCH_EXTENSION_NAME=x -Xptxas -v mk/pi05_mk.cu`. As before this is not the torch-extension build itself.

### STOP: acceptance 2 failed (the non-G code still changes between gn0 and gn1)
Per function, `fd2.py gn0.gi.dis gn1.gi.dis` (instruction text, whitespace-normalised):
| function | instrs gn0 / gn1 | result |
|---|---|---|
| `pi05_megakernel` (kernel body) | 16709 / 16709 | DIFFERENT, 3710 instruction positions differ |
| `gemm_phase` (A, F, H) | 1295 / 1295 | DIFFERENT, 244 positions differ (opcode sequence 1278 of 1295 equal) |
| `gemm_phase_wsplit<2>` (D) | 459 / 459 | byte-identical |
| `g_phase` | 293 / 591 | different (expected) |
gn0 vs gn2: kernel body 1463 positions differ (gn1: 3710), `gemm_phase` 244, wsplit identical. Per-stage (`stagediff.py`, kernel body incl. inlined callees; n = instruction count, op_chg = opcode changes, opnd = operand-only changes):
| stage | gn0 vs gn1: n_a/n_b, op_chg, opnd | gn0 vs gn2: op_chg, opnd |
|---|---|---|
| pro | 11750/11750, 169, 2147 | 0, 7 |
| A | 108/108, 12, 34 | 12, 34 |
| B | 550/550, 0, 1 | 0, 1 |
| C | 1038/1038, 0, 0 | 0, 0 |
| C3 | 528/528, 0, 0 | 0, 0 |
| D | 81/81, 0, 0 | 0, 0 |
| E | 444/444, 0, 2 | 0, 2 |
| F | 84/86, 12, 25 | 12, 25 |
| G (call site) | 74/74, 0, 3 | 0, 3 |
| H | 69/68, 11, 26 | 11, 26 |
| I | 1983/1982, 5, 342 | 5, 342 |
(The `callee:N` rows of `stagediff.out` are inlined helper lines; those with N in 800..863 are `g_phase` itself, the rest are noise at operand level, same as above; full table in `stagediff.out`.) So the heavy opcode reschedule of C seen in the gb0/gb1 inline case (606 op changes between gn0 and gb0) is gone: C and C3 are identical, D identical; what remains in the kernel body is mostly register renaming (operand-only, pro 2147 and I 342 for gn1) plus small opcode changes in pro / A / F / H / I. Not caused by anything inside the kernel's own source: only `g_phase` differs in source. Per the spec I stopped here and tried no workarounds. Whether these residual changes cost time on a GPU is not measured.

### gn0 vs gb0 (price of the noinline call itself, same `stagediff.py`)
Functions: kernel body 16709 vs 17018 instrs, `gemm_phase` 1295 vs 1298, `gemm_phase_wsplit` 459 vs 539, all DIFFERENT; gb0 has no `g_phase`. Stages (n_a gn0 / n_b gb0, op_chg, opnd): pro 11750/11762 84/1052; A 108/108 16/55; B 550/552 13/163; **C 1038/1038 606/511**; C3 528/532 55/383; D 81/79 13/31; E 444/444 7/261; F 84/81 35/39; G 74/337 266/26 (the loop is inline in gb0, a call in gn0); H 69/66 26/33; I 1983/2019 183/1506. `g_phase` body is 293 instructions in gn0 (callee bucket 862: 597 instrs in gb0's inlined `reduce_part4` lines).

### 1. Default unchanged (acceptance 1)
`diff <(cuobjdump -sass ../g-batch-cpu/post/gb0.cubin) gb0.sass` -> no output, SASS IDENTICAL (38905 lines).

### 3. Resources (`cuobjdump -res-usage`, `ptxas -v`)
| build | REG | STACK | spill stores / loads | g_phase |
|---|---|---|---|---|
| gb0 | 254 | 72 | 0 / 0 | n/a (inline) |
| gn0 | 254 | 72 | 0 / 0 | 0 B stack, 0 spill |
| gn1 | 254 | 72 | 0 / 0 | 0 B stack, 0 spill |
| gn2 | 254 | 72 | 0 / 0 | 0 B stack, 0 spill |
`cuobjdump -res-usage` lists only the entry kernel (callees are folded into it, no separate REG line for `g_phase`); `ptxas -v` "Function properties" gives 0 stack / 0 spills for `g_phase`, `gemm_phase`, `gemm_phase_wsplit`. Per-function REG of `g_phase`: not verified (not reported by either tool).

### 4. SASS order in `g_phase` (`order.py`; counts all `FADD` in the function, which includes non-reduction FADDs of the gelu code, so FADD totals are not the "64 reduction FADD" of the gb spec)
```
gn0: 293 instrs, LDG.E.128 16, FADD 72, STG.E.64 1: LDG before first FADD 8, before first STG.E.64 16
gn1: 591 instrs, LDG.E.128 32, FADD 144, STG.E.64 2: LDG before first FADD 14, before first STG.E.64 32
gn2: 591 instrs, LDG.E.128 32, FADD 144, STG.E.64 2: LDG before first FADD 14, before first STG.E.64 32
```
All 32 loads come before the first STG in gn1 and gn2. gn1 and gn2 give the same counts here (not distinguished by this metric; the sources differ, the instruction order inside was not diffed).

### 5. Python
```
k64 / k64-gb0 / k32 -> g_batch 0, g_noinline 0
k64-gb1 -> g_batch 1; k64-gn -> g_noinline 1; k64-gb1-gn -> 1/1; k64-gn-gb2 -> g_batch 2, g_noinline 1
bad: k64-gb3, k64-x, k64-gn-gn, k64-gb1-gb2, 64, "k64-", k64-GN -> ValueError
MegakernelArm (load stubbed) extra flags: k64 (); k64-gb1 ('-DPI05_G_BATCH=1',); k64-gn ('-DPI05_G_NOINLINE=1',);
  k64-gb1-gn ('-DPI05_G_BATCH=1','-DPI05_G_NOINLINE=1'); k64-gn-gb2 ('-DPI05_G_BATCH=2','-DPI05_G_NOINLINE=1')
CuteArm: k64, k64-gb1 ok; k64-gn, k64-gb1-gn -> ValueError (gn not supported)
ext names: k64 pi05_mk_p968_k64 (extra () unchanged); k64-gb1 hash 3d4e167f (unchanged, same extra tuple as before); k64-gn b7edbe9c; k64-gb1-gn e9409e81
```
`py_compile` of `megakernel.py`, `cute/arm.py`, `gbatch_equal.py` OK. `gbatch_equal.py` was not run (needs a GPU): not verified. No GPU run of any gn variant (correctness and timing): not verified.

## GPU results (main agent; P968 k64 fuse 18; bench = 5 rounds, arms shuffled per round)

Bitwise (job 8385, `gbatch_equal.py --arms megakernel`, 3 seeds): k64-gb1 / gb2 / gn / gb1-gn / gb2-gn all EQUAL to k64. PASS.

| variant | bench 8385 steady ms (vs k64) | bench 8386 steady ms (vs k64), list order reversed |
|---|---|---|
| k64 | 8.259 | 8.288 |
| k64-gb1 (inline) | 8.325 (+66) | 8.381 (+93) |
| k64-gn | 8.281 (+22) | 8.272 (-16) |
| k64-gb1-gn | **8.243 (-16)** | **8.204 (-84)** |
| k64-gb2-gn | 8.297 (+38) | 8.241 (-47) |

Runs: `runs/20261018T225... -p968-gnoinline`, `runs/20261018T232122Z-p968-gnoinline-rev`. Measured: k64-gb1-gn is the fastest
variant in both jobs (-16 / -84 us, 0.2 / 1.0 %); inline gb1 the slowest in both. Job-to-job spread of the same binary is large
(k64: 8.259 / 8.288 / 8.274 in job 8374), so the size of the gain is not pinned down; the sign is consistent in 2 of 2 jobs.
`probe/mk_stamps_ab.py` has a first-in-round penalty (no shuffle; the first variant reads 60-90 us slow in jobs 8383 / 8385 /
8386), so its per-stage deltas from these jobs are not used. Defaults unchanged (PI05_G_BATCH 0, PI05_G_NOINLINE 0).
