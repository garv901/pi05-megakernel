# SPEC: hoist the next-tile weight load above the partial stores in `gemm_phase` (pi05, CUDA + CuTe)

Owner: Sonnet subagent, CPU ONLY. Do not run sbatch, srun, or anything that uses a GPU (set `CUDA_VISIBLE_DEVICES=` on every
command that imports torch / cutlass). Do not commit. Repo root `/raid/garv901/vibe-megakernel`; pi05 package
`megabench/sota/pi05/` (below: `pi05/`, its own git repo, branch `main`). Python: `.venv/bin/python`.

## Why (measured, CuTe, P968 k64, jobs 8397 / 8398, `pi05/cute/runs/20261018T2359*-gmarks-p968-k64-loop{0,1}`)
Stage F (gate/up, N = 8192, K = 1024, 8 k chunks) gives 73% of warps 2 tiles. Per 2-tile warp, median: MMA tile 1 1.22 us,
partial stores tile 1 0.93 us, then `load_w` of tile 2, then **1.06 us waiting for the tile-2 weights** (`g_wnext->g_wrdy`),
MMA tile 2 1.22 us. The weights for tile 2 are only requested after the tile-1 store loop. Issuing them right after the
tile-1 MMA (before the stores) overlaps their L2 latency with the stores. `b` / `rB` is dead after the last MMA of a tile, so
the live set while the loads are in flight is acc + b, the same as during the MMA.

## The change (both implementations, behind a knob, default OFF)
Inside the tile loop of `gemm_phase` (NOT `gemm_phase_wsplit`):
- now: MMA -> store loop -> `if (ng + ng_step < NG) load_w(ng + ng_step, b)` -> counter publish (counted mode)
- hoist: MMA -> `if (ng + ng_step < NG) load_w(ng + ng_step, b)` -> store loop -> counter publish
Nothing else moves: same math, same store order, same publish position (after the stores), same first-tile load. Results
must stay bit-exact (weights are read-only `.nc` loads, stores go to `part`; no aliasing).

1. CUDA `pi05/mk/pi05_mk.cu`: macro `PI05_LOAD_HOIST` (default 0, `#ifndef` block with a one-line comment next to the
   other knobs, e.g. after `PI05_GEMM_FRAG`). With 0 the preprocessed source of `gemm_phase` must be unchanged.
2. CuTe `pi05/cute/gemm.py` `gemm_phase`: `Knobs.load_hoist: int = 0` (`pi05/cute/config.py`, validate 0/1). Get it into
   `gemm_phase` the same way other build-time knobs reach device code (look at how `d_wpt` / `wpre` are threaded; a new
   `cutlass.Constexpr` parameter passed at the three call sites in `kernel.py` / `ffn.py` is fine). With 0 the PTX must be
   byte-identical to the pre-edit tree. With the IKET loop markers on, keep `g_wnext` right after the hoisted load (so
   `g_wnext->g_wrdy` still measures the exposed weight latency) and `g_st` after the stores.
3. Variant suffix `lh`: `k64-lh` = load hoist on. `main` has no variant parser; add `parse_variant(variant) -> dict` to
   `pi05/megakernel.py` (`k<N>` plus optional suffix `lh`, at most once, ValueError otherwise), used by `MegakernelArm`
   (adds `-DPI05_LOAD_HOIST=1` to `extra` only when on, so plain `k64` keeps its extension name/hash) and by `cute/arm.py`
   (`Knobs(..., load_hoist=...)`). Put `load_hoist` in both arms' `config` dicts. Add `--load-hoist` to `cute/parity.py`,
   `cute/perf.py`, `cute/profile.py` the same way the other knobs are (parity compare file name gets `_lh{N}`).
   Reference for the shape of this plumbing (do not copy G-batch code): `git -C pi05 show g-batch-perf:megakernel.py`.

## Measurement tooling (CPU edits only; the main agent runs them)
4. Port `probe/mk_stamps_ab.py` from branch `g-batch-perf` (`git -C pi05 show g-batch-perf:probe/mk_stamps_ab.py`) to
   `main`, with one fix: the variant order is a fresh random permutation per round (`random.Random(seed)` with a fixed seed,
   record the per-round order in records.jsonl). Known bias without it: the first variant measures +60-90 us. Check that its
   imports (`common.build_mk_arm`, `mk_analyze.analyze`, `phases`) exist on `main` with the needed signatures; if
   `build_mk_arm` does not accept `k64-lh`, make it go through `parse_variant`. MK only (no CuTe arm) is fine.
5. New `pi05/variant_equal.py`: torch.equal of `actions` vs CUDA `k64` for each `--arms`/`--variants` entry and seed
   (default seeds 7301,8917,42811, prefix 968; CUDA `k64,k64-lh`, CuTe `k64,k64-lh`); prints EQUAL / DIFF per entry,
   `RESULT: PASS|FAIL`, exit 1 on any DIFF. Shape: `git -C pi05 show g-batch-perf:gbatch_equal.py`.
6. New `pi05/slurm/load_hoist.sbatch` (partition main, 1 GPU, `--time=00:25:00`, output to `pi05/runs/slurm-%j.log`,
   `source megabench/sota/slurm/env.sh`, `set -uo pipefail`, keep going after a failed step and print its exit code):
   (a) `variant_equal` (both arms); (b) `mk_stamps_ab` with `PROBE_PREFIX=968 MK_VARIANTS=k64,k64-lh`,
   `PROBE_OUT=pi05/probe/runs/<utc>-lhstamps`; (c) `python -m megabench.sota.pi05 --prefix 968 --arms megakernel,cute
   --mk-variants k64,k64-lh --rounds 5 --no-profile --tag p968-load-hoist`. Check that bench.py on `main` passes
   `--mk-variants` to the cute arm too; if it does not, say so in Results and do not restructure bench.py beyond what is
   needed for `k64-lh` to reach both arms.

## Acceptance (CPU; write a "## Results" section at the end of this file, every claim with its command or file)
Products in `pi05/runs/load-hoist-cpu/`.
- A. CUDA byte identity: before any edit save `pre/k64.cubin` + `cuobjdump -sass`; after: default build SASS == pre.
  Build: `nvcc -O3 -std=c++17 -lineinfo -DPI05_PREFIX=968 -DPI05_ATT_KEYS=64 [-DPI05_LOAD_HOIST=1] -arch=sm_100a -cubin
  $(cat inc.txt) -DTORCH_EXTENSION_NAME=x -Xptxas -v mk/pi05_mk.cu` (inc.txt = torch/python include paths; see
  `git -C pi05 show g-batch-perf:mk/SPEC-g-batch.md` Results for the exact recipe that worked).
- B. CUDA hoist build: ptxas `-v` REG / STACK / spill bytes for `gemm_phase` and the kernel, lh0 vs lh1; and in the
  lh1 SASS of `gemm_phase`, the 16 weight `LDG.E.128.CONSTANT` of the in-loop load sit before the first partial-store
  `STG.E.64` of the same iteration (give line numbers). Also report whether any SASS outside `gemm_phase` changed
  (gemm_phase is `__noinline__`, so ideally none).
- C. CuTe byte identity: full kernel P968 k64 PTX (plain build) pre == post with load_hoist 0. Use `compile_build` in
  `megabench/tests/test_pi05_cute_profile.py` as the pattern (fresh process, `CUTE_DSL_ARCH=sm_100a`,
  `CUTE_DSL_KEEP=ptx,cubin`, no cache). Snapshot pre before editing; compiles take minutes, run them in background shells.
- D. CuTe hoist build: REG / STACK / LDL / STL, lh0 vs lh1 (`cute/gemm_marks_check.py PLAIN.cubin MARK.cubin` may do it;
  otherwise `cuobjdump -res-usage` + count LDL/STL in SASS), and the same LDG-before-STG order check in stage F.
- E. Unit tests: extend or add `pi05/tests/` / `megabench/tests/` tests only if cheap: parse_variant good/bad cases,
  Knobs validation. Run the existing CPU tests you touch.
- F. `bash -n` on the sbatch; `python -c 'import ...'` smoke of the new modules with `CUDA_VISIBLE_DEVICES=`.

## Rules
- Keep the surrounding style (dense comments, existing idioms). Every new device path behind the knob.
- Files you may touch: `mk/pi05_mk.cu`, `megakernel.py`, `cute/{gemm,config,arm,kernel,ffn,parity,perf,profile}.py`,
  `probe/mk_stamps_ab.py` (new on main), `probe/common.py` (only if needed by 4), `variant_equal.py`, `slurm/load_hoist.sbatch`,
  tests, this spec's Results. Say why for anything else.
- If something in this spec is wrong about the code, stop at that point, write it in Results, and do not improvise a redesign.

## Results

All CPU (`CUDA_VISIBLE_DEVICES=` on every command). Nothing committed. Products in `runs/load-hoist-cpu/` (`pre/`, `post/`, `inc.txt` copied from `runs/g-batch-cpu/`). Snapshots were taken before the first source edit: CUDA `pre/k64.{cubin,sass,res}` + `pre/ptxas.txt`; CuTe `pre/cute_plain/` (plain) and `pre/cute_li/` (`CUTE_DSL_LINEINFO=1`), compiled from a copy of the tree (`$scratchpad/snap`, `rsync` of `pi05/` before editing) with `scratchpad/run_cute.sh` + `cute_build.py` (fresh process, `CUTE_DSL_ARCH=sm_100a CUTE_DSL_KEEP=ptx,cubin CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1`, no `CUTE_DSL_COMPILER_OPT`, `cpu_arm(Knobs(prefix=968, att_keys=64, ...), Stop())`; ~40 s per compile, not minutes).

CUDA build command (as in the g-batch Results): `nvcc -O3 -std=c++17 -lineinfo -DPI05_PREFIX=968 -DPI05_ATT_KEYS=64 [-DPI05_LOAD_HOIST=1] -arch=sm_100a -cubin $(cat inc.txt) -DTORCH_EXTENSION_NAME=x -Xptxas -v mk/pi05_mk.cu -o ...` (not the torch-extension build itself, so REG/STACK of the real extension build are not verified).

### A. CUDA byte identity: PASS
`diff runs/load-hoist-cpu/pre/k64.sass runs/load-hoist-cpu/post/lh0.sass` -> no output (38905 lines, `cuobjdump -sass`). Also pre `.res` / ptxas numbers equal lh0 (table below).

### B. CUDA hoist build
| build | kernel REG | STACK | spill st / ld | `gemm_phase` ptxas properties |
|---|---|---|---|---|
| pre / lh0 | 254 | 72 | 0 / 0 B | 0 stack, 0 / 0 B |
| lh1 | 254 | 72 | 0 / 0 B | 0 stack, 0 / 0 B |
(`post/lh{0,1}.ptxas`, `post/lh{0,1}.res`, `pre/k64.res`.)
Order check (`runs/load-hoist-cpu/order_cuda.txt`, from `cute/gemm_marks_check.disasm` on the `-lineinfo` cubins; line numbers are lines of `post/lhN.sass`): the iteration of the in-loop tile = 16 `STG.E.64` of `pi05_mk.cu:609` (the `float2` partial stores).
- lh0: STG lines 35757..36003, the 16 `LDG.E.128.CONSTANT` (load_w, loc `sm_32_intrinsics.hpp:131`) at 36143..36205: 0 before the first STG, 16 after the last.
- lh1: the 16 LDG at 35919..35957, STG at 35993..36189: **16 of 16 LDG before the first STG**, 0 after.
Outside `gemm_phase`: not checkable by that name. `cuobjdump -sass` of the cubin holds ONE function (`_Z15pi05_megakernel6Params`; ptxas inlined `gemm_phase` despite `__noinline__`, there is no `Used N registers` line for it), so SASS outside the gemm_phase body is not separable: the kernel goes 38900 -> 38884 instructions lh0 -> lh1 and branch targets after the first gemm_phase site shift, so a text diff is large (SequenceMatcher on address-stripped text: hundreds of hunks; not analysed further). Whether other code than the one inlined gemm_phase site changed was NOT verified. (Only one STG.E.64 group of line 609 exists in the cubin, i.e. one inlined gemm_phase site, 28 STG = 16 + 12 predicated.)

### C. CuTe byte identity: PASS (plain)
`cmp pre/cute_plain/*.ptx post/cute_lh0/*.ptx` -> identical (md5 9dff526611f5d074a25e8e77ca043682 both); `cmp` of the two `.cubin` -> identical. The `CUTE_DSL_LINEINFO=1` PTX differs (`pre/cute_li` vs `post/cute_lh0_li`, first at line 103) because the `.loc` lines of `gemm.py` shifted (source lines added); the lineinfo cubins have the same REG / STACK / LDL / STL (`gemm_marks_check.py pre/cute_li/*.cubin post/cute_lh0_li/*.cubin`: 0 differing lines).

### D. CuTe hoist build (P968 k64, full kernel)
| build | REG | STACK | LDL | STL |
|---|---|---|---|---|
| lh0 (plain, = pre) | 255 | 392 | 51 | 64 |
| lh1 (plain) | 255 | 400 | 55 | 70 |
(`cuobjdump -res-usage` + `grep -c LDL/STL` of `cuobjdump -sass` on `pre/cute_plain`, `post/cute_lh0`, `post/cute_lh1`; the lineinfo builds give the same numbers; per-line table `gemm_marks_lh0_vs_lh1_li.md`, columns "plain" = lh0, "marker" = lh1.) lh1: STACK +8, LDL +4, STL +6; the spill moved: `gemm.py:119` (load_w copy) STL 36 -> 18, `gemm.py:208` (partial store) STL 0 -> 18 and LDL 0 -> 14, plus a few in attention.py:154/185/186/188 (+4 STL, +6 LDL). 15 source lines differ. Whether this costs time is for the GPU A/B.
LDG-before-STG, `runs/load-hoist-cpu/order_cute.txt` (lineinfo cubins; 3 inlined gemm_phase sites, 28 `STG.E.64` of `gemm.py:208` each, in program order A, F, H; the site-to-stage mapping is by program order and the A site by `kernel.py:415` proximity, F vs H not separately proven by line info):
- lh0, all three sites: 0 in-loop `LDG.E.128.CONSTANT` (gemm.py:119) before the first STG, 16 after the last.
- lh1, all three sites incl. site 1 (= stage F): 16 before the first STG, 0 after the last (site 1: LDG sass lines 22915..22945, STG 22993..23207 in `post/cute_lh1_li` cubin).

### E. Tests: PASS
New `megabench/tests/test_pi05_load_hoist.py` (parse_variant good/bad, Knobs.load_hoist validation, CuteArm knobs/config on CPU, MegakernelArm passes `-DPI05_LOAD_HOIST=1` only for `-lh` with `load` stubbed): `CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest megabench.tests.test_pi05_load_hoist` -> 5 tests OK. The existing CuTe CPU tests were not run (they compile for minutes; the parts they touch, `Knobs` / `gemm_phase` signature, were compiled above).

### F. Smoke: PASS
`bash -n slurm/load_hoist.sbatch` ok. `CUDA_VISIBLE_DEVICES= .venv/bin/python -c 'import megabench.sota.pi05.variant_equal, ...cute.parity, ...cute.perf, ...cute.profile, ...cute.arm'` ok; `ast.parse` of `probe/mk_stamps_ab.py` ok (it runs at import, needs a GPU); `cute.parity --help` shows `--load-hoist`.

### Notes / deviations from the spec
- `gemm_phase` has FOUR call sites, not three (`kernel.py` A, `ffn.py` stage_d fallback `d_wpt == 1`, stage_f, stage_h); `load_hoist` is a required `Constexpr` parameter passed at all four.
- `gemm_phase` `__noinline__` does not survive ptxas (single function in the cubin), so spec item B "no SASS outside gemm_phase should change" cannot be shown literally, see B.
- CuTe compile takes ~40 s, not minutes.
- `bench.py` on main already passes `--mk-variants` to both the CUDA and the cute arm; unchanged. `probe/common.build_mk_arm` goes through `MegakernelArm`, hence `parse_variant`; `common.py` unchanged. `mk_stamps_ab.py`: default variants `k64,k64-lh`, `random.Random(20261019)` permutation per round, `round_order` records in records.jsonl; not run (GPU).
- `cute/parity.py` `build_flags` also adds `-DPI05_LOAD_HOIST=1` for the CUDA reference when `--load-hoist 1` (like wpre / d_wpt), so parity compares same-knob builds. The parity compare file name has `_lh{N}`.
- Not run (GPU): `variant_equal.py`, `mk_stamps_ab.py`, the sbatch, bitwise equality. Bit-exactness is by construction only.
- Pre-existing uncommitted changes (`cute/slurm/gemm_marks.sbatch`, `probe/phase_table.py`) are not mine.
