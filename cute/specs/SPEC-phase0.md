# SPEC phase 0: toolchain gate and parity harness

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md` (the whole plan, sections 2-4), then `../../mk/pi05_mk.cu` (the CUDA kernel we port).
This phase builds no part of the real kernel. It builds the scaffolding and three GPU probes that decide how the
later phases write CuTe code that is **bit-exact** to the CUDA kernel.

Paths below are relative to `megabench/sota/pi05/` unless absolute. Repo root: `/raid/garv901/vibe-megakernel`.
Python: `/raid/garv901/vibe-megakernel/.venv/bin/python` (cutlass-dsl 4.7.1, torch, nvcc at /usr/local/cuda/bin).

## Hard rules
- **No GPU.** Do not run anything that needs a GPU. Compile DSL kernels on CPU only:
  `CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a .venv/bin/python ...` (compile, no launch). CUDA extensions may be
  built on CPU (torch cpp_extension + nvcc, `TORCH_CUDA_ARCH_LIST=10.0`). The main agent runs the GPU jobs.
- Do not commit. Do not touch files outside `megabench/sota/pi05/` and `megabench/tests/`, with one exception:
  the `PI05_STOP` edit to `mk/pi05_mk.cu` (task 7).
- The default build of `mk/pi05_mk.cu` must stay byte-identical in SASS (task 7 check).
- **Idiom policy (user decision): fully idiomatic CuTe.** Use `cute.Tensor` + layouts for buffers, `cute.make_tiled_mma`
  with `cute.nvgpu.warp.MmaF16BF16Op` (bf16 in, f32 acc, 16x8x16), copy atoms (`cpasync.CopyG2SOp`, `warp.LdMatrix8x8x16bOp`,
  etc.), `cute.copy`, `cute.gemm`. Express the CUDA permutations as layouts. Use raw `cute.arch.*` / `llvm.inline_asm`
  only for sync primitives, `prefetch.global.L2`, `.cg` cache hints and exact-math cases with no CuTe equivalent.
  Write down every place where you had to leave the idiom, with the reason, in `cute/NOTES.md`.
- Match the surrounding style: short module docstrings, few comments, explain the *why* only.

## Reference material (read, do not copy blindly)
- DSL package source: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/` (cute, cute/arch, cute/nvgpu).
- DSL examples: `.venv/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/ampere/`
  (`flash_attention_v2.py`: ldmatrix/mma.sync/register softmax; `tensorop_gemm.py`; `cooperative_launch.py`;
  `inline_ptx.py`; `dynamic_smem_size.py`; `call_from_jit.py`).
- Earlier CuTe DSL work in this repo: `megabench/kernels/cute_fx1/FEASIBILITY.md` (feature table: red.release, ld.relaxed,
  fence asm, cp.async, cooperative launch, globaltimer), `toy_cute.py` (software grid barrier), `probe_features.py`,
  and the test pattern in `megabench/tests/test_cute_fx1_probe.py`.

## Deliverables

### 1. `cute/__init__.py` (empty) and `cute/config.py`
- Geometry constants as in `pi05_mk.cu:22-110` (T, ADIM, W, L, NSTEP, QD, HD, NQH, KVD, FF, NTHREADS=256, KC=128, CNT_*, ARR_*,
  NSSP, BAR_STRIDE, BAR_WORDS, KS_STRIDE/VS_STRIDE for the FA layout, ATT_ROWS=64, ATT_NRG=7).
- `@dataclass(frozen=True) class Knobs`: `prefix` (128), `att_keys` (64), `fuse` (18), `wpre` (15), `d_wpt` (2). Derived
  properties: `nk`, `att_nchunk`, `smem_bytes` (exact formula of `SMEM_BYTES` with PI05_ATT_FA=1). Validate as the CUDA
  static_asserts do.
- `workspace_layout(knobs) -> dict[name, (offset, nbytes)]` mirroring `make_layout()` (`pi05_mk.cu:1869`) exactly
  (same order, same 256 B alignment, includes `bar`). Names in the order of `layout_offsets()` plus `bar`, and `total`.

### 2. `cute/sync.py` (device helpers, used by every later phase)
DSL `@cute.jit`/`@dsl_user_op`-style device functions (pick what the DSL allows for inlined device helpers; see how
`cute.arch` helpers are written in the package):
- `grid_barrier(bar_ptr, epoch, nblocks)`: exact sequence of `pi05_mk.cu:301-327` with PI05_BARRIER=3
  (sync_threads; thread 0 `atom.add.release.gpu.global.u32` (result unused), spin `ld.acquire.gpu` until
  `(int)(v - epoch*nblocks) >= 0`; sync_threads).
- `wpre_arrive(bar_ptr)` / `wpre_wait(bar_ptr, target)` (`pi05_mk.cu:339-356`, without the stamp/PTRACE parts).
- `counter_publish(ptr)` = `red.release.gpu.global.add.u32 1`; `block_wait(ptr, target, backoff: Constexpr)` (`pi05_mk.cu:270-278`:
  thread 0 relaxed spin, optional `nanosleep 64`, then `fence.acq_rel.gpu`, then sync_threads).
- `prefetch_l2(base_ptr, nbytes)` grid-strided `prefetch.global.L2` per 128 B (`pi05_mk.cu:363-371`).
- `ld_cg` helpers for 32/64/128-bit loads from in-kernel-produced data (must lower to `ld.global.cg`), and a way to
  build CuTe copies whose global source uses `.cg` (cp.async `.cg` is already the CopyG2SOp default for 16 B? check).
- **First-barrier decision:** the CUDA kernel zeroes `bar`/`cnt` in-kernel then uses `cg grid.sync()` once. CuTe DSL has no
  grid.sync. Implement: the host zeroes the `cnt` and `bar` regions of the workspace (`ws[off:off+n].zero_()`) right
  before each launch, and the kernel uses the counter barrier from epoch 1. Bit-exactness is unaffected.

### 3. `cute/probes/smoke_coop.py` (GPU probe; CPU compile must pass)
One `@cute.kernel`, grid = SM count (runtime), 256 threads, `min_blocks_per_mp=1`, dynamic smem = `Knobs(968, 64).smem_bytes`
(and touch all of it), `cooperative=True`. Runtime arg `n_epochs` (default 1634 x 10). Three sub-tests chosen by a Constexpr:
- `barrier`: each epoch e, thread 0 of block b stores `e*1000+b` to `slot[e%2][b]`, `grid_barrier`, then every block reads
  all `slot[e%2][*]` with `.cg` loads and counts mismatches into an error word (atomic add).
- `wpre`: same, with `wpre_arrive` → 64 KB of `.nc` dummy loads into registers → `wpre_wait`.
- `counters`: blocks 0..73 produce rows and `counter_publish` on 7 counters; blocks 74..147 `block_wait` for
  `(e+1)*arrivals` and check the data (the B/C3 handoff pattern).
Host: torch tensors via `from_dlpack`, compile once with `cute.compile`, zero barrier words per launch as in task 2,
run 3 launches, report `errors`, `us_per_barrier` (CUDA event time / n_epochs), and whether the launch was cooperative.
CLI: `--compile-only`, `--out DIR` (writes `smoke.json`).

### 4. `cute/probes/math_parity.py` + `cute/probes/math_ref.cu` (GPU probe; CPU compile must pass)
Goal: for each scalar expression of the CUDA kernel, find a DSL formulation that is bit-identical to nvcc `-O3`
(default flags, no fast-math: `-fmad=true`, precise `expf/tanhf/sinf/cosf/powf`, IEEE `/`, `rsqrtf` = approx).
The CUDA side is an elementwise kernel (`math_ref.cu`, built with `torch.utils.cpp_extension.load` and flags
`-O3 -std=c++17 -lineinfo`, same as `megakernel.py`) that evaluates every expression below on input arrays. The DSL side
evaluates the same expressions, **with several candidate lowerings per expression** (e.g. `cute.math.exp(x, fastmath=False)`,
a libdevice `__nv_expf` call, ex2-based asm; plain `a*b+c` vs explicit fma vs separated rn ops). Expressions (copy the exact
C++ from the kernel; line numbers for reference):
- `rbf(x)` bf16 round-trip; `__floats2bfloat162_rn`; `(float)(double)`.
- `silu` (405), `gelu_tanh` (407-413, note nvcc contracts `x + kKappa*x3` and others), G epilogue
  `rbf(gelu_tanh(rbf(g))) * rbf(u)` (1791).
- adaRMS: `rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f))` and `__fadd_rn(__fmul_rn(__fmul_rn(h, rs), sc1), sh)` with `sc1=__fadd_rn(1, sc)` (496-517, 835-840).
- residual: `rbf(__fadd_rn(h, rbf(__fmul_rn(rbf(o), rbf(g)))))` (932).
- attention: `Mk ? rbf(s) * 0.0625f : -inf`, `rbf(expf(s - m))` incl. m=-inf/0 cases (1460-1481); combine
  `expf(w - M)`, `fmaf`, `num / den` (1711-1728).
- RoPE table: `1.0f / powf(10000.0f, (float)(2*j) / (float)HD)`, `__fmul_rn((float)(nvalid+t), inv)`, `rbf(cosf/sinf(ph))` (1168-1174);
  RoPE apply `rbf(rbf(x1*c) + rbf(-x2*s))` (1308-1309).
- time embedding FP64 block (1155-1166) incl. `pow`, `sin`, `cos`, cast to float.
- cond_matvec dot: chain of `fmaf` (806-809); action_in `v0 = fmaf(...)`, `rbf(v0 + b)` (958-964); action_out `fmaf` + Euler
  `__fadd_rn(x, __fmul_rn(-0.1f, vel))` (1025).
Inputs: about 1e6-1e7 values per expression from realistic ranges (scores ±50, exp args ≤ 0, gelu args ±10, rope phases
0..1100*... as in the kernel, h ~ N(0,1), etc.), plus edge cases (±0, -inf, denormals). Report per expression per candidate:
mismatch count and max ulp. Output `math.json`; pick the winning lowering per expression into `cute/mathx.py`
(you can only fill this in after the GPU run, so ship `mathx.py` with all candidates selectable by name, defaults =
your best guess). CLI: `--compile-only`, `--out DIR`.

### 5. `cute/probes/mma_tile.py` + `cute/probes/mma_ref.cu` (GPU probe; CPU compile must pass)
Decide whether idiomatic CuTe reproduces the CUDA tensor-core math bit-exactly.
(a) **GEMM tile** = one warp task of `gemm_phase` with PI05_GEMM_FRAG=2 (`pi05_mk.cu:424-609`): A [64 x 128] bf16
(rows >= 50 zero), B = 32 weight rows x 128 k, acc [64 x 32] f32. CUDA reference copies the exact loads, the
fragment-interleaved A staging (531-536), the mma loop (569-590) and the store. Mapping to reproduce: lane (g = lane/4,
t4 = lane%4) in 32-wide k slab `sl` holds physical k = `sl*32 + t4*8 + j`, j = 0..7, for both A and B. The first mma of
the slab uses j = {0,1} as logical k {2t4, 2t4+1} and j = {2,3} as logical {8+2t4, 9+2t4}; the second mma uses j = {4,5}
and {6,7} the same way. Accumulation order per acc: (sl0,mma0), (sl0,mma1), (sl1,mma0), ... . The CuTe version: a
`TiledMma` over one warp, and A/B **views whose K mode is permuted by a layout** so that `cute.gemm` over k-blocks 0..7
in order performs the same products. Use CuTe copies for staging. Also test K = 128 chunks at different `k0`.
(b) **wsplit tile** (`gemm_phase_wsplit<2>`, 621-741): two warps split the 4 slabs (2 each), partials to smem, fixed-order
float4 sum (warp 0 + warp 1). Same mapping.
(c) **FA unit** (1411-1536): one warp pair (dh = 0,1) on one m-tile: Q [16 x 256], K [64 x 256], V [64 x 256] in smem with
KS_STRIDE/VS_STRIDE padding; QK^T via ldmatrix x4 (natural k order, kk = 0..15), P given as an input bf16 [16 x 64]
packed into A fragments from the S accumulator layout (C->A repack as in 1474-1499), PV via ldmatrix x4 trans. CuTe version:
`LdMatrix` copy atoms + TiledMma, accumulator-to-operand retile (as in `flash_attention_v2.py`).
Inputs random bf16, several seeds. Output `mma.json`: per case, bit-equal yes/no, mismatch count, max abs diff.
CLI: `--compile-only`, `--out DIR`. If you cannot make a case idiomatic and bit-exact on paper, also write the
fallback (raw fragment code in the DSL) as a second candidate and report both.

### 6. `cute/mathx.py`
Device scalar helpers used by later phases: `rbf`, `fadd_rn`, `fmul_rn`, `fma`, `silu`, `gelu_tanh`, `expf`, `rsqrtf`, ...,
each with the candidate lowerings of task 4 selectable by a module-level dict, default = best guess.

### 7. `PI05_STOP` debug stop in `mk/pi05_mk.cu`
Macros `PI05_STOP_PH`, `PI05_STOP_STEP` (default 0), `PI05_STOP_LAYER` (default 0). Only when `PI05_STOP_PH` is defined:
at the stage points below, if `step == PI05_STOP_STEP && l == PI05_STOP_LAYER` (ignore both for ph 0), every block does
`cp_async_wait_all(); return;`. All blocks reach the same point, so no barrier is left waiting.
Stage ids (put the check right after the stage's work, before its trailing SYNC / counter wait of the next stage):
0 = prologue end (after step-0 action_in, line ~1224), 1 = A (after the gemm_phase call), 2 = B (after the B task loop),
3 = C3 (after the combine loop), 4 = D, 5 = E, 6 = F, 7 = G, 8 = H, 9 = I (after the whole I block, incl. the last-layer tail).
Without the macro the preprocessed code must be unchanged. **Check:** build the default k16/k32/k64 at P=128 on CPU
before and after your edit (`megakernel.load(128, k)` into a scratch `TORCH_EXTENSIONS_DIR`), compute
`cuobjdump -sass <so> | sha256sum`. Before == after is required. Also compare with `probe/sass_baseline.txt` and report
whether that matches (it may differ only by build environment; say so).

### 8. `cute/parity.py`
Library + CLI used by every later phase. `run_cuda(knobs, stop, seed) -> dict[name, torch.Tensor(uint8)]`: build/load the
CUDA ext with `extra=(-DPI05_STOP_PH=.., -DPI05_STOP_STEP=.., -DPI05_STOP_LAYER=..)` (`megakernel.load`), zeroed workspace,
inputs from `fixture.make_inputs(Geometry(prefix=...), seed)`, launch with fuse=knobs.fuse, return each workspace buffer
(by `layout_offsets()`) plus `actions`. Stage names `pro, A, B, C3, D, E, F, G, H, I` with `--step/--layer`, or `full`.
`run_cute(...)` is a stub that raises NotImplementedError until phase 1. `compare(a, b, names) -> report` byte-compares
(and for mismatches gives count, first index, max ulp interpreting the dtype of that buffer). Phase-0 CLI mode `--selfcheck`:
run CUDA at each stage twice and require byte-equality (determinism), and check `full` actions match the default build.
Also a test asserting `config.workspace_layout` == the ext's `layout_offsets()` (buildable on CPU).

### 9. Tests: `megabench/tests/test_pi05_cute_phase0.py`
CPU only (`CUDA_VISIBLE_DEVICES=`, `CUTE_DSL_ARCH=sm_100a`), subprocess like `test_cute_fx1_probe.py`, skip if no venv:
compile-only runs of tasks 3, 4, 5; layout equality (task 8); the SASS-unchanged check of task 7 can be a slow test marked
by an env var. All must pass before you finish.

### 10. `cute/slurm/phase0.sbatch`
Like `slurm/pi05.sbatch` (partition priority, 1 GPU, 30 min, `source megabench/sota/slurm/env.sh`). Writes
`cute/runs/<UTC ts>-p0/`: smoke.json (P=968/k64 smem), math.json, mma.json, parity selfcheck at P=128 k16 and P=968 k64
for stages pro, A, B, C3, I (step 0 layer 0) and full. Do not submit it.

### 11. `cute/NOTES.md`
Start the log: what was built, CPU compile results, register/smem usage of the probes (`CUTE_DSL_KEEP=cubin` dump +
`cuobjdump -res-usage`), idiom exceptions, open questions for the GPU run.

## Done when
All CPU tests pass, the SASS check of task 7 passes, and you report: files written, test output, the SASS hashes, every
idiom exception and every guess that the GPU run must confirm. Keep the report short and factual.
