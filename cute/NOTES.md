# CuTe port of pi05_mk.cu: Phase 0 log

Environment: cutlass-dsl 4.7.1, `CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a`, `.venv/bin/python`, no GPU used.

The phase sections are a dated log. CPU STACK / spill numbers before the phase 6 spill fix are superseded (full P=128/k16 build now: REG 255, STACK 392). Timing deltas between different jobs are cross-job and not a same-job comparison. Claims moved to the last section ("Unverified") must be rederived before use.

## Built
- `config.py`: constants, `Knobs(prefix, att_keys, fuse, wpre, d_wpt)` with validation, `workspace_layout(knobs)` (same order and 256 B alignment as `make_layout()`; plus `bar` and `total`). `smem_bytes` = 117248 for (968, 64), (128, 16), (128, 64).
- `sync.py`: counter grid barrier (`PI05_BARRIER=3`), WPRE arrive/wait, `counter_publish`, `block_wait`, `prefetch_l2`, `.cg` / `.nc` copy atoms and `ld_cg_*`.
- `mathx.py`: scalar helpers with selectable lowerings (`CANDIDATES`, `CHOICE`, `use()`); defaults are the first candidate of each entry (guesses until `math.json` exists).
- `probes/smoke_coop.py`, `probes/math_parity.py` + `math_ref.cu`, `probes/mma_tile.py` + `mma_ref.cu`.
- `parity.py`: `run_cuda`, `run_cute` (stub), `compare`, `--selfcheck`, `--layout-check`.
- `mk/pi05_mk.cu`: `PI05_STOP_PH/STEP/LAYER` debug stop (stage ids 0..9). Default SASS unchanged: k16/k32/k64 at P=128 hash to `probe/sass_baseline.txt` (slow test `PI05_CUTE_SLOW=1`). Builds with `-DPI05_STOP_PH=0/4/9` compile. At the last step and layer the stop-9 check is skipped (`if (!more) continue;`), so that run ends naturally; `parity --selfcheck` checks it equals the default actions.
- `slurm/phase0.sbatch` (not submitted), `megabench/tests/test_pi05_cute_phase0.py`.

## CPU results
- `python -m unittest megabench.tests.test_pi05_cute_phase0`: 6 pass, SASS test skipped by default; with `PI05_CUTE_SLOW=1` it passes.
- smoke_coop compiles for modes barrier / wpre / counters. PTX contains `atom.global.release.add.u32`, `red.release.gpu.global.add.u32`, `ld.global.cg.b32`, `ld.global.nc.v4.b32` x16 (wpre), `nanosleep.u32`, `.reqntid 256`, `.minnctapersm 1`.
- math_parity compiles all 23 expressions under every distinct profile (default, ptx, split, ex2, fast).
- mma_tile compiles gemm / wsplit / fa, each as `cute` and `raw`. PTX instruction counts: gemm 128 mma (cute and raw; same as CUDA), wsplit 64 per warp, fa 192 mma with 80 `ldmatrix.x4` (16 Q + 64 K) and 32 `ldmatrix.x4.trans` (V), matching the CUDA kernel's counts.
- Registers / static smem (`CUTE_DSL_KEEP=ptx,cubin`, `cuobjdump -res-usage`; dynamic smem is set by the launch):

| kernel | REG | SHARED (static) |
|---|---|---|
| smoke barrier / wpre / counters | 17 / 78 / 20 | 1024 |
| gemm cute / raw | 128 / 140 | 1024 / 0 |
| wsplit cute / raw | 122 / 104 | 1024 |
| fa cute / raw | 52 / 138 | 1024 |

  The math probe's usage was not recorded. The 1024 B static smem looks DSL-reserved (not requested by the kernels).

## Idiom exceptions
1. `nvvm_wrappers` (`W.atomic_add/red/load`) for release/acquire atomics; `inline_ptx` for `nanosleep` and `prefetch.global.L2` (no DSL op found).
2. Grid barrier first epoch: the host zeroes `bar` and `cnt` before each launch and the kernel starts at epoch 1 (the CUDA kernel zeroes them in-kernel behind a cooperative-groups `grid.sync`).
3. `mathx` "ptx" candidates are verbatim nvcc expansions of `expf` / `tanhf` as inline PTX; `zero_reg()` in `mma_tile` is an inline `mov.f32` because a literal accumulator becomes an immediate 'n' constraint that libNVVM rejects.
4. `mma_tile` `raw` candidates use inline `mma.sync` PTX with explicit per-lane fragments (the spec's fallback).
5. `cute` gemm/wsplit stage A in a plain row-major smem tile (stride 160) and read B straight from gmem with 32-bit copy atoms; the CUDA fragment-interleaved A layout and cp.async-to-smem B are not reproduced (they do not change the products). The wsplit partial tile is 64 rows per warp in smem (CUDA: 50), so smem differs; GPU only needs the products and the fixed-order sum.
6. Fake tensors need `stride_order=(2, 1, 0)` for row-major (default is column-major).

## Guesses the GPU run must confirm
- `KPERM = (2,4,2,2,4):(1,8,2,4,32)` applied by `cute.composition(t, (None, KPERM))` with `make_tiled_mma(..., permutation_mnk=(16,16,16))` gives the CUDA fragment-to-k assignment (same instruction operands) and CuTe's A/B fragment order (e, row/n, k-half) matches the PTX register order. Compile cannot check bit equality; `mma.json` decides gemm/cute, wsplit/cute, fa/cute vs raw.
- `launch(cooperative=True, min_blocks_per_mp=1, smem=117248)` sets the max-dynamic-smem attribute and launches cooperatively on 148 SMs (`smoke.json`).
- Release/acquire counter protocol (epoch * nblocks target) and `.cg` slot reads are race-free for ~16k epochs; `us_per_barrier` vs the CUDA kernel's barrier cost.
- Which `mathx` lowering is bit-identical per expression (`math.json`); defaults are guesses. FP64 `pow/sin/cos` go through libdevice; the CUDA time-embedding block uses nvcc's libm expansion.
- The tiled copy of `P` through `partition_C` into `rP` and the repack `((4,2),MMA_M,MMA_N/2)` match the CUDA C->A packing (fa/cute).
- `sVt` view with a dynamic `dh` column offset keeps 16 B alignment for `ldmatrix.trans` (compiled, runtime unchecked).
- `parity --selfcheck`: all stages are deterministic including the counter and barrier words, and the stop-at-end run equals the default actions.

## Not done
- No GPU run, so no bit-equality, timing or occupancy numbers.
- `run_cute` is a stub; the Phase 0 optional CPU PTX-vs-SASS comparison of math candidates was not done.

# Phase-0 GPU results (job 7745, `runs/20261016T223558Z-p0/`)
- smoke: cooperative launch OK on 148 SMs, 117248 B smem, 16340 epochs, 0 errors in all modes. barrier 1.43 us/epoch, wpre 1.89, counters 2.82 (different harness from the CUDA kernel; no CUDA barrier price is recorded here).
- math: every `mathx` default lowering is bit-exact for all 23 expressions (4e6 inputs each). Some `ex2` (5 of 23 expressions), `fast` (7) and `split` (3) candidates are not bit-exact (gelu_tanh split 9357 mismatches, action_out split 575241); not used.
- mma: gemm, wsplit and fa are bit-exact in both `cute` (TiledMma + permuted-K layout + copy atoms) and `raw`. The `cute` variants are the base of the real kernel. This confirms the K-permutation, launch and barrier guesses listed above.
- parity selfcheck: deterministic at pro, A, B, C3, I and full for P=128/k16 and P=968/k64; the stop-at-end build equals the default build.

# Phase 1: kernel skeleton + prologue (`pro`)
Built: `kernel.py` (megakernel, host jit, `Stop`, `SYNC`/`wpre` helpers, input/workspace specs), `rowops.py` (`warp_sum`, `block_sum`, `cond_matvec`, `action_in_task(s)`), `arm.py` (`CuteArm`), `parity.py` (`run_cute`, `--compare`, `--mask-tail`, `--timing`), `slurm/phase1.sbatch` (not submitted), `megabench/tests/test_pi05_cute_phase1.py`.
- Kernel order follows the P0..P3 prologue block of pi05_mk.cu (line numbers cited in the specs are stale): nvalid (same `block_sum`), P0 time embedding + RoPE, barrier epoch 1, P1/P2/P3 `cond_matvec`, `prefetch_l2` of layer-0 wq/wk/wv, step-0 `action_in_tasks` (TAIL + DEFNORM path). In-kernel `cnt`/`bar` zeroing is replaced by the host zeroing one contiguous slice cnt..bar (ssp lies between them and is rewritten every step).
- Scope decision: `fuse & 16` (DEFNORM) is required (`Knobs` raises otherwise); the non-defnorm row path (`action_in_row`, `residual_row`) is not ported. Other fuse bits remain knobs.
- `Stop(stage, step, layer)` accepts only `pro` for now (later phases extend `STAGES`); `Stop(None)` runs everything implemented, i.e. the same as `pro`.
- CPU compile (`Stop('pro')`, P=128/k16 and P=968/k64, identical code): REG 255, STACK 688, LOCAL 0 (cuobjdump); ptxas -v (cu13 ptxas on the DSL PTX): 1224 B spill stores, 1272 B spill loads. The CUDA kernel (full, not only `pro`) has REG 255, STACK 72. One `cond_matvec` alone spills 600 B, so the spill comes from the matvec (4 x 8 float4 weights + 40 accumulators) and not from the FP64 math. Not fixed in phase 1; a phase-6 item (candidates: stage weights per row, smaller task).
- Static smem 1024 B again (DSL), plus `red` / `s_valid` via `SmemAllocator`; dynamic smem 117248 B (`get_dyn_smem`).

Idiom exceptions (phase 1)
1. P3 weight/bias row selection (attn / ffn / final norm) builds a pointer from an integer address (`iterator.toint()`, `make_ptr(..., assumed_align)`) because one task picks one of three tensors at runtime.
2. Workspace views are built inside the kernel from the uint8 `ws` (`ws_views`) instead of being passed as separate kernel arguments (same bytes and offsets, avoids 20 extra kernel parameters).
3. Output lambdas use `tensor.__setitem__` (closures passed as Constexpr callbacks into `cond_matvec`).
4. `prefix_mask` is passed as a uint8 view of the bool tensor.

Guesses the GPU run must confirm
- `parity --compare --stages pro` byte-equal for all workspace buffers except `bar` (temb, tmid, cond, mod, xbuf, rope, h, ssp, cnt at `pro`; the others are zero). This depends on: `cond_matvec` fma order, `block_sum` order, `lane == s` selection, `.cg` vs `.nc` (no effect on bits), and the P3 pointer arithmetic.
- `launch(cooperative=True)` with 255 registers and 688 B stack is accepted, and the spills do not break correctness.
- `CuteArm.bind` compiles with `from_dlpack` tensors (compile with real CPU tensors verified; the GPU call path unverified).
- Per-launch cost of `CuteArm.launch` (one `zero_()` + compiled call); `--timing` prints CUDA vs CuTe `pro` times.

Not done in phase 1: stages A..I, `run_cute` beyond `pro`, bench registration, GPU run.

# Phase-1 GPU result (job 7747, `runs/20261016T230048Z-p1/`)
- `parity --compare --stages pro`: byte-equal CUDA vs CuTe for every buffer except `bar` in P=128/k16, P=968/k64 and P=128/k16 mask-tail 17, seeds 7301 and 7302 (6 cases). So the cond_matvec fma order, `block_sum` order, P3 pointer arithmetic and the cooperative launch with the 688 B stack are confirmed.
- Timing, `pro`-stop launch, median of 20: CUDA 0.158 ms, CuTe 0.200 ms (+42 us).

# Phase 2: GEMM engine, layer/step loops, stages A and B
Built: `gemm.py` (`gemm_phase`, `stage_a`, `load_w`, `a_layout`), `kernel.py` (step/layer loops, `phase_b`, `_after_a`, `STAGES = (pro, A, B)`), `rowops.py` (`reduce_part4`), `parity.py` (`--fuse`, non-vacuity report, `time_stop`), `slurm/phase2.sbatch` (not submitted), `megabench/tests/test_pi05_cute_phase2.py`. Existing tests adjusted: `Stop('A')` is now valid (phase-1 scope test uses `Stop('C3')` as the invalid stage).

## Loops, stops
- `step` / `l` are runtime `cutlass.range(..., unroll=1)` loops. A stop (stage, step, layer) is a trace-time constant: the outer trip count becomes `step+1`, the inner bound `layer+1` in the last step, and the stages after the stop stage are wrapped in `if keep` (`keep = (step != stop.step) | (l != stop.layer)`), so the stopping iteration returns exactly where the CUDA macro does (A: before the barrier / B; B: after the B tasks). `Stop('pro')` compiles no loop at all. No stop = A and B for all layers.
- WPRE bit 1: `pend` (runtime Boolean) defers the step-0/layer-0 barrier into the A GEMM; the epoch is consumed in the loop (`epoch + 1`, target = epoch * nblocks) exactly as the `wpre()` lambda; without the WPRE bit the barrier runs before the loop. `a_nmod` is always `mod` (step 0 row of norm 0); the switch to `nextmod` / next-step mod belongs to I (phase 4/5), so a run without a stop computes wrong values for layers >= 1 (stage C..I missing), as the spec allows.
- Hooks left: ATT_PREFILL=2 K/V prefill sits in `_after_a` (marked), between the L2 prefetches and `phase_b`.

## gemm_phase (all modes implemented: plain, norm, counter, WPRE; virtual N over up to three tensors)
- MMA formulation is the phase-0 one (TiledMma m16n8k16, `permutation_mnk=(16,16,16)`, KPERM `(2,4,2,2,4):(1,8,2,4,32)`, K steps in ascending kb order, so each accumulator sums in the CUDA order).
- B: straight global -> registers. `partition_B(composition(gB, (None, KPERM)))` has the layout `((2,2),4,(2,4)):((1,2),8*K,(4,32))`, and `make_fragment_B` is `((2,2),4,(2,4)):((1,2),32,(4,8))`. Regrouping the value mode and the m16 step into one mode of 8 gives `((2,2,2),4,4):((1,2,4),8K,32)` on gmem and `((2,2,2),4,4):((1,2,4),32,8)` in registers, i.e. 8 contiguous bf16 per (n-tile, slab) on both sides, so one `copy` with the `.nc` 128-bit atom does it: 16 `LDG.E.128.CONSTANT` per task (SASS). The layouts are asserted in `test_partition_layouts`.
- A tile layout choice: the **fragment-interleaved layout as a CuTe layout** (not the row-major fallback): `a_layout() = ((8,2,4),(2,4,2,(2,4))):((32,2,2048),(1,8,4,(256,512)))` over logical (m, k), i.e. uint4 [mt][sl][half][g*4+t4] of the CUDA kernel; `partition_A` gives `((2,2,2),4,8):((1,2,4),2048,256)` and the retiled copy source is 8 contiguous values per (mt, kb), so `make_tiled_copy_A` with a 128-bit universal atom is one LDS.128 per mma A operand. Staging stores are 4 x STS.128 per thread (the 2 pieces of rows r, r+8 are repacked in registers, as the CUDA code does). 16 KiB used (CUDA: 20 KiB with padding, unused here).
- SASS of the main loop (Stop('A'), fuse 18, P=128): 128 HMMA.16816.F32.BF16, 32 LDS.128, 32 STG.E.64 (rows < 50 predicated), 16 LDG.E.128.CONSTANT (next-task weights); CUDA's loop has 32 LDS.128 (4 mt x 4 sl x 2) and 128 HMMA, the same.
- Whole-kernel counts (include the prologue's cond_matvec): LDS.128 274, HMMA 128 for every stop A / B build.
- Norm mode: ss parts from `ssp[r][pc]`, xor 8, 4, 2, 1 with lane-in-half-warp = part index, broadcast from lane `tid & 16`, `X.ada_rs`, `fadd_rn(fmul_rn(fmul_rn(h, rs), fadd_rn(1, sc)), sh)` -> bf16 rn, rows >= 50 untouched; scale / shift / ss loads are `.cg`.
- Counter mode: `sync_warp`, lane 0 `red.release` on `cnt + ((n0/cdiv) % cmod)*CNT_STRIDE`, issued after the task's stores and after the next task's weight loads.
- Non-inlined device function: not possible, as far as found: `cute.jit` in cutlass-dsl 4.7.1 has no noinline option, so `gemm_phase` is inlined into the kernel (one instance per call site; phase 2 has one call site).

## Phase B
`phase_b` follows the B block of pi05_mk.cu: task = `gridDim-1-blockIdx` stepping by `gridDim`, `NTASK = 7`, 10 heads, `block_wait` on the head counter (FUSE_AB), `reduce_part4` (8 `.cg` float4 loads summed in order from +0.0), `rbf`, RoPE from the `rope` table (`.cg`) for heads < 9 via `X.rope_apply`, 64-bit stores of 4 bf16 to `q` / `ksuf` / `vsuf`. L2 prefetch of `wo[l]`, `prefix_k[l]`, `prefix_v[l]` before it.

## CPU numbers (cu13 ptxas -v on the DSL PTX; cuobjdump)
| build | REG | STACK | LOCAL | spill stores / loads (B) |
|---|---|---|---|---|
| pro (phase 1) | 255 | 688 | 0 | 1224 / 1272 |
| A, fuse 18 (P=128/k16 and P=968/k64 identical) | 255 | 704 | 0 | 1324 / 1384 |
| B, fuse 18 | 255 | 712 | 0 | 1384 / 1544 |
| A and B, fuse 19 | 255 | 688 | 0 | 1224 / 1280 |
The prologue's cond_matvec (600 B per instance) still dominates the spills; the GEMM adds at most about 160 B of stores in the fuse-18 builds and nothing visible with fuse 19. The CUDA full kernel has STACK 72. Not tuned (phase 6).

Idiom exceptions (phase 2)
1. `load_w` builds the weight tile pointer from an integer address (`iterator.toint()` + `make_ptr(..., assumed_align=16)`) for the runtime choice among three tensors (as P3 in phase 1), and re-wraps the `partition_B` iterator with `assumed_align=16` so the regrouped 8-value copy can be 128 bit (the lane offset `g*K + 8*t4` elements is 16 B aligned, but the DSL cannot see it).
2. The accumulator / `part` stores use `make_ptr(..., assumed_align=8)` per (mt, r8, nt) instead of `partition_C(gC)` (the partitioned pointer loses its alignment, which a 64-bit copy needs). The mapping equals `partition_C` (layout asserted: `((2,2),4,4):((1,8000),16000,8)` for a 1000-stride tile).
3. The smem A staging writes 128-bit units through a `(1024, 4)` u32 view of the same bytes as `a_layout()` (`local_tile` for the alignment), because the repack of two pieces into mma operands is a register permutation.
4. `phase_b` selects the q / ksuf / vsuf destination by integer address (dynamic head).

Guesses the GPU run must confirm
- Byte equality of `part` (stop A), and of `q`, `ksuf`, `vsuf` (stop B), for fuse 18 and fuse 19, including mask-tail 17 (RoPE positions shift). Depends on: the A fragment layout (logical coordinates through `a_layout`), the B regrouping, the norm-staging order, the 0-start sum of `reduce_part4`, `rope_apply`.
- Grid barrier accounting with the deferred WPRE barrier (epoch 5 consumed in A), and the counter hand-off A -> B (cnt equality at fuse 19).
- Whether the hoisted A fragment loads / extra spill hurt the A phase time (informative `--timing --stages B`).
- The `nonzero_frac` report: pro: temb tmid cond mod rope h ssp xbuf; A adds part; B adds q ksuf vsuf. All must be nonzero in the CUDA dump (a warning is printed otherwise).

Not done in phase 2: stages C3..I, the ATT_PREFILL hook body, the `nextmod` switch, WPRE bits 2/4/8, GPU run.

# Phase-2 GPU results (jobs 7748 fail, 7749 pass; `runs/20261016T232255Z-p2/`)
- 7748: compile failure, `prefix_k` / `prefix_v` are 4-D `[L, P, 1, HD]` in the real inputs (the phase-2 fake tensors had 3-D). Fixed; the fake-tensor compile path was removed. All compile tests now go through `arm.cpu_arm` (the exact `CuteArm.bind` path on CPU tensors whose shapes / dtypes / strides come from `fixture.make_inputs` under `FakeTensorMode`).
- 7749: pro, A, B byte-equal CUDA vs CuTe for P=128/k16 seeds 7301 / 7302, P=968/k64, P=128/k16 mask-tail 17 and P=128/k16 fuse 19 (counter hand-off A -> B). Non-vacuous (q / ksuf / vsuf about 99.6% nonzero). B-stop launch median: CUDA 0.173 ms, CuTe 0.224 ms (+51 us; the phase 1 `pro` gap was +42 us, a different job and a different build).

# Phase 3: attention (K/V prefill, C, C3)
Built: `attention.py` (`att_smem`, `mask_bytes`, `load_kv`, `prefill`, `fa_unit`, `phase_c`, `phase_c3`), `kernel.py` (`STAGES = (pro, A, B, C3)`, `_after_a` now ends with prefill + B, new `_after_b` = SYNC, C, SYNC unless FUSE_CC3, wg / wu L2 prefetch, C3), `parity.py` (stop C3; `expected()` adds stats opart attn, and cnt when fuse bit 2), `slurm/phase3.sbatch` (not submitted, `--time` 30 min), `megabench/tests/test_pi05_cute_phase3.py`.
- Smem carve (bytes as CUDA, `Knobs.smem_bytes` unchanged): Ks `(KEYS, 256):(264, 1)` at 0, Qs `(64, 256):(264, 1)`, Vs `(KEYS, 256):(264, 1)`, Mk uint8 `(KEYS)`. All are CuTe smem tensors over the same dynamic smem as the GEMM A tile.
- Prefill (ATT_PREFILL 2) runs in `_after_a` before B: blocks < NRG*NCHUNK, CTA sync, mask bytes, cp.async of prefix rows (rows P <= j < NK skipped, rows >= NK zero filled), commit. Unit loop: the block's first unit (`u == bid`) loads only rows `jlo..jhi` (P <= j < NK) and does not write Mk.
- FA formulation = the phase-0 `fa/cute` probe (TiledMma m16n8k16, ldmatrix A / B for S, `logical_divide` regroup of the C fragments into A fragments for P V, ldmatrix.trans for V), run per warp with `mt = warp & 3`, `dh = warp >> 2`; the fragment loads sit inside the constexpr k-loop (per kk for Q / K, per ks for V) so that ptxas can keep them short-lived. Softmax pieces use the `mathx` defaults (`attn_score`, `expf`, `rbf`, `comb_w`); row max is `cute.arch.fmax` (the same `max.f32` as `fmaxf`), quad exchange xor 1 then 2, `rsum` accumulated per n-tile in element order, then quad xor 1, 2.
- SASS counts (whole kernel, Stop C3): HMMA = 128 (gemm task) + 16 x KEYS/8 (S) + KEYS/16 x 16 (P V): 176 / 224 / 320 for k16 / k32 / k64, asserted in the tests. LDSM 40 / 64 / 112 = 16 (Q) + 16 x KEYS/16 (K) + 8 x KEYS/16 (V), the CUDA per-warp count.

CPU numbers (cuobjdump; cu13 ptxas -v on the DSL PTX); every build LOCAL 0, REG 255:
| build | STACK | spill stores / loads (B) |
|---|---|---|
| C3 P=128 k16 / k32 / k64, P=968 k64, fuse 18 | 712 | 1404 / 1628 |
| C3 P=128 k16 fuse 16 (barrier path) | 720 | 1412 / 1664 |
The phase-1 cond_matvec spills still dominate; attention adds about 20 B of stores to the B build.

Idiom exceptions (phase 3)
1. cp.async uses `cute.arch.cp_async_shared_global(dst, src, 16, "cg", cp_size=nbytes)` (src-size 0 = zero fill, as CUDA's `cp.async ... 16, src_bytes`), not a `CopyG2SOp` atom: a predicated atom copy skips masked pieces instead of zero filling them. Source addresses are `make_ptr(..., assumed_align=16)` from integer addresses (dynamic choice between prefix / ksuf / vsuf rows).
2. `stats`, `opart` and `attn` stores are 64-bit copies to `make_ptr(..., assumed_align=8)` tensors (as `part` in phase 2).
3. `jlo` / `jhi` clamps are written with dynamic `if`s (no min / max on dynamic Int32 inside the DSL helpers).

Guesses the GPU run must confirm
- Byte equality of `stats`, `opart`, `attn`, `cnt` at stop C3 for ATT_KEYS 16 / 32 / 64, fuse 18 and 16, mask-tail 17 / 200. Depends on: ldmatrix mapping with the dynamic m-tile offset (`local_tile(sQ, (16, HD), (mt, 0))`), the P C -> A regroup with 8 warps, `fmax` semantics (including +-0), the zero-fill cp.async, and the `pre` window logic.
- No smem race between the prefill (issued before B) and the previous layer's smem users (CTA sync at the start of the prefill, as CUDA).
- `cp_async_wait_group(0)` semantics equal CUDA's commit + wait_group 0.

Not done in phase 3: stages D..I, GPU run.

# Phase-3 GPU results (job 7759; `runs/20261016T233957Z-p3/`)
- B and C3 byte-equal CUDA vs CuTe in all 8 cases: P=128 k16 (seeds 7301, 7302), k32, k64; P=968 k64; P=128 k16 mask-tail 17; P=968 k64 mask-tail 200; P=128 k16 fuse 16 (CC3 off, barrier path). Non-vacuous (attn 99.8% nonzero).
- C3-stop launch median: P=128/k16 CUDA 0.185 ms vs CuTe 0.234 ms (raw 0.18544 / 0.23435); P=968/k64 CUDA 0.200 ms vs CuTe 0.240 ms (raw 0.19968 / 0.23965). B (job 7749) to C3 (job 7759) adds 12.5 us CUDA and 10.9 us CuTe (cross-job; no same-job B and C3 pair).

# Phase 4: stages D..I of a layer (non-last layers of step 0)
Built: `gemm.py` (`ws_stage_a`, `ws_load_w`, `gemm_phase_wsplit`), new `ffn.py` (`residual_tasks`, `stage_d` .. `stage_i`, modulation rows by byte address), `kernel.py` (`STAGES` D..I, `_tail` blocks, `a_nmod` / `pend` carried per layer, the phase-2 shortcut `a_nmod = step 0 mod` removed), `parity.py` (expected sets, `--wpre`, `--d-wpt`, `--timing --layer`, compare file name carries wpre / d_wpt / layer / stages), `config.py` (wsplit smem check only for `d_wpt > 1`; d_wpt 1 runs the plain `gemm_phase` as in CUDA), `slurm/phase4.sbatch` (not submitted, `--time` 30 min), `megabench/tests/test_pi05_cute_phase4.py`. The phase-1 Scope test now uses `Stop('Z')` as the invalid stage.
- Tail structure: blocks D, E, F, G, H, I, hand-off. Each block starts with the barrier that closes the previous one (so a stop, which returns before the trailing barrier, is a plain skip of the later blocks): D = SYNC (unless WPRE 8 and d_wpt > 1: deferred into the wsplit GEMM) + L2 prefetch of wd and mod rows 2l, 2l+1; E = SYNC, residual_tasks<QD/KC> (gate of norm 2l); F = SYNC unless WPRE 2 (deferred into the GEMM); G = SYNC unless FUSE_FG, prefetch of the next layer's wq / wk / wv (`ln = l+1 < n_layers ? l+1 : 0`), GeGLU tasks; H = SYNC unless WPRE 4 (deferred), prefetch of the next norm row; I = SYNC, residual_tasks<FF/KC> (gate of norm 2l+1); hand-off = `pend = True` (WPRE 1) or SYNC, `a_nmod` = mod row of norm 2(l+1). Epoch accounting: every barrier or deferral consumes one epoch, in CUDA order. Blocks up to the stop run unguarded, the rest in one `if keep` (several sibling `if keep` blocks produced an MLIR "operand does not dominate" error, so the tail is one jit function `_tail(lo, hi)` called twice).
- Stops D..I are limited to layers 0 .. L-2 (`Stop` validates); the last layer's tail and steps > 0 are phase 5 (marked `PHASE 5 HOOK` in `ffn.stage_i` and `kernel._tail`). Stops pro..C3 still emit only the stages up to the stop (their parity is checked at layer 0 only); stops D..I emit the whole layer so that earlier layers of the run are real.
- wsplit smem carve (bytes, as CUDA): A tile `a_layout()` over [0, 20480) (the 64 x AS_STRIDE bf16 area); partials `[warp][T=50][WS_STRIDE=40]` f32 at 20480 (64000 B, `mine` = row block `warp`; the group's WPT tiles are consecutive row blocks gi*WPT .. gi*WPT+WPT-1); weight buffers at 84480, per warp NSL*4*32 x 16 B laid out `[j][nt][lane]` (this lane's ((e,h,m16), nt, j) operands are one 128-bit LDS each, copied into the same register fragment layout as `load_w`). Named barriers: ids `1 + gi` (gi = warp / WPT), `32 * WPT` threads; the second barrier of a task (after `cp_async_wait_group(0)`) only when another task follows, as CUDA.
- wsplit A staging uses the plain mapping of the CUDA kernel (piece i = tid + 256 x, row i / 16, piece i % 16) and four 32-bit STS per piece into the same fragment-interleaved tile as `gemm_phase` (identical smem bytes).
- G: tasks `2 * FF/32` from the top block index, `jg = task >> 1`, `t = (task & 1) * 25 + tid / 8`, `c = jg * 32 + (tid % 8) * 4`; `reduce_part4<8>` of gate (col c) and up (col FF + c), `mathx.gelu_mul`, 64-bit store to `act`; `block_wait` on `cnt[CNT_GU + jg]` (target `it1 * ARR_GU`) when FUSE_FG.
- SASS / ptxas counts (CPU, cu13 ptxas -v on the DSL PTX; Stop I layer 1; all LOCAL 0, REG 255):
| build | STACK | HMMA | spill stores / loads (B) |
|---|---|---|---|
| P=128 k16 defaults (wpre 15, d_wpt 2, fuse 18) | 792 | 496 | 1612 / 1480 |
| P=968 k64 defaults | 792 | 640 | 1634 / 1502 |
| P=128 k16 wpre 0 | 792 | 496 | 1628 / 1488 |
| P=128 k16 d_wpt 1 (plain gemm_phase for D) | 936 | 560 | 2492 / 2332 |
| P=128 k16 d_wpt 4 | 792 | 464 | 1612 / 1480 |
| P=128 k16 fuse 22 / 23 | 784 | 496 | 1600 / 1472 |
HMMA = 3 x 128 (A, F, H) + D (64 for d_wpt 2, 32 for d_wpt 4, 128 for 1) + attention, asserted in the tests. The wsplit tile alone (PTX): 64 mma, 26 LDS.128 (16 A + 8 B + 2 reduction), 16 + 28 STS, bar.sync with ids and 64 threads, as the CUDA kernel. The phase-1 cond_matvec spills still dominate.

Idiom exceptions (phase 4)
1. wsplit staging stores are scalar 32-bit STS through a flat u32 view of the A tile (the CUDA code's interleave, `Ai[base]`), not a tiled copy: the plain mapping scatters the four words of a piece.
2. wsplit weights use `cp_async_shared_global` (as the phase-3 K/V loads) into a smem tensor with a static `(warp, lane, (e,h,m16), nt, j)` layout; the A / reduction smem reads are 128-bit copy atoms on `local_tile`s of static-layout smem tensors (a dynamic pointer offset loses the 16 B alignment the atom needs).
3. Modulation rows are addressed by Int64 byte address + `make_ptr` (the layer index is dynamic, the row pointer carried as `a_nmod`).
4. `residual_tasks` / G stores: 32-bit `h` pair and 64-bit `act` stores through `make_ptr` / u32 view, `h` loaded as u32 with `.cg` and unpacked with bit ops (the CUDA `hraw` path).

Guesses the GPU run must confirm
- Byte equality of part (D, F, H), h + ssp (E, I), act (G) at layers 0, 1, 16 for the listed configs; first suspects if not: the wsplit K-slab selection by the runtime `wi` (`kbd = 2 (wi NSL + j) + m` on the A partition while the B fragment slot is `2 j + m`), the weight-buffer lane layout, the in-order warp sums in the reduction, GeGLU tails (`gelu_tanh` vs CUDA tanhf), and the counter targets with FUSE_FG (ARR_GU).
- Epoch accounting for every WPRE value (the unit tests only compile): a mismatch deadlocks or races at the first deferred barrier (wpre 0 / 15 are both in the sbatch; wpre 8 alone is not).
- `cnt` equality at fuse 22 / 23 (F publishes into the CNT_GU counters).

Not done in phase 4: the last layer's tail (final norm, action_out, Euler update, action_in), steps > 0 (`a_nmod` for step + 1), the full run, GPU run.

# Phase 5: last-layer tail, step loop, full chunk, bench arm
Built: `rowops.py` (`final_norm_row`, `action_out_pair`, `action_tail_rows`; `action_in_task` now takes the x row and a `smem_x` flag), `ffn.stage_i` (l == L-1 branch), `kernel.py` (`_rest(lo, hi)` replaces phase 4's `_tail`: blocks after_a, after_b, D..I, hand-off; the Stop layer limit is gone; end-of-run `if pend: SYNC`), `arm.py` (`impl: cute` in `config`, runtime `n_steps` / `n_layers`), `parity.py` (`--n-steps`, `--n-layers`, `full` expected set, `--timing --stages full`), `bench.py` (arm `cute`, not a default arm), `README.md`, `slurm/phase5.sbatch` (not submitted, `--time` 30 min), `megabench/tests/test_pi05_cute_phase5.py`.
- Tail at `l == L-1` (constant L, as CUDA): prefetch action_out_w / action_in_w, `residual_tasks<FF/KC>`, SYNC, `action_tail_rows` (rows `t = blockIdx; t < T; t += gridDim`: final_norm_row into smem `hn`, two `action_out_pair` per warp (own CTA sync each, weights in registers before it), CTA sync, `action_in_task` from smem `xs = hn + W` for groups 2w, 2w+1 unless last step, CTA sync). `more = step + 1 < n_steps` is runtime. Hand-off block: not done when it is the last layer of the last step (kernel ends); the last layer of an earlier step sets `a_nmod = modstep + MODROW` (norm 0 of step + 1) and `pend` (WPRE 1) or SYNC; other layers as phase 4.
- Epoch accounting: every barrier / deferral consumes one epoch in CUDA order. Per step: 9 + 1 barriers per layer as phase 4; the last layer has one extra SYNC between the residual and the tail; the hand-off barrier of the last layer is the deferred one of step + 1 layer 0 (WPRE 1) or a SYNC. `it1 = step * n_layers + l + 1` continues across steps. A run truncated by `n_layers` (tail never runs) ends with `pend` set and takes the final SYNC, as CUDA.
- Stops: A, B, C3 of step 0 layer 0 keep the phase 2-3 builds (nothing after the stop is emitted, HMMA counts unchanged); every other stop emits the whole layer loop, so stops at later steps / layers run real earlier iterations.
- DSL finding: a subscript store rooted at the dict `v` (`v["ssp"][t, gi] = x`) makes the DSL treat `v` as written and rebuild its tensors in every region, which gave "operand does not dominate" IR errors in sibling `if` branches (this was also the real cause of the phase-4 sibling-`if keep` error). Stores now go through local aliases (`ssp = v["ssp"]`). The phase-4 `_tail` grouping is kept in `_rest`.
- CPU numbers (cuobjdump; cu13 ptxas -v; LOCAL 0, REG 255 everywhere):
| build | STACK | HMMA | spill stores / loads (B) |
|---|---|---|---|
| full P=128 k16 defaults | 952 | 496 | 1880 / 1736 |
| full P=128 k32 | 952 | 544 | 1880 / 1736 |
| full P=128 k64 / P=968 k64 | 952 | 640 | 1894 / 1750 |
| full P=128 k16 wpre 0 | 800 | 496 | 1636 / 1504 |
| full P=128 k16 d_wpt 1 | 944 | 560 | 2528 / 2436 |
| full P=128 k16 fuse 23 | 808 | 496 | 1640 / 1508 |
| stop I step 1 layer 17, P=128 k16 | 816 | 496 | 1660 / 1516 |
The full build spills about 268 B of stores more than stop I layer 1 (1612 B) and 220 B more than stop I step 1 layer 17 (1660 B); the cause (tail rows with 64 weight registers per call) is a guess.
- Bench arm: `bench.py --arms megakernel,cute --mk-variants k16 ...` builds `CuteArm(geo, v, "cuda")` and `bind(inputs)`; the DSL compile (`cute.compile`, about 9 s per variant in `runs/20261017T013644Z-p6/bench_p128.log`) happens there, i.e. in `build_arms`, before correctness and timing, and the build time is recorded in the `build` record. A launch is `zero_()` of the cnt..bar slice plus the compiled call (no allocation, no sync). The summary table lists `cute/k*`; they are excluded from the "baseline" set so they never count as a baseline. A CPU test builds the arm through `build_arms`.
- Timing in `parity --timing` now zeroes the whole workspace once before the warmups and no longer between launches (both kernels zero their own counters), so L2 is not flushed (also changes the phase 3 / 4 stop timings slightly).

Idiom exceptions (phase 5)
1. `action_out_pair` loads its 64 weights with 32-bit `.nc` copy atoms into registers before the CTA sync (as CUDA); `final_norm_row` / tail stores are 128-bit smem copies through `local_tile`.
2. `rowops.action_in_task` takes an `xrow` tensor and a Constexpr smem flag instead of two kernels.
3. Hand-off address of step + 1 by Int64 byte arithmetic on `modstep`.

Guesses the GPU run must confirm
- Byte equality of `h`, `ssp`, `xbuf` at I step 0 layer 17 and of everything at A / I step 1 layers 0 and 17; then `full` incl. `actions`. First suspects: `final_norm_row` rounding chain, shuffle broadcast, the Euler update order (`xbuf` read with .cg after other warps' writes in the same row: rows are per block, dims per warp, so no overlap), smem reuse of the A tile region by `hn` / `xs` (CTA syncs as CUDA).
- Epoch accounting at the step boundary with WPRE 0 and 15, and the end-of-run behavior (no deferred barrier left at the last step).
- Bench: `cute/k*` correctness against the FP32 oracle band; compile time of 3 variants in one process with the 30 minute smoke limit.

Not done in phase 5: GPU runs (phase 4 job 7781 and phase 5 pending), performance tuning (the prologue gap, register spills in cond_matvec / the tail).

# Phase 4 / 5 / smoke GPU results (recorded for phase 6)
- Phase 4 (job 7782, `runs/20261017T002522Z-p4/`): 17/17 compares byte-equal (D..I at layer 0, I at layers 1 / 16, seeds, mask-tail, wpre 0, d_wpt 1 / 4, fuse 16 / 22 / 23). Stop I at layer 16: P128/k16 CUDA 0.877 / CuTe 0.930 ms; P968/k64 0.941 / 1.010.
- Phase 5 (job 7784, `runs/20261017T002538Z-p5/`): 15/15 byte-equal incl. `full` with `actions` (all 12 configs). Full chunk (L2 not flushed, median 20): P128/k16 CUDA 7.665 / CuTe 7.778 ms (+1.5%); P968/k64 8.338 / 8.422 (+1.0%).
- Bench smoke (job 7786, `../runs/20261017T005220Z-cute-smoke/`): both arms pass with identical errors; steady megakernel 7.716 / cute 7.807 ms; isolated 7.722 / 7.807; cute shows 2 kernels per chunk (the host `zero_()` fill).

# Phase 6: performance parity
Built / changed: `perf.py` (new: stage ladder), `sync.exit_reset` + `config.BAR_EXIT_WORD` + `Knobs.host_zero` + `kernel.py` exit code (in-kernel counter reset), `arm.py` (`reset()`, bind zeroes cnt..bar once, `launch()` = compiled call only unless `host_zero`), `parity.py` (arm factories `cuda_arm` / `cute_arm`, `--host-zero`, `--check-reset`, `cnt_bar_nonzero`, compare excludes `cnt` + `bar` with the reset and requires all counter words zero), `rowops.cond_matvec` (spill fix), `slurm/phase6.sbatch` (not submitted), `megabench/tests/test_pi05_cute_phase6.py`.

## 1. Stage ladder (`python -m megabench.sota.pi05.cute.perf`)
Per config: stops pro, A, B, C3, D, E, F, G, H, I (step 0 layer 0), `I@1` (stop I at layer 1) and `full`; for each entry both arms are built and bound on the same inputs, 3 warmups, then N (default 20) CUDA / CuTe launches interleaved (CUDA, CuTe, CUDA, ...) with CUDA events, L2 not flushed, medians. One `LADDER {json}` line per entry, then a table of deltas (entry minus the previous entry; the first is its own time) for CUDA, CuTe and CuTe - CUDA, plus `ladder_p<P>_k<K>.json` with `--out`. `I@1 - I` is one extra layer, `full - I@1` is not a stage. Cost: one CUDA extension build per stop (torch cache) and one CuTe compile (about 20 s) per stop.

## 2. In-kernel counter reset (no host `zero_()` launch)
- Mechanism: the very last thing of every CTA (every build, stops included: the kernel has no early return, stops are trace-time / `keep` guards) is `sync.exit_reset(cnt, bar, flag, nb)`: CTA sync; thread 0 `atom.add.acq_rel.gpu` on `bar[BAR_EXIT_WORD]` (word 5152 = the unused second line of the bar area, so no layout change); the CTA whose old value is `nb - 1` (publishes through the smem word `s_valid`, free by then) zeroes all `cnt` words (5120) and all `bar` words (5184, the exit word included) with plain stores by its 256 threads, then `fence.acq_rel.gpu`. PTX: exactly one `atom.global.acq_rel.add.u32` (test asserts it, and 0 for `host_zero`).
- `bind()` zeroes cnt..bar once (`reset()` repeats it for recovery). `Knobs.host_zero=True` compiles without the exit code and zeroes cnt..bar from the host before each launch (phase 5 path, for A/B). ssp is not zeroed by the in-kernel path: it is fully written (`action_in_tasks`, 50 x 16 words) before any read; the host path still zeroes it.
- Safety argument. (a) A CTA reaches the exit add only after all of its own counter traffic is finished: every spin (`grid_barrier`, `wpre_wait`, `block_wait`) is thread 0's loop followed by a CTA sync, every add / red (`grid_barrier`, `wpre_arrive`, `counter_publish`) is issued by one thread after a CTA sync, and the exit begins with another CTA sync. A CTA that leaves a stop early or whose consumers never wait in a truncated run simply has no pending waits; one that issued adds nobody consumes has issued them already. The deferred final barrier (`pend`, `if pend: SYNC` at the end of an untruncated run) is taken before the exit, uniformly in all CTAs (pend depends only on uniform values). (b) The last CTA (old == nb - 1) therefore knows that all nb - 1 other CTAs have completed every spin and every add to cnt / bar: their exit adds are release operations, its own is acquire (acq_rel RMW chain on one word), so their counter writes are ordered before its zeroing stores and no CTA can still be reading or adding to a word it zeroes (they do not touch cnt / bar after the exit add; the exit word itself is only touched by the exit adds, and the last one is the zeroing CTA's own). (c) The next launch is stream ordered after the kernel, so it sees zero; the fence is belt and braces. (d) Cooperative launch co-residency means no CTA of this launch waits for a CTA that has not started (the barriers need it already); the exit protocol itself would also work without it.
- Not covered: a launch that is aborted (timeout / kill / exception) leaves counters non-zero; `arm.reset()` fixes that. Two different arms never share a workspace.
- Parity: `compare_stages` excludes `bar` and, with the reset, `cnt` (it ends zero; `--host-zero` compares it again, `p128_k16_hostzero_c3` in the sbatch covers the fuse-bit-2 `cnt` case) and requires `cnt_bar_nonzero_bytes == 0` after the CuTe launch (part of `equal`, printed as `counters_zero`). `--check-reset`: bind once, two back-to-back `full` launches with no host zeroing between, counters zero after each and launch 2 == launch 1 (all buffers except cnt / bar, and actions).

## 3. Spills
Method: stop builds and `cuobjdump -sass` (STL / LDL positions), `ptxas -v` on the DSL PTX; the same compile numbers as phase 4 / 5 (cuobjdump REG / STACK, ptxas spill bytes; all LOCAL 0, REG 255 unless noted).
- Spill fix for the phase-1 `cond_matvec` spills: a side-effecting `inline_ptx("// cond_matvec task")` at the top of the task loop body (no arithmetic or order change: same loads, same fma chain). `pro` alone: REG 255 / STACK 688 / spill 1224 B -> REG 155 / STACK 72 (CPU recompile, cuobjdump). Why it works (NVVM hoisting the loop-invariant `ld.shared` of the condition vectors) is a hypothesis, see the last section.
- `pro` alone has STACK 72 after the fix (the CUDA full kernel also has STACK 72, cuobjdump). The full P=128/k16 CuTe build has STACK 392 (CPU recompile; the P=128 k16 profile cubin shows 400, P=968/k64 392), so the tail and the layer stages still spill.
- Not tried: `maxrregcount`-like options (the DSL passes launch bounds `min_blocks_per_mp=1`, 256 threads, which already gives 255 registers; no way to give ptxas more), restructuring `stage_a` to keep fewer live values, splitting the weight prefetch.

## 4. Other overheads
- Host path: `MegakernelArm.launch` is one C++ call (`ext.launch(tensors, ...)`). `CuteArm.launch` is now one Python call of the JIT executor with 27 `from_dlpack` tensors + 3 ints (argument marshalling per call, unmeasured here) and no `zero_()` kernel (saves one memset kernel and its launch gap per chunk). With back-to-back launches the host cost is hidden; in per-launch event timing it is not. Candidate: `cute.compile(..., options="--enable-tvm-ffi")` with `from_dlpack(..., enable_tvm_ffi=True)` compiles on the CPU (4.0 s for the `pro` build, `tvm_ffi` 0.1.11 present) but is not wired in: the call convention (torch tensors vs the dlpack objects) and which stream the DSL launch then uses cannot be checked without a GPU. No CUDA graph (the CUDA arm does not use one).
- SASS, whole-kernel instruction counts (P128/k16 full; CUDA ext built CPU-only, `cuobjdump -sass`): cuobjdump text lines 37041 (CUDA, 18512 instruction lines) vs 27353 (CuTe, text lines; CUDA side recounted from the CPU-built ext, counts are of the exact mnemonic, e.g. `FFMA ` without `.RM/.SAT`); HMMA 216 vs 496 (CuTe fully unrolls kb x nt in A / F / H: 3 x 128 + 64 + 48, CUDA keeps loops); FFMA 4570 vs 4291; FADD 2094 vs 1167; SHFL.BFLY 1349 vs 689; LDS.128 344 vs 372; LDG.E.128.CONSTANT 182 vs 246; 32-bit LDG.E.CONSTANT 353 vs 79; STG.E.64 79 vs 122; WARPSYNC.COLLECTIVE 660 vs 5 (cause not investigated); 32-bit LDG.E.STRONG.GPU (`.cg`) 262 vs 46, LDG.E.64.STRONG.GPU 96 vs 96, LDG.E.128.STRONG.GPU 71 vs 83. The kernel is one function, so per-loop attribution (A / F / H inner loop, wsplit, FA unit, residual tasks, reduce_part4) needs `-lineinfo` mapping on both sides; not done. Nothing identified as "clearly codegen-only" to fix beyond the spill fix; the GPU ladder decides where to look (suspects: the unrolled gemm epilogue stores, CuTe's 32-bit weight loads in the tail vs the CUDA vector loads).

## Idiom exceptions (phase 6)
1. `W.atomic_add(..., sem="acq_rel")` and an smem flag word for the exit counter (no DSL grid-sync).
2. `inline_ptx("// cond_matvec task")` as an optimization barrier against NVVM hoisting of smem loads.
3. Stores in `exit_reset` go through the `cnt` / `bar` tensors passed as arguments (not through the dict `v`).

## Profiling (IKET timeline, Nsight Compute, compiler artifacts)

Files: `iket_marks.py` (range markers, trace-time no-ops unless `CUTE_DSL_COMPILER_OPT` contains `iket`, so plain builds are
byte-identical: checked by diffing the PTX of full and `pro` P=128 k16 before / after), `profile.py` (driver), `iket_summary.py`
(CPU summary of a decoded trace), `slurm/profile.sbatch` (the whole suite),
`tests/test_pi05_cute_profile.py` (CPU checks).

Ranges: stage ranges `pro_tab pro_t1 pro_t2 pro_mod pro_act A B C D E F G H I exit` (B = `_after_a`, C = `_after_b` incl.
C3, D..I = ffn stages; a stage range includes the barrier at its start) and nested barrier / wait ranges `sync` (grid_barrier),
`wpre` (wpre_wait), `cwait` (block_wait). Each push / pop is preceded by one volatile `ld.shared` (an LDS in SASS) so that the
timestamp is taken after the barrier has released (mkprobe/markers/defer-blocking.md). Switches: `PI05_IKET_CTA_STRIDE=N` records
only CTAs with `blockIdx.x % N == 0`; `PI05_IKET_SYNC=0` drops `sync / wpre / cwait`. Keep a run below ~2 M timestamps: whole grid costs about 0.9 M per launch and step, every 8th CTA and 10 steps about 1.1 M (the sbatch ESTIMATE lines).

gemm_phase sub-stage markers (`PI05_IKET_GEMM=1`, with iket; unset = PTX byte-identical to before, cmp-checked for the plain and the iket full build): point markers `g_0 g_wiss g_adone g_sync g_wrdy g_mma g_st g_wnext` in `gemm_phase` (`iket_marks.gmark`, not CTA-uniform, a settle LDS before each; `g_wrdy` is `gmark_dep`: the 64 u32 of the warp's rB are XOR-folded into one `st.volatile.shared` to a per-warp scratch word, so the timer runs after all weights arrived; upper bound of the weight latency). The scratch (32 B, `config.GEMM_SCRATCH`) is appended to `Knobs.smem_bytes` and addressed from the end of the dynamic smem (`get_dyn_smem_size`). `iket_summary.py --gemm` splits each A / F / H call into weight issue, wpre wait, stage_a, CTA sync, exposed weight latency, MMA, stores; `gemm_marks_check.py PLAIN.cubin MARK.cubin --ptx MARK.ptx` (CPU) compares REG / STACK / LDL / STL per source line and checks the marker SASS; `slurm/gemm_marks.sbatch` runs it all (P=128 k16, 30 min). Adding the markers changes ptxas register allocation (see the spec Result for the measured spill change), so a marker build is not a faithful spill / STACK sample of the unmarked kernel.

How to run (GPU, through Slurm only): `sbatch megabench/sota/pi05/cute/slurm/profile.sbatch`, result folder
`megabench/sota/pi05/cute/runs/<stamp>-prof/P<p>_k<k>/`. By hand:
- plain timing + dump: `CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all CUTE_DSL_DUMP_DIR=D CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1
  python -m megabench.sota.pi05.cute.profile --mode plain --launches 20`
- IKET: `.venv/bin/run-iket --output-dir D --clobber profile --postprocess all -- python -m megabench.sota.pi05.cute.profile --mode iket --n-steps 1`
  then `python megabench/sota/pi05/cute/iket_summary.py D --out D/summary.md`
- Nsight Compute: `CUTE_DSL_LINEINFO=1 ncu --set full --import-source yes --source-folders <repo> --profile-from-start off
  --clock-control none --cache-control all -k 'regex:kernel_cutlass_pi05_megakernel.*' --launch-count 1 -o X -f python -m
  megabench.sota.pi05.cute.profile --mode ncu` (the driver brackets only the measured launches with cudaProfilerStart / Stop).

Artifacts and how to open them:
- `iket_*/**.pftrace`: Perfetto timeline, one track per warp, one slice per range. Open at https://ui.perfetto.dev (load the file).
  `iket.decoded_results.json` / `iket_pid_*.trace.json` is the same data as JSON; `summary.md` is the text digest.
- `ncu/pi05_cute_full.ncu-rep`: open with `ncu-ui` (copy the file to a machine with the GUI). The Source page has a view selector
  for Python (needs `--import-source yes` and the source folder), PTX and SASS; line info is present because of
  `CUTE_DSL_LINEINFO=1`. `pi05_cute_full_details.csv` is the details page for reading without the GUI.
- `dsl/`: `*.ptx`, `*.cubin`, MLIR (`*.mlir`) and SASS written by `CUTE_DSL_KEEP=all` (the DSL names files by the host signature).
  `cuobjdump -sass X.cubin` gives SASS if no `.sass` file is there.
- `actions_*.pt`: action tensors; the sbatch checks iket == plain (1 step and 10 steps) and logs PASS / FAIL in `compare.log`.

# FUSE_ABL (fuse bit 32): A + grid barrier + B as one phase

Code: `abl.py` (`stage_a8`, `load_w_unit`, `abl_phase`), hooks in `kernel.py` (`abl_phase` instead of `gemm_phase`; `_after_a` skips the
SYNC and `phase_b`), `config.py` (`FUSE_ABL = 32`, Knobs validation, smem), `iket_marks.py` (range `AB`, 19 names), `parity.py` /
`perf.py` / `profile.py` (`--fuse 50`). Default builds (no bit 32) are byte-identical in PTX (dumped before the edit, `cmp` after;
also a test, `test_pi05_cute_abl.py`, when the pre-change dump exists).

Design: 80 units, unit u = (head hh = u // 8, pair p = u % 8); its 32 MMA columns are hb + 16 p .. +15 and hb + 128 + 16 p .. +15
(hb = 256 hh in the virtual N = 2560 space), so a RoPE pair (j, j + 128) is inside one unit. Unit u runs on CTA nb - 1 - u (the CTAs
`phase_b` used; loop `for u in range(nb - 1 - bid, 80, nb)`, one unit per CTA when nb >= 80). Per unit: weights (`load_w_unit`, same
rB fragment / kperm / 16 x 128-bit .nc loads, nt mode split into (b, a) with strides (8K, 128K); WPRE order as in `gemm_phase`), all 8
A chunks staged by `stage_a8` into 8 private 16 KB tiles (per-row rs computed once, loads one chunk ahead, per-element arithmetic as
`stage_a`), warp w = K chunk w runs the `gemm_phase` kb loop on tile w, the 8 fp32 partials go to smem, then 200 tasks (50 rows x 4 column
quads) sum s = 0..7 from +0.0 (`reduce_part4` order), rbf, RoPE (same `rope` entries as `phase_b`, heads < 9), bf16 stores to q / ksuf /
vsuf. No `part` writes, no counters. CTAs without a unit only take the deferred WPRE barrier step; the barrier before C is unchanged.
Prefill / prefetch run after the unit (prefill starts with a CTA sync, so the smem reuse is safe).

Deviations: (1) the fp32 partial buffer has row stride 40 (`PB_STRIDE`), not 32: 8 x 50 x 40 x 4 = 64,000 B, conflict-free float2 stores
(a stride-32 row pitch puts all 8 g rows of a quad on the same banks); it reuses the A-tile region, so smem is 8 x 16384 = 131,072 B
(`Knobs.smem_bytes`, only with bit 32). (2) the unit loop also handles nb < 80 (several units per CTA: weights of the next unit are
loaded after a CTA sync); never exercised on the 148-SM target. (3) `iket_summary.py` knows `AB`; `test_pi05_cute_profile.py` expects the
default build's names without `AB`.

Knobs / how to run: `--fuse 50` (= 18 + 32) in `parity.py`, `perf.py`, `profile.py`; the CUDA arm gets `fuse & 31` = 18. Stop A is rejected
(ValueError in `compile_kernel`) and skipped by the parity / perf ladders; stop B = after the fused phase and its prefetch / prefill.
`compare_stages` leaves `part` out of the stop B / C3 compares of FUSE_ABL builds (D / F / H overwrite it and still compare).
Sbatch: `slurm/abl_parity.sbatch`, `slurm/abl_perf.sbatch` (`CFG="968 64"` for the second config).

## Stage G load batching (`Knobs.g_batch`)
`g_batch` 1 makes each CTA of stage G issue the loads of both of its tasks (32 float4 `.cg` loads) before any math, so a 2-task CTA pays one L2 round trip instead of two; `g_batch` 2 loads task 2's up half after task 1's sums (about 96 floats live). Every element keeps the same arithmetic (8 partials from +0.0 in order, `gelu_mul`, bf16 pack) and the same tasks per CTA, so the result is bit-exact by construction; the `g_batch` 0 PTX is unchanged. The default stays 0 until the GPU A/B (`parity.py --g-batch`, `perf.py --g-batch`). CPU result (P968 k64): ptxas does not keep all loads ahead of the first FADD and STACK changes, see `specs/SPEC-g-batch.md` Results.

# Unverified — rederive before use
Moved here from the sections above; none of these is supported by a log or a file in the repo.
- Two fresh processes both compile the `pro` build in about 4.1 s (no DSL file-cache benefit). Rederive: time `arm.cpu_arm(Knobs(128, 16), Stop('pro'))` in two fresh processes.
- Before the spill fix, ptxas hoisted the 32 A fragment loads ahead of the HMMA chain in the stop-A loop at a cost of about 16 STL/LDL. Rederive: `cuobjdump -sass` of a stop-A build at the phase 2 commit, count STL/LDL in the HMMA loop.
- Root cause of the phase-1 `cond_matvec` spills: NVVM hoisted 80 loop-invariant `ld.shared.v4` (10 steps x 8 chunks) out of the task loop, about 320 live registers. Rederive: dump the PTX of `Stop('pro')` without the `inline_ptx` marker and count `ld.shared.v4` before the task loop; the spill drop itself (688 -> 72 STACK) is measured.
- Cause of the remaining STACK 72 in `pro`: libdevice FP64 `sin` / `cos` Payne-Hanek local array (`STL [R8]`, dynamic index), "the same as CUDA". Rederive: `cuobjdump -sass` of both and locate the STL with that dynamic index; only the equal STACK value is measured.
- Cause of the full-build extra spills (tail rows with 64 weight registers per call) and of the remaining full-build STACK 392. Rederive: stop builds plus STL/LDL positions in `cuobjdump -sass -lineinfo`.
- The DSL launch stream is the torch current stream in the bench. Rederive: read `cutlass.cutlass_dsl` launch code for the default stream, or run under `nsys` and check the stream id of the CuTe kernel and of the CUDA arm kernel.
- A host OOM at about 20 M IKET timestamps. Rederive: run `run-iket` with the whole grid and 10 steps and watch host memory (no log of this is kept).
- A CUDA counter-barrier price of 1.18 us. Rederive: the `megabench/mkprobe` price probes; not tied to this port.
- Per-launch host cost of `CuteArm.launch` (27 `from_dlpack` tensors + 3 ints) is hidden with back-to-back launches. Rederive: time N launches with events versus a CUDA-graph or `nsys` launch gap (unmeasured here).
