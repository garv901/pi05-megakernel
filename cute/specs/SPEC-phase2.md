# SPEC phase 2: GEMM engine, layer/step loops, stages A and B

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md`, `SPEC-phase1.md`, `../NOTES.md`, the current `kernel.py`, `rowops.py`, `probes/mma_tile.py`.
CUDA: `../../mk/pi05_mk.cu`. Same hard rules: **no GPU**, CPU compile only, no commits, no slurm submit, only files under
`megabench/sota/pi05/` and `megabench/tests/`, idiomatic CuTe with exceptions logged in `NOTES.md`.

## Phase-1 GPU result (job 7747, `runs/20261016T230048Z-p1/`) — record in NOTES.md
`pro` byte-equal CUDA vs CuTe for P=128/k16, P=968/k64, P=128/k16 mask-tail 17, seeds 7301/7302 (every buffer but `bar`).
Prologue wall time (pro-stop launch, median of 20): CUDA 0.158 ms, CuTe 0.200 ms (+42 us; likely the cond_matvec spills; phase 6).

## Goal
Add the step/layer loops and phase A (QKV GEMM with adaRMS norm staging) and phase B (split-K reduce + RoPE), with the
stops `A` and `B`, byte-equal to CUDA. Build `gemm_phase` as the reusable engine that phases F and H will call unchanged
in phase 4 (so implement every mode now, even those first exercised later).

## Deliverables

### 1. Loops and state (`kernel.py`, `pi05_mk.cu:1231-1256, 1838-1852`)
Runtime `step` / `l` loops (`cutlass.range`, not unrolled) over `n_steps` / `n_layers`; `it1`, `a_nmod` (= mod of step 0
after the prologue; it switches to `nextmod` / next-step mod in I, phase 4/5), the WPRE `a_pend` flag (layer 0 of step 0
has its barrier deferred into A, WPRE bit 1). Stops `A` and `B` at `(PI05_STOP_STEP, PI05_STOP_LAYER)`, exactly where the
CUDA macro returns. Bodies of C..I come in later phases; for now the loop body ends after B (a run without a stop just
runs A/B for all layers, that is fine).

### 2. `cute/gemm.py`: `gemm_phase` (`pi05_mk.cu:415-609`), all modes
- Virtual N over up to three weight tensors (`n1`, `n2` boundaries), K chunk `s = blockIdx % S`, `slot`, `nblk`, warp task
  stride `ng_step`, 32-column warp tasks, `part[s][t][n]` float2 stores for rows < 50 only, exactly as CUDA.
- Use the phase-0 `cute` gemm formulation (TiledMma m16n8k16, KPERM layout `(2,4,2,2,4):(1,8,2,4,32)`, bit-exact on GPU).
- **B operand straight from global into registers**, one 128-bit `.nc` load per (slab, n-tile) per lane, as `load_w`
  (438-450): the KPERM view of the weight rows partitioned by the TiledMma must give each thread 8 contiguous bf16 per
  slab, so the copy vectorizes to 128 bits. Next task's weights are loaded right after this task's stores (603), as CUDA.
- **A tile staging** (461-559, GEMM_FRAG 2): 1024 16-byte pieces, 4 per thread, thread → (row, piece) mapping of
  GEMM_FRAG 2 (`q = tid/16`, `pc = tid%16`, `r = (x>>1)*32 + (q>>3)*16 + (x&1)*8 + (q&7)`), `.cg` loads, rows >= 50 zero.
  Target smem layout: the fragment-interleaved tile, so one 128-bit LDS is one mma A operand (`uint4 [mt][sl][half][g*4+t4]`).
  Express it as a CuTe smem layout (the composition that makes `partition_A` of the TiledMma read contiguous 16 B per mma);
  if that is not possible, use the row-major tile of the phase-0 probe and log it. Either is bit-exact; report the
  LDS.128 / LDS count of the main loop from SASS for the choice you made.
- **Norm mode** (`ssp`, `nmod` non-null; 481-523): reproduce the order exactly: each thread's ss part for its row comes
  from `ssp[r][pc]`; the half-warp reduces the 16 parts by `shfl_xor` 8, 4, 2, 1 with **lane-in-half-warp = part index**,
  then broadcasts from lane `tid & 16`; `rs = rsqrtf(fadd_rn(v * (1/W), 1e-6))`; per element
  `fadd_rn(fmul_rn(fmul_rn(h, rs), fadd_rn(1, sc)), sh)` → bf16 rn; rows >= 50 untouched (zero). Use `mathx` defaults.
- **Counter mode** (`cnt`, `cdiv`, `cmod`; 604-607): `__syncwarp`, lane 0 `red.release` on `cnt + ((n0/cdiv)%cmod)*CNT_STRIDE`
  after the task's stores and after issuing the next weight loads.
- **WPRE mode** (452-460): `wpre_arrive`, first-task weight loads into registers, `wpre_wait`, then A staging.
- Check the generated code for spills; this function is `__noinline__` in CUDA. If the DSL can emit it as a separate
  non-inlined device function, try it and report the effect on registers / stack; otherwise inline.

### 3. Phase A call and phase B (`pi05_mk.cu:1243-1320`)
- A: `gemm_phase(a_nmod ? h : xn, W, wq[l], QD, wk[l], QD+KVD, wv[l], QD+2*KVD, part, smem, cnt if FUSE_AB, HD, 10, ssp, a_nmod, wpre if a_pend)`.
  Then `SYNC()` unless FUSE_AB.
- Before B: `prefetch_l2` of `wo[l]`, `prefix_k[l]`, `prefix_v[l]` (1259-1261). The ATT_PREFILL=2 K/V prefill
  (1262-1287) belongs to phase 3; leave a clearly marked hook at that position.
- B tasks (1288-1320): task = `gridDim-1-blockIdx` stepping by `gridDim`, `NTASK = ceil(50/8)`, 10 heads (8 q, k, v),
  `block_wait` on the head counter if FUSE_AB, `reduce_part4<8>` (.cg float4 loads, sum s = 0..7 in order), `rbf`, RoPE
  from the `rope` table for heads < 9, `pack_bf16x4` stores to `q`, `ksuf`, `vsuf`. Stop `B` after the loop.

### 4. `parity.py`
- `run_cute` supports stops `A`, `B` (and `pro`).
- Non-vacuity: the compare report also lists, per buffer, the fraction of nonzero bytes in the CUDA dump, and the CLI
  prints a warning if a buffer expected to be written at that stage is all zero. Expected-written sets:
  pro: temb tmid cond mod rope h ssp xbuf; A: + part; B: + q ksuf vsuf.
- Add `--fuse N` (default 18) passed to both kernels.

### 5. Tests (CPU) `megabench/tests/test_pi05_cute_phase2.py`
Compile `Stop('A')`, `Stop('B')` for P=128/k16 and P=968/k64, fuse 18 and 19; report REG/STACK/spill bytes (print, do not
assert on spills, but assert LOCAL==0 as before only if that still holds; if not, report). Keep phase 0/1 tests passing
and **run all three test modules at the end** and report that run.

### 6. `cute/slurm/phase2.sbatch` (do not submit)
`parity --compare --stages pro,A,B` for: P=128/k16 seeds 7301 7302; P=968/k64 seed 7301; P=128/k16 `--mask-tail 17`;
P=128/k16 `--fuse 19` (counter hand-off A→B); plus `--step 0 --layer 1` is not reachable yet (C..I missing), skip it.
Timing line: CUDA vs CuTe median wall time of a `B`-stop launch at P=128/k16, informative.

### 7. NOTES.md
Phase-1 GPU result, phase-2 build notes (smem layout choice and LDS counts, registers/spills, idiom exceptions, open guesses).

## Done when
All CPU tests (phase 0, 1, 2 modules) pass in one final run. Report: files, test command + result, registers/spills,
the A-tile layout choice with SASS LDS counts, idiom exceptions, open guesses, anything not done.
