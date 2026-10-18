# SPEC phase 1: the real kernel skeleton + prologue

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md`, `SPEC-phase0.md`, `../NOTES.md`, and the phase-0 code (`config.py`, `sync.py`, `mathx.py`,
`parity.py`, `probes/*`). CUDA source: `../../mk/pi05_mk.cu`. Same hard rules as phase 0: **no GPU**, CPU compile only
(`CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a`), no commits, no slurm submit, only files under `megabench/sota/pi05/`
and `megabench/tests/`, fully idiomatic CuTe with every exception logged in `NOTES.md`.

## Phase-0 GPU results (job 7745, `runs/20261016T223558Z-p0/`) — record these in NOTES.md
- smoke: cooperative launch OK on 148 SMs, 117248 B smem, 16340 epochs, 0 errors in all modes.
  barrier 1.43 us/epoch, wpre 1.89 us/epoch, counters 2.82 us/epoch (the CUDA price sheet says 1.18 us for the
  counter barrier; different harness, revisit in phase 6).
- math: every `mathx` **default** lowering is bit-exact for all 23 expressions (4e6 inputs each). Use the defaults
  everywhere; the `ex2`, `fast` and some `split` candidates are wrong (e.g. gelu_tanh split 9357 mismatches,
  action_out split 575241), do not use them.
- mma: gemm, wsplit and fa are bit-exact in both the `cute` (TiledMma + permuted-K layout + copy atoms) and `raw`
  variants. **Use the `cute` variants as the base for the real kernel.**
- parity selfcheck: deterministic at pro, A, B, C3, I and full for P=128/k16 and P=968/k64; the stop-at-end build gives
  the same actions as the default build.

## Goal of phase 1
Create the real megakernel module and host wrapper, implement everything up to stage `pro`, and make
`parity.run_cute` work so that CUDA vs CuTe at stage `pro` is byte-equal. Later phases add stages to this same kernel.

## Deliverables

### 1. `cute/kernel.py`
- `@cute.kernel pi05_megakernel(...)`: grid = SM count, 256 threads, `min_blocks_per_mp=1`, dynamic smem =
  `knobs.smem_bytes`, cooperative launch. Constexpr inputs: `Knobs` fields and a `Stop` (stage, step, layer) for parity
  builds (`Stop(None)` = production, runs to the end of what is implemented).
- Arguments: the 25 inputs in `megakernel.ORDER` order, `actions`, and the workspace. Pass workspace buffers as typed
  CuTe tensors made from views of the one `ws` uint8 tensor at the `config.workspace_layout` offsets (so the bytes are
  exactly where the CUDA kernel puts them and `parity` can dump both the same way). `n_steps`, `n_layers` are runtime
  Int32 as in CUDA.
- One dynamic smem buffer carved by hand with layouts (the CUDA kernel aliases the same bytes per phase; keep that).
- Barrier bookkeeping: an epoch counter in a register, `grid_barrier` from `sync.py`, the host zeroes `cnt` and `bar`
  before each launch (phase-0 decision). Provide a helper so later phases can call `SYNC()` / `wpre()` the way the
  CUDA macros do (`pi05_mk.cu:1081-1117`), minus stamps/PTRACE.
- `fuse` handling: `fuse & 16` (DEFNORM) is **required** (assert in `Knobs`): the non-defnorm row path
  (`action_in_row`, `residual_row`) is not ported. Log it as a scope decision in NOTES.md. Other fuse bits stay knobs.

### 2. Prologue, in this order (`pi05_mk.cu:1144-1229`)
- nvalid: every block counts `prefix_mask` with the same `block_sum` as CUDA.
- P0: time embedding (FP64, 1155-1166), RoPE table (1168-1174). Skip the in-kernel `cnt` zeroing (the host does it).
  First barrier = `SYNC()` epoch 1.
- P1/P2/P3: `cond_matvec` x3 (775-826 and 1180-1202), with the same row → (warp, block) task mapping, the same fma order
  (i, then s, then component x,y,z,w per acc), the same xor-butterfly warp_sum, `lane == s` selection, `+ bias`, then
  silu where CUDA has it. The P3 row → (norm, j) weight/bias mapping is in the lambdas at 1189-1201. Put this in
  `cute/rowops.py` as a reusable device function taking CuTe tensors (weights viewed as `[rows, W]`).
  Weight loads stay non-coherent (`.nc`) as `__ldg`; loads of in-kernel-produced data (`temb`, `tmid`, `cond`) use `.cg`.
- `prefetch_l2` of wq/wk/wv of layer 0 (1205-1207).
- Step-0 input: `action_in_tasks(p, noise, copy_x=true)` (945-977; TAIL 2 + DEFNORM path) → `h`, `ssp`, `xbuf`.
  Put `action_in_task(s)` in `rowops.py` too (phase 5 reuses it).
- Stop point `pro`: right after that, every block returns (drain cp.async first, as the CUDA macro).

### 3. `cute/arm.py` (host wrapper)
`class CuteArm` with the `MegakernelArm` interface (`megakernel.py`): `__init__(geo, variant, device, knobs=None, stop=None)`,
`bind(inputs)`, `launch()`, `outputs()`, `close()`. Compile once per (knobs, stop) and cache the compiled function
(in-process dict; also try the DSL's own disk cache if it has one and say whether it works). The per-launch host path
must be cheap: prebuilt dlpack/CuTe args, the `cnt`+`bar` zeroing as one `zero_()` on a contiguous slice if the layout
allows it (else two), then the compiled call on the current stream. Do not register it in `bench.py` yet.

### 4. `cute/parity.py`
Implement `run_cute(knobs, stop, seed, ...)` with the same return format as `run_cuda`. Add a CLI mode
`--compare --stages pro` that runs both and prints per-buffer equality. Compare every workspace buffer except `bar`
(epoch numbering differs by design) and `cnt` (only compare from the stage onward where both are meaningful; at
`pro` both are zero, so include it). Add `--mask-tail N`: zero the last N entries of `prefix_mask` (so nvalid < P and
the RoPE positions change); the CUDA and CuTe runs get the same inputs.

### 5. Tests `megabench/tests/test_pi05_cute_phase1.py` (CPU)
Compile the kernel with `Stop('pro')` for P=128/k16 and P=968/k64; check from the `CUTE_DSL_KEEP` cubin
(`cuobjdump -res-usage`) that there is no local memory / spill and report registers. Keep the phase-0 tests passing.

### 6. `cute/slurm/phase1.sbatch` (do not submit)
`parity --compare --stages pro` for P=128/k16, P=968/k64, and P=128/k16 with `--mask-tail 17`, seeds 7301 and 7302;
plus one timing line: CUDA vs CuTe wall time of a `pro`-stop launch (CUDA events, median of 20, after 3 warmups),
informative only. Output to `cute/runs/<UTC ts>-p1/`. Exit status nonzero if any compare fails.

### 7. NOTES.md
Phase-0 GPU results (above), phase-1 build notes, registers/smem, idiom exceptions, guesses for the GPU run.

## Done when
CPU tests pass. Report: files, test command + result, registers/spills, idiom exceptions, open guesses, anything
not done. Do not claim anything you did not verify.
