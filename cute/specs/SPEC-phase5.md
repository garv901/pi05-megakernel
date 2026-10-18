# SPEC phase 5: last-layer tail, step loop, full chunk, bench arm

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md`, `SPEC-phase4.md`, `../NOTES.md`, current `kernel.py`, `ffn.py`, `rowops.py`, `arm.py`, `parity.py`,
and `../../bench.py`, `../../megakernel.py`. CUDA: `../../mk/pi05_mk.cu`. Same hard rules: **no GPU**, CPU compile only via
`arm.cpu_arm`, no commits, no slurm submit (`--time <= 00:30:00`), only files under `megabench/sota/pi05/` and
`megabench/tests/`, idiomatic CuTe with exceptions logged in `NOTES.md`.

Phase 4's GPU job (7781) is queued while you work. If it fails you will get a separate fix request; do not change phase-4
code paths unless needed for phase 5, and keep them separable.

## Goal
The complete chunk: last-layer tail (TAIL 2, DEFNORM), steps 1..9, the end-of-run deferred barrier, `full` byte-equal to
CUDA (all workspace buffers + `actions`), then the `cute` arm in `bench.py` so the MegaBench band check and timing run.

## Deliverables

### 1. Last layer I branch (`pi05_mk.cu:1812-1829`, PI05_TAIL 2)
At `l == L-1` (constant L = 18, as CUDA; with a truncated `n_layers` the tail never runs, as in CUDA):
`prefetch_l2(action_out_w, ADIM*W*4)`, `prefetch_l2(action_in_w, W*ADIM*4)`, `residual_tasks<FF/KC>` (gate of norm 2l+1),
`SYNC()`, `more = step + 1 < n_steps`, `action_tail_rows(nextmod = modstep + 2L*3W, hn = smem, last_step = !more)`;
if `!more` skip the rest of this iteration (the kernel then ends; no further barrier); else `a_nmod = mod + (step+1)*MODROW`,
then `a_pend = true` (WPRE bit 1) or `SYNC()`.

### 2. `action_tail_rows` and helpers in `rowops.py` (`pi05_mk.cu:979-1060`)
- `final_norm_row` (981-1000): each lane loads `ssp[t*16 + (lane & 15)]`, `shfl_xor` 8,4,2,1, broadcast from lane 0,
  `rs = rsqrtf(fadd_rn(v*(1/W), 1e-6))`, thread tid handles columns `4*tid..4*tid+3` (h as uint2 .cg, scale/shift float4 .cg),
  exact rounding chain, float4 store to `hn` (smem, W floats).
- `action_out_pair` (1004-1030): weights for dims a0, a0+1 loaded to registers (`lane + 32j`, j = 0..31, .nc) **before** the
  CTA sync that publishes hn; dot in j order with fmaf, `warp_sum`, lanes 0/1: `vel = v + b[a]`,
  `xnew = fadd_rn(xbuf[t][a] (.cg), fmul_rn(-0.1f, vel))`, store xbuf, `xs[a]` (smem), and `actions[t][a]` on the last step.
- `action_tail_rows` (1046-1060): rows `t = blockIdx; t < T; t += gridDim`; final_norm_row; `pr = 0, 1` with `a0 = warp*4 + pr*2`
  (each call contains its own CTA sync); CTA sync; if not last step: `action_in_task(t, gi, xs, smem_x=true)` for
  `gi = warp*2, warp*2+1`; CTA sync. `xs = hn + W`.

### 3. Step loop and end of run
`modstep = mod + step*MODROW` per step; `it1` continues across steps (`step*n_layers + l + 1`); after the loops,
`if a_pend: SYNC()` (only reached by truncated runs). Stops at any `(step, layer)` for every stage, `I` at layer 17 includes
the tail; `full` = no stop. Remove the phase-4 `Stop` layer <= 16 restriction.

### 4. parity
`--step S` for all stages; `full` compares every buffer but `bar`, plus `actions`. Add `--n-steps`, `--n-layers`
(runtime truncation, both kernels) for debugging.

### 5. Bench arm
- `arm.CuteArm`: production path (no stop), `variant` = `k16/k32/k64` as `MegakernelArm`, knobs default otherwise;
  `config` dict like `MegakernelArm.config` plus `"impl": "cute"`. Host path per launch: one `zero_()` of the `cnt..bar`
  slice and the compiled call on the current stream; nothing else (no allocation, no sync).
- `bench.py`: add arm `cute` handled exactly like `megakernel` (same `--mk-variants`, same correctness / timing / profile
  code paths), but **not** in the default `--arms` list. Keep every other arm's behavior unchanged. Report how the
  compile time is kept out of the timed region (compile happens in `__init__`/`bind`, check).
- `README.md`: one line in the file table for `cute/`.

### 6. Tests (CPU) `megabench/tests/test_pi05_cute_phase5.py`
Compile `full` for P=128 k16/k32/k64 and P=968 k64 at defaults, plus P=128 k16 with wpre 0, d_wpt 1, fuse 23; compile stop
`I` at step 1 layer 17. A test that `bench.py --arms cute` argument parsing / arm construction path works on CPU up to the
point it needs a GPU (or a unit test of the arm factory). Print REG/STACK/spills. Run ALL test modules (phase 0-5) in one
final command and report that run.

### 7. `cute/slurm/phase5.sbatch` (do not submit)
1. `parity --compare` (exit nonzero on failure): `I` at step 0 layer 17; `A` and `I` at step 1 layers 0 and 17;
   `full` for P=128 k16 seeds 7301/7302, P=128 k32, P=128 k64, P=968 k64, P=128 k16 mask-tail 17, P=968 k64 mask-tail 200,
   and P=128 k16 with wpre 0 / d_wpt 1 / d_wpt 4 / fuse 16 / fuse 23.
2. Timing: full chunk CUDA vs CuTe median (CUDA events, L2 not flushed, 20 launches after 3 warmups) for P=128 k16 and
   P=968 k64; informative.
Under 30 minutes; most important runs first; elapsed per run. The bench smoke runs separately (I submit
`slurm/pi05.sbatch --arms megakernel,cute --mk-variants k16 --seeds 7301 --rounds 1 --iters 3 --steady 5 --tag cute-smoke`
afterwards; make sure that command line works).

### 8. NOTES.md
Phase-5 build notes (tail, epoch accounting across steps, regs/spills, idiom exceptions, open guesses).

## Done when
All CPU tests (phase 0-5 modules) pass in one final run. Short factual report as before.
