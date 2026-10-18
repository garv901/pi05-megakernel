# SPEC phase 4: the rest of the layer (D, E, F, G, H, I) and the layer loop

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md`, `SPEC-phase3.md`, `../NOTES.md`, current `kernel.py`, `gemm.py`, `attention.py`, `rowops.py`,
`parity.py`, and the phase-0 wsplit probe (`probes/mma_tile.py`, `wsplit/cute`: bit-exact on GPU). CUDA: `../../mk/pi05_mk.cu`.
Same hard rules: **no GPU**, CPU compile only via `arm.cpu_arm`, no commits, no slurm submit (`--time <= 00:30:00`), only
files under `megabench/sota/pi05/` and `megabench/tests/`, idiomatic CuTe with exceptions logged in `NOTES.md`.

## Phase-3 GPU result (job 7759, `runs/20261016T233957Z-p3/`) — record in NOTES.md
B and C3 byte-equal for P=128 k16 (seeds 7301/7302), k32, k64; P=968 k64; P=128 k16 mask-tail 17; P=968 k64 mask-tail 200;
P=128 k16 fuse 16. Non-vacuous (attn 99.8% nonzero). C3-stop launch: P=128/k16 CUDA 0.185 / CuTe 0.234 ms;
P=968/k64 CUDA 0.200 / CuTe 0.240 ms (B→C3 adds ~10 us in both; the gap is still the prologue).

## Goal
Complete one non-final layer: D, E, F, G, H, I (`pi05_mk.cu:1732-1847`, non-last-layer branches with DEFNORM), the
per-layer state hand-off (`a_nmod = nextmod`, `a_pend`) and all WPRE bits, so that stops `D`..`I` at any layer
`0 <= l <= 16` of step 0 are byte-equal to CUDA. The last layer's tail (`l == 17`, `action_tail_rows`) and steps > 0 are
phase 5: at `l == L-1` keep a clearly marked hook; a run that reaches it may do anything for now.

## Deliverables

### 1. D: o_proj (`pi05_mk.cu:1732-1751`)
- Barrier before D: `SYNC()` unless (WPRE bit 8 and D_WPT > 1), in which case it is the deferred `wpre()` passed into wsplit.
- `prefetch_l2(wd[l], W*FF*2)` and `prefetch_l2(modstep + 2l*3W, 6W*4)`.
- `D_WPT > 1`: `gemm_phase_wsplit<D_WPT>(attn, QD, wo[l], W, part, smem, wpre?)` (`621-741`) in `gemm.py`. Exact semantics:
  group `gi = warp / WPT`, `wi = warp % WPT`; weights per warp into its own smem buffer by cp.async (16 B) for slabs
  `wi*NSL + j`, issued before the A tile; A tile staged fragment-interleaved with the **plain** mapping
  `i = tid + x*256, r = i/16, pc = i%16` (not the GEMM_FRAG-2 mapping of gemm_phase); `cp.async.wait_group 0` + CTA sync;
  per task: `NSL` slabs, accumulators to `mine` (`[T=50][WS_STRIDE=40]` fp32 per warp, rows < 50 only), next-task weight
  cp.async issued after `__syncwarp`, **named barrier** `bar.sync 1+gi, 32*WPT`, then row quads `q = wi; q < 13; q += WPT`,
  `v = grp[warp 0 of group]` then `+= w = 1..WPT-1` in order, float4 store to `part[s][r][n0 + c4]`; if another task follows,
  `cp.async.wait_group 0` + the same named barrier. Smem carve exactly as CUDA: A tile in the `64 x AS_STRIDE` bf16 area,
  partials right after it, then the weight buffers (`[warp][NSL][4][32]` uint4). Reuse the phase-0 `wsplit/cute` formulation
  (it used 64-row partials; the real one has 50 rows and stride 40: use CUDA's).
- `D_WPT == 1`: plain `gemm_phase(attn, QD, wo[l], W, ...)` with an explicit `SYNC()` before it.
- `SYNC()` after D.

### 2. E: gated residual (`residual_tasks<QD/KC>`, `pi05_mk.cu:918-938`, call 1753-1754)
Gate = `modstep + 2l*3W + 2W`. Warp tasks `warp*gridDim + (gridDim-1-blockIdx)`, stride `gridDim*8`, `T*NSSP` tasks;
float2 partial loads s = 0..15 in order (.cg), gate float2, h as one u32 (.cg); exact rounding chain of 932-933;
bf16x2 store to h; `ss = warp_sum(fmaf(h1, h1, h0*h0))` (mathx `ss2` default); lane 0 writes `ssp[task]`.
Barrier after E: `SYNC()` unless WPRE bit 2 (then the deferred `wpre()` goes into F).

### 3. F: gate/up GEMM (1767-1774)
`gemm_phase(h, W, wg[l], FF, wu[l], 2FF, -, 2FF, part, smem, cnt+CNT_GU if FUSE_FG, cdiv 32, cmod FF/32, ssp, modstep+(2l+1)*3W, wpre if bit 2)`.
`SYNC()` after F unless FUSE_FG (fuse bit 4).

### 4. G: GeGLU (1775-1798)
`prefetch_l2` of next layer's wq/wk/wv (`ln = l+1 < n_layers ? l+1 : 0`); tasks `2*(FF/32)` from the top block index;
`block_wait(cnt + (CNT_GU + jg)*CNT_STRIDE, it1*ARR_GU)` if FUSE_FG; `GR = 25`, `t = (task&1)*GR + tid/8`, `c = jg*32 + (tid%8)*4`;
for `tid/8 < GR && t < T`: `reduce_part4<8>` of gate (N = 2FF, col c) and up (col FF + c), `pack_bf16x4(rbf(gelu_tanh(rbf(g))) * rbf(u))`
(mathx `gelu_mul` default) to `act[t][c]`. Barrier after G: `SYNC()` unless WPRE bit 4.

### 5. H: down GEMM (1799-1807)
`prefetch_l2(modstep + 2(l+1)*3W, 3W*4)` first (before the deferred barrier), then
`gemm_phase(act, FF, wd[l], W, -, W, -, W, part, smem, wpre if bit 4)` (S = 32 K chunks). `SYNC()` after H.

### 6. I: gated residual + hand-off (1808-1847, non-last branch)
`residual_tasks<FF/KC>(gate = modstep + (2l+1)*3W + 2W)`, `a_nmod = modstep + 2(l+1)*3W`; then `a_pend = true` (WPRE bit 1)
or `SYNC()`. Fix the phase-2 shortcut: A must stage with the current `a_nmod`, not step 0's mod.
Epoch accounting must match CUDA exactly for every WPRE value (`wpre()` increments the epoch and returns the counter target).

### 7. Stops and parity
- Stops `D, E, F, G, H, I` at `(step, layer)`, placed exactly where the CUDA macro returns (after the stage's work, before
  its trailing barrier / deferred barrier). Make `parity` accept `--layer L` for these stops.
- Expected-written: D: part; E: h, ssp; F: part; G: act; H: part; I: h, ssp.
- CUDA build flags: `parity` already adds `-DPI05_WPRE` / `-DPI05_D_WPT` for non-default knobs; check it also covers
  `fuse` (runtime, no flag) and that the default build stays flag-free.

### 8. Tests (CPU, `cpu_arm`) `megabench/tests/test_pi05_cute_phase4.py`
Compile stop `I` (layer 1) for: P=128 k16 and P=968 k64 at defaults; P=128 k16 with wpre 0, d_wpt 1, d_wpt 4, fuse 22
(18 + FG counters), fuse 23 (all counter hand-offs). Print REG/STACK/spills per build. Run ALL test modules (phase 0-4)
in one final command and report that run.

### 9. `cute/slurm/phase4.sbatch` (do not submit)
`parity --compare` (exit nonzero on any failure), all at step 0:
- P=128 k16 seed 7301: stages `D,E,F,G,H,I` at layer 0, then `I` at layers 1 and 16.
- P=128 k16 seed 7302 and P=968 k64 seed 7301: `I` at layers 0, 1, 16.
- P=128 k16 mask-tail 17, P=968 k64 mask-tail 200: `I` at layer 1.
- P=128 k16 variants at `I` layer 1: wpre 0; d_wpt 1; d_wpt 4; fuse 16; fuse 22; fuse 23.
- Timing: an `I`-stop launch at layer 16 (17 full layers) for P=128/k16 and P=968/k64, CUDA vs CuTe median.
Keep the whole job under 30 minutes (order the runs most important first; print elapsed per run).

### 10. NOTES.md
Phase-3 GPU results, phase-4 build notes (wsplit smem carve, named-barrier ids, regs/spills, idiom exceptions, open guesses).

## Done when
All CPU tests (phase 0-4 modules) pass in one final run. Short factual report as before.
