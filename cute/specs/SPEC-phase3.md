# SPEC phase 3: attention (prefill, C, C3)

> Audit note (2026-10-17): every `pi05_mk.cu` line number in this spec is stale (the file grew by +3 to +48 lines since it was written). Locate code by names and stage comments, not by line.

Read first: `../PLAN.md`, `SPEC-phase2.md`, `../NOTES.md`, current `kernel.py`, `gemm.py`, `rowops.py`, `arm.py`,
and the phase-0 FA probe (`probes/mma_tile.py`, `fa/cute` variant: bit-exact on GPU). CUDA: `../../mk/pi05_mk.cu`.
Same hard rules: **no GPU**, CPU compile only, no commits, no slurm submit, only files under `megabench/sota/pi05/` and
`megabench/tests/`, idiomatic CuTe with exceptions logged in `NOTES.md`. Compile tests go through `arm.cpu_arm` (real
binding path) only.

## Phase-2 GPU result (jobs 7748 fail, 7749 pass; `runs/20261016T232255Z-p2/`) — record in NOTES.md
7748: compile failure (3-D vs real 4-D prefix_k; fixed, test gap closed via `cpu_arm`). 7749: pro, A, B byte-equal for
P=128/k16 seeds 7301/7302, P=968/k64, P=128/k16 mask-tail 17, P=128/k16 fuse 19. Non-vacuous (q/ksuf/vsuf ~99.6% nonzero).
B-stop launch: CUDA 0.173 ms, CuTe 0.224 ms (+51 us, of which +42 us was already in the prologue).

## Goal
Add the attention K/V prefill, phase C (FA-style split-key attention, PI05_ATT_FA=1) and phase C3 (chunk combine),
with the stop `C3`, byte-equal to CUDA for ATT_KEYS 16, 32, 64. Put the device code in `cute/attention.py`.

## Deliverables

### 1. Prefill (ATT_PREFILL 2), at the phase-2 hook before the B tasks (`pi05_mk.cu:1262-1287`)
Blocks `< ATT_NRG * ATT_NCHUNK`: CTA sync (the A tile's smem is being reused), mask bytes `Mk[tid]` for tid < ATT_KEYS
(`j < P ? mask[j] != 0 : j < NK`), cp.async (16 B, `.cg`) of prefix K and V rows `j < P` of chunk `j0 = (blockIdx / ATT_NRG) * ATT_KEYS`
into `Ks` / `Vs`, zero-fill rows `j >= NK`, skip rows `P <= j < NK` (they are written by B, loaded later), commit.
Smem carve (same bytes as CUDA): `Ks` at 0 `[ATT_KEYS][KS_STRIDE]`, `Qs` after Ks `[ATT_ROWS][KS_STRIDE]`, `Vs` after Qs
`[ATT_KEYS][VS_STRIDE]`, `Mk` (uint8 [ATT_KEYS]) after Vs. Express these as CuTe smem tensors with padded row strides.
Then `SYNC()` after B (already there).

### 2. Phase C (`pi05_mk.cu:1356-1684`, FA branch only)
- Unit loop `u = blockIdx; u < ATT_NRG*ATT_NCHUNK; u += gridDim`; `rg = u % ATT_NRG`, `kc = u / ATT_NRG`, `r0 = rg*64`,
  `j0 = kc*ATT_KEYS`; `pre = (u == blockIdx)` → `jlo/jhi` window of rows still to load (1375-1387); non-pre units write `Mk`.
- Loads: K/V rows `jlo..jhi` (source prefix row for j < P, `ksuf`/`vsuf` row `j-P` for P <= j < NK, zero-fill for j >= NK),
  Q rows `r0..r0+63` from `q` viewed as `[QROWS=400][HD]` (row >= 400 zero-filled, address clamped to row 399), all cp.async,
  then wait-all + CTA sync. Use CuTe copy atoms (`CopyG2SOp` with zero-fill predication) on the smem tensors above.
- Per warp: `mt = warp & 3`, `dh = warp >> 2`; skip the work if `r0 + mt*16 >= QROWS` (but keep the CTA syncs).
- S = Q K^T with ldmatrix x4 (natural k order), loop `kk = 0..15` outer, n-tile pairs inner, exactly as the phase-0 `fa/cute`
  probe. Then for each element: `Mk[cc] ? rbf(s) * 0.0625f : -inf` (cc = nt*8 + 2*t4 + (e&1)), row max over the quad
  (`shfl_xor` 1 then 2), `msafe = (m == -inf) ? 0 : m`, `pr = rbf(expf(s - msafe))` (mathx defaults), `rsum[hf] += pr[2hf]; rsum[hf] += pr[2hf+1]`
  in n-tile order, then quad `shfl_xor` 1, 2 sums. Pack P to bf16 A fragments (C→A repack of the probe).
- Stats: lanes with `dh == 0 && t4 == 0` store `float2(rmax, rsum)` at `stats[(row * ATT_NCHUNK + kc) * 2]` for rows < 400.
- O_c = P V: `ks = 0..ATT_KEYS/16-1` outer, `np = 0..7` inner, ldmatrix x4 trans from `Vs` at the warp's 128-column half,
  acc pairs `(2np, 2np+1)` ← `(b0,b1)`, `(b2,b3)`; store float2 to `opart[(kc*QROWS + row)*HD + c]` for rows < 400.
- CTA sync; if FUSE_CC3 (fuse bit 2) thread 0 `counter_publish(cnt + (CNT_ATT + rg)*CNT_STRIDE)`. After the loop, `SYNC()`
  if not FUSE_CC3.

### 3. Phase C3 (`pi05_mk.cu:1686-1731`)
`prefetch_l2` of `wg[l]`, `wu[l]`; tasks `QROWS/4` from the top block index; `block_wait(cnt + (CNT_ATT + task*4/64)*CNT_STRIDE,
it1*ARR_ATT)` if FUSE_CC3; `i = (task*256 + tid)*4`, `r = i / HD`; load all `(m_c, l_c)` (.cg float2), `M` = max in chunk order,
`w_c = (m_c == -inf) ? 0 : expf(m_c - M)`, then o_c float4 loads in groups of CG = 8 chunks, `num = fma(w, o, num)` per component
and `den = fma(w, l_c, den)` in chunk order, store `pack_bf16x4(num/den)` to `attn`. Stop `C3` after this loop (before the
`SYNC` / WPRE-8 deferred barrier of D).

### 4. Knobs / config
`Knobs.validate` must reject combinations the CUDA static_asserts reject (`ATT_NRG * ATT_NCHUNK <= 148`, ATT_KEYS % 16 == 0,
ATT_KEYS <= 64, ATT_NT even). Valid sets to support: P=128 with k16/k32/k64; P=968 with k64 (k32 gives 7*32=224 > 148: invalid).

### 5. parity.py
Stop `C3`; expected-written adds `stats`, `opart`, `attn` (and `cnt` when fuse bit 2 is set).

### 6. Tests (CPU, via `cpu_arm`) `megabench/tests/test_pi05_cute_phase3.py`
Compile stop `C3` for P=128 k16/k32/k64 and P=968 k64 at fuse 18, plus P=128 k16 at fuse 16 (CC3 off: barrier path).
Report REG/STACK/spill bytes per build (print). Run ALL test modules (phase 0-3) in one final command and report that run.

### 7. `cute/slurm/phase3.sbatch` (do not submit; `--time` must be <= 00:30:00)
`parity --compare --stages B,C3` for: P=128 k16 seeds 7301/7302; P=128 k32; P=128 k64; P=968 k64; P=128 k16 mask-tail 17;
P=968 k64 mask-tail 200 (realistic padded prompt); P=128 k16 fuse 16. Timing line: C3-stop launch at P=128/k16 and P=968/k64.

### 8. NOTES.md
Phase-2 GPU results, phase-3 build notes (regs/spills, smem carve, idiom exceptions, open guesses).

## Done when
All CPU tests (phase 0-3 modules) pass in one final run. Short factual report as before.
