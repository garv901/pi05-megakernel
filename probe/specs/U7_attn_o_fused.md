# U7 spec: fused attention + o_proj microbenchmark (pi0.5 megakernel, P = 128)

## Goal
Decide whether fusing phases C (split-key attention), C3 (combine) and D (o_proj) of `mk/pi05_mk.cu` into one
communication-free phase is faster than the current C + D. In the fused phase each CTA recomputes the full attention
of one head (redundant across CTAs) and immediately multiplies it by its slice of Wo. The number to beat is the
in-situ production interval **C + D = 8.54 + 3.30 = 11.84 us/layer** (CUDA k16 build, P128, mean of runs stage0 + stage2; `probe/runs/analysis/tables.md` Table 1; it spans
from the B->C barrier exit to the D->E barrier exit, so it contains two barriers). The fused phase has one.

You write and compile code only (CPU). Do NOT run anything on a GPU and do NOT submit Slurm jobs; the main agent does that.

## Files you own (create only these; touch nothing else)
- `probe/ubench/attn_o_fused.cu`   standalone CUDA program, one binary, runtime flags
- `probe/slurm/u7.sbatch`          builds the binary and runs every config below into `probe/runs/<UTC ts>-u7/`
All paths relative to `/raid/garv901/vibe-megakernel/megabench/sota/pi05/`. Match the style of the existing
`probe/ubench/sync_cost.cu` (header comment with purpose + build line, `CK()` macro, globaltimer stamps, terse comments).

## Geometry (fixed, constexpr)
T = 50 tokens, NQH = 8 query heads, 1 shared K/V head, HD = 256, P = 128 prefix keys, NK = P + T = 178 keys,
W = 1024 (o_proj output width), QD = 2048 (o_proj input = 8 heads x 256), L = 18 layers (distinct buffers per layer).
Score scale 1/16 (= 1/sqrt(256)). No mask (prefix is fully valid at P = 128); keys j >= 178 in a padded block are -inf.
Query rows: head-major, head h = 50 rows padded to 64 (rows 50..63 zero, never stored).

Buffers (bf16 unless noted, random data from a fixed-seed host RNG; Q, K, V ~ N(0,1); Wo ~ N(0,1)/sqrt(2048)):
- `pk[L][P][HD]`, `pv[L][P][HD]`         prefix K/V
- `qsrc[L][NQH][T][HD]`, `ksrc[L][T][HD]`, `vsrc[L][T][HD]`   per-layer sources of Q and suffix K/V
- `qbuf[NQH][T][HD]`, `ksuf[T][HD]`, `vsuf[T][HD]`            working copies written in the pre-phase (emulates phase B)
- `wo[L][W][QD]`  row-major [n][k]: out[t][n] = sum_k attn[t][k] * wo[n][k]   (same as `p.wo` in pi05_mk.cu)
- fp32 partial outputs (see variants), a 512 MB cold pool for the L2 filler, an 16.8 MB region for the L2-pressure prefetch.

## Kernel structure
One persistent kernel: grid = 148 CTAs x 256 threads, `__launch_bounds__(256, 1)`, dynamic smem set with
`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`. Grid barrier = the production counter barrier (copy
`grid_barrier` from `mk/pi05_mk.cu:301-330`, PI05_BARRIER 3 path: CTA sync, thread-0 `red.release` add, `ld.acquire` poll
for epoch * gridDim.x, CTA sync). Each launch runs NIT iterations; iteration `it` uses layer `l = it % 18`:

```
pre-phase (not measured, all 148 CTAs):
  grid-stride copy qsrc[l] -> qbuf, ksrc[l] -> ksuf, vsrc[l] -> vsuf          (emulates phase B's writes, L2-resident)
  stream FILL_MB (default 34) of the cold pool with ld.global.nc, advancing through the pool each iteration
                                                                              (emulates the 34.6 MB/layer weight stream evicting L2)
  prefetch.global.L2 of wo[l] (4 MB) and pk[l], pv[l], spread over the grid (as pi05_mk.cu:1259-1261 does in phase B)
  if PF: each working CTA issues cp.async of everything it needs that does not depend on phase B:
         its Wo slice and the prefix K/V rows (j < P) it will hold in smem  (as PI05_ATT_PREFILL does, pi05_mk.cu:1322-1352)
barrier                                     <- block-0 stamp S0 after exit
measured phase (variant below); if L2P: all 148 CTAs first issue prefetch.global.L2 over the 16.8 MB region spread
  over the grid (emulates the Wg/Wu prefetch that pi05_mk.cu:1689 issues in C3)
barrier                                     <- block-0 stamp S1 after exit
```
Measured interval per iteration = S1 - S0 (block 0 is always a working CTA). Also per working CTA, stamps relative to
its own barrier exit: `t_q` (Q + all post-barrier loads landed), `t_att` (attention output finished, before o_proj),
`t_o` (partial stored). Discard the first 20 iterations of each launch; NIT = 200; 5 launches per config.

## Variants (the `--variant` flag)
Reuse the production building blocks: `mma16816`, `ldmatrix_x4`, `ldmatrix_x4_trans`, `cp_async16` (pi05_mk.cu:224-258),
and the FA unit structure (pi05_mk.cu:1356-1530, PI05_ATT_FA 1 path: warp = (m-tile w & 3, output-dim group w >> 2),
S and P in registers, quad-shuffle max/sum, C->A fragment repack for P.V). Rounding as production: s = bf16(q.k) / 16,
p = bf16(exp(s - m)), l = fp32 sum of p, attention output rounded to bf16 before o_proj.

**ctl**: measured phase is empty (barrier only). Calibrates the barrier + stamp cost.

**F1** (full head, all 256 dims, 16 column slices): working CTAs b < 128, head h = b / 16, slice j = b % 16.
  1. Wo slice = wo[l][64j .. 64j+64][256h .. 256h+256] -> smem (32 KB + padding).
  2. Q of head h [64 x 256] -> smem. K/V streamed in key blocks of KB keys (KB = 32 or 64, `--kb`), cp.async double buffered,
     FA2 online softmax (running max m, sum l, O accumulators rescaled by exp(m_old - m_new) per block).
     Warp w: m-tile w & 3, output dims [128 (w >> 2), +128); both dim halves compute S (as production).
  3. O /= l, round to bf16, store to smem (reuse the Q buffer) as the mma A operand.
  4. o_proj: [64 x 256] x [256 x 64]; warp w: m-tile w & 3, 32 output columns (w >> 2). Store rows < 50 as fp32 to
     `part1[h][50][1024]` at columns 64j .. 64j+64.  (Phase E would then sum 8 head partials instead of 16 chunks.)
  With PF: Wo slice and the first prefix key block are issued before the barrier.

**F2** (full head, one 128-dim half, 8 column slices; output layout identical to production phase D):
  working CTAs b < 128, chunk c = b / 8 (head h = c / 2, dim half dh = c % 2), slice j = b % 8.
  1. Wo slice = wo[l][128j .. 128j+128][128c .. 128c+128] -> smem (32 KB + padding).
  2. All 178 K rows (256 dims, for S) and all 178 V rows restricted to dims [128 dh, +128) resident in smem
     (~94 KB + ~48 KB), plus Q of head h [64 x 256]. No key loop; softmax over all keys at once
     (two-pass in registers, or 2-3 sub-blocks with online rescaling if registers demand it; choose, avoid spills, report).
     Warp w: m-tile w & 3, output dims [128 dh + 64 (w >> 2), +64); both compute S.
  3. O /= l, bf16, to smem.  4. o_proj [64 x 128] x [128 x 128]; warp w: m-tile w & 3, 64 columns (w >> 2).
     Store rows < 50 to `part2[c][50][1024]` columns 128j .. 128j+128 (= production `part[s][t][n]` with s = c).
  With PF: Wo slice and all prefix K/V rows (j < P) issued before the barrier; after it only Q and the 50 suffix rows.

Flags: `--variant ctl|F1|F2`, `--kb 32|64` (F1), `--pf 0|1`, `--solo 0|1` (1: only CTA 0 does the measured work and the
PF issue; every other CTA goes straight to the barriers; gives the isolated per-CTA chain latency), `--l2p 0|1`
(default 1), `--fill MB` (default 34), `--check` (see below).

## Correctness (`--check`)
Run one iteration for layers 0 and 17 with solo 0, sum the partials over heads (F1) or chunks (F2) into out[50][1024]
and compare with a host fp64 reference computed from the same bf16 inputs (scores q.k/16, softmax, P.V, attention
output rounded to bf16, then o_proj). Print rel L2 error and max abs error; FAIL if rel L2 > 2e-2 or any non-finite value.
The sbatch runs `--check` for F1 kb32, F1 kb64, F2 before the timing configs.

## Output
One line per config, `key=value` pairs, also appended to `results.txt` in the run dir:
`variant kb pf solo l2p fill interval_med interval_p10 interval_p90 work_max_med t_q_med t_att_med t_o_med smem regs`
(all times in us; work_max_med = median over iterations of the max over working CTAs of t_o; t_*_med = median over
iterations of the max over working CTAs). Print ptxas register / spill counts at build time into `build.txt`
(`-Xptxas -v`); target 0 spill stores. If F1 or F2 spills, say so in the header comment and in your report.

## sbatch (`probe/slurm/u7.sbatch`)
Copy the header of `probe/slurm/stage3.sbatch` (partition priority, 1 GPU, 16 CPUs, `--time=00:20:00`, job name
`p05probe-u7`, log to `probe/runs/slurm-%j.log`, `source megabench/sota/slurm/env.sh`). Build with
`nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v attn_o_fused.cu -o attn_o_fused` into the run dir.
Record `nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv`. Configs, each under `timeout 120`:
- checks: F1 kb32, F1 kb64, F2
- ctl solo 0
- F1 kb in {32, 64} x pf in {0, 1} x solo in {0, 1}
- F2 pf in {0, 1} x solo in {0, 1}
- F1 kb64 pf 1 l2p 0, F2 pf 1 l2p 0   (L2-pressure sensitivity)

## Report back (short)
Files written, build command and its ptxas summary (regs, spills, smem per variant), whether everything compiled, any
deviations from this spec and why, and anything you think will distort the measurement.
