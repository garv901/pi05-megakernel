# U8 spec: F3 = row-split fused attention + o_proj (extends U7)

## Goal
U7 (`probe/specs/U7_attn_o_fused.md`, results `probe/runs/20261017T003008Z-u7/results.txt`) showed that a CTA computing a
whole head's attention (64 query rows x 178 keys) takes ~8 us of mma.sync math (~60 mma per us per warp, ~2 TFLOP/s/SM),
so the fused phase (F2 pf=1: 13.15 us) loses to production C + D (11.84 us). F3 shrinks each CTA's attention by splitting the
**query rows** (tokens) across CTAs instead of keys. A query row's softmax needs all keys but no other row, so no combine is
needed and no CTA-to-CTA communication exists until phase E. Inside the CTA the 8 warps split the **keys** for S (no warp
computes S twice, unlike F2) and merge softmax max / sum through smem (the Qwen3 round-6 pattern), then split the **output
dims** for P.V. Expected per-warp mma count: S 48 + P.V 48 + o_proj 64 (F2: 384 + 96 + o_proj).

You write and compile code only (CPU). Do NOT run anything on a GPU and do NOT submit Slurm jobs.

## Files you own
- `probe/ubench/attn_o_fused.cu`: ADD variant F3 (`--variant F3`, template V = 3). Do not change the behaviour, timing or
  output format of ctl / F1 / F2. Update the header comment.
- `probe/slurm/u8.sbatch`: new; same header as `u7.sbatch` but job name `p05probe-u8`, run dir suffix `-u8`.
Paths relative to `/raid/garv901/vibe-megakernel/megabench/sota/pi05/`. Read the existing `attn_o_fused.cu` fully first
and reuse its helpers (`load_rows`, `load_q` pattern, `s_block`, `oproj` style, stamps, check path, Args, pre-phase).

## F3 work decomposition
Working CTAs b < 128: head h = b / 16, row block rb = (b / 4) % 4, Wo column slice j = b % 4.
- Row block rb = tokens [rb * 50 / 4, (rb + 1) * 50 / 4) = 12, 13, 12, 13 tokens -> one 16-row m-tile (rows past the block
  zero-filled, never stored). Q rows come from `qbuf[h][t][:]` (head-major, as F1/F2).
- Keys: all 178 (prefix 128 + suffix 50), padded to 192 = 8 warps x 24 keys (3 n-tiles each); keys >= 178 masked -inf.
- o_proj output: rows of this block x columns [256 j, 256 j + 256); K = head h's 256 input dims (wo columns 256h .. +256).
  Store fp32 to `part[h][t][256 j + c]` (same `part1[h][50][1024]` layout as F1; the check sums 8 head partials).
- Redundancy: each (head, row block) attention is computed by the 4 slice CTAs; each Wo element is read by the 4 row-block CTAs.

## F3 per-CTA steps (8 warps, warp w)
smem: K [192][264] bf16, V [192][264] bf16, Q/O [16][264] bf16, P [16][192 + 8] bf16, softmax stats [8 warps][16 rows] x
(max, sum) fp32. About 218 KB; must stay <= 232,448 B (assert it).
1. Loads (cp.async): Q block (16 rows), K and V rows. With `--pf 1`, prefix rows j < 128 and the zero rows 178..191 of K and
   V are issued before the barrier (as F2's `f2_issue_pre`); after it only Q and the 50 suffix rows. With pf 0, all after.
   Wo does not fit in smem; it is read from L2 (the pre-phase already prefetches all of wo[l] into L2).
2. S: warp w computes S[16 rows][keys 24w .. 24w+24) = Q K^T over 256 dims (16 k-steps x 3 n-tiles = 48 mma).
   s = bf16(q.k) / 16, masked keys -inf. Row max over the warp's 24 keys (quad shuffles), write to smem stats.
3. __syncthreads; every warp reads the 8 per-warp maxima of its rows in fixed warp order -> row max M (exact, all keys).
   p = bf16(exp(s - M)); per-warp row sums of p -> smem stats; P (bf16) -> smem P [row][key].
4. __syncthreads; l = sum of the 8 per-warp sums in fixed warp order. P.V: warp w computes O[16][dims 32w .. 32w+32)
   over all 192 keys (12 k-steps x 4 n-tiles = 48 mma; A = P via ldmatrix, B = V via ldmatrix.trans as in F1/F2).
   O /= l, round to bf16, store into the Q/O buffer (Q is dead after step 2; keep the needed __syncthreads).
5. o_proj: warp w computes out[16 rows][32 columns 256 j + 32 w ..] over K = 256 (16 k-steps x 4 n-tiles = 64 mma).
   A = O from smem (ldmatrix). B = Wo loaded straight from global/L2 into registers in two 128-k halves, the way
   `gemm_phase` `load_w` does in `mk/pi05_mk.cu:438-450` (uint4 b[4][4] per half: 32 columns x 128 k per warp), second half
   loaded while the first half's mma run. Issue the first half's loads as early as the register budget allows without
   spills (ideally before step 4's P.V); report where you put it. If registers force spills, fall back to staging the Wo
   half-slices through the dead K buffer with cp.async after step 2 and say so.
6. Store rows of the block (valid tokens only) as fp32 float2, as `oproj` does.
Stamps (thread 0 after a CTA sync, relative to its barrier exit): t_q (all post-barrier loads landed), t_att (O in smem),
t_o (partial stored), same arrays and output fields as F1/F2.

## Check
`--check --variant F3` must pass the existing fp64 host reference (sum of the 8 head partials) for layers 0 and 17 with
the same 2e-2 threshold. Run the check with pf 0 and pf 1 in the sbatch.

## u8.sbatch configs (each under `timeout 120`, all in one job so they are directly comparable)
- checks: F3 pf 0, F3 pf 1, F2 (regression)
- ctl solo 0
- F2 pf 1 solo 0                         (U7 best, same-job reference)
- F3 pf in {0, 1} x solo in {0, 1}
- F3 pf 1 l2p 0
Build with the same nvcc line as u7 (`-Xptxas -v` into build.txt).

## Report back (short)
Build result and ptxas per kernel (regs, spills, smem), where the Wo loads are issued, deviations and why, and anything
that could distort the measurement. Do not run or submit anything.
