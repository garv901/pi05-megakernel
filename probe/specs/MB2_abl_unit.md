# MB2 spec: standalone microbenchmark of the A -> C-release window (split-K + barrier vs fused A+B unit)

## Why
FUSE_ABL (CuTe fuse bit 32, `cute/abl.py`) fuses the QKV GEMM (A), the A->B grid barrier and the reduce/RoPE phase (B) into
one barrier-free phase. In the full chunk build it is +3.2 us/layer slower (+567 us/chunk P128 k16, +650 us P968 k64). IKET
(in-kernel timeline) of the full builds: A-release -> C-release window 8.29 us (fuse 18) vs 11.01 us (fuse 50); fused unit work
~9.8 us vs ~4.2 us for the split-K A stage; removed: barrier ~0.96 us + B ~3.14 us. MB2 rebuilds ONLY this window in a
small persistent kernel, so that each cost can be switched on and off:
- H3 staging latency chain: the unit stages 8 A chunks with adaRMS norm per CTA (baseline: 1). Is it L2-latency bound (MLP
  limited by 1-chunk register lookahead) or bandwidth / ALU bound?
- H4 spread: the C release is the max over 80 unit CTAs of (start + work); how does each variant amplify start jitter?
- H5 collisions: unit CTAs (bids 68..147) overlap prefill CTAs (bid < 84 at P128, < 112 at P968) and the post-unit L2 prefetch.
- Granularity: is there a middle point (c chunks per CTA, 1 < c < 8, cross-CTA reduce through local counters) that wins?
- Codegen: each variant here is its own kernel with its own register allocation; compare its unit time with the in-situ 9.8 us.

You write and compile code only (CPU). Do NOT run anything on a GPU and do NOT submit Slurm jobs.

## Files you own (paths relative to `/raid/garv901/vibe-megakernel/megabench/sota/pi05/`)
- `probe/ubench/abl_unit.cu` (new; single file, header comment = build line + one paragraph per variant, like `sync_cost.cu`)
- `probe/slurm/mb2.sbatch` (new; header as `probe/slurm/u8.sbatch`, job name `p05probe-mb2`, run dir suffix `-mb2`,
  `--time=00:30:00`, `--partition=priority`)
Read first: `probe/ubench/sync_cost.cu` (barrier, stamps, JSON output style), `probe/ubench/attn_o_fused.cu` (Args parsing,
check path), `mk/pi05_mk.cu:420-612` (`gemm_phase`, PI05_GEMM_FRAG == 2 path = production), `mk/pi05_mk.cu:1270-1345` (A, B tasks,
prefetch, prefill mode 2), `mk/pi05_mk.cu:255-300` (`block_wait`, `red_release`, acquire spin), `mk/pi05_mk.cu:755-765`
(`reduce_part4`), `cute/abl.py` (the fused unit: unit -> columns, stage_a8 order, smem partial buffer, 200 reduce tasks).
Copy helpers (mma16816, lds128, cp_async16, prefetch_l2, rbf, pack_bf16x4, RoPE expression) from `pi05_mk.cu` verbatim.

## Shapes (production expert layer, fixed constexpr)
T = 50 rows (MMA tile 64, rows >= 50 zero), K = W = 1024, KC = 128, NCH = 8 chunks, NSSP = 16, virtual N = 2560 = Q 2048 +
K 256 + V 256 = 10 heads x HD 256. 256 threads / 8 warps per CTA, grid = 148 CTAs (all SMs), 1 CTA/SM,
`__launch_bounds__(256, 1)`, cooperative launch (`cudaLaunchCooperativeKernel`; fail loudly if it cannot be co-resident).
Buffers: h [50][1024] bf16 (A input), h0 pristine copy, ssp [50][16] f32 (sum of squares of h over 64-column parts),
nmod [2 * 1024] f32 (scale then shift), w [2560][1024] bf16 (wq | wk | wv as one tensor, L2 resident after warm-up),
rope [50][128][2] f32 (cos, sin; theta 10000, freq_j = theta^(-2j/256), position t), part [8][50][2560] f32, out [50][2560]
bf16 (q | ksuf | vsuf as one tensor; heads 0..8 get RoPE, head 9 (V) does not), prefix pool for prefill (2 x 1024 x 256 bf16),
prefetch pool (see --pf_kb), counters, barrier word, stamps.
Data: deterministic RNG, h ~ N(0,1), w ~ N(0, 1/32), nmod scale ~ 0.1 N(0,1), shift ~ 0.1 N(0,1); ssp from h (host).

## One iteration (the kernel loops over --iters iterations; same structure for all variants)
1. prev phase (stand-in for phase I): if --fresh 1, residual tasks task = bid + k * 148 < 800 (row = task / 16, part = task % 16)
   copy 64 bf16 of h0 -> h and write ssp[row][part] (plain st.global; same values every iteration). Then optional
   cp.async.wait_group 0 + __syncthreads (drains the previous iteration's prefill before smem reuse).
2. G1 = production WPRE barrier: `wpre_arrive` (CTA sync, thread 0 atom.add.release), issue this CTA's weight loads into
   registers (variant specific, = `load_w`), `wpre_wait` (thread 0 acquire spin on epoch * 148, CTA sync). Stamp s0.
3. jitter: if --jitter_ns J > 0, every CTA spins (thread 0, then CTA sync) `hash(bid, iter) % (J + 1)` ns. Stamp s1.
4. variant work (below), stamps s2 .. s6.
5. G2 = plain grid barrier (stand-in for the barrier before C). Stamp s7 = G2 release observed.
Every stamp is taken by thread 0 after a __syncthreads followed by one volatile ld.shared whose result is unused (the "settle"
from `cute/iket_marks.py`: BAR.SYNC.DEFER_BLOCKING lets a warp read %globaltimer before the barrier releases), then
`mov.u64 %globaltimer`. stamps[iter][bid][8] u64. Stamps not reached by a CTA (idle CTA) = the previous stamp.

## Variants (template parameters so each gets its own register allocation; one kernel per supported combination)
**V0 = production split-K (fuse 18).** CTA bid owns chunk s = bid % 8, warp tile ng = (bid / 8) * 8 + warp, tiles ng < 80
(32 contiguous virtual columns), `load_w` before G1 wait (step 2); stage A chunk s with norm exactly as `gemm_phase`
PI05_GEMM_FRAG == 2 (one load round, rs from ssp shuffle tree, fragment-interleaved tile) (s2); MMA + fp32 partial store to
part[s][t][n] (s3); A->B grid barrier (s4), or with `--abc 1` no barrier and per-head counters as FUSE_AB (red.release per
tile, B task block_waits on its head's counter, target = 64 tiles x epoch) (s4 = after the wait); L2 prefetch of this CTA's
share of --pf_kb (`prefetch_l2`, spread over the grid); prefill (step "Prefill" below, CTAs bid < --prefill_ctas; production
mode 2 issues it before the B tasks); B: 70 tasks (10 heads x 7 slices of 8 rows), task = 147 - bid, `reduce_part4` of 8 partials
(.cg, s = 0..7 summed from +0.0), rbf, RoPE for heads < 9, 2 x 64-bit bf16 stores (s5); s6 = s5.
**V1 = fused unit (fuse 50), as `cute/abl.py`.** 80 units, u = 8 hh + p, columns 256 hh + 16 p + [0,16) and 256 hh + 128 + 16 p
+ [0,16); unit u on CTA `place(u)` (--place hi: 147 - u, production; spread: u * 148 / 80). Warp w = chunk w loads its 32 gathered
weight rows (step 2). Stage all 8 chunks with norm into 8 private 16 KB fragment-interleaved tiles (s2), modes below; warp w MMA
on tile w; __syncthreads; fp32 partials to smem [8][50][PB_STRIDE 40] reusing the tiles (s3); s4 = s3; 200 reduce tasks
(tid < 200: t = tid / 4, quad qd = tid % 4, x1 = tile cols 4 qd.., x2 = 16 + 4 qd..), sum s = 0..7 from +0.0, rbf, RoPE, stores (s5);
then (all CTAs, unit or not) L2 prefetch share and prefill (CTA sync first, it reuses smem) (s6). Non-unit CTAs skip 2..s5 work.
Staging modes (template): `--stage reg --la L`, L in {0, 1, 2, 4}: register staging; L chunks' raw loads (h 16B x 4 per thread,
nmod 4 float4) are in flight while chunk c is normalised (L = 1 is `stage_a8`'s pattern: c + 1 in flight, c + 2 issued after c
is stored; L = 0 = load, wait, normalise, store, fully serial). rs per row computed once (it does not depend on k0).
`--stage cpa`: all 8 chunks' raw h rows issued at once by cp.async (16 B, zero-fill rows >= 50) into 8 row-major tiles with
row stride AS_STRIDE = 160 bf16 (8 x 20 KB = 160 KB), wait_all + sync, normalise in place (thread mapping of PI05_GEMM_FRAG == 0),
sync, MMA with the PI05_GEMM_FRAG == 0 operand loads. nmod loads stay in registers. Must give byte-identical outputs.
`--norm 0` (both modes): skip the normalisation arithmetic (plain copy), to separate ALU from memory. Outputs then differ;
the check skips norm 0.
**H = hybrid, `--c` in {1, 2, 4}** (c = 8 is V1). G = 8 / c K-groups. 80 units u: kg = u % G, group gi = u / G covers
pair tiles [gi * 8 / c, (gi + 1) * 8 / c) (pair tile = V1's unit column set). Warp w: pair tile gi * 8 / c + w / c, chunk
kg * c + w % c. CTA stages its c chunks (register mode, la 1), MMA, stores the fp32 partial to part[chunk][t][virtual col]
(same layout as V0) (s3), __syncthreads, thread 0 red.release on counter[gi]; then block_wait(counter[gi], G * epoch) (s4);
the G CTAs of the group split the group's rows (kg takes rows [50 kg / G, 50 (kg + 1) / G)) and reduce all 8 partials in order
s = 0..7 from global (.cg), rbf, RoPE, store (s5); prefetch / prefill as V1 (s6). Placement as V1.

Prefill (all variants, CTAs bid < --prefill_ctas, keys --prefill_keys, 0 = off): __syncthreads, cp.async 16 B of
prefill_keys rows of K and of V (512 B each) from the prefix pool into smem past the region the variant is still using
(V0: past its A tile; V1/H: tiles are dead after the reduce), commit, no wait (drained in step 1 of the next iteration).
Prefetch: --pf_kb KB per iteration from a rotating pool of 24 x pf_kb (so lines really come from HBM, as the next layer's wo /
prefix K/V do in production), split over all 148 CTAs as `prefetch_l2` does.
`--active N` (V1, H): only units u < N do work (the others behave as non-unit CTAs). Default 80.
`--smem_kb S` dynamic smem pinned for every variant (default 132; cpa needs 160 and forces 160; print the effective value).

## Measurement and output
--iters 200 per launch, --launches 5; drop launch 0 and the first 10 iterations of each launch. Per iteration: R1 = min over CTAs
of s0, R2 = min over CTAs of s7, window = R2 - R1. Report median / p10 / p90 of window. Per stage k = 1..6: median over
iterations of the median over WORKING CTAs (V0 all with a tile or a task; V1/H unit CTAs) of (s_k - s_{k-1}), plus the median
over iterations of max over CTAs (s6) - R1 ("last arrival at G2"). Critical CTA = argmax s6 per iteration: report the 3 most
frequent bids with counts and their roles (unit / tile / B task / prefill / idle). Also an "idle" control: --variant ctl
(G1, jitter, prefetch/prefill only, G2) for the barrier-pair floor.
`cudaFuncGetAttributes` of the kernel used: numRegs, localSizeBytes (spills), print both.
One JSON line per run, like U3: {"u":"MB2","variant":"V0|V1|H|ctl", all knobs, "smem_kb", "regs", "local_bytes",
"window_us","window_p10","window_p90","st_us":[6 values],"last_arrival_us","crit":[[bid,count,role],...]}.
`--dump F` writes the raw stamps of the last launch (binary u64 [iters][148][8], plus a 1-line text header) for offline plots.

## Check (`--check`)
Runs 1 launch x 3 iterations. (a) V0 output vs a host reference (norm in float exactly as the device formula with 1/sqrtf on
host, bf16 rounding of the normalised A, GEMM in double, rbf, RoPE in double): max abs error <= 2e-2 * max |ref|.
(b) the selected variant's out == V0's out byte for byte (same inputs; norm 1 only). Print PASS/FAIL with the first mismatch.

## Build
`nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v probe/ubench/abl_unit.cu -o abl_unit`. It must compile on this CPU node
(`/usr/local/cuda/bin/nvcc`); report ptxas registers / spill bytes per kernel instantiation in your final message. Limit the
template instantiations to the combinations the sbatch uses.

## mb2.sbatch (all in one job, each run `timeout 120`, results appended to $D/results.txt, `nvidia-smi` clocks first)
Defaults for every run unless stated: --fresh 1 --pf_kb 4224 --prefill_keys 16 --prefill_ctas 84 --place hi --jitter_ns 0
--smem_kb 132 (= the P128 k16 production context). "P968" = --pf_kb 5216 --prefill_keys 64 --prefill_ctas 112.
- checks: V0; V0 --abc 1; V1 --stage reg --la {0,1,2,4}; V1 --stage cpa; H --c {1,2,4}
- A calibration: ctl; V0; V0 --abc 1; V1 (reg la 1); and the same four with P968
- B staging: V1 reg la {0,1,2,4}; V1 cpa; V1 reg la 1 --norm 0; V1 cpa --norm 0
- C contention: V1 reg la 1 and V1 cpa with --active {1, 8, 40, 80}
- D granularity: H --c {1,2,4}; V1 reg la 1; V1 cpa; (V0 is in A)
- E jitter: V0, V1 reg la 1, H --c 1 with --jitter_ns {0, 1000, 2000}
- F context: V0 and V1 (reg la 1) with {--fresh 0, --pf_kb 0, --prefill_keys 0, all three off}; V1 --place spread (P128 and P968)
- G smem pin: V0 with --smem_kb {117, 132, 160}
- one --dump each for V0 and V1 (defaults) into $D

## What to report back
Files written, compile output (registers / spills per instantiation), the template combinations built, any deviation from this
spec and why, and anything in the production code that contradicts this spec.
