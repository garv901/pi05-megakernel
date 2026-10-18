# Port the pi0.5 megakernel (`mk/pi05_mk.cu`) to CuTe DSL

## Context

`megabench/sota/pi05/mk/pi05_mk.cu` is Keisuke's v18 persistent cooperative megakernel for one pi0.5 action chunk
(CUDA, mma.sync + split-K + software grid barriers). At P=128 it runs 7.69 ms (k16) against 6.73 ms for
the FlashInfer graph+PDL baseline (`README.md`). We want the same kernel in CuTe DSL
(cutlass-dsl 4.7.1 in `/raid/garv901/vibe-megakernel/.venv`). Then we can (a) read and change it in idiomatic CuTe and
(b) use Blackwell features later (TMA, tcgen05, clusters) that are hard to reach from the hand-written CUDA.

Decisions (from the user):
- **Fidelity:** each stage must be bit-exact against the CUDA kernel (0 ulp on every workspace buffer written so far).
- **Style:** fully idiomatic CuTe. Use CuTe tensors and layouts for every buffer, `TiledMma` (warp `MmaF16BF16Op` 16x8x16)
  and copy atoms (`CpAsync`, `LdMatrix`). Write the CUDA kernel's k-permutations and fragment-interleaved tiles as
  layouts (composition / permuted modes), so the arithmetic order stays the same. Only the sync primitives (grid
  barrier, arrival counters, `prefetch.L2`, fences) use `cute.arch` or inline asm, because they have no CuTe form.
  If a construct cannot be made both idiomatic and bit-exact, log it as an exception in `NOTES.md` with the reason.
- **Scope:** port the production configuration plus a few ablation switches as `Constexpr`: `P`, `ATT_KEYS`, fuse bits,
  WPRE bits and `D_WPT`. Drop the dev-only paths: attention layout 0, cg/flag barriers 0-2, TAIL 0/1, GEMM_FRAG 0/1,
  PTRACE, CSTAMP, ATT_FA_PERM, ATT_FA_EXACTL and D_CTL.
- **After the port:** an optional Blackwell-native phase. It is only sketched here and gets its own review.

---

## 1. Model brief: the forward pass we emulate

pi0.5 (Physical Intelligence) = PaliGemma VLM prefix + a **Gemma-300M "action expert"** trained with flow matching.
Only the action expert runs in the timed region. The VLM prefill (SigLIP + Gemma-2B) is done in the untimed
fixture, and only its per-layer post-RoPE BF16 K/V and pad mask reach the kernel (`reference/pi05.py`, `fixture.py`).

Geometry (B=1): horizon T=50 action tokens, action dim 32, expert width W=1024, 18 layers, 8 query heads / 1 KV head
(MQA), head dim 256, FFN 4096 (GeGLU), prefix length P (128 for the SOTA bench, 968 graded), 10 Euler steps.

```
x = noise[50,32] (fp32);  dt = -0.1
for step k in 0..9:   t = 1 + k*dt
  cond = silu(Wto · silu(Wti · sincos_emb(t) + b) + b)           # fp32 [1024], time MLP
  h = bf16(Win · x + b)                                           # [50,1024], action_in (fp32 math)
  for layer l in 0..17:
    (sc,sh,g) = Wattn_norm[l] · cond + b                          # adaRMS modulation, fp32 [3x1024]
    n = bf16(rms(h) * (1+sc) + sh)
    q,k,v = n·Wq, n·Wk, n·Wv   ;  q,k = RoPE(q,k, pos = nvalid + t)   # bf16
    a = softmax_fp32( bf16(q·[Kprefix;k]^T) / 16 , mask ) · [Vprefix;v]   # 400 q rows (50 tok x 8 heads) over P+50 keys
    h = h + bf16(a·Wo) * bf16(g)
    (sc,sh,g) = Wffn_norm[l] · cond + b
    n = bf16(rms(h) * (1+sc) + sh)
    h = h + bf16( (gelu_tanh(n·Wg) * (n·Wu)) · Wd ) * bf16(g)
  (sc,sh,_) = Wfinal_norm · cond + b ;  n = bf16(rms(h)*(1+sc)+sh)
  v = Wout · n + b  (fp32 [50,32]);   x = x + dt*v
actions = x
```
Masking: suffix queries see every valid prefix key and all 50 suffix keys (bidirectional). Precision: expert GEMMs and
activations are BF16 with FP32 accumulation; time MLP, adaRMS modulation, action in/out, noise and x are FP32;
attention keeps BF16 scores and probabilities with an FP32 softmax.

Cost: weights are about 34.6 MB per layer (BF16), so 6.2 GB of streamed weights per chunk (180 layer passes). The
modulation matvec reads 465 MB of FP32 norm weights once. HBM floor is 0.84 ms (nominal 8 TB/s; 0.84 / 7.69 ms is about 11%). Why the kernel is far above the floor is not established here
(not measured as bandwidth-bound; per-stage times are in `probe/runs/analysis`).

## 2. Kernel semantics to preserve (production config, fuse=18)

One cooperative launch: grid = #SMs (148), 256 threads, 1 CTA/SM, dynamic smem = max(attention unit, wsplit GEMM,
10xW fp32). All intermediates live in one workspace laid out by `make_layout()` in `pi05_mk.cu`.

**Prologue:** count nvalid → P0: time embedding (FP64 sin/cos/pow → fp32), RoPE table (`rbf(cos/sin)`), zero
counters → SYNC0 (cg) → P1/P2 `cond_matvec` time MLP for all 10 t → SYNC → P3 every modulation row for all
10 steps x 37 norms (`mod`) → SYNC → step-0 `action_in_tasks` (h, ssp, xbuf ← noise) → barrier deferred into layer
0's A (WPRE bit 1).

**Per layer (9 grid barriers; "→" = barrier, "⇢" = arrival counter):**

| ph | work | split / tasks | output | sync after |
|---|---|---|---|---|
| A | QKV GEMM [50x1024]·[1024x2560], A tile = adaRMS(h) built while staging (ssp + mod) | 8 K-chunks of 128, 32-col warp tasks, mma m16n8k16 | `part[s][t][n]` fp32 | → |
| B | split-K reduce (fixed order), bf16, RoPE on q/k; attention K/V prefill (cp.async) before tasks | (head, 8-row) block tasks from the top block index | `q`, `ksuf`, `vsuf` | → |
| C | FA-style split-key attention: unit = (64 q rows, ATT_KEYS keys), 7 x ceil(NK/ATT_KEYS) units | warp = (m-tile, dim half), S/P in registers | `stats`(m,l), `opart` | ⇢ CNT_ATT per row group |
| C3 | combine chunks: Σ e^(m_c-M) o_c / Σ e^(m_c-M) l_c, bf16 | 4-row tasks, waits on its row-group counter | `attn` | → (inside D, WPRE 8) |
| D | o_proj via `gemm_phase_wsplit<2>` (2 warps share a K chunk, smem reduce) | 16 K-chunks | `part` | → |
| E | gated residual h += bf16(o)*bf16(g); per (row, 64-col) sum of squares | warp tasks | `h`, `ssp` | → (inside F, WPRE 2) |
| F | gate/up GEMM, A = adaRMS_ffn(h) staged | 8 K-chunks | `part` | → |
| G | reduce + gelu_tanh(gate)*up, bf16 | (32-col, half rows) tasks | `act` | → (inside H, WPRE 4) |
| H | down GEMM [50x4096]·[4096x1024] | 32 K-chunks | `part` | → |
| I | gated residual + ssp; last layer: → `action_tail_rows` (final norm, action_out, Euler, next action_in) | warp tasks / row blocks | `h`,`ssp`,`xbuf`,`actions` | → (inside next A, WPRE 1) |

Plus `prefetch.global.L2` of the next phase's weights, prefix K/V and modulation rows at fixed points.

**Sync contract:** grid barrier = CTA sync, thread 0 does `atom.add.release.gpu` on one counter, then spins with
`ld.acquire` until it reaches `epoch*gridDim`, then CTA sync. The WPRE form splits this into arrive → first-tile weight
loads → wait. Arrival counters: producer CTA sync + `red.release.gpu`; consumer thread 0 does relaxed spin +
`fence.acq_rel.gpu` + CTA sync. Targets are `(iteration+1)*arrivals`, so counters are never reset inside a launch. Any
data produced by another CTA in this launch is read with `.cg` (L2) loads or `cp.async.cg`, never `.nc`/L1.

**Bit-exactness hot spots to reproduce exactly:**
- GEMM k-permutation: lane (g,t4) loads 8 contiguous k per 32-wide slab for both A and B. In CuTe this is a permuted K
  layout shared by the A and B views.
- Fragment-interleaved A tile (GEMM_FRAG 2) and the wsplit smem partial reduce order (warp 0 + warp 1).
- Split-K reduction order (s = 0..S-1), stats/opart combine order (chunk order, CG=8 groups).
- Every `rbf()` rounding point, `__fadd_rn`/`__fmul_rn` (no FMA contraction there) versus plain `a*b+c` (nvcc
  contracts it to FMA) and the explicit `fmaf` calls.
- libm semantics: `expf`, `tanhf`, `sinf/cosf`, `powf`, FP64 `sin/cos/pow`, `rsqrtf` (approx), IEEE `/`.

---

## 3. Code layout (new)

`megabench/sota/pi05/cute/`:
- `config.py`: geometry constants, `Constexpr` switch set (`P`, `ATT_KEYS`, `FUSE`, `WPRE`, `D_WPT`), derived sizes,
  smem budget, and a workspace layout that mirrors `make_layout()`. It is asserted equal to the CUDA ext's
  `layout_offsets()` in tests.
- `sync.py`: grid barrier, `WPre` arrive/wait, counter `publish`/`block_wait`, `prefetch_l2`, `.cg` load helpers
  (`cute.arch.red/load/atomic` + `llvm.inline_asm`, following `megabench/kernels/cute_fx1/toy_cute.py`).
- `mathx.py`: CUDA-parity scalar math (rbf, gelu_tanh, silu, rn add/mul, expf/tanhf/rsqrt lowering chosen in Phase 0).
- `gemm.py`: `gemm_phase` (plain / norm-staging / counter / WPRE) and `gemm_phase_wsplit` as TiledMma + permuted layouts.
- `attention.py`: prefill, FA unit (ldmatrix atoms, register softmax), C3 combine. Borrow structure from the
  cutlass DSL `examples/ampere/flash_attention_v2.py`.
- `rowops.py`: `cond_matvec`, `residual_tasks`, `action_in_task(s)`, `final_norm_row`, `action_out_pair`, `action_tail_rows`.
- `kernel.py`: `@cute.kernel pi05_megakernel` (dynamic `cutlass.range` step/layer loops, a stage stop for parity),
  `@cute.jit` host with `launch(cooperative=True, min_blocks_per_mp=1)`, compile cache keyed by the Constexpr set.
- `arm.py`: `CuteArm`, the same interface as `MegakernelArm` (`megakernel.py`), registered in `bench.py` as arm `cute`.
- `parity.py`: GPU script that runs CUDA vs CuTe with the same inputs and stage stop and byte-compares workspace buffers.
- `NOTES.md`: per-phase log, idiom exceptions, SASS/regs/spills, timings.

Tests (CPU, `CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a`, like `megabench/tests/test_cute_fx1_probe.py`):
`megabench/tests/test_pi05_cute_*.py`. They check that each module compiles, the layout equals `layout_offsets()`,
and that STACK does not grow versus the recorded numbers (`cuobjdump -res-usage` on the `CUTE_DSL_KEEP=cubin` dump). The kernel still has spills (full P=128/k16 build: REG 255, STACK 392).

**CUDA-side change (debug only):** the `PI05_STOP_PH` / `PI05_STOP_STEP` / `PI05_STOP_LAYER` macros in `mk/pi05_mk.cu` (stage ids 0..9) exit all
blocks cleanly after a given stage, passed through the existing `extra` flags of `megakernel.load()`. The default
build's SASS hash must stay equal to `probe/sass_baseline.txt`. The CuTe kernel gets the same stop as a `Constexpr`.

---

## 4. Phases

Each phase ends with a parity job: CUDA vs CuTe, same seed, `stop` at the listed stages, 0-ulp compare of the listed
buffers at P=128 (k16) and P=968 (k64). I review the result before the next phase starts.

**Phase 0: toolchain gate and harness (small).**
- GPU smoke in CuTe DSL: cooperative 148x256 launch, 117 KB dynamic smem, counter grid barrier + WPRE split and arrival
  counters, run 1634 times in a loop without deadlock.
- Math-parity microkernel: CUDA (nvcc -O3) vs DSL for expf, tanhf, sinf/cosf, powf, FP64 sin/cos/pow, rsqrtf, IEEE
  div, bf16 rn, and FMA contraction rules, over about 1e7 inputs each. Pick the DSL lowering for each (libdevice call,
  `fastmath` flag, or asm) and record it in `mathx.py`.
- Single-tile mma probe: one 64x32 tile with K=128, CUDA `gemm_phase` order vs CuTe TiledMma with the permuted-K
  layout, plus the FA ldmatrix path. Exit gate: bit-exact. Otherwise decide the idiom exception now.
- `PI05_STOP` in CUDA (SASS hash check), `config.py` layout, `parity.py`, CPU test skeleton.

**Phase 1: prologue.** nvalid, P0 (temb, rope, counter zeroing), `cond_matvec` x3, step-0 `action_in_tasks`.
Parity: `temb, tmid, cond, mod, rope, h, ssp, xbuf` at stop = prologue end.

**Phase 2: GEMM engine + A/B.** `gemm_phase` with norm staging, WPRE arrive/load/wait, optional counter publish (FUSE
bit 1); phase B reduce + RoPE + attention K/V prefill issue (ATT_PREFILL 2). Parity: `part` after A, `q/ksuf/vsuf`
after B (step 0, layer 0).

**Phase 3: attention C + C3.** FA unit with prefilled prefix rows, mask, register softmax, stats/opart, CNT_ATT
counters, combine. Parity: `stats, opart, attn` after C3, at ATT_KEYS 16/32/64.

**Phase 4: rest of the layer and the layer loop.** D (`gemm_phase_wsplit<D_WPT>`, WPRE 8), E residual, F gate/up
(WPRE 2, counters if FUSE bit 4), G, H (WPRE 4), I, L2 prefetches, epoch bookkeeping. Parity: every buffer after
each of D..I for layer 0, then `h, ssp` after layers 1, 17.

**Phase 5: tail, step loop, full chunk.** `action_tail_rows`, `a_nmod` hand-off, 10 steps. Parity: `actions`
bit-exact vs CUDA for 3 seeds; MegaBench band pass via `bench.py` (`cute` arm next to `megakernel`); mutation check.

**Phase 6: performance parity.** Same-job interleaved timing vs CUDA k16/k32/k64 at P=128 and P=968
(`bench.py` method, L2 flushed). Check regs/spills/SASS diff of hot loops. Close codegen gaps while keeping 0 ulp.
Target: within about 1% of CUDA (`specs/SPEC-phase6.md`). Then re-run 2-3 ablation switches (WPRE, D_WPT, ATT_KEYS) to confirm the CuTe
version reproduces the CUDA A/B signs.

**Phase 7 (optional, separate review): Blackwell-native.** Sketch only. Correctness becomes band-only. Candidates: TMA (`cp.async.bulk` + mbarrier) weight streaming into
smem for A/F/H instead of 16 B register loads; tcgen05 UMMA with M padded 50→64 and TMEM accumulators (frees the
255-reg budget); cluster-multicast A tiles across split-K CTAs; `cp.async.bulk.prefetch.L2`. Plan it in detail only
after Phase 6.

## 5. Execution method

Per phase: I write `cute/specs/SPEC-phaseN.md` (exact CUDA line ranges, layouts, parity stages) → Sonnet subagent
implements and passes CPU compile tests (no GPU) → I review the diff → GPU parity/timing job via slurm (run-gpu-job
skill) → `NOTES.md` entry → user review. No commits unless asked.

## 6. Verification

- CPU: `CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a .venv/bin/python -m pytest megabench/tests/test_pi05_cute_*.py`.
- GPU parity: `python -m megabench.sota.pi05.cute.parity --prefix 128 --att-keys 16 --stop <stage> --seeds 7301`
  (slurm job). Pass = all listed buffers byte-equal.
- End to end: `sbatch slurm/pi05.sbatch --tag cute-smoke` with the `cute` arm. Pass = band check passes, `actions`
  equal to the megakernel arm, timing reported next to k16/k32/k64.

## 7. Main risks

- DSL codegen (MLIR→NVVM→ptxas) differs from nvcc: register pressure at the 255 cap, unroll/ordering, and FMA
  contraction. Phase 0 math/mma probes and Phase 6 SASS review cover this.
- DSL compile time for one large kernel; mitigate with a compile cache per Constexpr set.
