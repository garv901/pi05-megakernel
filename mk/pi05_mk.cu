// Single-launch persistent megakernel for the pi0.5 10-step flow-matching
// [Ported from keisuke's vibesys run 2026-10-09 (v18, sha 04bc95a0); only change:
//  the prefix length is the PI05_PREFIX compile-time define.]
// action chunk (Gemma-300M action expert with adaRMS time conditioning).
//
// One cooperative launch; grid-wide barriers separate dependent phases.
// BF16 GEMMs use mma.sync m16n8k16 with FP32 accumulation and split-K
// partials that the following phase reduces in a fixed order.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;

// ---------------------------------------------------------------- geometry
constexpr int T = 50;          // horizon (action tokens)
constexpr int ADIM = 32;       // action dim
constexpr int W = 1024;        // expert hidden
constexpr int L = 18;          // layers
constexpr int NSTEP = 10;      // denoise steps
constexpr int QD = 2048;       // 8 heads x 256
constexpr int HD = 256;        // head dim
constexpr int NQH = 8;
constexpr int KVD = 256;
constexpr int FF = 4096;       // expert intermediate
// Prefix length (valid + padded prefix K/V rows). The graded case is 968
// (3 x 256 image tokens + 200 text tokens); the SOTA comparison builds 128.
#ifndef PI05_PREFIX
#define PI05_PREFIX 968
#endif
constexpr int P = PI05_PREFIX;  // prefix length
constexpr int NK = P + T;      // keys seen by every suffix query
constexpr int NORMS = 2 * L + 1;  // adaRMS norms per step (attn/ffn per layer + final)
constexpr int MODROW = NORMS * 3 * W;  // modulation floats per step
constexpr int QROWS = T * NQH;  // 400 query rows (token-major, head-minor)

constexpr int NTHREADS = 256;
constexpr int NWARPS = NTHREADS / 32;
constexpr int KC = 128;        // split-K chunk of every GEMM unit

// attention tiling
// Unit = (80 query rows, ATT_KEYS keys); 5 x ceil(1018 / ATT_KEYS) units run in one
// wave on the 148 blocks. Same-job sweep (dev/NOTES.md round 3): 64 keys (16 chunks,
// 80 units) 9.93 ms beat 48 keys (110 units) 10.05 and 128 keys 10.97. Dev builds override.
#ifndef PI05_ATT_KEYS
#define PI05_ATT_KEYS 64
#endif
// Attention unit layout (round 12): 0 = block-wide (80 rows, warp w computes S for key n-tile w
// over all rows, row max / sum through smem with 3 __syncthreads, P through smem); 1 = FA2-style
// (64 rows = 4 m-tiles, 7 row groups, the last holding 16 valid rows): warp w owns m-tile w & 3 and
// output dims [128 (w >> 2), +128); S and P stay in registers (quad-shuffle max / sum, C->A fragment
// repack), no __syncthreads between the load wait and the release. Both dim halves compute S.
// Same-job A/B (dev/NOTES.md round 12): 1 = 8.843 ms vs 0 = 9.114-9.118 (-3.0%), C 12.78 -> 10.48 us/layer;
// unit QK+softmax+PV 4.89 -> 3.35 us. With PI05_ATT_PREFILL 2: 8.815-8.820 ms (-3.3%).
#ifndef PI05_ATT_FA
#define PI05_ATT_FA 1
#endif
constexpr int ATT_KEYS = PI05_ATT_KEYS;     // keys per chunk (multiple of 16)
constexpr int ATT_NCHUNK = (NK + ATT_KEYS - 1) / ATT_KEYS;  // 16 at P=968, ATT_KEYS=64
#if PI05_ATT_FA
constexpr int ATT_ROWS = 64;                // query rows per unit (4 m-tiles, one per warp pair)
constexpr int ATT_NRG = (QROWS + ATT_ROWS - 1) / ATT_ROWS;  // 7 (rows >= QROWS zero-filled, never stored)
static_assert(QROWS % 16 == 0 && ATT_KEYS <= 64 && NWARPS == 2 * (ATT_ROWS / 16) && HD == 256, "FA unit");
#else
constexpr int ATT_ROWS = 80;                // query rows per unit (5 m-tiles)
constexpr int ATT_NRG = QROWS / ATT_ROWS;   // 5
static_assert(QROWS % ATT_ROWS == 0, "attention tiling");
#endif
constexpr int ATT_NT = ATT_KEYS / 8;        // QK^T key n-tiles per unit
constexpr int ATT_NTW = (ATT_NT + NWARPS - 1) / NWARPS;  // n-tiles per warp (warps past ATT_NT idle)
static_assert(ATT_KEYS % 16 == 0, "attention tiling");
static_assert(ATT_NRG * ATT_NCHUNK <= 148, "attention units must fit one wave");
// bf16 elements: conflict-free 16B frags (layout 0) / conflict-free ldmatrix rows (layout 1)
constexpr int KS_STRIDE = PI05_ATT_FA ? HD + 8 : HD + 32;
constexpr int VS_STRIDE = HD + 8;           // ldmatrix.trans rows
constexpr int PS_STRIDE = ATT_KEYS + 8;
constexpr int ATT_PS_ELEMS = PI05_ATT_FA ? 0 : ATT_ROWS * PS_STRIDE;    // P tile (bf16), layout 0 only
constexpr int ATT_WRED_BYTES = PI05_ATT_FA ? 0 : NWARPS * ATT_ROWS * 4;  // [NWARPS][ATT_ROWS] fp32, layout 0 only

constexpr size_t SMEM_C1 = (size_t)ATT_KEYS * KS_STRIDE * 2 + (size_t)ATT_ROWS * KS_STRIDE * 2 +
                           (size_t)ATT_KEYS * VS_STRIDE * 2 + (size_t)ATT_PS_ELEMS * 2 +
                           (size_t)ATT_WRED_BYTES + ATT_KEYS;
// gemm_phase_wsplit<2> footprint (A tile + partials + weights; asserted there).
constexpr size_t SMEM_C2 = (size_t)64 * (KC + 32) * 2 + (size_t)NWARPS * T * 40 * 4 + (size_t)NWARPS * 2 * 4 * 32 * 16;
constexpr size_t SMEM_T = (size_t)NSTEP * W * 4;
constexpr size_t SMEM_BYTES = SMEM_C1 > SMEM_C2 ? (SMEM_C1 > SMEM_T ? SMEM_C1 : SMEM_T)
                                                : (SMEM_C2 > SMEM_T ? SMEM_C2 : SMEM_T);

// Arrival counters replacing the reduce/combine grid barriers. Producers
// publish (stores, fence, red.release) and consumers wait with ld.acquire for
// (iteration + 1) * arrivals, so counters never need resetting inside a launch.
constexpr int CNT_STRIDE = 32;                 // one 128B line per counter
constexpr int CNT_QKV = 0;                     // 10 heads (8 q, k, v): 8 groups x 8 K chunks
constexpr int CNT_ATT = 16;                    // ATT_NRG (5 or 7) query row groups: ATT_NCHUNK key chunks
constexpr int CNT_GU = 32;                     // 128 gate/up 32-col groups: 2 groups x 8 K chunks
constexpr int NCNT = CNT_GU + FF / 32;
constexpr unsigned ARR_QKV = (HD / 32) * (W / KC);
constexpr unsigned ARR_ATT = ATT_NCHUNK;
constexpr unsigned ARR_GU = 2 * (W / KC);
constexpr int FUSE_AB = 1, FUSE_CC3 = 2, FUSE_FG = 4, FUSE_BACKOFF = 8;
// Non-final residuals run as grid-parallel (row, column group) tasks and the next
// GEMM normalizes h while staging A (no xn, no row-local block_sum).
constexpr int FUSE_DEFNORM = 16;
constexpr int NSSP = W / 64;  // sum-of-squares parts per row (64-column residual tasks)

// Grid barrier primitive (dev arms, round 7): 0 = cg grid.sync; 1 = flat epoch flags (one 128B
// line per block, st.release, warp 0 polls every flag with ld.acquire); 2 = hierarchical flags
// (block 0 polls the flags, then releases a generation word every block polls); 3 = monotonic
// atom.add.release counter with ld.acquire polling. The first barrier stays cg.grid.sync so the
// per-launch zeroing of the barrier words is ordered before their first use. Same-job A/B
// (dev/NOTES.md round 7): 3 = 9.48 ms vs cg 9.79, flat flags 11.77, hier 13.28. The counter add
// has no return value (ATOMG ... RZ), so polling starts without waiting for the add's round trip
// through the 148-deep same-address queue, which cg's returning ATOM + flip test must do.
#ifndef PI05_BARRIER
#define PI05_BARRIER 3
#endif
// Attention K/V prefill (round 8): 0 = every unit loads K, V, Q and the mask after the B->C
// barrier; 1 = each block's first unit issues its prefix K/V rows (j < P), the zero-filled tail
// rows (j >= NK) and the mask before that barrier (none depends on B), leaving Q and the suffix
// K/V rows on the post-barrier path. Same-job A/B (dev/NOTES.md round 8): 1 = 9.243 ms vs 0 = 9.37-9.39
// (bit-identical); the post-barrier load segment of a unit drops from 2.06 to 1.16 us. 2 (round 12) =
// the same copies issued before phase B's task loop instead of after it (with the 112-unit FA layout,
// blocks 78..111 run both a B task and a prefill). Same-job A/B with FA 1 (dev/NOTES.md round 12):
// 2 = 8.815-8.820 ms vs 1 = 8.843 (B 3.88 -> 3.74 us/layer; a0 3.16), bit-identical.
#ifndef PI05_ATT_PREFILL
#define PI05_ATT_PREFILL 2
#endif
// o_proj (phase D) warp split-K (round 10): 1 = gemm_phase (one warp per 32-column task);
// 2 / 4 = gemm_phase_wsplit, WPT warps of a block share each task's 128-k chunk. Same-job A/B
// (dev/NOTES.md round 10): 2 = 9.229 ms vs 1 = 9.369 (-1.5%), 4 = 9.348 (two passes at 9-10 blocks/chunk).
#ifndef PI05_D_WPT
#define PI05_D_WPT 2
#endif
// gemm_phase A tile layout (round 11): 0 = row-major [64][AS_STRIDE] (each mma A operand assembled
// from two LDS.128 by register MOVs); 1 = fragment-interleaved as in gemm_phase_wsplit (one LDS.128 =
// one A operand, 4-way bank-conflicted 32-bit staging stores); 2 = same tile, thread tid stages rows
// (r, r + 8) pairs (r = 32 (x / 2) + 16 (q / 8) + q % 8 (+ 8), q = tid / 16) and stores 128-bit fragments
// (conflict-free). Same products and summation order (bit-identical). Same-job A/B (dev/NOTES.md
// round 11): 2 = 9.151-9.154 ms vs 0 = 9.259 (-1.13..-1.18%), 1 = 9.208-9.212 (gemm_phase loop MOVs 248 -> 0).
#ifndef PI05_GEMM_FRAG
#define PI05_GEMM_FRAG 2
#endif
// gemm_phase next-tile weight load: 0 = after the partial-store loop, 1 = right after the tile's last MMA (before the stores;
// b is dead there, so the L2 latency overlaps the stores). Same math / store order / counter publish position (bit-identical).
// Default 1 since job 8779 (shim A/B, 30 L2-flushed launches): 8.192 ms vs 8.304 (-1.36%), 0 spills; variant -nolh = 0.
#ifndef PI05_LOAD_HOIST
#define PI05_LOAD_HOIST 1
#endif
// gemm_phase task order inside a K chunk: 0 = slot-major (ng = slot * NWARPS + warp: the first blocks' 8 warps take the first
// tiles, so A's 640 tiles fill 80 SMs and 68 idle); 1 = warp-major (ng = warp * nblk + slot: every block's low warps first,
// 4-5 busy warps on every SM). Same tiles, chunks and math (bit-identical); only the SM / warp that runs a tile changes.
// Default 1 since job 8794 (bench, L2 flushed): 8.056 ms vs 8.166 (-1.35%), 0 spills; U9: fewer warps per SM storing
// partials at once (A phase 3.20 -> 2.75 us); variant -nosp = 0.
#ifndef PI05_GEMM_SPREAD
#define PI05_GEMM_SPREAD 1
#endif
// Per-step tail and step-0 input with the grid-parallel residual (fuse bit 16), round 14: 0 = row-serial
// (residual_row last-layer branch: final norm, action_out, Euler, action_in + attention norm on 50 blocks);
// 1 = last-layer residual_tasks writes h / ssp, SYNC, action_out + Euler as (row, 16 dims) block tasks
// normalizing h from ssp, SYNC, action_in as (row, 64-column) warp tasks writing h / ssp, and layer 0's
// A GEMM stages the attention norm (step 0: the same action_in tasks on the noise); 2 = as 1 with
// action_out, Euler and action_in fused per row block (one barrier fewer). Same-job A/B (dev/NOTES.md round 14):
// 2 = 8.569 ms vs 0 = 8.813-8.815 (-2.8%), 1 = 8.687-8.691 (-1.4%); tail 27.9 -> 9.7 us/step, step-0 input 21.3 -> 4.7 us.
#ifndef PI05_TAIL
#define PI05_TAIL 2
#endif
// First-tile weight issue across the preceding grid barrier (round 18), bits 1 = A (QKV), 2 = F (gate/up),
// 4 = H (down), 8 = D (o_proj): the barrier before that GEMM runs inside the GEMM function as CTA sync,
// thread 0's release add, every warp's first-tile weight loads (registers; D: cp.async to smem), thread 0's
// acquire poll, CTA sync. Same counter targets, epochs and stamps as SYNC(); counter barrier only.
// Same-job A/B (dev/NOTES.md round 18, 2 jobs x 3 interleaved rounds): 15 = 8.331-8.335 ms vs 0 = 8.539-8.543
// (-2.42..-2.46%), 7 = -2.0%, 13 = -1.3%, 6 = -0.5%, 2 = +1.2%; 0 ulp vs 0 on every stagewise dump.
#ifndef PI05_WPRE
#define PI05_WPRE 15
#endif
#if PI05_BARRIER != 3
#undef PI05_WPRE
#define PI05_WPRE 0
#endif
// Debug-only per-block attention sub-phase stamps (round 8): CST_N globaltimer values per
// (step, layer, block) written after the 4096-entry tstamp area. Never set in production.
#ifdef PI05_CSTAMP
constexpr int CST_N = 12;
#endif
#if PI05_BARRIER
constexpr int BAR_STRIDE = 32;              // u32 words per flag (one 128B line)
constexpr int BAR_WORDS = 160 * BAR_STRIDE + 2 * BAR_STRIDE;  // flags, then one generation/counter line
#endif

struct Params {
  // inputs (read-only)
  const float* action_in_w; const float* action_in_b;
  const float* time_in_w; const float* time_in_b;
  const float* time_out_w; const float* time_out_b;
  const float* action_out_w; const float* action_out_b;
  const bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd;
  const float *attn_norm_w, *attn_norm_b, *ffn_norm_w, *ffn_norm_b, *final_norm_w, *final_norm_b;
  const float* noise;
  const bf16* prefix_k; const bf16* prefix_v;
  const uint8_t* prefix_mask;
  // output
  float* actions;
  // scratch (candidate-owned workspace)
  float* temb; float* tmid; float* cond; float* mod; float* xbuf; float* rope;
  bf16* h; bf16* xn; float* ssp; float* part; bf16* q; bf16* ksuf; bf16* vsuf;
  float* scores; float* stats; float* opart; bf16* attn; bf16* act;
  unsigned int* cnt;  // monotonic arrival counters (zeroed in init), CNT_STRIDE apart
  int n_steps; int n_layers;  // debug truncation (10 / 18 in production)
  int fuse;  // FUSE_* bits: which epilogues hand off through counters instead of a grid barrier
  unsigned long long* tstamp; int tmax;  // optional phase timestamps (block 0), null in production
#ifdef PI05_CSTAMP
  unsigned long long* cst;  // [step * n_layers + l][block][CST_N] when the timing buffer is large enough
#endif
#ifdef PI05_PTRACE
  unsigned long long* ptr;  // probe: [barrier][block][2] globaltimer at arrive / exit (after tstamp's 4096 slots)
#endif
#ifdef PI05_PFTRACE
  unsigned long long* pf;   // probe: [step*layers+l][stream 0..3][block][warp][2] globaltimer at prefetch issue start / end
#endif
#if PI05_BARRIER
  unsigned int* bar;  // barrier flags / counter (BAR_WORDS), zeroed before the first cg barrier
#endif
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}

// 16-byte global->shared copy through L2 (coherent for in-kernel producers);
// src_bytes = 0 zero-fills the destination.
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ unsigned ld_relaxed(const unsigned* ptr) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(unsigned* ptr) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(ptr) : "memory");
}
// Whole block waits until *c >= target: thread 0 spins relaxed, then one acquire
// fence; the CTA barrier extends it to the block. Releases come from one thread
// after a warp/CTA barrier over the published stores (CUTLASS semaphore pattern).
__device__ __forceinline__ void block_wait(const unsigned* c, unsigned target, bool backoff) {
  if (threadIdx.x == 0) {
    while (ld_relaxed(c) < target) {
      if (backoff) __nanosleep(64);
    }
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  }
  __syncthreads();
}

#if PI05_BARRIER
__device__ __forceinline__ void st_release(unsigned* ptr, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;\n" ::"l"(ptr), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* ptr) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}
// Spin until (int)(*ptr - target) >= 0 with acquire loads; debug builds trap after ~2^26 polls.
__device__ __forceinline__ void spin_acquire(const unsigned* ptr, unsigned target) {
#ifdef PI05_BAR_DEBUG
  unsigned n = 0;
  while ((int)(ld_acquire(ptr) - target) < 0) if (++n > (1u << 26)) __trap();
#else
  while ((int)(ld_acquire(ptr) - target) < 0) {}
#endif
}
// Grid barrier number `epoch` (1, 2, ... after the zeroing cg barrier). Block writes are
// published by thread 0's release after the CTA barrier; the polling threads' acquire loads
// plus the closing CTA barrier extend visibility to the block.
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned epoch) {
  __syncthreads();
#if PI05_BARRIER == 1
  if (threadIdx.x == 0) st_release(bar + blockIdx.x * BAR_STRIDE, epoch);
  if (threadIdx.x < 32) {
    for (int b = threadIdx.x; b < gridDim.x; b += 32) spin_acquire(bar + b * BAR_STRIDE, epoch);
    __syncwarp();
  }
#elif PI05_BARRIER == 2
  unsigned* gen = bar + 160 * BAR_STRIDE;
  if (threadIdx.x == 0) st_release(bar + blockIdx.x * BAR_STRIDE, epoch);
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    for (int b = threadIdx.x; b < gridDim.x; b += 32) spin_acquire(bar + b * BAR_STRIDE, epoch);
    __syncwarp();
    if (threadIdx.x == 0) st_release(gen, epoch);
  }
  if (threadIdx.x == 0) spin_acquire(gen, epoch);
#else
  unsigned* cnt = bar + 160 * BAR_STRIDE;
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory");
    spin_acquire(cnt, epoch * gridDim.x);
  }
#endif
  __syncthreads();
}
#endif

#if PI05_WPRE
// Deferred grid barrier handed to a GEMM phase: counter, poll target, block-0 stamp slot (or null).
struct WPre {
  unsigned* bar; unsigned target; unsigned long long* stamp;
#ifdef PI05_PTRACE
  unsigned long long* pt;  // this block's [arrive, exit] slot pair (probe)
#endif
};
// grid_barrier's opening half (CTA sync, thread 0's release add); the caller issues its loads, then wpre_wait.
__device__ __forceinline__ void wpre_arrive(const WPre& w) {
  __syncthreads();
#ifdef PI05_PTRACE
  if (w.pt != nullptr && threadIdx.x == 0) w.pt[0] = globaltimer();
#endif
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(w.bar) : "memory");
  }
}
__device__ __forceinline__ void wpre_wait(const WPre& w) {
  if (threadIdx.x == 0) spin_acquire(w.bar, w.target);
  __syncthreads();
  if (w.stamp != nullptr && threadIdx.x == 0) *w.stamp = globaltimer();
#ifdef PI05_PTRACE
  if (w.pt != nullptr && threadIdx.x == 0) w.pt[1] = globaltimer();
#endif
}
#define WPRE_PARAM , WPre wp = WPre{nullptr, 0u, nullptr}
#else
#define WPRE_PARAM
#endif

// Warm L2 with a weight range that a later phase streams (spread over the grid).
__device__ __forceinline__ void prefetch_l2(const void* base, size_t bytes) {
  constexpr size_t CH = 128;
  const size_t n = bytes / CH;
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const char* b = reinterpret_cast<const char*>(base);
#ifndef PI05_ABL_NOPF
  for (size_t i = gt; i < n; i += nt) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(b + i * CH));
#endif
}

// Volatile so the scheduler keeps fragment loads next to their mma (bounds registers).
__device__ __forceinline__ uint4 lds128(const void* smem_ptr) {
  uint4 r;
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Block-wide sum (all threads receive the result). red must hold NWARPS floats.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NWARPS; ++i) s += red[i];
  return s;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float gelu_tanh(float x) {
  const float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
  const float kKappa = 0.044715f;
  float x3 = x * x * x;
  float inner = kBeta * (x + kKappa * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// ------------------------------------------------------------- GEMM phase
// part[s][t][n] = sum_{k in chunk s} A[t][k] * Wt[n][k]   (t < 50)
// The virtual N dimension is split across up to three weight tensors.
// Block b owns K chunk s = b % S (KC = 128) and stages A[:, chunk] in smem
// once; its warps take 32-column units of that chunk. Within a 32-wide k slab
// each lane loads 8 contiguous bf16; A and B use the same k permutation so the
// dot product is unchanged.
constexpr int AS_STRIDE = KC + 32;  // bf16; 16B fragment loads are conflict-free

__device__ __noinline__ void gemm_phase(const bf16* __restrict__ A, int K, const bf16* w0, int n1, const bf16* w1, int n2,
                           const bf16* w2, int N, float* __restrict__ part, bf16* As,
                           unsigned* cnt = nullptr, int cdiv = 1, int cmod = 1,
                           const float* ssp = nullptr, const float* nmod = nullptr WPRE_PARAM) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int NG = N / 32, S = K / KC;
  const int s = blockIdx.x % S, slot = blockIdx.x / S;
  const int nblk = (gridDim.x - s + S - 1) / S;  // blocks sharing chunk s
  const int k0 = s * KC;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  int ng = PI05_GEMM_SPREAD ? warp * nblk + slot : slot * NWARPS + warp;
  const int ng_step = nblk * NWARPS;

  auto load_w = [&](int ngi, uint4 (&b)[4][4]) {
    const int n0 = ngi * 32;
    const bf16* Wb;
    int nl;
    if (n0 < n1) { Wb = w0; nl = n0; }
    else if (n0 < n2) { Wb = w1; nl = n0 - n1; }
    else { Wb = w2; nl = n0 - n2; }
#pragma unroll
    for (int sl = 0; sl < 4; ++sl)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        b[sl][nt] = __ldg(reinterpret_cast<const uint4*>(Wb + (size_t)(nl + nt * 8 + g) * K + k0 + sl * 32 + t4 * 8));
  };

  uint4 b[4][4];
#if PI05_WPRE
  if (wp.bar != nullptr) {
    wpre_arrive(wp);
    if (ng < NG) load_w(ng, b);
    wpre_wait(wp);
  } else
#endif
  if (ng < NG) load_w(ng, b);
  // Stage A[0:50, k0:k0+128] (rows 50..63 zero).
  // Norm mode (ssp != null): A is the residual h and the staged tile is its adaRMS
  // norm, rs[r] from the 8 per-chunk sums of squares ssp[r][0..7] (fixed order, as
  // block_sum) and scale/shift = nmod[k], nmod[W + k]; all loads issued in one round.
  {
    constexpr int PIECES = 64 * (KC / 8);  // 1024 16B pieces
    uint4 av[PIECES / NTHREADS];
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
#if PI05_GEMM_FRAG == 2
      const int q = threadIdx.x / (KC / 8), pc = threadIdx.x % (KC / 8);
      const int r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7);
#else
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
#endif
      av[x] = r < T ? __ldcg(reinterpret_cast<const uint4*>(A + (size_t)r * K + k0 + pc * 8)) : zero;
    }
    // Thread tid stages rows tid/16 + 16x (x < 4) at the same 8 columns (pc = tid % 16), so
    // its scale/shift are 4 float4 and each half-warp owns one row per x: lane j of the
    // half-warp loads ss part j and a fixed shuffle tree, broadcast from lane 0, gives rs.
    static_assert(KC / 8 == 16 && NSSP == 16, "norm staging layout");
    if (ssp != nullptr) {
      const int pc = threadIdx.x % (KC / 8), r0 = threadIdx.x / (KC / 8);
      float ssv[PIECES / NTHREADS];
#pragma unroll
      for (int x = 0; x < PIECES / NTHREADS; ++x) {
#if PI05_GEMM_FRAG == 2
        const int r = (x >> 1) * 32 + (r0 >> 3) * 16 + (x & 1) * 8 + (r0 & 7);
#else
        const int r = r0 + x * (NTHREADS / (KC / 8));
#endif
        ssv[x] = r < T ? __ldcg(ssp + r * NSSP + pc) : 0.f;
      }
      const float4* sp = reinterpret_cast<const float4*>(nmod + k0 + pc * 8);
      const float4 sc0 = __ldcg(sp), sc1 = __ldcg(sp + 1), sh0 = __ldcg(sp + W / 4), sh1 = __ldcg(sp + W / 4 + 1);
      const float sc[8] = {__fadd_rn(1.0f, sc0.x), __fadd_rn(1.0f, sc0.y), __fadd_rn(1.0f, sc0.z), __fadd_rn(1.0f, sc0.w),
                           __fadd_rn(1.0f, sc1.x), __fadd_rn(1.0f, sc1.y), __fadd_rn(1.0f, sc1.z), __fadd_rn(1.0f, sc1.w)};
      const float sh[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
      const int hbase = threadIdx.x & 16;  // lane 0 of this half-warp
#pragma unroll
      for (int x = 0; x < PIECES / NTHREADS; ++x) {
        float v = ssv[x];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        v = __shfl_sync(0xffffffffu, v, hbase);
        const float rs = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
#if PI05_GEMM_FRAG == 2
        if ((x >> 1) * 32 + (r0 >> 3) * 16 + (x & 1) * 8 + (r0 & 7) < T) {
#else
        if (r0 + x * (NTHREADS / (KC / 8)) < T) {
#endif
          uint32_t* wv = reinterpret_cast<uint32_t*>(&av[x]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            __nv_bfloat162 hv2 = *reinterpret_cast<__nv_bfloat162*>(&wv[e]);
            const float o0 = __fadd_rn(__fmul_rn(__fmul_rn(__low2float(hv2), rs), sc[2 * e]), sh[2 * e]);
            const float o1 = __fadd_rn(__fmul_rn(__fmul_rn(__high2float(hv2), rs), sc[2 * e + 1]), sh[2 * e + 1]);
            hv2 = __floats2bfloat162_rn(o0, o1);
            wv[e] = *reinterpret_cast<uint32_t*>(&hv2);
          }
        }
      }
    }
#if PI05_GEMM_FRAG == 2
    // Fragment-interleaved tile (layout of gemm_phase_wsplit): uint4 [mt][sl][half][g * 4 + t4] =
    // {row g, row g + 8} x 2 words; this thread's pieces x = 2j, 2j + 1 are rows (r, r + 8), r % 16 < 8.
    {
      const int q = threadIdx.x / (KC / 8), pc = threadIdx.x % (KC / 8);
      uint4* A4 = reinterpret_cast<uint4*>(As);
#pragma unroll
      for (int j = 0; j < PIECES / NTHREADS / 2; ++j) {
        const uint4 u = av[2 * j], v = av[2 * j + 1];
        uint4* dst = A4 + ((j * 2 + (q >> 3)) * 4 + (pc >> 2)) * 2 * 32 + (q & 7) * 4 + (pc & 3);
        dst[0] = make_uint4(u.x, v.x, u.y, v.y);
        dst[32] = make_uint4(u.z, v.z, u.w, v.w);
      }
    }
#elif PI05_GEMM_FRAG
    // Fragment-interleaved tile (layout of gemm_phase_wsplit): uint4 [mt][sl][half][g * 4 + t4].
    static_assert(64 * KC * 2 <= 64 * AS_STRIDE * 2, "interleaved A tile");
    uint32_t* Ai = reinterpret_cast<uint32_t*>(As);
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      const int base = (((r >> 4) * 4 + (pc >> 2)) * 2 * 32 + (r & 7) * 4 + (pc & 3)) * 4 + ((r >> 3) & 1);
      Ai[base] = av[x].x;
      Ai[base + 2] = av[x].y;
      Ai[base + 32 * 4] = av[x].z;
      Ai[base + 32 * 4 + 2] = av[x].w;
    }
#else
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      *reinterpret_cast<uint4*>(As + r * AS_STRIDE + pc * 8) = av[x];
    }
#endif
  }
  __syncthreads();
  for (; ng < NG; ng += ng_step) {
    float acc[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int sl = 0; sl < 4; ++sl) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
#if PI05_GEMM_FRAG
        const uint4* af = reinterpret_cast<const uint4*>(As) + (mt * 4 + sl) * 2 * 32 + lane;
        const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[sl][nt].x, b[sl][nt].y);
          mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[sl][nt].z, b[sl][nt].w);
        }
#else
        const uint4 a0 = lds128(As + (mt * 16 + g) * AS_STRIDE + sl * 32 + t4 * 8);
        const uint4 a1 = lds128(As + (mt * 16 + g + 8) * AS_STRIDE + sl * 32 + t4 * 8);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma16816(acc[mt][nt], a0.x, a1.x, a0.y, a1.y, b[sl][nt].x, b[sl][nt].y);
          mma16816(acc[mt][nt], a0.z, a1.z, a0.w, a1.w, b[sl][nt].z, b[sl][nt].w);
        }
#endif
      }
    }
    const int n0 = ng * 32;
#if PI05_LOAD_HOIST
    if (ng + ng_step < NG) load_w(ng + ng_step, b);
#endif
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int r0 = mt * 16 + g, c = n0 + nt * 8 + 2 * t4;
        if (r0 < T)
          *reinterpret_cast<float2*>(part + ((size_t)s * T + r0) * N + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T)
          *reinterpret_cast<float2*>(part + ((size_t)s * T + r0 + 8) * N + c) =
              make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
#if !PI05_LOAD_HOIST
    if (ng + ng_step < NG) load_w(ng + ng_step, b);
#endif
    if (cnt != nullptr) {  // publish this 32-column partial tile (fence overlaps the next loads)
      __syncwarp();
      if (lane == 0) red_release(cnt + ((n0 / cdiv) % cmod) * CNT_STRIDE);
    }
  }
}

// Phase D (o_proj) only fills about half the grid: 32 32-column tasks per K chunk against
// ~74 warps. gemm_phase_wsplit<WPT> splits each task's 128-k chunk over WPT warps of one
// block (warp wi of a group runs k slabs [wi*4/WPT, (wi+1)*4/WPT)). The groups then reduce
// the WPT fp32 partial tiles through smem in fixed warp order and split the 50-row store.
// Plain mode only (no norm, no counters); same part[s][t][n] layout as gemm_phase.
// The weights go to smem by cp.async (issued before the A tile, so no registers stay live
// across the staging: register-resident weights there perturb gemm_phase's codegen, round 10),
// and the A tile is stored fragment-interleaved so one LDS.128 is one mma A operand.
constexpr int WS_STRIDE = 40;  // fp32; conflict-free float2 fragment writes and float4 row reads
template <int WPT>
__device__ __noinline__ void gemm_phase_wsplit(const bf16* __restrict__ A, int K, const bf16* __restrict__ Wt, int N,
                                               float* __restrict__ part, bf16* As WPRE_PARAM) {
  constexpr int NGRP = NWARPS / WPT, NSL = 4 / WPT;
  static_assert(NWARPS % WPT == 0 && 4 % WPT == 0, "warp split");
  // smem: A tile (16 KB interleaved, in the 64 x AS_STRIDE area), partials [warp][T][WS_STRIDE], weights.
  static_assert(64 * AS_STRIDE * 2 + NWARPS * T * WS_STRIDE * 4 + NWARPS * NSL * 4 * 32 * 16 <= SMEM_BYTES, "smem");
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int gi = warp / WPT, wi = warp % WPT;
  const int NG = N / 32, S = K / KC;
  const int s = blockIdx.x % S, slot = blockIdx.x / S;
  const int nblk = (gridDim.x - s + S - 1) / S;
  const int k0 = s * KC;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  int ng = slot * NGRP + gi;
  const int ng_step = nblk * NGRP;
  float* red = reinterpret_cast<float*>(As + 64 * AS_STRIDE);
  float* mine = red + warp * T * WS_STRIDE;
  const float* grp = red + gi * WPT * T * WS_STRIDE;
  uint4* wb = reinterpret_cast<uint4*>(red + NWARPS * T * WS_STRIDE) + warp * NSL * 4 * 32;  // [j][nt][lane]

  auto load_w = [&](int ngi) {
    const int n0 = ngi * 32;
#pragma unroll
    for (int j = 0; j < NSL; ++j)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        cp_async16(wb + (j * 4 + nt) * 32 + lane, Wt + (size_t)(n0 + nt * 8 + g) * K + k0 + (wi * NSL + j) * 32 + t4 * 8, 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

#if PI05_WPRE
  if (wp.bar != nullptr) {
    wpre_arrive(wp);  // the weight buffers are free after the CTA sync (C / C3 are done in this block)
    if (ng < NG) load_w(ng);
    wpre_wait(wp);
  } else
#endif
  if (ng < NG) load_w(ng);
  {
    constexpr int PIECES = 64 * (KC / 8);
    uint4 av[PIECES / NTHREADS];
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      av[x] = r < T ? __ldcg(reinterpret_cast<const uint4*>(A + (size_t)r * K + k0 + pc * 8)) : zero;
    }
    // Fragment-interleaved tile: uint4 [mt][sl][half][g * 4 + t4] holds the k16 step's A operand
    // {row g, row g + 8} x {word 2 half, word 2 half + 1} of lane (g, t4).
    uint32_t* Ai = reinterpret_cast<uint32_t*>(As);
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      const int base = (((r >> 4) * 4 + (pc >> 2)) * 2 * 32 + (r & 7) * 4 + (pc & 3)) * 4 + ((r >> 3) & 1);
      Ai[base] = av[x].x;
      Ai[base + 2] = av[x].y;
      Ai[base + 32 * 4] = av[x].z;
      Ai[base + 32 * 4 + 2] = av[x].w;
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  for (; ng < NG; ng += ng_step) {  // uniform within a group (ng depends on gi only)
    float acc[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int j = 0; j < NSL; ++j) {
      const int sl = wi * NSL + j;
      uint4 b[4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) b[nt] = lds128(wb + (j * 4 + nt) * 32 + lane);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const uint4* af = reinterpret_cast<const uint4*>(As) + (mt * 4 + sl) * 2 * 32 + lane;
        const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[nt].x, b[nt].y);
          mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[nt].z, b[nt].w);
        }
      }
    }
    const int n0 = ng * 32;
    __syncwarp();
    if (ng + ng_step < NG) load_w(ng + ng_step);  // own buffer, already read into registers
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int r0 = mt * 16 + g, c = nt * 8 + 2 * t4;
        if (r0 < T) *reinterpret_cast<float2*>(mine + r0 * WS_STRIDE + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T)
          *reinterpret_cast<float2*>(mine + (r0 + 8) * WS_STRIDE + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
    asm volatile("bar.sync %0, %1;\n" ::"r"(1 + gi), "r"(32 * WPT) : "memory");
    // Row quad q (rows 4q .. 4q+3, lane / 8 picks the row, lane % 8 the float4 column).
    const int rr = lane >> 3, c4 = (lane & 7) * 4;
#pragma unroll
    for (int q = wi; q < (T + 3) / 4; q += WPT) {
      const int r = q * 4 + rr;
      if (r < T) {
        float4 v = *reinterpret_cast<const float4*>(grp + r * WS_STRIDE + c4);
#pragma unroll
        for (int w = 1; w < WPT; ++w) {
          const float4 u = *reinterpret_cast<const float4*>(grp + (w * T + r) * WS_STRIDE + c4);
          v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
        }
        *reinterpret_cast<float4*>(part + ((size_t)s * T + r) * N + n0 + c4) = v;
      }
    }
    if (ng + ng_step < NG) {
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      asm volatile("bar.sync %0, %1;\n" ::"r"(1 + gi), "r"(32 * WPT) : "memory");
    }
  }
}

template <int S>
__device__ __forceinline__ float reduce_part(const float* part, int N, int t, int c) {
  float x[S];
#pragma unroll
  for (int s = 0; s < S; ++s) x[s] = __ldcg(part + ((size_t)s * T + t) * N + c);
  float v = 0.f;
#pragma unroll
  for (int s = 0; s < S; ++s) v += x[s];
  return v;
}

// Four consecutive columns of reduce_part with all S loads in flight.
template <int S>
__device__ __forceinline__ float4 reduce_part4(const float* part, int N, int t, int c) {
  float4 x[S];
#pragma unroll
  for (int s = 0; s < S; ++s) x[s] = __ldcg(reinterpret_cast<const float4*>(part + ((size_t)s * T + t) * N + c));
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int s = 0; s < S; ++s) { v.x += x[s].x; v.y += x[s].y; v.z += x[s].z; v.w += x[s].w; }
  return v;
}

__device__ __forceinline__ uint2 pack_bf16x4(float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  return make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}

// ------------------------------------------------- FP32 matvec x 10 conds
// out[s][r] = act(sum_k Wrow(r)[k] * in[s][k] + bias(r)) for 10 condition rows.
// Each warp handles 4 consecutive rows; the 10 inputs are staged in smem.
template <typename RowFn, typename BiasFn, typename OutFn>
__device__ void cond_matvec(const float* in_global, float* smem, int rows, RowFn rowfn, BiasFn biasfn, OutFn outfn) {
  for (int i = threadIdx.x; i < NSTEP * W / 4; i += NTHREADS)
    reinterpret_cast<float4*>(smem)[i] = __ldcg(reinterpret_cast<const float4*>(in_global) + i);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * NWARPS + (threadIdx.x >> 5);
  const int nw = gridDim.x * NWARPS;
  for (int task = gw; task < rows / 4; task += nw) {
    const int r0 = task * 4;
    float bias[4], mine[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) bias[j] = __ldg(biasfn(r0 + j));
    float4 wv[4][8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float* wr = rowfn(r0 + j);
#pragma unroll
      for (int i = 0; i < 8; ++i) wv[j][i] = __ldg(reinterpret_cast<const float4*>(wr + i * 128 + lane * 4));
    }
    float acc[4][NSTEP];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int s = 0; s < NSTEP; ++s) acc[j][s] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
      for (int s = 0; s < NSTEP; ++s) {
        const float4 c = reinterpret_cast<const float4*>(smem + s * W + i * 128)[lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[j][s] = fmaf(wv[j][i].x, c.x, acc[j][s]);
          acc[j][s] = fmaf(wv[j][i].y, c.y, acc[j][s]);
          acc[j][s] = fmaf(wv[j][i].z, c.z, acc[j][s]);
          acc[j][s] = fmaf(wv[j][i].w, c.w, acc[j][s]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int s = 0; s < NSTEP; ++s) {
        const float v = warp_sum(acc[j][s]);
        if (lane == s) mine[j] = v;
      }
    if (lane < NSTEP) {
#pragma unroll
      for (int j = 0; j < 4; ++j) outfn(lane, r0 + j, mine[j] + bias[j]);
    }
  }
  __syncthreads();
}

// ------------------------------------------------------ row-local phases
// adaRMS normalization of one row held as 4 floats per thread (cols tid+256*i).
__device__ __forceinline__ void ada_rms_row(const float hv[4], const float* modrow, float* red, float out[4]) {
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss = fmaf(hv[i], hv[i], ss);
  ss = block_sum(ss, red);
  const float rs = rsqrtf(__fadd_rn(ss * (1.0f / W), 1e-6f));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = threadIdx.x + NTHREADS * i;
    const float scale = __ldcg(modrow + c), shift = __ldcg(modrow + W + c);
    out[i] = rbf(__fadd_rn(__fmul_rn(__fmul_rn(hv[i], rs), __fadd_rn(1.0f, scale)), shift));
  }
}

// h <- action_in(x) (BF16), xn <- attn adaRMS of layer 0 for this step.
__device__ void action_in_row(const Params& p, int t, const float* xs, const float* modrow, float* red) {
  float hv[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int n = threadIdx.x + NTHREADS * i;
    const float* wr = p.action_in_w + (size_t)n * ADIM;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < ADIM; ++k) v = fmaf(xs[k], __ldg(wr + k), v);
    hv[i] = rbf(v + __ldg(p.action_in_b + n));
    p.h[t * W + n] = __float2bfloat16_rn(hv[i]);
  }
  float o[4];
  ada_rms_row(hv, modrow, red, o);
#pragma unroll
  for (int i = 0; i < 4; ++i) p.xn[t * W + threadIdx.x + NTHREADS * i] = __float2bfloat16_rn(o[i]);
}

// Reduce o_proj/down partials for row t, apply the gated residual, then the
// next norm. which: 0 = after attention (ffn norm next), 1 = after MLP.
template <int S>
__device__ void residual_row(const Params& p, int t, const float* gate, const float* nextmod, float* red,
                             float* xs /*smem >= W+64 floats*/, int step, int last_layer) {
  float hv[4], ov[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ov[i] = reduce_part<S>(p.part, W, t, threadIdx.x + NTHREADS * i);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = threadIdx.x + NTHREADS * i;
    const float o = rbf(ov[i]);
    const float gt = rbf(__ldcg(gate + c));
    hv[i] = rbf(__fadd_rn(b2f(__ldcg(p.h + t * W + c)), rbf(__fmul_rn(o, gt))));
    p.h[t * W + c] = __float2bfloat16_rn(hv[i]);
  }
  float o[4];
  ada_rms_row(hv, nextmod, red, o);
  if (!last_layer) {
#pragma unroll
    for (int i = 0; i < 4; ++i) p.xn[t * W + threadIdx.x + NTHREADS * i] = __float2bfloat16_rn(o[i]);
    return;
  }
  // Final norm output -> action_out (FP32) -> Euler update.
  float* hn = xs;            // W floats
  float* xrow = xs + W;      // ADIM floats
#pragma unroll
  for (int i = 0; i < 4; ++i) hn[threadIdx.x + NTHREADS * i] = o[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int a = warp; a < ADIM; a += NWARPS) {
    const float* wr = p.action_out_w + (size_t)a * W;
    float v = 0.f;
    for (int k = lane; k < W; k += 32) v = fmaf(hn[k], __ldg(wr + k), v);
    v = warp_sum(v);
    if (lane == 0) {
      const float vel = v + __ldg(p.action_out_b + a);
      const float xnew = __fadd_rn(__ldcg(p.xbuf + t * ADIM + a), __fmul_rn(-0.1f, vel));
      p.xbuf[t * ADIM + a] = xnew;
      xrow[a] = xnew;
    }
  }
  __syncthreads();
  if (step + 1 < p.n_steps) {
    action_in_row(p, t, xrow, p.mod + (size_t)(step + 1) * MODROW, red);
  } else {
    if (threadIdx.x < ADIM) p.actions[t * ADIM + threadIdx.x] = xrow[threadIdx.x];
  }
}

// Grid-parallel gated residual (non-final norms): one warp task per (row t, 64-column
// group c) over all blocks, highest block indices first; each lane reduces two adjacent
// columns (float2 partials, split-K order as residual_row, so h is unchanged) and
// ssp[t][c] gets the group's sum of squares. The consumer GEMM applies the norm while
// staging A (gemm_phase norm mode), summing the NSSP parts in order.
template <int S>
__device__ __forceinline__ void residual_tasks(const Params& p, const float* gate) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int task = warp * gridDim.x + (gridDim.x - 1 - blockIdx.x); task < T * NSSP; task += gridDim.x * NWARPS) {
    const int t = task / NSSP, c = (task % NSSP) * 64 + lane * 2;
    float2 x[S];
#pragma unroll
    for (int s = 0; s < S; ++s) x[s] = __ldcg(reinterpret_cast<const float2*>(p.part + ((size_t)s * T + t) * W + c));
    const float2 gv = __ldcg(reinterpret_cast<const float2*>(gate + c));
    const unsigned hraw = __ldcg(reinterpret_cast<const unsigned*>(p.h + t * W + c));
    const __nv_bfloat162 hold = *reinterpret_cast<const __nv_bfloat162*>(&hraw);
    float2 v = make_float2(0.f, 0.f);
#pragma unroll
    for (int s = 0; s < S; ++s) { v.x += x[s].x; v.y += x[s].y; }
    const float h0 = rbf(__fadd_rn(__low2float(hold), rbf(__fmul_rn(rbf(v.x), rbf(gv.x)))));
    const float h1 = rbf(__fadd_rn(__high2float(hold), rbf(__fmul_rn(rbf(v.y), rbf(gv.y)))));
    *reinterpret_cast<__nv_bfloat162*>(p.h + t * W + c) = __floats2bfloat162_rn(h0, h1);
    const float ss = warp_sum(fmaf(h1, h1, h0 * h0));
    if (lane == 0) p.ssp[task] = ss;
  }
}

#if PI05_TAIL
// Per-step tail (round 14) on the deferred-norm machinery. h <- action_in(x) (BF16) as one warp task per
// (row t, 64-column group) over all blocks: each lane computes two adjacent columns (k-order of
// action_in_row) and ssp[t][group] gets the group's sum of squares (layout of residual_tasks), so the
// next A GEMM stages the attention norm. xsrc = noise (step 0, also copied to xbuf) or xbuf.
__device__ __forceinline__ void action_in_task(const Params& p, int t, int gi, const float* xs, bool smem_x) {
  const int lane = threadIdx.x & 31;
  const int n = gi * 64 + lane * 2;
  float4 xv[ADIM / 4], w0[ADIM / 4], w1[ADIM / 4];
#pragma unroll
  for (int i = 0; i < ADIM / 4; ++i) {
    xv[i] = smem_x ? reinterpret_cast<const float4*>(xs)[i] : __ldcg(reinterpret_cast<const float4*>(xs + t * ADIM) + i);
    w0[i] = __ldg(reinterpret_cast<const float4*>(p.action_in_w + (size_t)n * ADIM) + i);
    w1[i] = __ldg(reinterpret_cast<const float4*>(p.action_in_w + (size_t)(n + 1) * ADIM) + i);
  }
  const float2 bv = __ldg(reinterpret_cast<const float2*>(p.action_in_b + n));
  float v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int i = 0; i < ADIM / 4; ++i) {
    v0 = fmaf(xv[i].x, w0[i].x, v0); v0 = fmaf(xv[i].y, w0[i].y, v0);
    v0 = fmaf(xv[i].z, w0[i].z, v0); v0 = fmaf(xv[i].w, w0[i].w, v0);
    v1 = fmaf(xv[i].x, w1[i].x, v1); v1 = fmaf(xv[i].y, w1[i].y, v1);
    v1 = fmaf(xv[i].z, w1[i].z, v1); v1 = fmaf(xv[i].w, w1[i].w, v1);
  }
  const float h0 = rbf(v0 + bv.x), h1 = rbf(v1 + bv.y);
  *reinterpret_cast<__nv_bfloat162*>(p.h + t * W + n) = __floats2bfloat162_rn(h0, h1);
  const float ss = warp_sum(fmaf(h1, h1, h0 * h0));
  if (lane == 0) p.ssp[t * NSSP + gi] = ss;
}

__device__ __forceinline__ void action_in_tasks(const Params& p, const float* xsrc, bool copy_x) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int task = warp * gridDim.x + (gridDim.x - 1 - blockIdx.x); task < T * NSSP; task += gridDim.x * NWARPS) {
    const int t = task / NSSP, gi = task % NSSP;
    action_in_task(p, t, gi, xsrc, false);
    if (copy_x && gi == 0) p.xbuf[t * ADIM + lane] = __ldcg(xsrc + t * ADIM + lane);
  }
}

// Final adaRMS norm of row t from h and ssp (rs: the 16 parts in gemm_phase's staging order), written
// to hn (smem, W floats) as FP32 of the BF16-rounded value, as ada_rms_row's output.
__device__ __forceinline__ void final_norm_row(const Params& p, int t, const float* fmod, float* hn) {
  const int lane = threadIdx.x & 31;
  float v = __ldcg(p.ssp + t * NSSP + (lane & 15));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  v = __shfl_sync(0xffffffffu, v, 0);
  const float rs = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
  const int c = threadIdx.x * 4;
  const uint2 hraw = __ldcg(reinterpret_cast<const uint2*>(p.h + t * W + c));
  const float4 sc = __ldcg(reinterpret_cast<const float4*>(fmod + c));
  const float4 sh = __ldcg(reinterpret_cast<const float4*>(fmod + W + c));
  const __nv_bfloat162 h01 = *reinterpret_cast<const __nv_bfloat162*>(&hraw.x);
  const __nv_bfloat162 h23 = *reinterpret_cast<const __nv_bfloat162*>(&hraw.y);
  float4 o;
  o.x = rbf(__fadd_rn(__fmul_rn(__fmul_rn(__low2float(h01), rs), __fadd_rn(1.0f, sc.x)), sh.x));
  o.y = rbf(__fadd_rn(__fmul_rn(__fmul_rn(__high2float(h01), rs), __fadd_rn(1.0f, sc.y)), sh.y));
  o.z = rbf(__fadd_rn(__fmul_rn(__fmul_rn(__low2float(h23), rs), __fadd_rn(1.0f, sc.z)), sh.z));
  o.w = rbf(__fadd_rn(__fmul_rn(__fmul_rn(__high2float(h23), rs), __fadd_rn(1.0f, sc.w)), sh.w));
  reinterpret_cast<float4*>(hn)[threadIdx.x] = o;
}

// action_out dims a0, a0 + 1 of row t (lane k-stride and warp_sum of residual_row) and the Euler
// update of xbuf (actions on the last step); xs (smem) receives the new x when non-null.
__device__ __forceinline__ void action_out_pair(const Params& p, int t, int a0, const float* hn, float* xs, bool last_step) {
  const int lane = threadIdx.x & 31;
  float w0[W / 32], w1[W / 32];
#pragma unroll
  for (int j = 0; j < W / 32; ++j) {
    w0[j] = __ldg(p.action_out_w + (size_t)a0 * W + lane + 32 * j);
    w1[j] = __ldg(p.action_out_w + (size_t)(a0 + 1) * W + lane + 32 * j);
  }
  __syncthreads();  // hn staged
  float v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int j = 0; j < W / 32; ++j) {
    const float hk = hn[lane + 32 * j];
    v0 = fmaf(hk, w0[j], v0);
    v1 = fmaf(hk, w1[j], v1);
  }
  v0 = warp_sum(v0);
  v1 = warp_sum(v1);
  if (lane < 2) {
    const int a = a0 + lane;
    const float vel = (lane ? v1 : v0) + __ldg(p.action_out_b + a);
    const float xnew = __fadd_rn(__ldcg(p.xbuf + t * ADIM + a), __fmul_rn(-0.1f, vel));
    p.xbuf[t * ADIM + a] = xnew;
    if (xs != nullptr) xs[a] = xnew;
    if (last_step) p.actions[t * ADIM + a] = xnew;
  }
}

#if PI05_TAIL == 1
// Block task (row t, 16 action dims): final norm of row t in smem, warp w -> dims 2w, 2w + 1.
__device__ __forceinline__ void action_out_tasks(const Params& p, const float* fmod, float* hn, bool last_step) {
  const int warp = threadIdx.x >> 5;
  for (int task = blockIdx.x; task < 2 * T; task += gridDim.x) {
    const int t = task >> 1;
    final_norm_row(p, t, fmod, hn);
    action_out_pair(p, t, (task & 1) * 16 + warp * 2, hn, nullptr, last_step);
    __syncthreads();  // hn reuse
  }
}
#else
// Fused tail, one block per row t: final norm, all 32 action_out dims (4 per warp), Euler, then
// action_in of the row (warp w -> 64-column groups 2w, 2w + 1) and its ssp; no barrier in between.
__device__ __forceinline__ void action_tail_rows(const Params& p, const float* fmod, float* hn, bool last_step) {
  const int warp = threadIdx.x >> 5;
  float* xs = hn + W;
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    final_norm_row(p, t, fmod, hn);
#pragma unroll 1
    for (int pr = 0; pr < 2; ++pr) action_out_pair(p, t, warp * 4 + pr * 2, hn, xs, last_step);
    __syncthreads();  // xs complete
    if (!last_step) {
#pragma unroll 1
      for (int gi = warp * 2; gi < warp * 2 + 2; ++gi) action_in_task(p, t, gi, xs, true);
    }
    __syncthreads();  // hn / xs reuse
  }
}
#endif
#endif

// Debug stop (parity harness): with PI05_STOP_PH defined, every block returns right after stage PH of
// (PI05_STOP_STEP, PI05_STOP_LAYER) (both ignored for stage 0, the prologue end), before that stage's
// trailing barrier. 1 = A, 2 = B, 3 = C3, 4 = D, 5 = E, 6 = F, 7 = G, 8 = H, 9 = I (a run that ends at the
// last step's last layer ends there anyway). Absent in production builds.
#ifdef PI05_STOP_PH
#ifndef PI05_STOP_STEP
#define PI05_STOP_STEP 0
#endif
#ifndef PI05_STOP_LAYER
#define PI05_STOP_LAYER 0
#endif
#define STOP_PRO() { if (PI05_STOP_PH == 0) { cp_async_wait_all(); return; } }
#define STOP_AT(ph) { if ((ph) == PI05_STOP_PH && step == PI05_STOP_STEP && l == PI05_STOP_LAYER) { cp_async_wait_all(); return; } }
#else
#define STOP_PRO()
#define STOP_AT(ph)
#endif

// ------------------------------------------------------------- the kernel
__global__ void __launch_bounds__(NTHREADS, 1) pi05_megakernel(Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ float red[NWARPS];
  __shared__ int s_valid;
  cg::grid_group grid = cg::this_grid();
  int tcount = 1;
  if (p.tstamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0) p.tstamp[0] = globaltimer();
#ifdef PI05_PTRACE
#define PT_A() do { __syncthreads(); if (p.ptr != nullptr && threadIdx.x == 0) \
    p.ptr[((size_t)tcount * gridDim.x + blockIdx.x) * 2] = globaltimer(); } while (0)
#define PT_E() do { if (p.ptr != nullptr && threadIdx.x == 0) \
    p.ptr[((size_t)tcount * gridDim.x + blockIdx.x) * 2 + 1] = globaltimer(); } while (0)
#else
#define PT_A() do {} while (0)
#define PT_E() do {} while (0)
#endif
#ifdef PI05_PFTRACE
// stream: 0 = B (Wo + prefix K/V), 1 = C3 (Wg + Wu), 2 = D (Wd + mod rows), 3 = G (next layer Wq/Wk/Wv). Lane 0 of each warp stamps.
#define PF_IDX(stream) ((((size_t)(step * p.n_layers + l) * 4 + (stream)) * gridDim.x + blockIdx.x) * NWARPS + (threadIdx.x >> 5)) * 2
#define PF_B(stream) do { if (p.pf != nullptr && (threadIdx.x & 31) == 0) p.pf[PF_IDX(stream)] = globaltimer(); } while (0)
#define PF_E(stream) do { if (p.pf != nullptr && (threadIdx.x & 31) == 0) p.pf[PF_IDX(stream) + 1] = globaltimer(); } while (0)
#else
#define PF_B(stream) do {} while (0)
#define PF_E(stream) do {} while (0)
#endif
#if PI05_BARRIER
  unsigned bar_epoch = 0;
#define SYNC0()                                                                    \
  do {                                                                             \
    PT_A();                                                                        \
    grid.sync();                                                                   \
    PT_E();                                                                        \
    if (p.tstamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && tcount < p.tmax) \
      p.tstamp[tcount] = globaltimer();                                            \
    ++tcount;                                                                      \
  } while (0)
#define SYNC()                                                                     \
  do {                                                                             \
    PT_A();                                                                        \
    grid_barrier(p.bar, ++bar_epoch);                                              \
    PT_E();                                                                        \
    if (p.tstamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && tcount < p.tmax) \
      p.tstamp[tcount] = globaltimer();                                            \
    ++tcount;                                                                      \
  } while (0)
  if (threadIdx.x == 0) {
    p.bar[blockIdx.x * BAR_STRIDE] = 0u;
    if (blockIdx.x == 0) p.bar[160 * BAR_STRIDE] = 0u;
  }
#if PI05_WPRE
  // The next SYNC() handed to a GEMM phase (same epoch / stamp-slot bookkeeping as SYNC()).
  auto wpre = [&]() {
    const WPre w{p.bar + 160 * BAR_STRIDE, (++bar_epoch) * gridDim.x,
                 (p.tstamp != nullptr && blockIdx.x == 0 && tcount < p.tmax) ? p.tstamp + tcount : nullptr
#ifdef PI05_PTRACE
                 , p.ptr != nullptr ? p.ptr + ((size_t)tcount * gridDim.x + blockIdx.x) * 2 : nullptr
#endif
                };
    ++tcount;
    return w;
  };
#endif
#else
#define SYNC()                                                                     \
  do {                                                                             \
    PT_A();                                                                        \
    grid.sync();                                                                   \
    PT_E();                                                                        \
    if (p.tstamp != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && tcount < p.tmax) \
      p.tstamp[tcount] = globaltimer();                                            \
    ++tcount;                                                                      \
  } while (0)
#define SYNC0() SYNC()
#endif
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * NTHREADS + tid;
  const int gthreads = gridDim.x * NTHREADS;
  const int lane = tid & 31, warp = tid >> 5;
#ifdef PI05_CSTAMP
#define CST(k)                                                                                     \
  do {                                                                                             \
    if (p.cst != nullptr && tid == 0)                                                              \
      p.cst[((size_t)(step * p.n_layers + l) * gridDim.x + blockIdx.x) * CST_N + (k)] = globaltimer(); \
  } while (0)
#else
#define CST(k) do {} while (0)
#endif

  // Valid prefix count (every block, no barrier needed).
  {
    int c = 0;
    for (int i = tid; i < P; i += NTHREADS) c += p.prefix_mask[i] ? 1 : 0;
    float s = block_sum((float)c, red);
    if (tid == 0) s_valid = (int)(s + 0.5f);
    __syncthreads();
  }
  const int nvalid = s_valid;

  // ---- P0: sinusoidal time embedding (FP64 -> FP32) and RoPE tables.
  for (int i = gtid; i < NSTEP * (W / 2); i += gthreads) {
    const int s = i / (W / 2), j = i % (W / 2);
    const float tf = (float)(1.0 + (double)s * (-1.0 / NSTEP));
    const int steps = W / 2;
    const double step = 1.0 / (double)(steps - 1);
    const double frac = j < steps / 2 ? step * (double)j : 1.0 - step * (double)(steps - j - 1);
    const double period = 4e-3 * pow(4.0 / 4e-3, frac);
    const double freq = 1.0 / period * 2.0 * 3.141592653589793;
    const double ph = (double)tf * freq;
    p.temb[s * W + j] = (float)sin(ph);
    p.temb[s * W + W / 2 + j] = (float)cos(ph);
  }
  for (int i = gtid; i < NCNT * CNT_STRIDE; i += gthreads) p.cnt[i] = 0u;
  for (int i = gtid; i < T * (HD / 2); i += gthreads) {
    const int t = i / (HD / 2), j = i % (HD / 2);
    const float inv = 1.0f / powf(10000.0f, (float)(2 * j) / (float)HD);
    const float ph = __fmul_rn((float)(nvalid + t), inv);
    p.rope[i * 2] = rbf(cosf(ph));
    p.rope[i * 2 + 1] = rbf(sinf(ph));
  }
  SYNC0();

  float* smf = reinterpret_cast<float*>(smem_raw);
  bf16* smb = reinterpret_cast<bf16*>(smem_raw);
  // ---- P1/P2: time MLP for all 10 times (FP32).
  cond_matvec(p.temb, smf, W, [&](int r) { return p.time_in_w + (size_t)r * W; },
              [&](int r) { return p.time_in_b + r; },
              [&](int s, int r, float v) { p.tmid[s * W + r] = silu(v); });
  SYNC();
  cond_matvec(p.tmid, smf, W, [&](int r) { return p.time_out_w + (size_t)r * W; },
              [&](int r) { return p.time_out_b + r; },
              [&](int s, int r, float v) { p.cond[s * W + r] = silu(v); });
  SYNC();
  // ---- P3: every adaRMS modulation for every step: mod[s][norm][3W].
  cond_matvec(
      p.cond, smf, MODROW,
      [&](int r) {
        const int n = r / (3 * W), j = r % (3 * W);
        const float* base = n == 2 * L ? p.final_norm_w : ((n & 1) ? p.ffn_norm_w : p.attn_norm_w) + (size_t)(n >> 1) * 3 * W * W;
        return base + (size_t)j * W;
      },
      [&](int r) {
        const int n = r / (3 * W), j = r % (3 * W);
        const float* bias = n == 2 * L ? p.final_norm_b : ((n & 1) ? p.ffn_norm_b : p.attn_norm_b) + (size_t)(n >> 1) * 3 * W;
        return bias + j;
      },
      [&](int s, int r, float v) { p.mod[(size_t)s * MODROW + r] = v; });
  SYNC();

  // ---- step 0 input: x = noise, h = action_in(x), xn = attn norm (layer 0).
  prefetch_l2(p.wq, (size_t)QD * W * 2);
  prefetch_l2(p.wk, (size_t)KVD * W * 2);
  prefetch_l2(p.wv, (size_t)KVD * W * 2);
#if PI05_TAIL
  if (p.fuse & FUSE_DEFNORM) action_in_tasks(p, p.noise, true);
  else
#endif
  {
    float* xs = smf;
    for (int t = blockIdx.x; t < T; t += gridDim.x) {
      if (tid < ADIM) {
        const float v = __ldg(p.noise + t * ADIM + tid);
        xs[tid] = v;
        p.xbuf[t * ADIM + tid] = v;
      }
      __syncthreads();
      action_in_row(p, t, xs, p.mod, red);
      __syncthreads();
    }
  }
  STOP_PRO()
#if PI05_WPRE & 1
  bool a_pend = true;  // this barrier runs inside the next A GEMM
#else
  SYNC();
#endif

  const int g = lane >> 2, t4 = lane & 3;
  const bool defnorm = (p.fuse & FUSE_DEFNORM) != 0;
#if PI05_TAIL
  const float* a_nmod = defnorm ? p.mod : nullptr;  // A stages norm(h) with this modulation when set
#else
  const float* a_nmod = nullptr;  // A stages norm(h) with this modulation when set (after a deferred I)
#endif
  for (int step = 0; step < p.n_steps; ++step) {
    const float* modstep = p.mod + (size_t)step * MODROW;
    for (int l = 0; l < p.n_layers; ++l) {
      const unsigned it1 = (unsigned)(step * p.n_layers + l + 1);  // counter epoch of this layer
      const bool backoff = (p.fuse & FUSE_BACKOFF) != 0;
      // ---- A: QKV projections; every 32-col partial tile arrives on its head's counter.
#if PI05_WPRE & 1
      WPre wa{nullptr, 0u, nullptr};
      if (a_pend) { wa = wpre(); a_pend = false; }
#endif
      gemm_phase(a_nmod != nullptr ? p.h : p.xn, W, p.wq + (size_t)l * QD * W, QD, p.wk + (size_t)l * KVD * W, QD + KVD,
                 p.wv + (size_t)l * KVD * W, QD + 2 * KVD, p.part, smb,
                 (p.fuse & FUSE_AB) ? p.cnt + CNT_QKV * CNT_STRIDE : nullptr, HD, 10,
                 a_nmod != nullptr ? p.ssp : nullptr, a_nmod
#if PI05_WPRE & 1
                 , wa
#endif
                 );
      STOP_AT(1)
      if (!(p.fuse & FUSE_AB)) SYNC();
      // ---- B (no barrier): once a head's 64 tiles have arrived, reduce, BF16 round, RoPE.
      // Tasks (head, 10-row slice) go to the highest block indices, which QKV leaves idle.
      PF_B(0);
      prefetch_l2(p.wo + (size_t)l * W * QD, (size_t)W * QD * 2);
      prefetch_l2(p.prefix_k + (size_t)l * P * HD, (size_t)P * HD * 2);
      prefetch_l2(p.prefix_v + (size_t)l * P * HD, (size_t)P * HD * 2);
      PF_E(0);
#if PI05_ATT_PREFILL == 2
      // The prefill below (same copies and smem ordering), issued before the B tasks instead.
      if (blockIdx.x < ATT_NRG * ATT_NCHUNK) {
        __syncthreads();
        bf16* Ks = smb;
        bf16* Vs = Ks + ATT_KEYS * KS_STRIDE + ATT_ROWS * KS_STRIDE;
        uint8_t* Mk = reinterpret_cast<uint8_t*>(Vs + ATT_KEYS * VS_STRIDE + ATT_PS_ELEMS) + ATT_WRED_BYTES;
        const int j0 = (blockIdx.x / ATT_NRG) * ATT_KEYS;
        const bf16* pk = p.prefix_k + (size_t)l * P * HD;
        const bf16* pv = p.prefix_v + (size_t)l * P * HD;
        if (tid < ATT_KEYS) {
          const int j = j0 + tid;
          Mk[tid] = j < P ? (p.prefix_mask[j] != 0) : (j < NK);
        }
        for (int i = tid; i < ATT_KEYS * 32; i += NTHREADS) {
          const int jj = i >> 5, pc = i & 31, j = j0 + jj;
          if (j >= P && j < NK) continue;
          const bf16* ksrc = j < P ? pk + (size_t)j * HD : p.ksuf;
          const bf16* vsrc = j < P ? pv + (size_t)j * HD : p.vsuf;
          const int nb = j < P ? 16 : 0;
          cp_async16(Ks + jj * KS_STRIDE + pc * 8, ksrc + pc * 8, nb);
          cp_async16(Vs + jj * VS_STRIDE + pc * 8, vsrc + pc * 8, nb);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
      }
#endif
      {
        const int NQKV = QD + 2 * KVD;
        constexpr int RS = NTHREADS / (HD / 8);  // 8 rows x 32 column quads per task
        constexpr int NTASK = (T + RS - 1) / RS;
        for (int task = gridDim.x - 1 - blockIdx.x; task < 10 * NTASK; task += gridDim.x) {
          const int hh = task / NTASK, t = (task % NTASK) * RS + tid / (HD / 8), j = (tid % (HD / 8)) * 4;
          if (p.fuse & FUSE_AB) block_wait(p.cnt + (CNT_QKV + hh) * CNT_STRIDE, it1 * ARR_QKV, backoff);
          if (t < T) {
            const int c = hh * HD + j;
            const float4 v1 = reduce_part4<W / KC>(p.part, NQKV, t, c);
            const float4 v2 = reduce_part4<W / KC>(p.part, NQKV, t, c + HD / 2);
            float x1[4] = {rbf(v1.x), rbf(v1.y), rbf(v1.z), rbf(v1.w)};
            float x2[4] = {rbf(v2.x), rbf(v2.y), rbf(v2.z), rbf(v2.w)};
            float o1[4], o2[4];
            if (hh < 9) {
              const float4* rp = reinterpret_cast<const float4*>(p.rope + (t * (HD / 2) + j) * 2);
              const float4 r01 = __ldcg(rp), r23 = __ldcg(rp + 1);
              const float cs[4] = {r01.x, r01.z, r23.x, r23.z}, sn[4] = {r01.y, r01.w, r23.y, r23.w};
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                o1[e] = rbf(rbf(x1[e] * cs[e]) + rbf(-x2[e] * sn[e]));
                o2[e] = rbf(rbf(x2[e] * cs[e]) + rbf(x1[e] * sn[e]));
              }
            } else {
#pragma unroll
              for (int e = 0; e < 4; ++e) { o1[e] = x1[e]; o2[e] = x2[e]; }
            }
            bf16* dst = hh < 8 ? p.q + t * QD + hh * HD : (hh == 8 ? p.ksuf + t * KVD : p.vsuf + t * KVD);
            *reinterpret_cast<uint2*>(dst + j) = pack_bf16x4(o1[0], o1[1], o1[2], o1[3]);
            *reinterpret_cast<uint2*>(dst + j + HD / 2) = pack_bf16x4(o2[0], o2[1], o2[2], o2[3]);
          }
        }
      }
      STOP_AT(2)
#if PI05_ATT_PREFILL == 1
      // Prefill of this block's first attention unit (u = blockIdx.x): prefix K/V rows, the
      // zero-filled rows past NK and the mask, none of which B writes. Smem ordering: Ks/Vs/Mk
      // overlap gemm_phase's A tile (As = smb), which every warp reads until its last mma of
      // phase A; B and the L2 prefetches use no dynamic smem. The __syncthreads below orders
      // those reads before the cp.async writes, and nothing touches Ks/Vs/Mk again until the
      // unit's cp_async_wait_all (whose wait_group 0 also drains this committed group) and the
      // __syncthreads that follows it. The unit then loads only Q and rows P <= j < NK.
      if (blockIdx.x < ATT_NRG * ATT_NCHUNK) {
        __syncthreads();
        bf16* Ks = smb;
        bf16* Vs = Ks + ATT_KEYS * KS_STRIDE + ATT_ROWS * KS_STRIDE;
        uint8_t* Mk = reinterpret_cast<uint8_t*>(Vs + ATT_KEYS * VS_STRIDE + ATT_PS_ELEMS) + ATT_WRED_BYTES;
        const int j0 = (blockIdx.x / ATT_NRG) * ATT_KEYS;
        const bf16* pk = p.prefix_k + (size_t)l * P * HD;
        const bf16* pv = p.prefix_v + (size_t)l * P * HD;
        if (tid < ATT_KEYS) {
          const int j = j0 + tid;
          Mk[tid] = j < P ? (p.prefix_mask[j] != 0) : (j < NK);
        }
        for (int i = tid; i < ATT_KEYS * 32; i += NTHREADS) {
          const int jj = i >> 5, pc = i & 31, j = j0 + jj;
          if (j >= P && j < NK) continue;
          const bf16* ksrc = j < P ? pk + (size_t)j * HD : p.ksuf;
          const bf16* vsrc = j < P ? pv + (size_t)j * HD : p.vsuf;
          const int nb = j < P ? 16 : 0;
          cp_async16(Ks + jj * KS_STRIDE + pc * 8, ksrc + pc * 8, nb);
          cp_async16(Vs + jj * VS_STRIDE + pc * 8, vsrc + pc * 8, nb);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
      }
#endif
      CST(9);
      SYNC();
      CST(0);
      // ---- C: split-key attention. Unit = (ATT_ROWS query rows, ATT_KEYS keys):
      // scores = bf16(q.k)/16 (masked), chunk-local max m_c, p = bf16(exp(s - m_c)),
      // l_c = sum p, o_c = p.V. The combine phase rescales by exp(m_c - M).
      {
        bf16* Ks = smb;
        bf16* Qs = Ks + ATT_KEYS * KS_STRIDE;
        bf16* Vs = Qs + ATT_ROWS * KS_STRIDE;
#if PI05_ATT_FA
        uint8_t* Mk = reinterpret_cast<uint8_t*>(Vs + ATT_KEYS * VS_STRIDE);
#else
        bf16* Ps = Vs + ATT_KEYS * VS_STRIDE;
        float* wred = reinterpret_cast<float*>(Ps + ATT_ROWS * PS_STRIDE);  // [NWARPS][ATT_ROWS]
        uint8_t* Mk = reinterpret_cast<uint8_t*>(wred + NWARPS * ATT_ROWS);
#endif
        const bf16* pk = p.prefix_k + (size_t)l * P * HD;
        const bf16* pv = p.prefix_v + (size_t)l * P * HD;
        for (int u = blockIdx.x; u < ATT_NRG * ATT_NCHUNK; u += gridDim.x) {
          const int rg = u % ATT_NRG, kc = u / ATT_NRG;
          const int r0 = rg * ATT_ROWS, j0 = kc * ATT_KEYS;
#if PI05_ATT_PREFILL
          // Prefilled unit: only rows P <= j < NK remain (empty unless the chunk holds suffix keys).
          const bool pre = u == blockIdx.x;
          const int jlo = pre ? min(max(P - j0, 0), ATT_KEYS) : 0;
          const int jhi = pre ? min(max(NK - j0, 0), ATT_KEYS) : ATT_KEYS;
          if (!pre && tid < ATT_KEYS) {
#else
          constexpr int jlo = 0, jhi = ATT_KEYS;
          if (tid < ATT_KEYS) {
#endif
            const int j = j0 + tid;
            Mk[tid] = j < P ? (p.prefix_mask[j] != 0) : (j < NK);
          }
          {
            // K, Q and V rows issued together (no register staging): 32 x 16B per row.
            for (int i = jlo * 32 + tid; i < jhi * 32; i += NTHREADS) {
              const int jj = i >> 5, pc = i & 31, j = j0 + jj;
              const bf16* ksrc = j < P ? pk + (size_t)j * HD : p.ksuf + (j < NK ? j - P : 0) * HD;
              const bf16* vsrc = j < P ? pv + (size_t)j * HD : p.vsuf + (j < NK ? j - P : 0) * HD;
              const int nb = j < NK ? 16 : 0;
              cp_async16(Ks + jj * KS_STRIDE + pc * 8, ksrc + pc * 8, nb);
              cp_async16(Vs + jj * VS_STRIDE + pc * 8, vsrc + pc * 8, nb);
            }
            for (int i = tid; i < ATT_ROWS * 32; i += NTHREADS) {
              const int rr = i >> 5, pc = i & 31;
#if PI05_ATT_FA
              const int r = r0 + rr;  // rows >= QROWS (last row group) are zero-filled
              cp_async16(Qs + rr * KS_STRIDE + pc * 8, p.q + (size_t)min(r, QROWS - 1) * HD + pc * 8, r < QROWS ? 16 : 0);
#else
              cp_async16(Qs + rr * KS_STRIDE + pc * 8, p.q + (size_t)(r0 + rr) * HD + pc * 8, 16);
#endif
            }
            cp_async_wait_all();
          }
          __syncthreads();
          CST(1);
#if PI05_ATT_FA
          // Warp (m-tile mt, dim half dh); m-tiles wholly past QROWS (last row group) skip all work.
          const int mt = warp & 3, dh = warp >> 2;
          const int ra = r0 + mt * 16 + g, rb = ra + 8;
          if (r0 + mt * 16 < QROWS) {
            // S = Q K^T for the m-tile's 16 rows and all ATT_KEYS keys. Q A fragments and K B
            // fragments by ldmatrix (natural k order within each 16-k step).
            float sc[ATT_NT][4];
#pragma unroll
            for (int nt = 0; nt < ATT_NT; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) sc[nt][e] = 0.f;
#ifndef PI05_ATT_FA_PERM
            static_assert(ATT_NT % 2 == 0, "K fragments load n-tile pairs");
            const bf16* qa = Qs + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS_STRIDE + (lane >> 4) * 8;
            const bf16* kb = Ks + ((lane >> 4) * 8 + (lane & 7)) * KS_STRIDE + ((lane >> 3) & 1) * 8;
#pragma unroll 4
            for (int kk = 0; kk < HD / 16; ++kk) {
              uint32_t a0, a1, a2, a3;
              ldmatrix_x4(a0, a1, a2, a3, qa + kk * 16);
#pragma unroll
              for (int np = 0; np < ATT_NT / 2; ++np) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, kb + np * 16 * KS_STRIDE + kk * 16);
                mma16816(sc[2 * np], a0, a1, a2, a3, b0, b1);
                mma16816(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
              }
            }
#else
            // Dev check only: layout 0's k permutation and mma order per tile (bit-identical S).
#pragma unroll 2
            for (int ks = 0; ks < HD / 32; ++ks) {
              const uint4 a0 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g) * KS_STRIDE + ks * 32 + t4 * 8);
              const uint4 a1 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g + 8) * KS_STRIDE + ks * 32 + t4 * 8);
#pragma unroll
              for (int nt = 0; nt < ATT_NT; ++nt) {
                const uint4 bk = *reinterpret_cast<const uint4*>(Ks + (nt * 8 + g) * KS_STRIDE + ks * 32 + t4 * 8);
                mma16816(sc[nt], a0.x, a1.x, a0.y, a1.y, bk.x, bk.y);
                mma16816(sc[nt], a0.z, a1.z, a0.w, a1.w, bk.z, bk.w);
              }
            }
#endif
            // Scale, mask, row max (rows g / g + 8) over the quad.
            float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nt = 0; nt < ATT_NT; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int cc = nt * 8 + 2 * t4 + (e & 1);
                const float v = Mk[cc] ? rbf(sc[nt][e]) * 0.0625f : -INFINITY;
                sc[nt][e] = v;
                rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
              }
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 1));
              rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 2));
            }
            CST(2);
            // p = bf16(exp(s - m)) packed as PV A fragments: k-step k takes n-tiles 2k (a0, a1) and
            // 2k + 1 (a2, a3); l = fp32 sum of p.
            // Branch-free: a fully masked row (m = -inf) subtracts 0 and masked keys give expf(-inf) = 0,
            // the same values as layout 0's guarded expression.
            uint32_t pa[ATT_NT][2];
            float rsum[2] = {0.f, 0.f};
            const float msafe[2] = {rmax[0] == -INFINITY ? 0.f : rmax[0], rmax[1] == -INFINITY ? 0.f : rmax[1]};
#pragma unroll
            for (int nt = 0; nt < ATT_NT; ++nt) {
              float pr[4];
#pragma unroll
              for (int e = 0; e < 4; ++e) pr[e] = rbf(expf(sc[nt][e] - msafe[e >> 1]));
#ifdef PI05_ATT_FA_EXACTL
              // Dev check only: layout 0's summation order (per n-tile quad sums, then n-tiles in order).
#pragma unroll
              for (int hf = 0; hf < 2; ++hf) {
                float v = pr[2 * hf] + pr[2 * hf + 1];
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                rsum[hf] += v;
              }
#else
              rsum[0] += pr[0]; rsum[0] += pr[1];
              rsum[1] += pr[2]; rsum[1] += pr[3];
#endif
              const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]);
              const __nv_bfloat162 hi = __floats2bfloat162_rn(pr[2], pr[3]);
              pa[nt][0] = *reinterpret_cast<const uint32_t*>(&lo);
              pa[nt][1] = *reinterpret_cast<const uint32_t*>(&hi);
            }
#ifndef PI05_ATT_FA_EXACTL
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              rsum[hf] += __shfl_xor_sync(0xffffffffu, rsum[hf], 1);
              rsum[hf] += __shfl_xor_sync(0xffffffffu, rsum[hf], 2);
            }
#endif
            CST(3);
            if (dh == 0 && t4 == 0) {
              if (ra < QROWS) *reinterpret_cast<float2*>(p.stats + ((size_t)ra * ATT_NCHUNK + kc) * 2) = make_float2(rmax[0], rsum[0]);
              if (rb < QROWS) *reinterpret_cast<float2*>(p.stats + ((size_t)rb * ATT_NCHUNK + kc) * 2) = make_float2(rmax[1], rsum[1]);
            }
            // O_c = P V for the m-tile's 16 rows and this warp's 128 output dims.
            float acc[16][4];
#pragma unroll
            for (int nt = 0; nt < 16; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
            for (int ks = 0; ks < ATT_KEYS / 16; ++ks) {
              const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
              for (int np = 0; np < 8; ++np) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4_trans(b0, b1, b2, b3, Vs + krow * VS_STRIDE + dh * 128 + np * 16 + (lane >> 4) * 8);
                mma16816(acc[2 * np], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b0, b1);
                mma16816(acc[2 * np + 1], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b2, b3);
              }
            }
            CST(4);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
              const int c = dh * 128 + nt * 8 + 2 * t4;
              if (ra < QROWS) *reinterpret_cast<float2*>(p.opart + ((size_t)kc * QROWS + ra) * HD + c) = make_float2(acc[nt][0], acc[nt][1]);
              if (rb < QROWS) *reinterpret_cast<float2*>(p.opart + ((size_t)kc * QROWS + rb) * HD + c) = make_float2(acc[nt][2], acc[nt][3]);
            }
          }
#else
          // S = Q K^T for this warp's ATT_NTW key n-tiles (8 keys each) and all 80 rows.
          float sc[5][ATT_NTW][4];
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int nt = 0; nt < ATT_NTW; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) sc[mt][nt][e] = 0.f;
          if (warp * ATT_NTW < ATT_NT) {
#pragma unroll 2
            for (int ks = 0; ks < HD / 32; ++ks) {
              uint4 bk[ATT_NTW];
#pragma unroll
              for (int nt = 0; nt < ATT_NTW; ++nt)
                bk[nt] = *reinterpret_cast<const uint4*>(Ks + ((warp * ATT_NTW + nt) * 8 + g) * KS_STRIDE + ks * 32 + t4 * 8);
#pragma unroll
              for (int mt = 0; mt < 5; ++mt) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g) * KS_STRIDE + ks * 32 + t4 * 8);
                const uint4 a1 = *reinterpret_cast<const uint4*>(Qs + (mt * 16 + g + 8) * KS_STRIDE + ks * 32 + t4 * 8);
#pragma unroll
                for (int nt = 0; nt < ATT_NTW; ++nt) {
                  mma16816(sc[mt][nt], a0.x, a1.x, a0.y, a1.y, bk[nt].x, bk[nt].y);
                  mma16816(sc[mt][nt], a0.z, a1.z, a0.w, a1.w, bk[nt].z, bk[nt].w);
                }
              }
            }
          }
          // Scale, mask, warp-local row max -> wred (key columns past ATT_KEYS are -inf).
          float rmax[5][2];
#pragma unroll
          for (int mt = 0; mt < 5; ++mt) {
            rmax[mt][0] = rmax[mt][1] = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < ATT_NTW; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int cc = (warp * ATT_NTW + nt) * 8 + 2 * t4 + (e & 1);
                const float v = (cc < ATT_KEYS && Mk[cc]) ? rbf(sc[mt][nt][e]) * 0.0625f : -INFINITY;
                sc[mt][nt][e] = v;
                rmax[mt][e >> 1] = fmaxf(rmax[mt][e >> 1], v);
              }
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              rmax[mt][hf] = fmaxf(rmax[mt][hf], __shfl_xor_sync(0xffffffffu, rmax[mt][hf], 1));
              rmax[mt][hf] = fmaxf(rmax[mt][hf], __shfl_xor_sync(0xffffffffu, rmax[mt][hf], 2));
              if (t4 == 0) wred[warp * ATT_ROWS + mt * 16 + g + hf * 8] = rmax[mt][hf];
            }
          }
          __syncthreads();
          CST(2);
          float rsum[5][2];
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const int rr = mt * 16 + g + hf * 8;
              float m = wred[rr];
#pragma unroll
              for (int w2 = 1; w2 < NWARPS; ++w2) m = fmaxf(m, wred[w2 * ATT_ROWS + rr]);
              rmax[mt][hf] = m;
              rsum[mt][hf] = 0.f;
            }
          __syncthreads();  // everyone has read wred maxima
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int nt = 0; nt < ATT_NTW; ++nt) {
              if ((warp * ATT_NTW + nt) >= ATT_NT) continue;  // warp-uniform: no keys here
              float pr[4];
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const float m = rmax[mt][e >> 1];
                pr[e] = (m == -INFINITY || sc[mt][nt][e] == -INFINITY) ? 0.f : rbf(expf(sc[mt][nt][e] - m));
                rsum[mt][e >> 1] += pr[e];
              }
              const int rr = mt * 16 + g, cc = (warp * ATT_NTW + nt) * 8 + 2 * t4;
              *reinterpret_cast<__nv_bfloat162*>(Ps + rr * PS_STRIDE + cc) = __floats2bfloat162_rn(pr[0], pr[1]);
              *reinterpret_cast<__nv_bfloat162*>(Ps + (rr + 8) * PS_STRIDE + cc) = __floats2bfloat162_rn(pr[2], pr[3]);
            }
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              float v = rsum[mt][hf];
              v += __shfl_xor_sync(0xffffffffu, v, 1);
              v += __shfl_xor_sync(0xffffffffu, v, 2);
              if (t4 == 0) wred[warp * ATT_ROWS + mt * 16 + g + hf * 8] = v;
            }
          __syncthreads();
          CST(3);
          if (tid < ATT_ROWS) {
            float v = 0.f;
#pragma unroll
            for (int w2 = 0; w2 < NWARPS; ++w2) v += wred[w2 * ATT_ROWS + tid];
            p.stats[((r0 + tid) * ATT_NCHUNK + kc) * 2 + 1] = v;
          }
          if (t4 == 0 && warp == 0) {
#pragma unroll
            for (int mt = 0; mt < 5; ++mt)
#pragma unroll
              for (int hf = 0; hf < 2; ++hf)
                p.stats[((r0 + mt * 16 + g + hf * 8) * ATT_NCHUNK + kc) * 2] = rmax[mt][hf];
          }
          // O_c = P V for this warp's 32 output dims.
          float acc[5][4][4];
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
          for (int ks = 0; ks < ATT_KEYS / 16; ++ks) {
            uint32_t bfr[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
              const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
              const int ncol = warp * 32 + np * 16 + (lane >> 4) * 8;
              ldmatrix_x4_trans(bfr[np * 2][0], bfr[np * 2][1], bfr[np * 2 + 1][0], bfr[np * 2 + 1][1],
                                Vs + krow * VS_STRIDE + ncol);
            }
#pragma unroll
            for (int mt = 0; mt < 5; ++mt) {
              const bf16* pa = Ps + (mt * 16 + g) * PS_STRIDE + ks * 16 + 2 * t4;
              const uint32_t a0 = *reinterpret_cast<const uint32_t*>(pa);
              const uint32_t a1 = *reinterpret_cast<const uint32_t*>(pa + 8 * PS_STRIDE);
              const uint32_t a2 = *reinterpret_cast<const uint32_t*>(pa + 8);
              const uint32_t a3 = *reinterpret_cast<const uint32_t*>(pa + 8 * PS_STRIDE + 8);
#pragma unroll
              for (int nt = 0; nt < 4; ++nt) mma16816(acc[mt][nt], a0, a1, a2, a3, bfr[nt][0], bfr[nt][1]);
            }
          }
          CST(4);
#pragma unroll
          for (int mt = 0; mt < 5; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
              const int rr = r0 + mt * 16 + g, c = warp * 32 + nt * 8 + 2 * t4;
              float* o = p.opart + ((size_t)kc * QROWS + rr) * HD + c;
              *reinterpret_cast<float2*>(o) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
              *reinterpret_cast<float2*>(o + 8 * HD) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
            }
#endif
          __syncthreads();  // publish stats/opart of this (row group, key chunk) unit
          if (tid == 0 && (p.fuse & FUSE_CC3)) red_release(p.cnt + (CNT_ATT + rg) * CNT_STRIDE);
          CST(5);
        }
      }
      if (!(p.fuse & FUSE_CC3)) SYNC();
      // ---- C3 (no barrier): combine key chunks once a row group's ATT_NCHUNK units arrived:
      // out = sum_c e^(m_c-M) o_c / sum_c e^(m_c-M) l_c. Tasks of 4 rows go to the
      // highest block indices, which attention leaves idle.
      PF_B(1);
      prefetch_l2(p.wg + (size_t)l * FF * W, (size_t)FF * W * 2);
      prefetch_l2(p.wu + (size_t)l * FF * W, (size_t)FF * W * 2);
      PF_E(1);
      constexpr int CR = NTHREADS * 4 / HD;  // 4 rows per task, one column quad per thread
      for (int task = gridDim.x - 1 - blockIdx.x; task < QROWS / CR; task += gridDim.x) {
        if (p.fuse & FUSE_CC3) block_wait(p.cnt + (CNT_ATT + task * CR / ATT_ROWS) * CNT_STRIDE, it1 * ARR_ATT, backoff);
        CST(6);
        const int i = (task * NTHREADS + tid) * 4, r = i / HD;
        // Pass 1: chunk maxima/sums (row-contiguous, float4 pairs). Pass 2: o_c in groups of
        // CG loads in flight; both passes accumulate in chunk order (deterministic).
        const float2* st = reinterpret_cast<const float2*>(p.stats + (size_t)r * ATT_NCHUNK * 2);
        float wgt[ATT_NCHUNK], ls[ATT_NCHUNK];
#pragma unroll
        for (int c = 0; c < ATT_NCHUNK; ++c) {
          const float2 v = __ldcg(st + c);
          wgt[c] = v.x; ls[c] = v.y;
        }
        float M = wgt[0];
#pragma unroll
        for (int c = 1; c < ATT_NCHUNK; ++c) M = fmaxf(M, wgt[c]);
        float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
        float den = 0.f;
#pragma unroll
        for (int c = 0; c < ATT_NCHUNK; ++c) wgt[c] = wgt[c] == -INFINITY ? 0.f : expf(wgt[c] - M);
        constexpr int CG = 8;
#pragma unroll
        for (int c0 = 0; c0 < ATT_NCHUNK; c0 += CG) {
          float4 o[CG];
#pragma unroll
          for (int c = 0; c < CG; ++c)
            if (c0 + c < ATT_NCHUNK) o[c] = __ldcg(reinterpret_cast<const float4*>(p.opart + (size_t)(c0 + c) * QROWS * HD + i));
#pragma unroll
          for (int c = 0; c < CG; ++c)
            if (c0 + c < ATT_NCHUNK) {
              const float w = wgt[c0 + c];
              num.x = fmaf(w, o[c].x, num.x); num.y = fmaf(w, o[c].y, num.y);
              num.z = fmaf(w, o[c].z, num.z); num.w = fmaf(w, o[c].w, num.w);
              den = fmaf(w, ls[c0 + c], den);
            }
        }
        *reinterpret_cast<uint2*>(p.attn + i) = pack_bf16x4(num.x / den, num.y / den, num.z / den, num.w / den);
        CST(7);
      }
      CST(8);
      STOP_AT(3)
#if !(PI05_WPRE & 8) || !(PI05_D_WPT > 1) || defined(PI05_D_CTL)
      SYNC();
#endif
      CST(10);
      // ---- D: o_proj. Warm wd (the shorter E no longer hides it) and the modulation rows
      // of norms 2l, 2l+1 (E gate, F staging scale/shift, I gate), long evicted by the weights.
      PF_B(2);
      prefetch_l2(p.wd + (size_t)l * W * FF, (size_t)W * FF * 2);
      prefetch_l2(modstep + (size_t)(2 * l) * 3 * W, (size_t)6 * W * 4);
      PF_E(2);
#if PI05_D_WPT > 1 && defined(PI05_D_CTL)
      // Dev control (never in production): the split path is compiled but D runs gemm_phase.
      if (p.fuse & 1024) gemm_phase_wsplit<PI05_D_WPT>(p.attn, QD, p.wo + (size_t)l * W * QD, W, p.part, smb);
      else gemm_phase(p.attn, QD, p.wo + (size_t)l * W * QD, W, nullptr, W, nullptr, W, p.part, smb);
#elif PI05_D_WPT > 1 && (PI05_WPRE & 8)
      gemm_phase_wsplit<PI05_D_WPT>(p.attn, QD, p.wo + (size_t)l * W * QD, W, p.part, smb, wpre());
#elif PI05_D_WPT > 1
      gemm_phase_wsplit<PI05_D_WPT>(p.attn, QD, p.wo + (size_t)l * W * QD, W, p.part, smb);
#else
      gemm_phase(p.attn, QD, p.wo + (size_t)l * W * QD, W, nullptr, W, nullptr, W, p.part, smb);
#endif
      STOP_AT(4)
      SYNC();
      // ---- E: gated residual, FFN adaRMS.
      if (defnorm) {
        residual_tasks<QD / KC>(p, modstep + (size_t)(2 * l) * 3 * W + 2 * W);
      } else {
        for (int t = blockIdx.x; t < T; t += gridDim.x) {
          residual_row<QD / KC>(p, t, modstep + (size_t)(2 * l) * 3 * W + 2 * W,
                       modstep + (size_t)(2 * l + 1) * 3 * W, red, smf, step, 0);
          __syncthreads();
        }
      }
      STOP_AT(5)
#if !(PI05_WPRE & 2)
      SYNC();
#endif
      // ---- F: gate/up projections.
      // Gate tile j and up tile j arrive on the same counter.
      gemm_phase(defnorm ? p.h : p.xn, W, p.wg + (size_t)l * FF * W, FF, p.wu + (size_t)l * FF * W, 2 * FF, nullptr,
                 2 * FF, p.part, smb, (p.fuse & FUSE_FG) ? p.cnt + CNT_GU * CNT_STRIDE : nullptr, 32, FF / 32,
                 defnorm ? p.ssp : nullptr, defnorm ? modstep + (size_t)(2 * l + 1) * 3 * W : nullptr
#if PI05_WPRE & 2
                 , wpre()
#endif
                 );
      STOP_AT(6)
      if (!(p.fuse & FUSE_FG)) SYNC();
      // ---- G (no barrier): gelu_tanh(gate) * up per 32-col group once its 16 tiles arrived.
      {
        const int ln = (l + 1 < p.n_layers) ? l + 1 : 0;
        PF_B(3);
        prefetch_l2(p.wq + (size_t)ln * QD * W, (size_t)QD * W * 2);
        prefetch_l2(p.wk + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
        prefetch_l2(p.wv + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
        PF_E(3);
      }
      {
        constexpr int GR = (T + 1) / 2;  // tasks: (32-col group, half of the rows), one quad per thread
        for (int task = gridDim.x - 1 - blockIdx.x; task < 2 * (FF / 32); task += gridDim.x) {
          const int jg = task >> 1, t = (task & 1) * GR + tid / 8, c = jg * 32 + (tid % 8) * 4;
          if (p.fuse & FUSE_FG) block_wait(p.cnt + (CNT_GU + jg) * CNT_STRIDE, it1 * ARR_GU, backoff);
          if (tid / 8 < GR && t < T) {
            const float4 gv = reduce_part4<W / KC>(p.part, 2 * FF, t, c);
            const float4 uv = reduce_part4<W / KC>(p.part, 2 * FF, t, FF + c);
            *reinterpret_cast<uint2*>(p.act + t * FF + c) =
                pack_bf16x4(rbf(gelu_tanh(rbf(gv.x))) * rbf(uv.x), rbf(gelu_tanh(rbf(gv.y))) * rbf(uv.y),
                            rbf(gelu_tanh(rbf(gv.z))) * rbf(uv.z), rbf(gelu_tanh(rbf(gv.w))) * rbf(uv.w));
          }
        }
      }
      STOP_AT(7)
#if !(PI05_WPRE & 4)
      SYNC();
#endif
      // ---- H: down projection (warm the next norm's modulation row for I and the next A).
      // (WPRE 4: this prefetch is issued before the barrier, which runs inside gemm_phase.)
      prefetch_l2(modstep + (size_t)(2 * (l + 1)) * 3 * W, (size_t)3 * W * 4);
      gemm_phase(p.act, FF, p.wd + (size_t)l * W * FF, W, nullptr, W, nullptr, W, p.part, smb
#if PI05_WPRE & 4
                 , nullptr, 1, 1, nullptr, nullptr, wpre()
#endif
                 );
      STOP_AT(8)
      SYNC();
      // ---- I: gated residual, next norm (or final norm, action_out, Euler update, next action_in).
      {
        const bool last = (l == L - 1);
        const float* nextmod = last ? modstep + (size_t)(2 * L) * 3 * W : modstep + (size_t)(2 * (l + 1)) * 3 * W;
#if PI05_TAIL
        if (defnorm && last) {
          prefetch_l2(p.action_out_w, (size_t)ADIM * W * 4);
          prefetch_l2(p.action_in_w, (size_t)W * ADIM * 4);
          residual_tasks<FF / KC>(p, modstep + (size_t)(2 * l + 1) * 3 * W + 2 * W);
          SYNC();
          const bool more = step + 1 < p.n_steps;
#if PI05_TAIL == 1
          action_out_tasks(p, nextmod, smf, !more);
          if (!more) continue;  // final actions written; kernel exit
          SYNC();
          action_in_tasks(p, p.xbuf, false);
#else
          action_tail_rows(p, nextmod, smf, !more);
          if (!more) continue;
#endif
          a_nmod = p.mod + (size_t)(step + 1) * MODROW;
        } else
#endif
        if (defnorm && !last) {
          residual_tasks<FF / KC>(p, modstep + (size_t)(2 * l + 1) * 3 * W + 2 * W);
          a_nmod = nextmod;
        } else {
          for (int t = blockIdx.x; t < T; t += gridDim.x) {
            residual_row<FF / KC>(p, t, modstep + (size_t)(2 * l + 1) * 3 * W + 2 * W, nextmod, red, smf, step,
                         last ? 1 : 0);
            __syncthreads();
          }
          a_nmod = nullptr;
        }
      }
      STOP_AT(9)
#if PI05_WPRE & 1
      a_pend = true;
#else
      SYNC();
#endif
    }
  }
#if PI05_WPRE & 1
  if (a_pend) SYNC();  // truncated (debug) runs end on this barrier, as without WPRE
#endif
}

// ------------------------------------------------------------- host side
static int g_grid = 0;

int64_t workspace_bytes();

static size_t align_up(size_t x) { return (x + 255) & ~(size_t)255; }

struct Layout {
  size_t temb, tmid, cond, mod, xbuf, rope, h, xn, part, q, ksuf, vsuf, scores, stats, opart, attn, act, cnt, ssp, total;
#if PI05_BARRIER
  size_t bar;
#endif
};

static Layout make_layout() {
  Layout o;
  size_t off = 0;
  auto take = [&](size_t bytes) { size_t at = off; off = align_up(off + bytes); return at; };
  o.temb = take(NSTEP * W * 4);
  o.tmid = take(NSTEP * W * 4);
  o.cond = take(NSTEP * W * 4);
  o.mod = take((size_t)NSTEP * MODROW * 4);
  o.xbuf = take(T * ADIM * 4);
  o.rope = take(T * (HD / 2) * 2 * 4);
  o.h = take(T * W * 2);
  o.xn = take(T * W * 2);
  size_t part = 0;
  part = std::max(part, (size_t)(W / KC) * T * (QD + 2 * KVD));
  part = std::max(part, (size_t)(QD / KC) * T * W);
  part = std::max(part, (size_t)(W / KC) * T * 2 * FF);
  part = std::max(part, (size_t)(FF / KC) * T * W);
  o.part = take(part * 4);
  o.q = take(T * QD * 2);
  o.ksuf = take(T * KVD * 2);
  o.vsuf = take(T * KVD * 2);
  o.scores = take((size_t)QROWS * ATT_NCHUNK * ATT_KEYS * 4);
  o.stats = take((size_t)QROWS * ATT_NCHUNK * 2 * 4);
  o.opart = take((size_t)ATT_NCHUNK * QROWS * HD * 4);
  o.attn = take(T * QD * 2);
  o.act = take(T * FF * 2);
  o.cnt = take((size_t)NCNT * CNT_STRIDE * 4);
  o.ssp = take((size_t)T * NSSP * 4);
#if PI05_BARRIER
  o.bar = take((size_t)BAR_WORDS * 4);
#endif
  o.total = off;
  return o;
}

int64_t workspace_bytes() { return (int64_t)make_layout().total; }

int64_t setup(int64_t device) {
  cudaSetDevice((int)device);
  cudaFuncSetAttribute(pi05_megakernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pi05_megakernel, NTHREADS, SMEM_BYTES);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, (int)device);
  TORCH_CHECK(per_sm >= 1, "megakernel does not fit on an SM");
  g_grid = prop.multiProcessorCount;  // one persistent block per SM
  TORCH_CHECK(g_grid >= FF / KC, "grid must cover every split-K chunk");
#if PI05_BARRIER
  TORCH_CHECK(g_grid <= 160, "barrier flag array holds 160 blocks");
#endif
  return g_grid;
}

template <typename TT>
static TT* ptr(const torch::Tensor& t) { return reinterpret_cast<TT*>(t.data_ptr()); }

void launch(std::vector<torch::Tensor> in, torch::Tensor actions, torch::Tensor ws, int64_t n_steps,
            int64_t n_layers, torch::Tensor tstamp, int64_t fuse) {
  TORCH_CHECK(in.size() == 25, "expected 25 input tensors");
  Params p;
  int i = 0;
  p.action_in_w = ptr<const float>(in[i++]); p.action_in_b = ptr<const float>(in[i++]);
  p.time_in_w = ptr<const float>(in[i++]); p.time_in_b = ptr<const float>(in[i++]);
  p.time_out_w = ptr<const float>(in[i++]); p.time_out_b = ptr<const float>(in[i++]);
  p.action_out_w = ptr<const float>(in[i++]); p.action_out_b = ptr<const float>(in[i++]);
  p.wq = ptr<const bf16>(in[i++]); p.wk = ptr<const bf16>(in[i++]); p.wv = ptr<const bf16>(in[i++]);
  p.wo = ptr<const bf16>(in[i++]); p.wg = ptr<const bf16>(in[i++]); p.wu = ptr<const bf16>(in[i++]);
  p.wd = ptr<const bf16>(in[i++]);
  p.attn_norm_w = ptr<const float>(in[i++]); p.attn_norm_b = ptr<const float>(in[i++]);
  p.ffn_norm_w = ptr<const float>(in[i++]); p.ffn_norm_b = ptr<const float>(in[i++]);
  p.final_norm_w = ptr<const float>(in[i++]); p.final_norm_b = ptr<const float>(in[i++]);
  p.noise = ptr<const float>(in[i++]);
  p.prefix_k = ptr<const bf16>(in[i++]); p.prefix_v = ptr<const bf16>(in[i++]);
  p.prefix_mask = ptr<const uint8_t>(in[i++]);
  p.actions = ptr<float>(actions);
  const Layout o = make_layout();
  unsigned char* base = ptr<unsigned char>(ws);
  p.temb = (float*)(base + o.temb); p.tmid = (float*)(base + o.tmid); p.cond = (float*)(base + o.cond);
  p.mod = (float*)(base + o.mod); p.xbuf = (float*)(base + o.xbuf); p.rope = (float*)(base + o.rope);
  p.h = (bf16*)(base + o.h); p.xn = (bf16*)(base + o.xn); p.part = (float*)(base + o.part);
  p.q = (bf16*)(base + o.q); p.ksuf = (bf16*)(base + o.ksuf); p.vsuf = (bf16*)(base + o.vsuf);
  p.scores = (float*)(base + o.scores); p.stats = (float*)(base + o.stats); p.opart = (float*)(base + o.opart);
  p.attn = (bf16*)(base + o.attn); p.act = (bf16*)(base + o.act);
  p.cnt = (unsigned int*)(base + o.cnt);
  p.ssp = (float*)(base + o.ssp);
#if PI05_BARRIER
  p.bar = (unsigned int*)(base + o.bar);
#endif
  p.n_steps = (int)n_steps; p.n_layers = (int)n_layers; p.fuse = (int)fuse;
  p.tstamp = tstamp.numel() > 0 ? reinterpret_cast<unsigned long long*>(tstamp.data_ptr()) : nullptr;
  p.tmax = (int)tstamp.numel();
#ifdef PI05_PTRACE
  p.ptr = tstamp.numel() >= 4096 + (int64_t)1700 * g_grid * 2 ? p.tstamp + 4096 : nullptr;
  if (p.ptr != nullptr) p.tmax = 4096;
#endif
#ifdef PI05_PFTRACE
  {
    const int64_t pf_off = 4096 + (int64_t)1700 * g_grid * 2;
    const int64_t pf_n = (int64_t)n_steps * n_layers * 4 * g_grid * NWARPS * 2;
    p.pf = tstamp.numel() >= pf_off + pf_n ? p.tstamp + pf_off : nullptr;
  }
#endif
#ifdef PI05_CSTAMP
  p.cst = tstamp.numel() >= 4096 + (int64_t)n_steps * n_layers * g_grid * CST_N ? p.tstamp + 4096 : nullptr;
  if (p.cst != nullptr) p.tmax = 4096;
#endif
  void* args[] = {&p};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaLaunchCooperativeKernel((void*)pi05_megakernel, dim3(g_grid), dim3(NTHREADS), args,
                                                SMEM_BYTES, stream);
  TORCH_CHECK(err == cudaSuccess, "pi05 megakernel launch failed: ", cudaGetErrorString(err));
}

std::vector<int64_t> layout_offsets() {
  const Layout o = make_layout();
  return {(int64_t)o.temb, (int64_t)o.tmid, (int64_t)o.cond, (int64_t)o.mod, (int64_t)o.xbuf, (int64_t)o.rope,
          (int64_t)o.h, (int64_t)o.xn, (int64_t)o.part, (int64_t)o.q, (int64_t)o.ksuf, (int64_t)o.vsuf,
          (int64_t)o.scores, (int64_t)o.stats, (int64_t)o.opart, (int64_t)o.attn, (int64_t)o.act, (int64_t)o.cnt, (int64_t)o.ssp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_bytes", &workspace_bytes);
  m.def("setup", &setup);
  m.def("launch", &launch);
  m.def("layout_offsets", &layout_offsets);
}
