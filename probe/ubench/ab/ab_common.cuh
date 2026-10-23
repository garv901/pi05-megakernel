// ab_common.cuh: shared validity harness for the standalone A+B (QKV projection + reduce + RoPE) phase study (plan: help-me-exploire-and-transient-bentley.md).
// Modeled on probe/ubench/fg/fg_common.cuh. One persistent cooperative cluster-2 kernel (148 CTAs x 256 thr, 1 CTA/SM, 197632 B dynamic smem as the
// FG_UMMA megakernel) runs 180 layer iterations (10 steps x 18 layers, distinct per-layer weights of real size). Per iteration, barriers as in the megakernel:
//   H  dummy: 8 MB Wd read + synthetic H partials part[32][50][1024] bf16                                -> SYNC  stamp S0
//   I  residual_tasks<32> (null, k0) or the NF form (A' = h*(1+scale) + shift rows; null_apw, k1)       -> I->A WPRE split barrier, stamp S1 (wpre_wait)
//   A  variant slot (null: barrier only; k0: gemm_phase A; k1: fused A+B on tcgen05), then
//   B  (k0 only: SYNC S2, then reduce / RoPE / store)  -> SYNC stamp S3 (the A->C barrier for the fused variants, B->C for k0)
//   C  dummy: K/V prefix TMA wait + HMMA burn (--ccyc)                                                   -> SYNC S4
//   D  dummy: 4 MB Wo read (+ prefetch of the next H's Wd)                                               -> SYNC S5
//   EFG dummy: next-layer Wq/Wk/Wv L2 prefetch + spin (--egcyc)                                          -> SYNC S6 -> next iteration's H
// then NCAL known-delay calibration rounds. Stamps are clock64 by thread 0 of every CTA at each barrier exit (smem array, flushed at the end).
// -DFG_STAMPS=0 removes them (gate 2 unstamped twin). Per launch the host emits JSON gate lines; see ab_analyze.py.
#pragma once
#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
namespace cg = cooperative_groups;
typedef __nv_bfloat16 bf16;
typedef unsigned long long ull;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

#ifndef FG_STAMPS
#define FG_STAMPS 1
#endif

// ---------------------------------------------------------------- geometry (as mk/pi05_mk.cu)
constexpr int T = 50, W = 1024, L = 18, NSTEP = 10, QD = 2048, HD = 256, KVD = 256, FF = 4096;
constexpr int NTHREADS = 256, NWARPS = 8, KC = 128, NSSP = W / 64;
constexpr int CNT_STRIDE = 32, CNT_GU = 32, NCNT = CNT_GU + FF / 32;
constexpr unsigned ARR_GU = 2 * (W / KC);
constexpr int FUSE_AB = 1, FUSE_CC3 = 2, FUSE_FG = 4, FUSE_BACKOFF = 8, FUSE_DEFNORM = 16;
constexpr int FUSE_DEFAULT = 18;  // the graded build: FUSE_FG off (real F->G barrier), DEFNORM on
// megakernel defines that the verbatim code below is written against
#define PI05_BARRIER 3
#define PI05_WPRE 15
#define PI05_PART_BF16 1
constexpr int BAR_STRIDE = 32;
constexpr int BAR_WORDS = 160 * BAR_STRIDE + 2 * BAR_STRIDE;

// ---------------------------------------------------------------- stamp slots (per CTA)
constexpr int NIT = NSTEP * L, NPH = 7, NCAL = 128;
constexpr int SL_START = 0, SL_PRO = 1, SL_IT = 2, SL_CAL = SL_IT + NIT * NPH, SL_END = SL_CAL + 2 * NCAL, FG_NSLOT = SL_END + 1;
enum { ABV_NULL = 0, ABV_APW = 1, ABV_K0 = 2, ABV_K1 = 16 };  // variant ids; k1 file variants start at ABV_K1
constexpr int NQKV = QD + 2 * KVD, P = 968, NKVU = 143, ABA0 = 108, KVU_ROWS = 96;  // QKV width, prefix length, K2 attention units, first CTA of the 40-CTA A designs
constexpr size_t AB_SMEM_DYN = 197632;  // FG_SMEM of the FG_UMMA megakernel (cluster launch)

struct Params {  // the field names the verbatim code uses
  bf16* h; float* ssp; float* part; bf16* act; bf16* xn; unsigned* cnt; int fuse; float* mod;
  bf16* q; bf16* ksuf; bf16* vsuf; const float* rope; bf16* ap;
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
// phase stamps are SM clock64 (cycles); converted with the per-launch globaltimer/clock64 slope over the ~8 ms span
#define TS_NOW() ((unsigned long long)clock64())
__device__ __forceinline__ void fg_spin(long long cyc) {
  const long long t0 = clock64();
  while (clock64() - t0 < cyc) {}
}
// C-phase filler: the first `nw` warps issue HMMAs on register data until the clock64 deadline (heat comparable to the megakernel's
// attention phase); the others idle-spin. nw = 0 is fg_spin.
__device__ __forceinline__ void fg_burn(long long cyc, int nw) {
  if ((int)(threadIdx.x >> 5) >= nw) { fg_spin(cyc); return; }
  float c[4][4] = {};
  uint32_t a0 = 0x3f803f80u + threadIdx.x, a1 = 0x3f803f81u, a2 = 0x3f803f82u, a3 = 0x3f803f83u, b0 = 0x3f803f84u, b1 = 0x3f803f85u;
  const long long t0 = clock64();
  while (clock64() - t0 < cyc) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  if (c[0][0] == 12345.f && c[1][1] == 6789.f && c[2][2] == 1.f && c[3][3] == 2.f) a0 = 0;  // keep the accumulators live
}

__device__ __forceinline__ float gelu_tanh_ref(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

// ---- VERBATIM from mk/pi05_mk.cu (helpers shared by the context phases and K0) ----
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
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
  if (w.stamp != nullptr && threadIdx.x == 0) *w.stamp = TS_NOW();
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

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

#if PI05_PART_BF16
// bf16 part[] accessors; idx counts bf16 elements, so the layout part[s][t][n] is unchanged. The fp32 build keeps its
// original float stores and loads verbatim (any source change there reschedules the register-saturated kernel).
__device__ __forceinline__ void part_st2(float* part, size_t idx, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(part) + idx) = __floats2bfloat162_rn(a, b);
}
__device__ __forceinline__ void part_st4(float* part, size_t idx, float4 v) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
  *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(part) + idx) =
      make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}
__device__ __forceinline__ float part_ld(const float* part, size_t idx) {
  return __uint_as_float((unsigned)__ldcg(reinterpret_cast<const unsigned short*>(part) + idx) << 16);
}
__device__ __forceinline__ float2 part_ld2(const float* part, size_t idx) {
  const unsigned u = __ldcg(reinterpret_cast<const unsigned*>(reinterpret_cast<const bf16*>(part) + idx));
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ float4 part_ld4(const float* part, size_t idx) {
  const uint2 u = __ldcg(reinterpret_cast<const uint2*>(reinterpret_cast<const bf16*>(part) + idx));
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                     __uint_as_float(u.y & 0xffff0000u));
}
#endif

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
#if PI05_PART_BF16
    for (int s = 0; s < S; ++s) x[s] = part_ld2(p.part, ((size_t)s * T + t) * W + c);
#else
    for (int s = 0; s < S; ++s) x[s] = __ldcg(reinterpret_cast<const float2*>(p.part + ((size_t)s * T + t) * W + c));
#endif
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

// ------------------------------------------------------------- shared tcgen05 / mbarrier / TMA helpers (from fg_k1.cu / pi05_mk.cu FG_UMMA)
__device__ unsigned g_ab_to;  // mbarrier wait timeouts (host reads after every launch)
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
  for (ull i = 0; i < (1ull << 22) && !ok; ++i)
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) atomicAdd(&g_ab_to, 1u);
}
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ ull sdesc(uint32_t addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ uint32_t idesc(int N, int M) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
__device__ __forceinline__ bool elect_one() { uint32_t e; asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(e)); return e != 0; }
__device__ __forceinline__ void st_shared_b16(uint32_t a, unsigned short v) { asm volatile("st.shared.b16 [%0], %1;" ::"r"(a), "h"(v) : "memory"); }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// ------------------------------------------------------------- A' writer (the NF form of I) and the L2 / K/V helpers
// NF I: residual_tasks + A' = bf16(h * (1 + scale_next)) into the Ap workspace (+ shift rows). Ap row r: token t (t < 31), t + 1 (t >= 31); rows 31 and 51 = shift.
// Identical to fg_E of mk/pi05_mk.cu (cnt / group_wait arguments dropped).
template <int S, bool WRITE_AP = true>
__device__ __forceinline__ void ab_E(const Params& p, const float* gate, const float* modF) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int task = warp * gridDim.x + (gridDim.x - 1 - blockIdx.x); task < T * NSSP; task += gridDim.x * NWARPS) {
    const int t = task / NSSP, c = (task % NSSP) * 64 + lane * 2;
    const float2 sc = __ldcg(reinterpret_cast<const float2*>(modF + c));  // issued with the other loads
    float2 x[S];
#pragma unroll
    for (int s = 0; s < S; ++s) x[s] = part_ld2(p.part, ((size_t)s * T + t) * W + c);
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
    *reinterpret_cast<__nv_bfloat162*>(p.ap + (t >= 31 ? t + 1 : t) * W + c) = __floats2bfloat162_rn(__fmul_rn(h0, __fadd_rn(1.0f, sc.x)), __fmul_rn(h1, __fadd_rn(1.0f, sc.y)));
  }
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i < W / 2; i += NTHREADS) {
      const float2 sh = __ldcg(reinterpret_cast<const float2*>(modF + W + 2 * i));
      *reinterpret_cast<__nv_bfloat162*>(p.ap + 31 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
      *reinterpret_cast<__nv_bfloat162*>(p.ap + 51 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
    }
  asm volatile("fence.proxy.async.global;" ::: "memory");  // generic-proxy writes -> visible to the TMA (async proxy) after the barrier
}

// prefetch_l2 over the CTAs [0, nparts) only (the CTAs that have no A work in the 40 / 80 CTA designs)
__device__ __forceinline__ void prefetch_l2_part(const void* base, size_t bytes, int nparts) {
  if ((int)blockIdx.x >= nparts) return;
  constexpr size_t CH = 128;
  const size_t n = bytes / CH;
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)nparts * NTHREADS;
  const char* b = reinterpret_cast<const char*>(base);
  for (size_t i = gt; i < n; i += nt) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(b + i * CH));
}

__shared__ __align__(8) ull ab_kvbar;
// K2 attention prefix K / V: CTAs < 143 pull a 96-row x 256 K tile and V tile (48 KB each) into smem with 8 bulk copies (as ak_pre's 8 2D TMAs)
// onto one mbarrier. All threads of the CTA must call it (it starts with a CTA barrier: the smem must be free).
__device__ __noinline__ void ab_kv_issue(const bf16* pk, const bf16* pv, int l, bf16* smb) {
  if (blockIdx.x >= NKVU) return;
  const uint32_t base = (su32(smb) + 1023) & ~1023u, bar = su32(&ab_kvbar);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_expect(bar, 2 * KVU_ROWS * HD * 2);
    const size_t off = (size_t)l * P * HD + (size_t)((blockIdx.x % 10) * KVU_ROWS) * HD;
    constexpr uint32_t PIECE = KVU_ROWS * HD * 2 / 4;
#pragma unroll
    for (int cb = 0; cb < 4; ++cb) {
      bulk_g2s(base + cb * PIECE, reinterpret_cast<const char*>(pk + off) + cb * PIECE, PIECE, bar);
      bulk_g2s(base + KVU_ROWS * HD * 2 + cb * PIECE, reinterpret_cast<const char*>(pv + off) + cb * PIECE, PIECE, bar);
    }
  }
}
__device__ __forceinline__ void ab_kv_wait() {
  if (blockIdx.x >= NKVU) return;
  mbar_wait(su32(&ab_kvbar), 0);
}

// ------------------------------------------------------------- harness device code
struct AbArgs {
  const bf16 *wq, *wk, *wv, *wo, *wd, *pk, *pv;
  float* mod; bf16* h; float* ssp; float* part; bf16 *q, *ksuf, *vsuf; const float* rope; bf16* ap;
  unsigned* bar; unsigned* sink; unsigned* kscnt; float* kspart;
  ull* stamps;   // [gridDim][FG_NSLOT]
  ull* gtv;      // [0] globaltimer at post-prologue, [1] at end, [2..2+2*NCAL) canary reads at the calibration barriers
  int variant, cold, ctx, fuse, cwarps;
  long long ccyc, egcyc, ncal;
  CUtensorMap maps[5];  // wq, wk, wv (4D, box {64, 8, 2, 4}), a32, a20 (tcgen05 variants)
};
constexpr int CTX_WO_PF = 1, CTX_WD_PF = 2, CTX_DPART = 4, CTX_KV = 8, CTX_ALL = 15;

__device__ __forceinline__ uint32_t fg_hash(uint32_t i) { uint32_t h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return h; }

// H-like phase: 8 MB Wd read (L2-warm via the previous D's prefetch) + synthetic H partials part[32][50][1024] bf16 (the real I input).
__device__ __forceinline__ void ab_phase_H(const Params& p, const bf16* wd, int it, unsigned* sink, int ctx) {
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const uint4* w4 = reinterpret_cast<const uint4*>(wd);
  uint32_t acc = 0;
#pragma unroll 4
  for (size_t i = gt; i < (size_t)W * FF * 2 / 16; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (ctx & CTX_DPART)
    for (size_t i = gt; i < (size_t)(FF / KC) * T * W / 2; i += nt) {
      const uint32_t hv = fg_hash((uint32_t)i + (uint32_t)it * 7919u);
      part_st2(p.part, 2 * i, ((int)(hv & 0xffff) - 32768) * (0.02f / 32768.f), ((int)(hv >> 16) - 32768) * (0.02f / 32768.f));
    }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
__device__ __forceinline__ void ab_phase_D(const bf16* wo, unsigned* sink) {  // 4 MB Wo read
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const uint4* w4 = reinterpret_cast<const uint4*>(wo);
  uint32_t acc = 0;
#pragma unroll 4
  for (size_t i = gt; i < (size_t)W * QD * 2 / 16; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 0x9e3779b9u) sink[1] = acc;
}

__device__ __forceinline__ void ab_I(const Params& p, const AbArgs& a, const float* gate, const float* modA) {
  if (a.variant == ABV_NULL || a.variant == ABV_K0) residual_tasks<FF / KC>(p, gate);
  else ab_E<FF / KC>(p, gate, modA);
}

template <class VT>
__global__ void __launch_bounds__(NTHREADS, 1) ab_kernel(const __grid_constant__ AbArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
#if FG_STAMPS
  __shared__ ull stm[FG_NSLOT];
  __shared__ ull gtm[2 * NCAL];
#endif
  cg::grid_group grid = cg::this_grid();
  bf16* smb = reinterpret_cast<bf16*>(smem_raw);
  const int tid = threadIdx.x;
  Params p{a.h, a.ssp, a.part, nullptr, nullptr, nullptr, 0, a.mod, a.q, a.ksuf, a.vsuf, a.rope, a.ap};
  unsigned bar_epoch = 0;
#if FG_STAMPS
#define FG_STAMP(sl) do { if (tid == 0) stm[(sl)] = clock64(); } while (0)
#else
#define FG_STAMP(sl) do {} while (0)
#endif
#define SYNC(sl) do { grid_barrier(a.bar, ++bar_epoch); FG_STAMP(sl); } while (0)
  auto wpre = [&](int sl) {  // next barrier handed to a phase (split arrive / wait); the stamp is written in wpre_wait
    return WPre{a.bar + 160 * BAR_STRIDE, (++bar_epoch) * gridDim.x,
#if FG_STAMPS
                &stm[sl]
#else
                nullptr
#endif
               };
  };
  FG_STAMP(SL_START);
  if (tid == 0) {
    a.bar[blockIdx.x * BAR_STRIDE] = 0u;
    if (blockIdx.x == 0) a.bar[160 * BAR_STRIDE] = 0u;
  }
  grid.sync();  // zeroing barrier (as the megakernel's first); the prologue is stamped separately
  FG_STAMP(SL_PRO);
#if FG_STAMPS
  if (blockIdx.x == 0 && tid == 0) a.gtv[0] = globaltimer();
#endif
  for (int it = 0; it < NIT; ++it) {
    const int l = it % L, lp = (l + L - 1) % L;
    const int sb = SL_IT + it * NPH;
    // ---- H (previous layer's down projection): Wd read + synthetic partials
    ab_phase_H(p, a.wd + (size_t)lp * W * FF, it, a.sink, a.ctx);
    SYNC(sb + 0);  // S0
    // ---- I: gated residual of layer lp (+ A' for this layer's attention norm in the NF variants), then the I->A split barrier
    ab_I(p, a, a.mod + (size_t)((2 * lp + 1) * 3 + 2) * W, a.mod + (size_t)(2 * l) * 3 * W);
    {
      const WPre w = wpre(sb + 1);  // S1
      VT::A(p, a, l, it, w, smb);
    }
    if constexpr (VT::kHasK0) {
      if (a.variant == ABV_K0) { SYNC(sb + 2); VT::B(p, a, l, smb); } else FG_STAMP(sb + 2);
    } else FG_STAMP(sb + 2);
    SYNC(sb + 3);  // S3: A->C (fused / null) or B->C (k0)
    // ---- C dummy: K/V prefix wait + HMMA burn
    if (a.ctx & CTX_KV) ab_kv_wait();
    fg_burn(a.ccyc * 85 / 100, a.cwarps);
    fg_burn(a.ccyc * 15 / 100, a.cwarps);
    SYNC(sb + 4);  // S4
    // ---- D dummy: Wo read; prefetch of the next H's Wd
    if (a.ctx & CTX_WD_PF) prefetch_l2(a.wd + (size_t)l * W * FF, (size_t)W * FF * 2);
    ab_phase_D(a.wo + (size_t)l * W * QD, a.sink);
    SYNC(sb + 5);  // S5
    // ---- E..G dummy: next-layer Wq / Wk / Wv L2 prefetch (as G) + spin
    {
      const int ln = (l + 1 < L) ? l + 1 : 0;
      prefetch_l2(a.wq + (size_t)ln * QD * W, (size_t)QD * W * 2);
      prefetch_l2(a.wk + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
      prefetch_l2(a.wv + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
    }
    fg_burn(a.egcyc, a.cwarps);
    SYNC(sb + 6);  // S6
  }
  // ---- known-delay calibration: round r = {empty barrier | spin ncal, barrier}; same stamp path as the real phases
  for (int r = 0; r < NCAL; ++r) {
    SYNC(SL_CAL + 2 * r);
#if FG_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r] = globaltimer();
#endif
    fg_spin(a.ncal);
    SYNC(SL_CAL + 2 * r + 1);
#if FG_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r + 1] = globaltimer();
#endif
  }
  FG_STAMP(SL_END);
#if FG_STAMPS
  if (blockIdx.x == 0 && tid == 0) {
    a.gtv[1] = globaltimer();
    for (int i = 0; i < 2 * NCAL; ++i) a.gtv[2 + i] = gtm[i];
  }
  __syncthreads();
  for (int i = tid; i < FG_NSLOT; i += NTHREADS) a.stamps[(size_t)blockIdx.x * FG_NSLOT + i] = stm[i];
#endif
#undef SYNC
#undef FG_STAMP
}

// null A slot (every non-fused variant): split barrier kept, no A / B work; the B-position prefetches and the K/V TMA run on the CTAs with no A work
__device__ __forceinline__ void ab_null_A(const AbArgs& a, int l, const WPre& w, bf16* smb, int nparts) {
  wpre_arrive(w);
  wpre_wait(w);
  prefetch_l2_part(a.wo + (size_t)l * W * QD, (size_t)W * QD * 2, nparts);
  prefetch_l2_part(a.pk + (size_t)l * P * HD, (size_t)P * HD * 2, nparts);
  prefetch_l2_part(a.pv + (size_t)l * P * HD, (size_t)P * HD * 2, nparts);
  if (a.ctx & CTX_KV) ab_kv_issue(a.pk, a.pv, l, smb);
}

// ------------------------------------------------------------- init / reference kernels
__global__ void ab_init_bf16(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(((int)(fg_hash((uint32_t)i + seed) & 0xffff) - 32768) * (scale / 32768.f));
}
__global__ void ab_init_f32(float* p, size_t n, uint32_t seed, float scale, float bias) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = bias + ((int)(fg_hash((uint32_t)i + seed) & 0xffff) - 32768) * (scale / 32768.f);
}
// fp32 CUDA-core reference of A+B for the last layer: AnT[k][t] = bf16(adaRMS-normed h) as the K0 staging computes it (rs from the 16 ssp parts summed
// serially), then x1 = Wrow(hh*256 + j) . A, x2 = Wrow(hh*256 + j + 128) . A, RoPE (heads 0..8, unless norope) in fp32 on the bf16-rounded cos / sin table, compared
// with the kernel's bf16 q / ksuf / vsuf. Tolerance from the kernel's own roundings (bf16 A' / partials, x rounding, RoPE intermediates); x2 safety.
__global__ void ab_ref_norm(const bf16* h, const float* ssp, const float* nmod, float* AnT) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * W; i += gridDim.x * blockDim.x) {
    const int t = i / W, k = i % W;
    float v = 0.f;
    for (int c = 0; c < NSSP; ++c) v += ssp[t * NSSP + c];
    const float rs = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
    const float o = __fadd_rn(__fmul_rn(__fmul_rn(__bfloat162float(h[i]), rs), __fadd_rn(1.0f, nmod[k])), nmod[W + k]);
    AnT[k * T + t] = __bfloat162float(__float2bfloat16_rn(o));
  }
}
__global__ void ab_ref_qkv(const bf16* wq, const bf16* wk, const bf16* wv, const float* rope, const float* AnT, const bf16* q, const bf16* ksuf, const bf16* vsuf,
                           int use_rope, unsigned* maxerr, unsigned* maxref, unsigned* nbad, unsigned* nsent, unsigned* maxratio) {
  const int hh = blockIdx.x / 128, j = blockIdx.x % 128, t = threadIdx.x;  // block = one (head, rope pair), thread = one token
  if (t >= T) return;
  const bf16* w1 = hh < 8 ? wq + (size_t)(hh * 256 + j) * W : hh == 8 ? wk + (size_t)j * W : wv + (size_t)j * W;
  const bf16* w2 = w1 + (size_t)128 * W;
  float a1 = 0.f, a2 = 0.f, s1 = 0.f, s2 = 0.f;
  for (int k = 0; k < W; ++k) {
    const float a = AnT[k * T + t];
    const float p1 = a * __bfloat162float(w1[k]), p2 = a * __bfloat162float(w2[k]);
    a1 += p1; a2 += p2; s1 += fabsf(p1); s2 += fabsf(p2);
  }
  float r1 = a1, r2 = a2;
  if (hh < 9 && use_rope) {
    const float cs = rope[(t * 128 + j) * 2], sn = rope[(t * 128 + j) * 2 + 1];
    r1 = a1 * cs - a2 * sn; r2 = a2 * cs + a1 * sn;
  }
  const bf16* dst = hh < 8 ? q + (size_t)t * QD + hh * HD : hh == 8 ? ksuf + (size_t)t * KVD : vsuf + (size_t)t * KVD;
  const bf16 g1 = dst[j], g2 = dst[j + 128];
  if (*reinterpret_cast<const unsigned short*>(&g1) == 0x7f7f) atomicAdd(nsent, 1u);
  if (*reinterpret_cast<const unsigned short*>(&g2) == 0x7f7f) atomicAdd(nsent, 1u);
  const float o1 = __bfloat162float(g1), o2 = __bfloat162float(g2);
  const float e1 = fabsf(o1 - r1), e2 = fabsf(o2 - r2);
  const float e = fmaxf(e1, e2);
  atomicMax(maxerr, isfinite(e) ? __float_as_uint(e) : 0x7f800000u);
  atomicMax(maxref, __float_as_uint(fmaxf(fabsf(r1), fabsf(r2))));
  const float dx = (1.f / 256.f) * (s1 + s2) + (1.f / 128.f) * (fabsf(a1) + fabsf(a2));
  const float tol1 = 2.f * (dx + (1.f / 256.f) * fabsf(r1)) + 1e-6f, tol2 = 2.f * (dx + (1.f / 256.f) * fabsf(r2)) + 1e-6f;
  if (!(e1 <= tol1) || !(e2 <= tol2)) atomicAdd(nbad, 1u);
  const float ratio = fminf(fmaxf(e1 / tol1, e2 / tol2), 1e30f);
  atomicMax(maxratio, isfinite(ratio) ? __float_as_uint(ratio) : 0x7f800000u);
}

// ------------------------------------------------------------- host harness
static double fg_median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double fg_pct(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }

template <class VT> static const char* ab_vname(int v) {
  if (v == ABV_NULL) return "null";
  if (v == ABV_APW) return "null_apw";
  if (v == ABV_K0) return "k0";
  const char* n = VT::vname(v);
  return n ? n : "var";
}
template <class VT> static int ab_parse(const std::string& s) {
  if (s == "null") return ABV_NULL;
  if (s == "null_apw") return ABV_APW;
  if (s == "k0") return VT::kHasK0 ? ABV_K0 : -1;
  return VT::parse(s);
}
static bool ab_produces(int v) { return v == ABV_K0 || v >= ABV_K1; }

struct FgRec {
  int variant, idx; double ev_ms = 0, nspc = 0, ab_ns = 0, stage_ns[NPH] = {0}, calib_cyc = 0, calib_gt = 0, prologue_ns = 0, closure = 0;
  double maxerr = 0, maxref = 0, maxratio = 0; unsigned nbad = 0, nsent = 0, timeouts = 0; long ord_viol = 0; double ord_min_ns = 0;
};

template <class VT>
int ab_main(int argc, char** argv) {
  int reps = 20, cold = 0, ctx = CTX_ALL, cwarps = 0; long long ncal = 8700, ccyc = 9000, egcyc = 12000; std::string out = ".", name = std::string(VT::kName) + (FG_STAMPS ? "" : "_ns"), vlist = "null,null_apw";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i]; const char* v = argv[i + 1];
    if (k == "--reps") reps = atoi(v); else if (k == "--cold") cold = atoi(v); else if (k == "--ctx") ctx = atoi(v);
    else if (k == "--cwarps") cwarps = atoi(v); else if (k == "--ncal") ncal = atoll(v); else if (k == "--ccyc") ccyc = atoll(v); else if (k == "--egcyc") egcyc = atoll(v);
    else if (k == "--out") out = v; else if (k == "--name") name = v; else if (k == "--variants") vlist = v;
    else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  std::vector<int> vs;
  for (size_t p = 0; p < vlist.size();) {
    size_t q = vlist.find(',', p); if (q == std::string::npos) q = vlist.size();
    const std::string s = vlist.substr(p, q - p);
    vs.push_back(ab_parse<VT>(s));
    if (vs.back() < 0) { fprintf(stderr, "bad variant %s\n", s.c_str()); return 2; }
    p = q + 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(ab_kernel<VT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AB_SMEM_DYN));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ab_kernel<VT>, NTHREADS, AB_SMEM_DYN));
  fprintf(stderr, "%s: %s grid=%d occ/SM=%d stamps=%d reps=%d cold=%d ctx=%d ncal=%lld ccyc=%lld egcyc=%lld cwarps=%d\n", name.c_str(), prop.name, grid, occ, FG_STAMPS, reps, cold, ctx, ncal, ccyc, egcyc, cwarps);
  if (occ < 1) { fprintf(stderr, "kernel does not fit\n"); return 1; }
  std::string jpath = out + "/" + name + ".jsonl";
  FILE* jf = fopen(jpath.c_str(), "w"); if (!jf) { perror(jpath.c_str()); return 1; }
#define JOUT(...) do { fprintf(jf, __VA_ARGS__); fprintf(stdout, __VA_ARGS__); } while (0)

  const size_t nwq = (size_t)L * QD * W, nwkv = (size_t)L * KVD * W, nwo = (size_t)L * W * QD, nwd = (size_t)L * W * FF, npk = (size_t)L * P * HD;
  bf16 *wq, *wk, *wv, *wo, *wd, *pk, *pv, *h, *q, *ksuf, *vsuf, *ap; float *mod, *ssp, *part, *AnT, *rope; unsigned *bar, *sink, *kscnt; ull *stamps, *gtv; char* flush;
  CK(cudaMalloc(&wq, nwq * 2)); CK(cudaMalloc(&wk, nwkv * 2)); CK(cudaMalloc(&wv, nwkv * 2)); CK(cudaMalloc(&wo, nwo * 2)); CK(cudaMalloc(&wd, nwd * 2));
  CK(cudaMalloc(&pk, npk * 2)); CK(cudaMalloc(&pv, npk * 2));
  CK(cudaMalloc(&h, (size_t)T * W * 2));
  CK(cudaMalloc(&q, (size_t)T * QD * 2)); CK(cudaMalloc(&ksuf, (size_t)T * KVD * 2)); CK(cudaMalloc(&vsuf, (size_t)T * KVD * 2));
  CK(cudaMalloc(&ap, (size_t)64 * W * 2)); CK(cudaMemset(ap, 0, (size_t)64 * W * 2));
  CK(cudaMalloc(&rope, (size_t)T * 128 * 2 * 4));
  CK(cudaMalloc(&mod, (size_t)2 * L * 3 * W * 4)); CK(cudaMalloc(&ssp, (size_t)T * NSSP * 4));
  CK(cudaMalloc(&part, (size_t)8 * T * 2 * FF * 2)); CK(cudaMalloc(&AnT, (size_t)W * T * 4));
  CK(cudaMalloc(&bar, BAR_WORDS * 4)); CK(cudaMalloc(&sink, 64)); CK(cudaMalloc(&kscnt, 64 * 1024));
  CK(cudaMalloc(&stamps, (size_t)grid * FG_NSLOT * 8)); CK(cudaMalloc(&gtv, (2 + 2 * NCAL) * 8));
  const size_t flush_bytes = (size_t)2 * prop.l2CacheSize; CK(cudaMalloc(&flush, flush_bytes));
  CK(cudaMemset(part, 0, (size_t)8 * T * 2 * FF * 2)); CK(cudaMemset(bar, 0, BAR_WORDS * 4)); CK(cudaMemset(sink, 0, 64));
  ab_init_bf16<<<1184, 256>>>(wq, nwq, 13, 0.05f); ab_init_bf16<<<1184, 256>>>(wk, nwkv, 14, 0.05f); ab_init_bf16<<<1184, 256>>>(wv, nwkv, 15, 0.05f);
  ab_init_bf16<<<1184, 256>>>(wo, nwo, 16, 0.05f); ab_init_bf16<<<1184, 256>>>(wd, nwd, 17, 0.05f);
  ab_init_bf16<<<1184, 256>>>(pk, npk, 18, 0.5f); ab_init_bf16<<<1184, 256>>>(pv, npk, 19, 0.5f);
  ab_init_f32<<<148, 256>>>(mod, (size_t)2 * L * 3 * W, 21, 0.3f, 0.1f);
  {  // rope table as the megakernel's prologue: rbf(cos), rbf(sin) of (nvalid + t) * inv_freq(j)
    std::vector<float> rt((size_t)T * 128 * 2);
    for (int t = 0; t < T; ++t) for (int j = 0; j < 128; ++j) {
      const float inv = 1.0f / powf(10000.0f, (float)(2 * j) / (float)HD);
      const float ph = (float)(900 + t) * inv;
      auto rb = [](float x) { return __bfloat162float(__float2bfloat16_rn(x)); };
      rt[((size_t)t * 128 + j) * 2] = rb(cosf(ph)); rt[((size_t)t * 128 + j) * 2 + 1] = rb(sinf(ph));
    }
    CK(cudaMemcpy(rope, rt.data(), rt.size() * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaDeviceSynchronize());
  unsigned *dref; CK(cudaMalloc(&dref, 32));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  AbArgs args{}; 
  args.wq = wq; args.wk = wk; args.wv = wv; args.wo = wo; args.wd = wd; args.pk = pk; args.pv = pv; args.mod = mod; args.h = h; args.ssp = ssp; args.part = part;
  args.q = q; args.ksuf = ksuf; args.vsuf = vsuf; args.rope = rope; args.ap = ap; args.bar = bar; args.sink = sink; args.kscnt = kscnt; args.stamps = stamps; args.gtv = gtv;
  args.variant = 0; args.cold = cold; args.ctx = ctx; args.fuse = 0; args.cwarps = cwarps; args.ccyc = ccyc; args.egcyc = egcyc; args.ncal = ncal;
  VT::prepare(args, grid, ap);
  std::vector<FgRec> recs;
  std::vector<ull> st((size_t)grid * FG_NSLOT), gt(2 + 2 * NCAL);
  const std::string rawdir = out + "/raw"; { std::string c = "mkdir -p " + rawdir; if (system(c.c_str())) return 1; }
  std::vector<char> dumped(ABV_K1 + 64, 0);

  auto run = [&](int variant, int idx, bool record) {
    CK(cudaMemsetAsync(flush, 0, flush_bytes));
    ab_init_bf16<<<148, 256>>>(h, (size_t)T * W, 31, 1.0f);
    CK(cudaMemsetAsync(q, 0x7f, (size_t)T * QD * 2)); CK(cudaMemsetAsync(ksuf, 0x7f, (size_t)T * KVD * 2)); CK(cudaMemsetAsync(vsuf, 0x7f, (size_t)T * KVD * 2));
    CK(cudaMemsetAsync(kscnt, 0, 64 * 1024));
    CK(cudaMemsetAsync(stamps, 0, (size_t)grid * FG_NSLOT * 8));
    args.variant = variant;
    CK(cudaEventRecord(e0));
    {
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(grid); lc.blockDim = dim3(NTHREADS); lc.dynamicSmemBytes = AB_SMEM_DYN; lc.stream = 0;
      cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
      at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
      lc.attrs = at; lc.numAttrs = 2;
      CK(cudaLaunchKernelEx(&lc, ab_kernel<VT>, args));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    if (!record) return;
    unsigned to = 0; CK(cudaMemcpyFromSymbol(&to, g_ab_to, 4)); { unsigned z = 0; CK(cudaMemcpyToSymbol(g_ab_to, &z, 4)); }
    if (to) fprintf(stderr, "WARNING: %u mbarrier wait timeouts (variant %s launch %d)\n", to, ab_vname<VT>(variant), idx);
    // fp32 reference of the last layer (iteration 179 = layer 17)
    CK(cudaMemset(dref, 0, 32));
    ab_ref_norm<<<200, 256>>>(h, ssp, mod + (size_t)(2 * (L - 1)) * 3 * W, AnT);
    ab_ref_qkv<<<10 * 128, 64>>>(wq + (size_t)(L - 1) * QD * W, wk + (size_t)(L - 1) * KVD * W, wv + (size_t)(L - 1) * KVD * W, rope, AnT, q, ksuf, vsuf,
                                  VT::norope(variant) ? 0 : 1, dref, dref + 1, dref + 2, dref + 3, dref + 4);
    unsigned hr[5]; CK(cudaMemcpy(hr, dref, 20, cudaMemcpyDeviceToHost));
    FgRec r; r.variant = variant; r.idx = idx; r.ev_ms = ms; r.timeouts = to;
    r.maxerr = *reinterpret_cast<float*>(&hr[0]); r.maxref = *reinterpret_cast<float*>(&hr[1]); r.nbad = hr[2]; r.nsent = hr[3]; r.maxratio = *reinterpret_cast<float*>(&hr[4]);
    if (idx == 0 && !dumped[variant]) {  // outputs of the first recorded launch, for the cross-binary comparison against k0 (ab_analyze.py)
      dumped[variant] = 1;
      std::vector<unsigned short> o((size_t)T * (QD + 2 * KVD));
      CK(cudaMemcpy(o.data(), q, (size_t)T * QD * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(o.data() + (size_t)T * QD, ksuf, (size_t)T * KVD * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(o.data() + (size_t)T * (QD + KVD), vsuf, (size_t)T * KVD * 2, cudaMemcpyDeviceToHost));
      char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%s.out.u16", rawdir.c_str(), VT::kBin, ab_vname<VT>(variant));
      FILE* f = fopen(fn, "wb"); fwrite(o.data(), 2, o.size(), f); fclose(f);
    }
#if FG_STAMPS
    CK(cudaMemcpy(st.data(), stamps, st.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gt.data(), gtv, gt.size() * 8, cudaMemcpyDeviceToHost));
    const ull* s0 = st.data();  // CTA 0
    r.nspc = (double)(gt[1] - gt[0]) / (double)(s0[SL_END] - s0[SL_PRO]);
    const double total_ns = (double)(s0[SL_END] - s0[SL_START]) * r.nspc;
    r.closure = (total_ns - ms * 1e6) / (ms * 1e6);
    r.prologue_ns = (double)(s0[SL_PRO] - s0[SL_START]) * r.nspc;
    for (int k = 0; k < NPH; ++k) {
      double sum = 0;
      for (int i = 0; i < NIT; ++i) { const int sl = SL_IT + i * NPH + k; sum += (double)(s0[sl] - s0[sl - 1]); }
      r.stage_ns[k] = sum / NIT * r.nspc;
    }
    r.ab_ns = r.stage_ns[1] + r.stage_ns[2] + r.stage_ns[3];  // I .. A .. B, I->A barrier exit .. B->C barrier exit
    std::vector<double> pe, pd, ge, gd;
    for (int qq = 0; qq < NCAL; ++qq) {
      const int a0 = SL_CAL + 2 * qq;
      pe.push_back((double)(s0[a0] - s0[a0 - 1])); pd.push_back((double)(s0[a0 + 1] - s0[a0]));
      if (qq > 0) ge.push_back((double)(gt[2 + 2 * qq] - gt[2 + 2 * qq - 1]));
      gd.push_back((double)(gt[2 + 2 * qq + 1] - gt[2 + 2 * qq]));
    }
    r.calib_cyc = fg_median(pd) - fg_median(pe);
    r.calib_gt = (fg_median(gd) - fg_median(ge)) / ((double)ncal * r.nspc);
    r.ord_min_ns = 1e18;
    for (int c = 0; c < grid; ++c)
      for (int i = 1; i < FG_NSLOT; ++i) {
        const long long d = (long long)(st[(size_t)c * FG_NSLOT + i] - st[(size_t)c * FG_NSLOT + i - 1]);
        if (d < 0) ++r.ord_viol;
        r.ord_min_ns = std::min(r.ord_min_ns, d * r.nspc);
      }
    char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%s_%03d.cta0.u64", rawdir.c_str(), name.c_str(), ab_vname<VT>(variant), idx);
    FILE* f = fopen(fn, "wb"); fwrite(s0, 8, FG_NSLOT, f); fclose(f);
    if (idx >= reps - 2) {
      snprintf(fn, sizeof fn, "%s/%s_%s_%03d.allcta.u64", rawdir.c_str(), name.c_str(), ab_vname<VT>(variant), idx);
      f = fopen(fn, "wb"); fwrite(st.data(), 8, st.size(), f); fclose(f);
    }
#endif
    VT::post_run(variant, idx, reps, out, name, r.nspc);
    recs.push_back(r);
  };
  for (int v : vs) for (int i = 0; i < 3; ++i) run(v, -1, false);  // warm-up
  for (int r = 0; r < reps; ++r) {
    std::vector<int> ord = vs;
    if (r & 1) std::reverse(ord.begin(), ord.end());  // ABAB: alternate the order each round
    for (int v : ord) run(v, r, true);
  }

  // ---- per-launch records and gates
  auto gate = [&](const FgRec& r, const char* g, bool pass, const char* valfmt, double a, double b = 0, double c = 0) {
    char vb[256]; snprintf(vb, sizeof vb, valfmt, a, b, c);
    JOUT("{\"launch\":%d,\"variant\":\"%s\",\"gate\":\"%s\",\"pass\":%s,\"val\":%s}\n", r.idx, ab_vname<VT>(r.variant), g, pass ? "true" : "false", vb);
  };
  double nspc_med = 0; { std::vector<double> all; for (auto& r : recs) all.push_back(r.nspc); nspc_med = fg_median(all); }
  auto var_nspc = [&](int v) { std::vector<double> x; for (auto& r : recs) if (r.variant == v) x.push_back(r.nspc); return fg_median(x); };
  int nfail = 0;
  for (auto& r : recs) {
    JOUT("{\"rec\":\"launch\",\"name\":\"%s\",\"launch\":%d,\"variant\":\"%s\",\"event_ms\":%.6f,\"nspc\":%.6f,\"prologue_ns\":%.1f,"
         "\"stage_ns\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],\"AB_ns\":%.1f,\"timeouts\":%u,\"maxratio\":%.4f,\"cold\":%d,\"ctx\":%d,\"cwarps\":%d,\"ccyc\":%lld,\"egcyc\":%lld}\n",
         name.c_str(), r.idx, ab_vname<VT>(r.variant), r.ev_ms, r.nspc, r.prologue_ns, r.stage_ns[0], r.stage_ns[1], r.stage_ns[2], r.stage_ns[3],
         r.stage_ns[4], r.stage_ns[5], r.stage_ns[6], r.ab_ns, r.timeouts, r.maxratio, cold, ctx, cwarps, ccyc, egcyc);
    // work check: k0 / k1 must match the fp32 reference (and leave no sentinel); null variants must leave the output at the sentinel.
    const bool produces = ab_produces(r.variant);
    const bool wp = produces ? (r.nbad == 0 && r.nsent == 0 && std::isfinite(r.maxerr) && r.timeouts == 0) : (r.nsent == (unsigned)(T * NQKV));
    gate(r, "work", wp, "{\"maxerr\":%.6g,\"maxref\":%.6g,\"nbad\":%.0f}", r.maxerr, r.maxref, (double)r.nbad); nfail += !wp;
#if FG_STAMPS
    const bool g1 = fabs(r.closure) < 0.01; gate(r, "closure", g1, "%.5f", r.closure); nfail += !g1;
    const bool g4 = fabs(r.calib_cyc / (double)ncal - 1.0) < 0.02;
    gate(r, "known_delay", g4, "{\"readback_over_N\":%.5f,\"gt_canary_over_N\":%.5f}", r.calib_cyc / (double)ncal, r.calib_gt); nfail += !g4;
    const double vn = var_nspc(r.variant);
    const bool g5 = fabs(r.nspc / vn - 1.0) < 0.015; gate(r, "clock_stable", g5, "{\"nspc\":%.6f,\"dev_from_variant_median\":%.5f}", r.nspc, r.nspc / vn - 1.0); nfail += !g5;
    const bool g6 = r.ord_viol == 0; gate(r, "ordering", g6, "{\"violations\":%.0f,\"min_interval_ns\":%.1f}", (double)r.ord_viol, r.ord_min_ns); nfail += !g6;
#endif
  }
  // ---- set-level gates per variant against null: clock_cross, delta_agreement (L0 events vs L1 stamps), per layer iteration
  for (int v : vs) {
    if (v == ABV_NULL) continue;
    std::vector<double> evv, evn, l1v, l1n, ev_pair, allv, alln;
    std::vector<double> st_v[NPH], st_n[NPH];
    for (auto& r : recs) {
      double tot = 0; for (int k = 0; k < NPH; ++k) tot += r.stage_ns[k];
      if (r.variant == v) { evv.push_back(r.ev_ms); l1v.push_back(r.ab_ns); allv.push_back(tot); for (int k = 0; k < NPH; ++k) st_v[k].push_back(r.stage_ns[k]); }
      if (r.variant == ABV_NULL) { evn.push_back(r.ev_ms); l1n.push_back(r.ab_ns); alln.push_back(tot); for (int k = 0; k < NPH; ++k) st_n[k].push_back(r.stage_ns[k]); }
    }
    if (evn.empty()) continue;
    const double L0 = (fg_median(evv) - fg_median(evn)) * 1e3 / NIT;
    for (size_t i = 0; i < std::min(evv.size(), evn.size()); ++i) ev_pair.push_back((evv[i] - evn[i]) * 1e3 / NIT);
#if FG_STAMPS
    const double cdev = var_nspc(v) / var_nspc(ABV_NULL) - 1.0;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"clock_cross\",\"pass\":%s,\"val\":{\"nspc\":%.6f,\"null_nspc\":%.6f,\"dev\":%.5f}}\n",
         ab_vname<VT>(v), fabs(cdev) < 0.015 ? "true" : "false", var_nspc(v), var_nspc(ABV_NULL), cdev);
    const double Lwin = (fg_median(l1v) - fg_median(l1n)) / 1e3;
    const double Lall = (fg_median(allv) - fg_median(alln)) / 1e3;
    const bool g3 = fabs(L0 - Lall) < 0.2;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"delta_agreement\",\"pass\":%s,\"val\":{\"L0_us\":%.4f,\"L1_all_us\":%.4f,\"diff_us\":%.4f,"
         "\"L1_window_us\":%.4f,\"outside_window_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f,"
         "\"stage_delta_us\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]}}\n",
         ab_vname<VT>(v), g3 ? "true" : "false", L0, Lall, L0 - Lall, Lwin, Lall - Lwin, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9),
         (fg_median(st_v[0]) - fg_median(st_n[0])) / 1e3, (fg_median(st_v[1]) - fg_median(st_n[1])) / 1e3, (fg_median(st_v[2]) - fg_median(st_n[2])) / 1e3,
         (fg_median(st_v[3]) - fg_median(st_n[3])) / 1e3, (fg_median(st_v[4]) - fg_median(st_n[4])) / 1e3, (fg_median(st_v[5]) - fg_median(st_n[5])) / 1e3,
         (fg_median(st_v[6]) - fg_median(st_n[6])) / 1e3);
    nfail += !g3;
#else
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"L0_delta\",\"pass\":true,\"val\":{\"L0_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f}}\n",
         ab_vname<VT>(v), L0, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9));
#endif
  }
  for (int v : vs) {
    std::vector<double> ev; for (auto& r : recs) if (r.variant == v) ev.push_back(r.ev_ms);
    JOUT("{\"rec\":\"summary\",\"name\":\"%s\",\"variant\":\"%s\",\"stamps\":%d,\"n\":%zu,\"event_ms_med\":%.6f,\"event_ms_p10\":%.6f,\"event_ms_p90\":%.6f,"
         "\"per_layer_us_med\":%.4f}\n", name.c_str(), ab_vname<VT>(v), FG_STAMPS, ev.size(), fg_median(ev), fg_pct(ev, 0.1), fg_pct(ev, 0.9), fg_median(ev) * 1e3 / NIT);
  }
  JOUT("{\"rec\":\"done\",\"name\":\"%s\",\"failed_gates\":%d,\"nspc_median\":%.6f}\n", name.c_str(), nfail, nspc_med);
  fclose(jf);
  return 0;
}
