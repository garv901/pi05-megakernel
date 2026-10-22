// fg_common.cuh: shared validity harness for the standalone F+G phase study (plan: yes-work-in-plan-ticklish-bubble.md).
// One persistent cooperative kernel (148 CTAs x 256 threads, 1 CTA/SM, 117248 B dynamic smem as in the megakernel) runs 180 layer
// iterations (10 steps x 18 layers, distinct per-layer weights). Per iteration, with a grid barrier after each phase:
//   C  dummy phase (spin ~10.5 us; prefetch_l2 of Wo early, Wg/Wu late unless the cold flag is set)  -> stamp S0
//   D  4 MB Wo read + synthetic D partials (part[16][50][1024] bf16)                                  -> stamp S1
//   E  real residual_tasks<16> on those partials -> h, ssp; then E->F is a WPRE split barrier        -> stamp S2 (in wpre_wait)
//   F  variant slot (null: barrier only; calib: spin N cycles; k0 / later: the real F)               -> stamp S3 (F->G barrier)
//   G  next-layer Wq/Wk/Wv prefetch (all variants) + the variant's G reduce / gelu*up                -> WPRE of H
//   H  8 MB Wd read + act chunk read; G->H is a WPRE split barrier (first Wd loads between arrive and wait) -> stamp S4
//      H->next C barrier                                                                              -> stamp S5
// then NCAL calibration rounds {barrier; spin 0 | barrier; spin N | barrier} (known-delay readback).
// Stamps: block-0-style clock64 by thread 0 of EVERY CTA at every barrier exit, kept in a smem array and flushed after the last
// barrier (no global traffic during the run). -DFG_STAMPS=0 removes them (gate 2 builds). globaltimer is read only at post-prologue
// and at the end (span ~8 ms, ns/cycle calibration) and, as the stale-read canary, at the calibration barriers.
// Per launch the host emits JSON gate lines {"gate","pass","val"} (+launch/variant). See fg_analyze.py for gate 2 (cross-binary).
#pragma once
#include <cooperative_groups.h>
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
constexpr int NIT = NSTEP * L, NPH = 6, NCAL = 128;
constexpr int SL_START = 0, SL_PRO = 1, SL_IT = 2, SL_CAL = SL_IT + NIT * NPH, SL_END = SL_CAL + 2 * NCAL, FG_NSLOT = SL_END + 1;
enum { FGV_NULL = 0, FGV_K0 = 1, FGV_CALIB = 2 };
constexpr size_t FG_SMEM_DYN = 117248;  // SMEM_C2 of the megakernel (gemm_phase_wsplit footprint), the kernel's dynamic smem

struct Params {  // the field names the verbatim code uses
  bf16* h; float* ssp; float* part; bf16* act; bf16* xn; unsigned* cnt; int fuse; float* mod;
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

// ------------------------------------------------------------- harness device code
struct FgArgs {
  const bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd;
  float* mod; bf16* h; float* ssp; float* part; bf16* act;
  unsigned* bar; unsigned* sink;
  ull* stamps;   // [gridDim][FG_NSLOT]
  ull* gtv;      // [0] globaltimer at post-prologue, [1] at end, [2..2+2*NCAL) canary reads at the calibration barriers
  int variant, cold, ctx, fuse, cwarps;  // cwarps: warps that burn HMMA during the C phase instead of idling (power / clock context)
  long long ccyc, ncal;
};
constexpr int CTX_WO_PF = 1, CTX_WD_PF = 2, CTX_DPART = 4, CTX_ACT_RD = 8, CTX_ALL = 15;

__device__ __forceinline__ uint32_t fg_hash(uint32_t i) { uint32_t h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return h; }

// D-like phase: 4 MB Wo read (L2-warm via the early prefetch) + synthetic D partials part[16][50][1024] bf16 (the real E input).
__device__ __forceinline__ void fg_phase_D(const Params& p, const bf16* wo, int it, unsigned* sink, int ctx) {
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const uint4* w4 = reinterpret_cast<const uint4*>(wo);
  uint32_t acc = 0;
#pragma unroll 4
  for (size_t i = gt; i < (size_t)W * QD * 2 / 16; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (ctx & CTX_DPART)
    for (size_t i = gt; i < (size_t)(QD / KC) * T * W / 2; i += nt) {
      const uint32_t hv = fg_hash((uint32_t)i + (uint32_t)it * 7919u);
      part_st2(p.part, 2 * i, ((int)(hv & 0xffff) - 32768) * (0.02f / 32768.f), ((int)(hv >> 16) - 32768) * (0.02f / 32768.f));
    }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__device__ __forceinline__ void fg_null_F(const WPre& w) {  // barrier kept (arrive / wait / stamp), no F work
  wpre_arrive(w);
  wpre_wait(w);
}

template <class VT>
__global__ void __launch_bounds__(NTHREADS, 1) fg_kernel(FgArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
#if FG_STAMPS
  __shared__ ull stm[FG_NSLOT];
  __shared__ ull gtm[2 * NCAL];
#endif
  cg::grid_group grid = cg::this_grid();
  bf16* smb = reinterpret_cast<bf16*>(smem_raw);
  const int tid = threadIdx.x;
  Params p{a.h, a.ssp, a.part, a.act, nullptr, nullptr, a.fuse, a.mod};
  unsigned bar_epoch = 0;
  // stamp slots are compile-time offsets from the loop base, so the stamps add no live loop state
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
  const float* nmodbase = a.mod;
  unsigned it1 = 0; const bool backoff = false; (void)it1; (void)backoff;
  for (int it = 0; it < NIT; ++it) {
    const int l = it % L;
    const int sb = SL_IT + it * NPH;
    // ---- C: dummy phase; Wo prefetch (the real B stream) first, Wg/Wu prefetch after the "attention" (as C3)
    if (a.ctx & CTX_WO_PF) prefetch_l2(a.wo + (size_t)l * W * QD, (size_t)W * QD * 2);
    fg_burn(a.ccyc * 85 / 100, a.cwarps);
    if (!a.cold) {
      prefetch_l2(a.wg + (size_t)l * FF * W, (size_t)FF * W * 2);
      prefetch_l2(a.wu + (size_t)l * FF * W, (size_t)FF * W * 2);
    }
    fg_burn(a.ccyc * 15 / 100, a.cwarps);
    SYNC(sb + 0);  // S0
    // ---- D
    if (a.ctx & CTX_WD_PF) prefetch_l2(a.wd + (size_t)l * W * FF, (size_t)W * FF * 2);
    fg_phase_D(p, a.wo + (size_t)l * W * QD, it, a.sink, a.ctx);
    SYNC(sb + 1);  // S1
    // ---- E: real residual_tasks on the synthetic D partials
#ifdef FG_HOOKS
    VT::E(p, a, l, nmodbase);  // variant-owned E (must do what residual_tasks does; may add work for its own variants)
#else
    residual_tasks<QD / KC>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W);
#endif
    // ---- F (E->F split barrier inside; S2)
    {
      const WPre w = wpre(sb + 2);  // S2
      VT::F(p, a, l, w, smb);
    }
    if (VT::kFgBarrier) SYNC(sb + 3); else FG_STAMP(sb + 3);  // S3
    // ---- G: next-layer prefetch (all variants), then the variant's reduce / gelu*up
    {
      const int ln = (l + 1 < L) ? l + 1 : 0;
      prefetch_l2(a.wq + (size_t)ln * QD * W, (size_t)QD * W * 2);
      prefetch_l2(a.wk + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
      prefetch_l2(a.wv + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
    }
    VT::G(p, a, tid, it1, backoff);
    // ---- H: G->H split barrier (first Wd loads between arrive and wait), then the rest of the 8 MB Wd read + act chunk
    {
      const WPre w = wpre(sb + 4);  // S4
      const size_t gt = (size_t)blockIdx.x * NTHREADS + tid, nt = (size_t)gridDim.x * NTHREADS;
      const uint4* w4 = reinterpret_cast<const uint4*>(a.wd + (size_t)l * W * FF);
      const size_t n4 = (size_t)W * FF * 2 / 16;
      uint4 b[4];
      wpre_arrive(w);
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = __ldg(w4 + gt + j * nt);
      wpre_wait(w);
      uint32_t acc = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) acc ^= b[j].x ^ b[j].y ^ b[j].z ^ b[j].w;
      if (a.ctx & CTX_ACT_RD) {  // this CTA's 128-wide K chunk of act (50 rows), the H A-tile read
        const int s = blockIdx.x % (FF / KC);
        for (int i = tid; i < T * (KC / 8); i += NTHREADS) {
          const uint4 v = __ldcg(reinterpret_cast<const uint4*>(a.act + (size_t)(i / (KC / 8)) * FF + s * KC + (i % (KC / 8)) * 8));
          acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
      }
#pragma unroll 2
      for (size_t i = gt + 4 * nt; i < n4; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
      if (acc == 0x9e3779b9u) a.sink[1] = acc;
    }
    SYNC(sb + 5);  // S5
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

// ------------------------------------------------------------- init / reference kernels
__global__ void fg_init_bf16(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(((int)(fg_hash((uint32_t)i + seed) & 0xffff) - 32768) * (scale / 32768.f));
}
__global__ void fg_init_f32(float* p, size_t n, uint32_t seed, float scale, float bias) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = bias + ((int)(fg_hash((uint32_t)i + seed) & 0xffff) - 32768) * (scale / 32768.f);
}
// fp32 CUDA-core reference of F+G for layer l (pattern of k_ref in gemm_tc2.cu): AnT[k][t] = bf16(adaRMS-normed h) as the staging
// computes it (rs from the 16 ssp parts summed serially here), then g = Wg.A, u = Wu.A, act_ref = gelu_tanh(g) * u, all fp32.
__global__ void fg_ref_norm(const bf16* h, const float* ssp, const float* nmod, float* AnT) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * W; i += gridDim.x * blockDim.x) {
    const int t = i / W, k = i % W;
    float v = 0.f;
    for (int c = 0; c < NSSP; ++c) v += ssp[t * NSSP + c];
    const float rs = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
    const float o = __fadd_rn(__fmul_rn(__fmul_rn(__bfloat162float(h[i]), rs), __fadd_rn(1.0f, nmod[k])), nmod[W + k]);
    AnT[k * T + t] = __bfloat162float(__float2bfloat16_rn(o));
  }
}
__global__ void fg_ref_gu(const bf16* wg, const bf16* wu, const float* AnT, const bf16* act, unsigned* maxerr, unsigned* maxref,
                          unsigned* nbad, unsigned* nsent) {
  const int n = blockIdx.x, t = threadIdx.x;  // block = one feature, thread = one token
  if (t >= T) return;
  const bf16* g = wg + (size_t)n * W; const bf16* u = wu + (size_t)n * W;
  float ag = 0.f, au = 0.f, sg = 0.f, su = 0.f;
  for (int k = 0; k < W; ++k) {
    const float a = AnT[k * T + t];
    const float pg = a * __bfloat162float(g[k]), pu = a * __bfloat162float(u[k]);
    ag += pg; au += pu; sg += fabsf(pg); su += fabsf(pu);
  }
  const float gl = gelu_tanh_ref(ag), ref = gl * au;
  const bf16 raw = act[t * FF + n];
  if (*reinterpret_cast<const unsigned short*>(&raw) == 0x7f7f) atomicAdd(nsent, 1u);  // untouched sentinel
  const float got = __bfloat162float(raw);
  const float err = fabsf(got - ref);
  atomicMax(maxerr, isfinite(err) ? __float_as_uint(err) : 0x7f800000u);
  atomicMax(maxref, __float_as_uint(fabsf(ref)));
  // tolerance from the kernel's own rounding: the 8 bf16 split-K partials (each <= 2^-9 relative to its magnitude, bounded by
  // sum |a w|), rbf(g) / rbf(u), and the final bf16 round; gelu' <= 1.13; x2 safety.
  const float dg = (1.f / 512.f) * (sg + fabsf(ag)), du = (1.f / 512.f) * (su + fabsf(au));
  const float tol = 2.f * (1.13f * dg * fabsf(au) + fabsf(gl) * du + (1.f / 256.f) * fabsf(ref)) + 1e-6f;
  if (!(err <= tol)) atomicAdd(nbad, 1u);
}

// ------------------------------------------------------------- host harness
static double fg_median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double fg_pct(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }
#ifdef FG_HOOKS  // additive, macro-guarded extension points (used by fg_k1.cu only; no effect on a build without FG_HOOKS)
static const char* fg_ext_vname(int v);  // defined by the variant file
static int fg_ext_parse(const std::string& s);
#define FG_VNAME_FALLBACK(v) fg_ext_vname(v)
#else
#define FG_VNAME_FALLBACK(v) "var"
#endif
static const char* fg_vname(int v) { return v == FGV_NULL ? "null" : v == FGV_K0 ? "k0" : v == FGV_CALIB ? "calib" : FG_VNAME_FALLBACK(v); }

struct FgRec {
  int variant, idx; double ev_ms = 0, nspc = 0, fg_ns = 0, f_ns = 0, g_ns = 0, stage_ns[NPH] = {0}, calib_cyc = 0, calib_gt = 0, prologue_ns = 0, closure = 0;
  double maxerr = 0, maxref = 0; unsigned nbad = 0, nsent = 0; long ord_viol = 0; double ord_min_ns = 0;
};

template <class VT>
int fg_main(int argc, char** argv) {
  int reps = 20, cold = 0, ctx = CTX_ALL, cwarps = 0; long long ncal = 8700, ccyc = 19000; std::string out = ".", name = std::string(VT::kName) + (FG_STAMPS ? "" : "_ns"), vlist = "null,k0,calib";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i]; const char* v = argv[i + 1];
    if (k == "--reps") reps = atoi(v); else if (k == "--cold") cold = atoi(v); else if (k == "--ctx") ctx = atoi(v);
    else if (k == "--cwarps") cwarps = atoi(v); else if (k == "--ncal") ncal = atoll(v); else if (k == "--ccyc") ccyc = atoll(v); else if (k == "--out") out = v;
    else if (k == "--name") name = v; else if (k == "--variants") vlist = v;
    else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  std::vector<int> vs;
  for (size_t p = 0; p < vlist.size();) {
    size_t q = vlist.find(',', p); if (q == std::string::npos) q = vlist.size();
    const std::string s = vlist.substr(p, q - p);
    vs.push_back(s == "null" ? FGV_NULL : s == "k0" ? FGV_K0 : s == "calib" ? FGV_CALIB : -1);
#ifdef FG_HOOKS
    if (vs.back() < 0) vs.back() = fg_ext_parse(s);
#endif
    if (vs.back() < 0) { fprintf(stderr, "bad variant %s\n", s.c_str()); return 2; }
    p = q + 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(fg_kernel<VT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)VT::kSmem));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fg_kernel<VT>, NTHREADS, VT::kSmem));
  fprintf(stderr, "%s: %s grid=%d occ/SM=%d stamps=%d reps=%d cold=%d ctx=%d ncal=%lld ccyc=%lld cwarps=%d\n", name.c_str(), prop.name, grid, occ, FG_STAMPS, reps, cold, ctx, ncal, ccyc, cwarps);
  if (occ < 1) { fprintf(stderr, "kernel does not fit\n"); return 1; }
  std::string jpath = out + "/" + name + ".jsonl";
  FILE* jf = fopen(jpath.c_str(), "w"); if (!jf) { perror(jpath.c_str()); return 1; }
#define JOUT(...) do { fprintf(jf, __VA_ARGS__); fprintf(stdout, __VA_ARGS__); } while (0)

  const size_t nwg = (size_t)L * FF * W, nwq = (size_t)L * QD * W, nwkv = (size_t)L * KVD * W, nwo = (size_t)L * W * QD, nwd = (size_t)L * W * FF;
  bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd, *h, *act; float *mod, *ssp, *part, *AnT; unsigned *bar, *sink, *cnt4; ull *stamps, *gtv; char* flush;
  CK(cudaMalloc(&wq, nwq * 2)); CK(cudaMalloc(&wk, nwkv * 2)); CK(cudaMalloc(&wv, nwkv * 2)); CK(cudaMalloc(&wo, nwo * 2));
  CK(cudaMalloc(&wg, nwg * 2)); CK(cudaMalloc(&wu, nwg * 2)); CK(cudaMalloc(&wd, nwd * 2));
  CK(cudaMalloc(&h, (size_t)T * W * 2)); CK(cudaMalloc(&act, (size_t)T * FF * 2));
  CK(cudaMalloc(&mod, (size_t)2 * L * 3 * W * 4)); CK(cudaMalloc(&ssp, (size_t)T * NSSP * 4));
  CK(cudaMalloc(&part, (size_t)8 * T * 2 * FF * 2)); CK(cudaMalloc(&AnT, (size_t)W * T * 4));
  CK(cudaMalloc(&bar, BAR_WORDS * 4)); CK(cudaMalloc(&sink, 64)); CK(cudaMalloc(&cnt4, 64));
  CK(cudaMalloc(&stamps, (size_t)grid * FG_NSLOT * 8)); CK(cudaMalloc(&gtv, (2 + 2 * NCAL) * 8));
  const size_t flush_bytes = (size_t)2 * prop.l2CacheSize; CK(cudaMalloc(&flush, flush_bytes));
  CK(cudaMemset(part, 0, (size_t)8 * T * 2 * FF * 2)); CK(cudaMemset(bar, 0, BAR_WORDS * 4)); CK(cudaMemset(sink, 0, 64));
  fg_init_bf16<<<1184, 256>>>(wg, nwg, 11, 0.1f); fg_init_bf16<<<1184, 256>>>(wu, nwg, 12, 0.1f);
  fg_init_bf16<<<1184, 256>>>(wq, nwq, 13, 0.05f); fg_init_bf16<<<1184, 256>>>(wk, nwkv, 14, 0.05f); fg_init_bf16<<<1184, 256>>>(wv, nwkv, 15, 0.05f);
  fg_init_bf16<<<1184, 256>>>(wo, nwo, 16, 0.05f); fg_init_bf16<<<1184, 256>>>(wd, nwd, 17, 0.05f);
  fg_init_f32<<<148, 256>>>(mod, (size_t)2 * L * 3 * W, 21, 0.3f, 0.1f);
  CK(cudaDeviceSynchronize());
  unsigned *dref; CK(cudaMalloc(&dref, 16));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  FgArgs args{wq, wk, wv, wo, wg, wu, wd, mod, h, ssp, part, act, bar, sink, stamps, gtv, 0, cold, ctx, FUSE_DEFAULT, cwarps, ccyc, ncal};
#ifdef FG_HOOKS
  VT::prepare(args, grid);
#endif
  std::vector<FgRec> recs;
  std::vector<ull> st((size_t)grid * FG_NSLOT), gt(2 + 2 * NCAL);
  const std::string rawdir = out + "/raw"; { std::string c = "mkdir -p " + rawdir; if (system(c.c_str())) return 1; }

  auto run = [&](int variant, int idx, bool record) {
    CK(cudaMemsetAsync(flush, 0, flush_bytes));
    fg_init_bf16<<<148, 256>>>(h, (size_t)T * W, 31, 1.0f);
    CK(cudaMemsetAsync(act, 0x7f, (size_t)T * FF * 2));
    CK(cudaMemsetAsync(stamps, 0, (size_t)grid * FG_NSLOT * 8));
    args.variant = variant;
    void* kargs[] = {&args};
    CK(cudaEventRecord(e0));
#ifdef FG_CLUSTER  // cooperative launch with clusters of FG_CLUSTER CTAs (grid stays 148)
    {
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(grid); lc.blockDim = dim3(NTHREADS); lc.dynamicSmemBytes = VT::kSmem; lc.stream = 0;
      cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
      at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = FG_CLUSTER; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
      lc.attrs = at; lc.numAttrs = 2;
      CK(cudaLaunchKernelEx(&lc, fg_kernel<VT>, args));
    }
#else
    CK(cudaLaunchCooperativeKernel((void*)fg_kernel<VT>, dim3(grid), dim3(NTHREADS), kargs, VT::kSmem, 0));
#endif
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    if (!record) return;
    // fp32 reference of the last layer (act after iteration 179 = layer 17)
    CK(cudaMemset(dref, 0, 16));
    fg_ref_norm<<<200, 256>>>(h, ssp, mod + (size_t)(2 * (L - 1) + 1) * 3 * W, AnT);
    fg_ref_gu<<<FF, 64>>>(wg + (size_t)(L - 1) * FF * W, wu + (size_t)(L - 1) * FF * W, AnT, act, dref, dref + 1, dref + 2, dref + 3);
    unsigned hr[4]; CK(cudaMemcpy(hr, dref, 16, cudaMemcpyDeviceToHost));
    FgRec r; r.variant = variant; r.idx = idx; r.ev_ms = ms;
    r.maxerr = *reinterpret_cast<float*>(&hr[0]); r.maxref = *reinterpret_cast<float*>(&hr[1]); r.nbad = hr[2]; r.nsent = hr[3];
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
    r.f_ns = r.stage_ns[3]; r.g_ns = r.stage_ns[4]; r.fg_ns = r.stage_ns[3] + r.stage_ns[4];  // S3 = E->F..F->G barrier exit, S4 = G..G->H barrier exit
    // calibration readback (clock64 stamps) and the globaltimer canary
    std::vector<double> pe, pd, ge, gd;
    for (int q = 0; q < NCAL; ++q) {
      const int a0 = SL_CAL + 2 * q;
      pe.push_back((double)(s0[a0] - s0[a0 - 1])); pd.push_back((double)(s0[a0 + 1] - s0[a0]));
      if (q > 0) ge.push_back((double)(gt[2 + 2 * q] - gt[2 + 2 * q - 1]));
      gd.push_back((double)(gt[2 + 2 * q + 1] - gt[2 + 2 * q]));
    }
    r.calib_cyc = fg_median(pd) - fg_median(pe);
    r.calib_gt = (fg_median(gd) - fg_median(ge)) / ((double)ncal * r.nspc);  // canary: globaltimer readback / expected (1.0 = fine)
    r.ord_min_ns = 1e18;
    for (int c = 0; c < grid; ++c)
      for (int i = 1; i < FG_NSLOT; ++i) {
        const long long d = (long long)(st[(size_t)c * FG_NSLOT + i] - st[(size_t)c * FG_NSLOT + i - 1]);
        if (d < 0) ++r.ord_viol;
        r.ord_min_ns = std::min(r.ord_min_ns, d * r.nspc);
      }
    char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%s_%03d.cta0.u64", rawdir.c_str(), name.c_str(), fg_vname(variant), idx);
    FILE* f = fopen(fn, "wb"); fwrite(s0, 8, FG_NSLOT, f); fclose(f);
    if (idx >= reps - 2) {  // all-CTA stamps for the last two launches of each variant
      snprintf(fn, sizeof fn, "%s/%s_%s_%03d.allcta.u64", rawdir.c_str(), name.c_str(), fg_vname(variant), idx);
      f = fopen(fn, "wb"); fwrite(st.data(), 8, st.size(), f); fclose(f);
    }
#endif
#ifdef FG_HOOKS
    VT::post_run(variant, idx, reps, out, name, r.nspc);
#endif
    recs.push_back(r);
  };
  for (int v : vs) for (int i = 0; i < 3; ++i) run(v, -1, false);  // warm-up (clocks, caches, first-launch costs)
  for (int r = 0; r < reps; ++r) {
    std::vector<int> ord = vs;
    if (r & 1) std::reverse(ord.begin(), ord.end());  // ABAB: alternate the order each round
    for (int v : ord) run(v, r, true);
  }

  // ---- per-launch records and gates
  auto gate = [&](const FgRec& r, const char* g, bool pass, const char* valfmt, double a, double b = 0, double c = 0) {
    char vb[256]; snprintf(vb, sizeof vb, valfmt, a, b, c);
    JOUT("{\"launch\":%d,\"variant\":\"%s\",\"gate\":\"%s\",\"pass\":%s,\"val\":%s}\n", r.idx, fg_vname(r.variant), g, pass ? "true" : "false", vb);
  };
  double nspc_med = 0; { std::vector<double> all; for (auto& r : recs) all.push_back(r.nspc); nspc_med = fg_median(all); }
  auto var_nspc = [&](int v) { std::vector<double> x; for (auto& r : recs) if (r.variant == v) x.push_back(r.nspc); return fg_median(x); };
  int nfail = 0;
  for (auto& r : recs) {
    JOUT("{\"rec\":\"launch\",\"name\":\"%s\",\"launch\":%d,\"variant\":\"%s\",\"event_ms\":%.6f,\"nspc\":%.6f,\"prologue_ns\":%.1f,"
         "\"stage_ns\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],\"F_ns\":%.1f,\"G_ns\":%.1f,\"cold\":%d,\"ctx\":%d,\"cwarps\":%d,\"ccyc\":%lld}\n",
         name.c_str(), r.idx, fg_vname(r.variant), r.ev_ms, r.nspc, r.prologue_ns, r.stage_ns[0], r.stage_ns[1], r.stage_ns[2], r.stage_ns[3],
         r.stage_ns[4], r.stage_ns[5], r.f_ns, r.g_ns, cold, ctx, cwarps, ccyc);
    // 7: work check. k0 must match the fp32 reference; null / calib must leave act at the sentinel.
    const bool produces = r.variant != FGV_NULL && r.variant != FGV_CALIB;
    const bool wp = produces ? (r.nbad == 0 && r.nsent == 0 && std::isfinite(r.maxerr)) : (r.nsent == (unsigned)(T * FF));
    gate(r, "work", wp, "{\"maxerr\":%.6g,\"maxref\":%.6g,\"nbad\":%.0f}", r.maxerr, r.maxref, (double)r.nbad); nfail += !wp;
#if FG_STAMPS
    const bool g1 = fabs(r.closure) < 0.01; gate(r, "closure", g1, "%.5f", r.closure); nfail += !g1;
    const bool g4 = fabs(r.calib_cyc / (double)ncal - 1.0) < 0.02;
    gate(r, "known_delay", g4, "{\"readback_over_N\":%.5f,\"gt_canary_over_N\":%.5f}", r.calib_cyc / (double)ncal, r.calib_gt); nfail += !g4;
    const double vn = var_nspc(r.variant);  // gate 5a: launch-to-launch stability within a variant (clock64 -> ns slope per launch)
    const bool g5 = fabs(r.nspc / vn - 1.0) < 0.015; gate(r, "clock_stable", g5, "{\"nspc\":%.6f,\"dev_from_variant_median\":%.5f}", r.nspc, r.nspc / vn - 1.0); nfail += !g5;
    const bool g6 = r.ord_viol == 0; gate(r, "ordering", g6, "{\"violations\":%.0f,\"min_interval_ns\":%.1f}", (double)r.ord_viol, r.ord_min_ns); nfail += !g6;
#endif
  }
  // ---- set-level gates per variant against null: 5b clock_cross, 3 delta_agreement (L0 events vs L1 stamps), per layer iteration
  for (int v : vs) {
    if (v == FGV_NULL) continue;
    std::vector<double> evv, evn, l1v, l1n, f1v, f1n, g1v, g1n, ev_pair, allv, alln;
    std::vector<double> st_v[NPH], st_n[NPH];
    for (auto& r : recs) {
      double tot = 0; for (int k = 0; k < NPH; ++k) tot += r.stage_ns[k];
      if (r.variant == v) { evv.push_back(r.ev_ms); l1v.push_back(r.fg_ns); f1v.push_back(r.f_ns); g1v.push_back(r.g_ns); allv.push_back(tot); for (int k = 0; k < NPH; ++k) st_v[k].push_back(r.stage_ns[k]); }
      if (r.variant == FGV_NULL) { evn.push_back(r.ev_ms); l1n.push_back(r.fg_ns); f1n.push_back(r.f_ns); g1n.push_back(r.g_ns); alln.push_back(tot); for (int k = 0; k < NPH; ++k) st_n[k].push_back(r.stage_ns[k]); }
    }
    if (evn.empty()) continue;
    const double L0 = (fg_median(evv) - fg_median(evn)) * 1e3 / NIT;  // us per layer iteration
    for (size_t i = 0; i < std::min(evv.size(), evn.size()); ++i) ev_pair.push_back((evv[i] - evn[i]) * 1e3 / NIT);
#if FG_STAMPS
    const double cdev = var_nspc(v) / var_nspc(FGV_NULL) - 1.0;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"clock_cross\",\"pass\":%s,\"val\":{\"nspc\":%.6f,\"null_nspc\":%.6f,\"dev\":%.5f}}\n",
         fg_vname(v), fabs(cdev) < 0.015 ? "true" : "false", var_nspc(v), var_nspc(FGV_NULL), cdev);
    const double Lwin = (fg_median(l1v) - fg_median(l1n)) / 1e3;    // F+G window only (E->F barrier exit .. G->H barrier exit)
    const double Lall = (fg_median(allv) - fg_median(alln)) / 1e3;  // all six stage periods of the iteration
    const bool g3 = fabs(L0 - Lall) < 0.2;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"delta_agreement\",\"pass\":%s,\"val\":{\"L0_us\":%.4f,\"L1_all_us\":%.4f,\"diff_us\":%.4f,"
         "\"L1_window_us\":%.4f,\"outside_window_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f,\"F_us\":%.4f,\"G_us\":%.4f,\"null_F_us\":%.4f,\"null_G_us\":%.4f,"
         "\"stage_delta_us\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]}}\n",
         fg_vname(v), g3 ? "true" : "false", L0, Lall, L0 - Lall, Lwin, Lall - Lwin, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9), fg_median(f1v) / 1e3, fg_median(g1v) / 1e3,
         fg_median(f1n) / 1e3, fg_median(g1n) / 1e3,
         (fg_median(st_v[0]) - fg_median(st_n[0])) / 1e3, (fg_median(st_v[1]) - fg_median(st_n[1])) / 1e3, (fg_median(st_v[2]) - fg_median(st_n[2])) / 1e3,
         (fg_median(st_v[3]) - fg_median(st_n[3])) / 1e3, (fg_median(st_v[4]) - fg_median(st_n[4])) / 1e3, (fg_median(st_v[5]) - fg_median(st_n[5])) / 1e3);
    nfail += !g3;
#else
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"L0_delta\",\"pass\":true,\"val\":{\"L0_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f}}\n",
         fg_vname(v), L0, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9));
#endif
  }
  for (int v : vs) {
    std::vector<double> ev; for (auto& r : recs) if (r.variant == v) ev.push_back(r.ev_ms);
    JOUT("{\"rec\":\"summary\",\"name\":\"%s\",\"variant\":\"%s\",\"stamps\":%d,\"n\":%zu,\"event_ms_med\":%.6f,\"event_ms_p10\":%.6f,\"event_ms_p90\":%.6f,"
         "\"per_layer_us_med\":%.4f}\n", name.c_str(), fg_vname(v), FG_STAMPS, ev.size(), fg_median(ev), fg_pct(ev, 0.1), fg_pct(ev, 0.9), fg_median(ev) * 1e3 / NIT);
  }
  JOUT("{\"rec\":\"done\",\"name\":\"%s\",\"failed_gates\":%d,\"nspc_median\":%.6f}\n", name.c_str(), nfail, nspc_med);
  fclose(jf);
  return 0;
}
