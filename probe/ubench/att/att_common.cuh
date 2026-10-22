// att_common.cuh: shared validity harness for the standalone ATTENTION phase study (phase C split-key attention + C3 combine of the pi0.5
// megakernel, mk/pi05_mk.cu). Same structure and gate discipline as probe/ubench/fg/fg_common.cuh.
// One persistent cooperative kernel (148 CTAs x 256 threads, 1 CTA/SM, 117248 B dynamic smem by default = SMEM_C2 of the megakernel;
// --smem / --cluster reproduce the FG_UMMA build's 197632 B and cluster-of-2 launch) runs 180 layer iterations (10 steps x 18 layers,
// distinct per-layer prefix K/V: L = 18 layers x [968][256] bf16 K and V = 17.8 MB, as in situ). Per iteration, grid barriers between phases:
//   A  dummy QKV GEMM (spin --acyc cycles)                                                                          -> stamp S0 (A->B barrier)
//   B  harness: L2 prefetch of Wo + this layer's prefix K/V (as the megakernel's B), VT::B_pre (K0: the PREFILL=2 K/V + mask cp.async
//      prefill into smem), dummy B tasks on the highest 70 blocks (write q [50][2048], ksuf, vsuf with real shapes, bounded random data;
//      --bcyc spin per task block); the last iteration also poisons stats / opart (NaN) and resets attn (sentinel)   -> stamp S1 (B->C)
//   C  VT::C  (K0: the unit loop, red.release on CNT_ATT + rg after each unit) [VT::kCCBarrier: an extra grid barrier]  -> stamp S2 (own end of C)
//      harness: L2 prefetch of Wg / Wu (as the megakernel's C3); VT::C3 (K0: block_wait on the row group's counter + combine -> attn)
//   D  harness D-like: prefetch Wd; C3->D is a WPRE split barrier (first Wo loads between arrive and wait), then the attn A-tile read
//      (this CTA's 128-wide K chunk of attn [50][2048]), the rest of the 4 MB Wo read and synthetic D partials -> stamp S3 (at wpre_wait exit)
//      D->E barrier                                                                                                   -> stamp S4
//   tail1 dummy E..G (spin --tcyc), barrier -> S5; prefetch next layer's Wq/Wk/Wv; tail2 dummy G..I (spin --tcyc), barrier -> S6
// then NCAL calibration rounds {barrier; spin 0 | barrier; spin N | barrier} (known-delay readback).
// Stage k (stage_ns[k]) = stamp k - stamp k-1 (S6 of the previous iteration for k = 0). The attention window of the megakernel's breakdown
// (C_attn interval, 10.6 us) is stage 2 + stage 3 here (B->C barrier exit .. C3->D barrier exit).
// Stamps: clock64 by thread 0 of EVERY CTA at every barrier exit, kept in a smem array and flushed after the last barrier (no global traffic
// during the run). -DATT_STAMPS=0 removes them (gate 2 builds). globaltimer is read only post-prologue and at the end (span ~8 ms, ns/cycle
// calibration) and, as the stale-read canary, at the calibration barriers. -DATT_MARKS=1 (separate binary, never used for gate numbers)
// additionally lets the variant stamp sub-phases of C / C3 into a smem array (ATT_MARK) for the stage timeline.
// Per launch the host emits JSON gate lines {"gate","pass","val"} (+launch/variant). See att_analyze.py for gate 2 (cross-binary).
//
// ---- Extension interface (variant files att_k1.cu ...; build with -DATT_HOOKS) ----
//   struct VT { static constexpr const char* kName; static constexpr bool kCCBarrier; static constexpr size_t kSmem; static constexpr int kCluster;
//     __device__ static void B_pre(const Params&, const AttArgs&, int l, bf16* smb);
//     __device__ static void C (const Params&, const AttArgs&, int l, int it, bf16* smb, ull* mk);
//     __device__ static void C3(const Params&, const AttArgs&, int l, int it, bf16* smb, ull* mk);
//     (ATT_HOOKS only:) static void prepare(AttArgs&, int grid); static void post_run(int variant, int idx, int reps, const std::string& out,
//     const std::string& name, double nspc); }  plus static const char* att_ext_vname(int); static int att_ext_parse(const std::string&);
//   Variant ids: ATTV_NULL / ATTV_K0 / ATTV_CALIB are reserved; extension variants use ATTV_EXT + i (ATTV_EXT = 16). Every variant must write p.attn
//   [400][256] bf16 (row = token * 8 + head, the megakernel's attn layout consumed by D) from p.q, the prefix K/V of layer l, p.ksuf / p.vsuf and
//   the mask; stats / opart / cnt are scratch the variant may use freely (the epoch of layer iteration `it` is it + 1; counters are zeroed per
//   launch; NCNT lines of CNT_STRIDE words). A variant that needs a grid-wide barrier between C and C3 sets kCCBarrier (null / calib run it too).
//   VT::kCluster > 1 requests a cooperative cluster launch with that cluster size (the --cluster flag overrides); kSmem is the dynamic smem
//   (the --smem flag raises it). The harness issues the L2 prefetches of Wo / prefix K/V before B_pre and of Wg / Wu between C and C3 for every
//   variant (they belong to the megakernel's B / C3 phases and warm the stream the following phases read).
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

#ifndef ATT_STAMPS
#define ATT_STAMPS 1
#endif
#ifndef ATT_MARKS
#define ATT_MARKS 0
#endif

// ---------------------------------------------------------------- geometry (as mk/pi05_mk.cu, graded config P968 / k64)
constexpr int T = 50, W = 1024, L = 18, NSTEP = 10, QD = 2048, HD = 256, KVD = 256, FF = 4096, NQH = 8;
constexpr int P = 968, NK = P + T, QROWS = T * NQH;
constexpr int NTHREADS = 256, NWARPS = 8, KC = 128;
#ifndef ATT_KEYS_DEF
#define ATT_KEYS_DEF 64
#endif
constexpr int ATT_KEYS = ATT_KEYS_DEF, ATT_NCHUNK = (NK + ATT_KEYS - 1) / ATT_KEYS;  // 16
constexpr int ATT_ROWS = 64, ATT_NRG = (QROWS + ATT_ROWS - 1) / ATT_ROWS;              // 64 query rows per unit, 7 row groups (PI05_ATT_FA = 1)
constexpr int ATT_NT = ATT_KEYS / 8;
constexpr int KS_STRIDE = HD + 8, VS_STRIDE = HD + 8;
static_assert(QROWS % 16 == 0 && ATT_KEYS <= 64 && NWARPS == 2 * (ATT_ROWS / 16) && HD == 256, "FA unit");
static_assert(ATT_NRG * ATT_NCHUNK <= 148, "attention units must fit one wave");
constexpr int CNT_STRIDE = 32, CNT_QKV = 0, CNT_ATT = 16, CNT_GU = 32, NCNT = CNT_GU + FF / 32;
constexpr unsigned ARR_ATT = ATT_NCHUNK;
constexpr int FUSE_AB = 1, FUSE_CC3 = 2, FUSE_FG = 4, FUSE_BACKOFF = 8, FUSE_DEFNORM = 16;
constexpr int FUSE_DEFAULT = 18;  // the graded build: FUSE_CC3 (counter hand-off C -> C3) + FUSE_DEFNORM
// megakernel defines that the verbatim code is written against
#define PI05_BARRIER 3
#define PI05_WPRE 15
#define PI05_ATT_FA 1
#define PI05_ATT_PREFILL 2
constexpr int BAR_STRIDE = 32;
constexpr int BAR_WORDS = 160 * BAR_STRIDE + 2 * BAR_STRIDE;

// ---------------------------------------------------------------- stamp slots (per CTA)
constexpr int NIT = NSTEP * L, NPH = 7, NCAL = 128, ATT_NMARK = 8;
constexpr int SL_START = 0, SL_PRO = 1, SL_IT = 2, SL_CAL = SL_IT + NIT * NPH, SL_END = SL_CAL + 2 * NCAL, ATT_NSLOT = SL_END + 1;
enum { ATTV_NULL = 0, ATTV_K0 = 1, ATTV_CALIB = 2, ATTV_EXT = 16 };
constexpr size_t ATT_SMEM_DYN = 117248;  // SMEM_C2 of the megakernel (gemm_phase_wsplit footprint), the plain build's dynamic smem
constexpr size_t ATT_SMEM_UMMA = 197632;  // SMEM_FGU of the FG_UMMA build

struct Params {  // the field names the verbatim code uses
  const bf16* prefix_k; const bf16* prefix_v; const uint8_t* prefix_mask;
  bf16* q; bf16* ksuf; bf16* vsuf; bf16* attn; float* stats; float* opart; float* part; unsigned* cnt; int fuse;
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TS_NOW() ((unsigned long long)clock64())
__device__ __forceinline__ void att_spin(long long cyc) {
  const long long t0 = clock64();
  while (clock64() - t0 < cyc) {}
}
// phase filler: the first `nw` warps issue HMMAs on register data until the clock64 deadline (heat); the others idle-spin. nw = 0 is att_spin.
__device__ __forceinline__ void att_burn(long long cyc, int nw) {
  if ((int)(threadIdx.x >> 5) >= nw) { att_spin(cyc); return; }
  float c[4][4] = {};
  uint32_t a0 = 0x3f803f80u + threadIdx.x, a1 = 0x3f803f81u, a2 = 0x3f803f82u, a3 = 0x3f803f83u, b0 = 0x3f803f84u, b1 = 0x3f803f85u;
  const long long t0 = clock64();
  while (clock64() - t0 < cyc) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  if (c[0][0] == 12345.f && c[1][1] == 6789.f && c[2][2] == 1.f && c[3][3] == 2.f) a0 = 0;
}

// sub-phase marks (ATT_MARKS=1 builds only): thread 0 writes clock64 into this iteration's mark slots (smem); a no-op otherwise
#if ATT_MARKS
#define ATT_MARK(mk, k) do { if (threadIdx.x == 0 && (mk) != nullptr) (mk)[(k)] = clock64(); } while (0)
#else
#define ATT_MARK(mk, k) do {} while (0)
#endif

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
// Whole block waits until *c >= target: thread 0 spins relaxed, then one acquire fence; the CTA barrier extends it to the block.
__device__ __forceinline__ void block_wait(const unsigned* c, unsigned target, bool backoff) {
  if (threadIdx.x == 0) {
    while (ld_relaxed(c) < target) {
      if (backoff) __nanosleep(64);
    }
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  }
  __syncthreads();
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* ptr) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}
__device__ __forceinline__ void spin_acquire(const unsigned* ptr, unsigned target) {
  while ((int)(ld_acquire(ptr) - target) < 0) {}
}
// Grid barrier number `epoch` (PI05_BARRIER 3: monotonic atom.add.release counter, ld.acquire polling).
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned epoch) {
  __syncthreads();
  unsigned* cnt = bar + 160 * BAR_STRIDE;
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory");
    spin_acquire(cnt, epoch * gridDim.x);
  }
  __syncthreads();
}
// Deferred grid barrier handed to a GEMM phase: counter, poll target, stamp slot (or null).
struct WPre { unsigned* bar; unsigned target; unsigned long long* stamp; };
__device__ __forceinline__ void wpre_arrive(const WPre& w) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(w.bar) : "memory");
  }
}
__device__ __forceinline__ void wpre_wait(const WPre& w) {
  if (threadIdx.x == 0) spin_acquire(w.bar, w.target);
  __syncthreads();
  if (w.stamp != nullptr && threadIdx.x == 0) *w.stamp = TS_NOW();
}
// Warm L2 with a weight range that a later phase streams (spread over the grid).
__device__ __forceinline__ void prefetch_l2(const void* base, size_t bytes) {
  constexpr size_t CH = 128;
  const size_t n = bytes / CH;
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const char* b = reinterpret_cast<const char*>(base);
  for (size_t i = gt; i < n; i += nt) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(b + i * CH));
}
// 16-byte global->shared copy through L2; src_bytes = 0 zero-fills the destination.
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint2 pack_bf16x4(float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  return make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}

// ------------------------------------------------------------- harness device code
struct AttArgs {
  const bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd;
  const bf16 *prefix_k, *prefix_v; const uint8_t* prefix_mask;
  bf16 *q, *ksuf, *vsuf, *attn; float *stats, *opart, *part;
  unsigned *bar, *cnt, *sink;
  ull* stamps;   // [gridDim][ATT_NSLOT]
  ull* gtv;      // [0] globaltimer at post-prologue, [1] at end, [2..2+2*NCAL) canary reads at the calibration barriers
  ull* marks;    // [gridDim][NIT * ATT_NMARK] (ATT_MARKS builds)
  int variant, ctx, fuse, cwarps;  // cwarps: warps that burn HMMA in the dummy phases instead of idling
  long long acyc, bcyc, tcyc, ncal;
};
constexpr int CTX_WO_PF = 1, CTX_WD_PF = 2, CTX_DPART = 4, CTX_NEXT_PF = 8, CTX_ALL = 15;
constexpr float ATT_QK_SCALE = 2.4f;  // q, K uniform +-2.4: raw q.k std = 16 * 2.4^2 / 3 = 30.7, scaled score std = 1.9 (peaked softmax over ~1000 keys)

__device__ __forceinline__ uint32_t att_hash(uint32_t i) { uint32_t h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return h; }
__device__ __forceinline__ float att_u(uint32_t hv, float scale) { return ((int)(hv & 0xffff) - 32768) * (scale / 32768.f); }

// B-like dummy: the megakernel's B tasks (10 heads x 7 slices of 8 rows, highest block indices first) write q (8 heads), ksuf, vsuf as
// [t][head*256 + j] bf16 with the real layouts; the values are bounded random data (distinct per iteration).
__device__ __forceinline__ void att_phase_B(const Params& p, const AttArgs& a, int it) {
  constexpr int RS = NTHREADS / (HD / 8), NTASK = (T + RS - 1) / RS;  // 8 rows x 32 column quads per task
  bool any = false;
  uint32_t lacc = 0;
  for (int task = gridDim.x - 1 - blockIdx.x; task < 10 * NTASK; task += gridDim.x) {
    any = true;
    const int hh = task / NTASK, t = (task % NTASK) * RS + threadIdx.x / (HD / 8), j = (threadIdx.x % (HD / 8)) * 4;
    if (t < T) {
      {  // the B reduce's loads: 8 split-K bf16 partial quads for each of the two column halves (L2-latency bound, in flight together); values are
         // discarded (only kept live) so q / ksuf / vsuf stay hash data independent of the dummy partials
        const bf16* part = reinterpret_cast<const bf16*>(p.part);
        constexpr int NQKV = QD + 2 * KVD;
        const int c = hh * HD + j;
        uint2 v[16];
#pragma unroll
        for (int s = 0; s < W / KC; ++s) {
          v[s] = __ldcg(reinterpret_cast<const uint2*>(part + ((size_t)s * T + t) * NQKV + c));
          v[8 + s] = __ldcg(reinterpret_cast<const uint2*>(part + ((size_t)s * T + t) * NQKV + c + HD / 2));
        }
#pragma unroll
        for (int s = 0; s < 16; ++s) lacc ^= v[s].x ^ v[s].y;
      }
      const float sc = hh < 9 ? ATT_QK_SCALE : 1.0f;
      float o1[4], o2[4];
      const uint32_t base = (uint32_t)((((it * 10 + hh) * T + t) * HD + j) * 2) ;
#pragma unroll
      for (int e = 0; e < 4; ++e) { o1[e] = att_u(att_hash(base + 2 * e + 1), sc); o2[e] = att_u(att_hash(base + 2 * e + 1 + 0x7f001), sc); }
      bf16* dst = hh < 8 ? p.q + t * QD + hh * HD : (hh == 8 ? p.ksuf + t * KVD : p.vsuf + t * KVD);
      *reinterpret_cast<uint2*>(dst + j) = pack_bf16x4(o1[0], o1[1], o1[2], o1[3]);
      *reinterpret_cast<uint2*>(dst + j + HD / 2) = pack_bf16x4(o2[0], o2[1], o2[2], o2[3]);
    }
  }
  if (lacc == 0x9e3779b9u) a.sink[2] = lacc;
  if (any) att_spin(a.bcyc);  // the rest of the B task time (RoPE math) of the busy blocks
}

// D-like phase: split-barrier C3 -> D (first Wo loads issued between arrive and wait), then this CTA's 128-wide K chunk of attn (the D A-tile
// read: attn [50][2048] bf16, S = 16 chunks) and the rest of the 4 MB Wo read, then synthetic D partials part[16][50][1024] bf16.
__device__ __forceinline__ void att_phase_D(const Params& p, const AttArgs& a, const bf16* wo, int it, const WPre& w) {
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const uint4* w4 = reinterpret_cast<const uint4*>(wo);
  const size_t n4 = (size_t)W * QD * 2 / 16;
  uint4 b[4];
  wpre_arrive(w);
#pragma unroll
  for (int j = 0; j < 4; ++j) b[j] = __ldg(w4 + gt + j * nt);
  wpre_wait(w);
  uint32_t acc = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) acc ^= b[j].x ^ b[j].y ^ b[j].z ^ b[j].w;
  {
    const int s = blockIdx.x % (QD / KC);
    for (int i = threadIdx.x; i < T * (KC / 8); i += NTHREADS) {
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p.attn + (size_t)(i / (KC / 8)) * QD + s * KC + (i % (KC / 8)) * 8));
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
#pragma unroll 2
  for (size_t i = gt + 4 * nt; i < n4; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (a.ctx & CTX_DPART)
    for (size_t i = gt; i < (size_t)(QD / KC) * T * W / 2; i += nt) {
      const uint32_t hv = att_hash((uint32_t)i + (uint32_t)it * 7919u);
      __nv_bfloat162 v2 = __floats2bfloat162_rn(att_u(hv, 0.02f), att_u(hv >> 16, 0.02f));
      *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(p.part) + 2 * i) = v2;
    }
  if (acc == 0x9e3779b9u) a.sink[0] = acc;
}

template <class VT>
__global__ void __launch_bounds__(NTHREADS, 1) att_kernel(AttArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
#if ATT_STAMPS
  __shared__ ull stm[ATT_NSLOT];
  __shared__ ull gtm[2 * NCAL];
#endif
#if ATT_MARKS
  __shared__ ull mkm[NIT * ATT_NMARK];
#endif
  cg::grid_group grid = cg::this_grid();
  bf16* smb = reinterpret_cast<bf16*>(smem_raw);
  const int tid = threadIdx.x;
  Params p{a.prefix_k, a.prefix_v, a.prefix_mask, a.q, a.ksuf, a.vsuf, a.attn, a.stats, a.opart, a.part, a.cnt, a.fuse};
  unsigned bar_epoch = 0;
#if ATT_STAMPS
#define ATT_STAMP(sl) do { if (tid == 0) stm[(sl)] = clock64(); } while (0)
#else
#define ATT_STAMP(sl) do {} while (0)
#endif
#define SYNC(sl) do { grid_barrier(a.bar, ++bar_epoch); ATT_STAMP(sl); } while (0)
  auto wpre = [&](int sl) {  // next barrier handed to a phase (split arrive / wait); the stamp is written in wpre_wait
    return WPre{a.bar + 160 * BAR_STRIDE, (++bar_epoch) * gridDim.x,
#if ATT_STAMPS
                &stm[sl]
#else
                nullptr
#endif
               };
  };
  ATT_STAMP(SL_START);
#if ATT_MARKS
  for (int i = tid; i < NIT * ATT_NMARK; i += NTHREADS) mkm[i] = 0ull;
#endif
  if (tid == 0) {
    a.bar[blockIdx.x * BAR_STRIDE] = 0u;
    if (blockIdx.x == 0) a.bar[160 * BAR_STRIDE] = 0u;
  }
  grid.sync();  // zeroing barrier (as the megakernel's first); the prologue is stamped separately
  ATT_STAMP(SL_PRO);
#if ATT_STAMPS
  if (blockIdx.x == 0 && tid == 0) a.gtv[0] = globaltimer();
#endif
  for (int it = 0; it < NIT; ++it) {
    const int l = it % L;
    const int sb = SL_IT + it * NPH;
#if ATT_MARKS
    ull* mk = &mkm[it * ATT_NMARK];
#else
    ull* mk = nullptr;
#endif
    // ---- A: dummy QKV GEMM
    att_burn(a.acyc, a.cwarps);
    SYNC(sb + 0);  // S0
    // ---- B: Wo + prefix K/V L2 prefetch (all variants), the variant's pre-B copies, dummy B tasks
    if (a.ctx & CTX_WO_PF) prefetch_l2(a.wo + (size_t)l * W * QD, (size_t)W * QD * 2);
    prefetch_l2(a.prefix_k + (size_t)l * P * HD, (size_t)P * HD * 2);
    prefetch_l2(a.prefix_v + (size_t)l * P * HD, (size_t)P * HD * 2);
    VT::B_pre(p, a, l, smb);
    att_phase_B(p, a, it);
    if (it == NIT - 1) {  // coverage / race canary for the checked layer: attn back to the sentinel, stats and opart to NaN
      const size_t gt = (size_t)blockIdx.x * NTHREADS + tid, nt = (size_t)gridDim.x * NTHREADS;
      for (size_t i = gt; i < (size_t)QROWS * HD / 2; i += nt) reinterpret_cast<unsigned*>(a.attn)[i] = 0x7f7f7f7fu;
      for (size_t i = gt; i < (size_t)ATT_NCHUNK * QROWS * HD; i += nt) reinterpret_cast<unsigned*>(a.opart)[i] = 0x7fc00000u;
      for (size_t i = gt; i < (size_t)QROWS * ATT_NCHUNK * 2; i += nt) reinterpret_cast<unsigned*>(a.stats)[i] = 0x7fc00000u;
    }
    SYNC(sb + 1);  // S1
    // ---- C
    ATT_MARK(mk, 0);
    VT::C(p, a, l, it, smb, mk);
    if (VT::kCCBarrier) grid_barrier(a.bar, ++bar_epoch);
    ATT_STAMP(sb + 2);  // S2: this CTA's end of C (+ the optional C -> C3 barrier)
    // ---- C3: Wg / Wu L2 prefetch (all variants), then the combine
    prefetch_l2(a.wg + (size_t)l * FF * W, (size_t)FF * W * 2);
    prefetch_l2(a.wu + (size_t)l * FF * W, (size_t)FF * W * 2);
    VT::C3(p, a, l, it, smb, mk);
    // ---- D (C3 -> D split barrier, S3 at its exit)
    if (a.ctx & CTX_WD_PF) prefetch_l2(a.wd + (size_t)l * W * FF, (size_t)W * FF * 2);
    {
      const WPre w = wpre(sb + 3);
      att_phase_D(p, a, a.wo + (size_t)l * W * QD, it, w);
    }
    SYNC(sb + 4);  // S4
    // ---- tail: dummy E..I with the G next-layer prefetch between the two halves
    att_burn(a.tcyc, a.cwarps);
    SYNC(sb + 5);  // S5
    if (a.ctx & CTX_NEXT_PF) {
      const int ln = (l + 1 < L) ? l + 1 : 0;
      prefetch_l2(a.wq + (size_t)ln * QD * W, (size_t)QD * W * 2);
      prefetch_l2(a.wk + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
      prefetch_l2(a.wv + (size_t)ln * KVD * W, (size_t)KVD * W * 2);
    }
    att_burn(a.tcyc, a.cwarps);
    SYNC(sb + 6);  // S6
  }
  // ---- known-delay calibration: round r = {empty barrier | spin ncal, barrier}; same stamp path as the real phases
  for (int r = 0; r < NCAL; ++r) {
    SYNC(SL_CAL + 2 * r);
#if ATT_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r] = globaltimer();
#endif
    att_spin(a.ncal);
    SYNC(SL_CAL + 2 * r + 1);
#if ATT_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r + 1] = globaltimer();
#endif
  }
  ATT_STAMP(SL_END);
#if ATT_STAMPS
  if (blockIdx.x == 0 && tid == 0) {
    a.gtv[1] = globaltimer();
    for (int i = 0; i < 2 * NCAL; ++i) a.gtv[2 + i] = gtm[i];
  }
  __syncthreads();
  for (int i = tid; i < ATT_NSLOT; i += NTHREADS) a.stamps[(size_t)blockIdx.x * ATT_NSLOT + i] = stm[i];
#endif
#if ATT_MARKS
  for (int i = tid; i < NIT * ATT_NMARK; i += NTHREADS) a.marks[(size_t)blockIdx.x * NIT * ATT_NMARK + i] = mkm[i];
#endif
#undef SYNC
#undef ATT_STAMP
}

// ------------------------------------------------------------- init / reference kernels
__global__ void att_init_bf16(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(att_u(att_hash((uint32_t)i + seed), scale));
}
// prefix mask: ~6% of the prefix keys masked (padding), the first 256 always valid
__global__ void att_init_mask(uint8_t* m) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < P; j += gridDim.x * blockDim.x) m[j] = (j < 256) || (att_hash(j + 99u) % 16u != 0u);
}
// fp32 CUDA-core reference of attention for layer l = L - 1 (the last iteration) and the 400 query rows: block = one row (token t = r / 8, head r % 8),
// thread = one key (scores) then one output dim. s_j = q.k_j / 16 in fp32 (bf16 inputs, exact), w_j = exp(s_j - max), out = sum w v / sum w.
// Tolerance from the kernels' own rounding (derived, not fitted): K0 rounds the raw score to bf16 (relative 2^-9, so |ds_j| <= |s_j| / 512 after the
// exact x/16), rounds p = exp(s - m_c) to bf16 (relative 2^-9; the row sum l_c uses the same rounded p) and the output to bf16. To first order the
// output error is sum_j w_j e_j (v_j - out) / Z with |e_j| <= (|s_j| + 1) / 512, plus the final bf16 round |out| / 512; x2 safety on both.
__global__ void att_ref(const bf16* q, const bf16* ksuf, const bf16* vsuf, const bf16* pk, const bf16* pv, const uint8_t* mask, const bf16* attn,
                        unsigned* maxerr, unsigned* maxref, unsigned* nbad, unsigned* nsent) {
  __shared__ float qs[HD], w[NK], sj[NK], red[NWARPS];
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  qs[tid] = __bfloat162float(q[(size_t)r * HD + tid]);
  __syncthreads();
  float m = -INFINITY;
  for (int j = tid; j < NK; j += NTHREADS) {
    float s = -INFINITY;
    if (j >= P || mask[j]) {
      const bf16* k = j < P ? pk + (size_t)j * HD : ksuf + (size_t)(j - P) * HD;
      float acc = 0.f;
      for (int d = 0; d < HD; ++d) acc = fmaf(qs[d], __bfloat162float(k[d]), acc);
      s = acc * 0.0625f;
    }
    sj[j] = s; m = fmaxf(m, s);
  }
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  m = red[0];
  for (int i = 1; i < NWARPS; ++i) m = fmaxf(m, red[i]);
  __syncthreads();
  float z = 0.f;
  for (int j = tid; j < NK; j += NTHREADS) { const float e = sj[j] == -INFINITY ? 0.f : expf(sj[j] - m); w[j] = e; z += e; }
  for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
  if (lane == 0) red[warp] = z;
  __syncthreads();
  z = 0.f;
  for (int i = 0; i < NWARPS; ++i) z += red[i];
  const int d = tid;
  float o = 0.f;
  for (int j = 0; j < NK; ++j) {
    const float v = __bfloat162float(j < P ? pv[(size_t)j * HD + d] : vsuf[(size_t)(j - P) * HD + d]);
    o = fmaf(w[j], v, o);
  }
  const float ref = o / z;
  float terr = 0.f;
  for (int j = 0; j < NK; ++j) {
    if (w[j] == 0.f) continue;
    const float v = __bfloat162float(j < P ? pv[(size_t)j * HD + d] : vsuf[(size_t)(j - P) * HD + d]);
    terr = fmaf(w[j] * ((fabsf(sj[j]) + 1.f) * (1.f / 512.f)), fabsf(v - ref), terr);
  }
  const float tol = 2.f * (terr / z) + 2.f * fabsf(ref) * (1.f / 512.f) + 1e-6f;
  const bf16 raw = attn[(size_t)r * HD + d];
  if (*reinterpret_cast<const unsigned short*>(&raw) == 0x7f7f) atomicAdd(nsent, 1u);  // untouched sentinel
  const float got = __bfloat162float(raw);
  const float err = fabsf(got - ref);
  atomicMax(maxerr, isfinite(err) ? __float_as_uint(err) : 0x7f800000u);
  atomicMax(maxref, __float_as_uint(fabsf(ref)));
  if (!(err <= tol)) atomicAdd(nbad, 1u);
}

// ------------------------------------------------------------- host harness
static double att_median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double att_pct(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }
#ifdef ATT_HOOKS  // additive, macro-guarded extension points (used by variant files; no effect on a build without ATT_HOOKS)
static const char* att_ext_vname(int v);  // defined by the variant file
static int att_ext_parse(const std::string& s);
#define ATT_VNAME_FALLBACK(v) att_ext_vname(v)
#else
#define ATT_VNAME_FALLBACK(v) "var"
#endif
static const char* att_vname(int v) { return v == ATTV_NULL ? "null" : v == ATTV_K0 ? "k0" : v == ATTV_CALIB ? "calib" : ATT_VNAME_FALLBACK(v); }

struct AttRec {
  int variant, idx; double ev_ms = 0, nspc = 0, c_ns = 0, stage_ns[NPH] = {0}, calib_cyc = 0, calib_gt = 0, prologue_ns = 0, closure = 0;
  double maxerr = 0, maxref = 0; unsigned nbad = 0, nsent = 0; long ord_viol = 0; double ord_min_ns = 0; bool cnt_ok = true;
};

template <class VT>
int att_main(int argc, char** argv) {
  int reps = 20, ctx = CTX_ALL, cwarps = 0, cluster = VT::kCluster; size_t smem = VT::kSmem;
  long long ncal = 35000, acyc = 5500, bcyc = 1500, tcyc = 13700;
  std::string out = ".", name = std::string(VT::kName) + (ATT_STAMPS ? "" : "_ns"), vlist = "null,k0,calib";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i]; const char* v = argv[i + 1];
    if (k == "--reps") reps = atoi(v); else if (k == "--ctx") ctx = atoi(v);
    else if (k == "--cwarps") cwarps = atoi(v); else if (k == "--ncal") ncal = atoll(v); else if (k == "--acyc") acyc = atoll(v);
    else if (k == "--bcyc") bcyc = atoll(v); else if (k == "--tcyc") tcyc = atoll(v); else if (k == "--out") out = v;
    else if (k == "--name") name = v; else if (k == "--variants") vlist = v; else if (k == "--cluster") cluster = atoi(v);
    else if (k == "--smem") smem = std::max(smem, (size_t)atoll(v));
    else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  std::vector<int> vs;
  for (size_t p = 0; p < vlist.size();) {
    size_t q = vlist.find(',', p); if (q == std::string::npos) q = vlist.size();
    const std::string s = vlist.substr(p, q - p);
    vs.push_back(s == "null" ? ATTV_NULL : s == "k0" ? ATTV_K0 : s == "calib" ? ATTV_CALIB : -1);
#ifdef ATT_HOOKS
    if (vs.back() < 0) vs.back() = att_ext_parse(s);
#endif
    if (vs.back() < 0) { fprintf(stderr, "bad variant %s\n", s.c_str()); return 2; }
    p = q + 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(att_kernel<VT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, att_kernel<VT>, NTHREADS, smem));
  fprintf(stderr, "%s: %s grid=%d occ/SM=%d stamps=%d marks=%d reps=%d ctx=%d ncal=%lld acyc=%lld bcyc=%lld tcyc=%lld cwarps=%d smem=%zu cluster=%d\n", name.c_str(),
          prop.name, grid, occ, ATT_STAMPS, ATT_MARKS, reps, ctx, ncal, acyc, bcyc, tcyc, cwarps, smem, cluster);
  if (occ < 1) { fprintf(stderr, "kernel does not fit\n"); return 1; }
  std::string jpath = out + "/" + name + ".jsonl";
  FILE* jf = fopen(jpath.c_str(), "w"); if (!jf) { perror(jpath.c_str()); return 1; }
#define JOUT(...) do { fprintf(jf, __VA_ARGS__); fprintf(stdout, __VA_ARGS__); } while (0)

  const size_t nwg = (size_t)L * FF * W, nwq = (size_t)L * QD * W, nwkv = (size_t)L * KVD * W, nwo = (size_t)L * W * QD, nwd = (size_t)L * W * FF;
  const size_t nkv = (size_t)L * P * HD;
  bf16 *wq, *wk, *wv, *wo, *wg, *wu, *wd, *pk, *pv, *qb, *ksuf, *vsuf, *attn; uint8_t* mask; float *stats, *opart, *part;
  unsigned *bar, *cnt, *sink; ull *stamps, *gtv, *marks; char* flush;
  CK(cudaMalloc(&wq, nwq * 2)); CK(cudaMalloc(&wk, nwkv * 2)); CK(cudaMalloc(&wv, nwkv * 2)); CK(cudaMalloc(&wo, nwo * 2));
  CK(cudaMalloc(&wg, nwg * 2)); CK(cudaMalloc(&wu, nwg * 2)); CK(cudaMalloc(&wd, nwd * 2));
  CK(cudaMalloc(&pk, nkv * 2)); CK(cudaMalloc(&pv, nkv * 2)); CK(cudaMalloc(&mask, P));
  CK(cudaMalloc(&qb, (size_t)T * QD * 2)); CK(cudaMalloc(&ksuf, (size_t)T * KVD * 2)); CK(cudaMalloc(&vsuf, (size_t)T * KVD * 2));
  CK(cudaMalloc(&attn, (size_t)QROWS * HD * 2));
  CK(cudaMalloc(&stats, (size_t)QROWS * ATT_NCHUNK * 2 * 4)); CK(cudaMalloc(&opart, (size_t)ATT_NCHUNK * QROWS * HD * 4));
  CK(cudaMalloc(&part, (size_t)8 * T * 2 * FF * 2));
  CK(cudaMalloc(&bar, BAR_WORDS * 4)); CK(cudaMalloc(&cnt, NCNT * CNT_STRIDE * 4)); CK(cudaMalloc(&sink, 64));
  CK(cudaMalloc(&stamps, (size_t)grid * ATT_NSLOT * 8)); CK(cudaMalloc(&gtv, (2 + 2 * NCAL) * 8));
  CK(cudaMalloc(&marks, (size_t)grid * NIT * ATT_NMARK * 8));
  const size_t flush_bytes = (size_t)2 * prop.l2CacheSize; CK(cudaMalloc(&flush, flush_bytes));
  CK(cudaMemset(part, 0, (size_t)8 * T * 2 * FF * 2)); CK(cudaMemset(bar, 0, BAR_WORDS * 4)); CK(cudaMemset(sink, 0, 64));
  CK(cudaMemset(marks, 0, (size_t)grid * NIT * ATT_NMARK * 8));
  for (bf16* wptr : {wq, wk, wv}) CK(cudaMemset(wptr, 0x3c, (wptr == wq ? nwq : nwkv) * 2));
  CK(cudaMemset(wo, 0x3c, nwo * 2)); CK(cudaMemset(wg, 0x3c, nwg * 2)); CK(cudaMemset(wu, 0x3c, nwg * 2)); CK(cudaMemset(wd, 0x3c, nwd * 2));
  att_init_bf16<<<1184, 256>>>(pk, nkv, 41, ATT_QK_SCALE); att_init_bf16<<<1184, 256>>>(pv, nkv, 42, 1.0f);
  att_init_mask<<<8, 256>>>(mask);
  CK(cudaDeviceSynchronize());
  unsigned *dref; CK(cudaMalloc(&dref, 16));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  AttArgs args{wq, wk, wv, wo, wg, wu, wd, pk, pv, mask, qb, ksuf, vsuf, attn, stats, opart, part, bar, cnt, sink, stamps, gtv, marks,
               0, ctx, FUSE_DEFAULT, cwarps, acyc, bcyc, tcyc, ncal};
#ifdef ATT_HOOKS
  VT::prepare(args, grid);
#endif
  std::vector<AttRec> recs;
  std::vector<ull> st((size_t)grid * ATT_NSLOT), gt(2 + 2 * NCAL);
  const std::string rawdir = out + "/raw"; { std::string c = "mkdir -p " + rawdir; if (system(c.c_str())) return 1; }

  auto run = [&](int variant, int idx, bool record) {
    CK(cudaMemsetAsync(flush, 0, flush_bytes));
    CK(cudaMemsetAsync(attn, 0x7f, (size_t)QROWS * HD * 2));
    CK(cudaMemsetAsync(cnt, 0, NCNT * CNT_STRIDE * 4));
    CK(cudaMemsetAsync(stamps, 0, (size_t)grid * ATT_NSLOT * 8));
    args.variant = variant;
    void* kargs[] = {&args};
    CK(cudaEventRecord(e0));
    if (cluster > 1) {  // cooperative launch with clusters of `cluster` CTAs (grid stays 148)
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(grid); lc.blockDim = dim3(NTHREADS); lc.dynamicSmemBytes = smem; lc.stream = 0;
      cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
      at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = cluster; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
      lc.attrs = at; lc.numAttrs = 2;
      CK(cudaLaunchKernelEx(&lc, att_kernel<VT>, args));
    } else {
      CK(cudaLaunchCooperativeKernel((void*)att_kernel<VT>, dim3(grid), dim3(NTHREADS), kargs, smem, 0));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    if (!record) return;
    // fp32 reference of the last layer (attn after iteration 179 = layer 17; q / ksuf / vsuf are the last iteration's)
    CK(cudaMemset(dref, 0, 16));
    att_ref<<<QROWS, NTHREADS>>>(qb, ksuf, vsuf, pk + (size_t)(L - 1) * P * HD, pv + (size_t)(L - 1) * P * HD, mask, attn, dref, dref + 1, dref + 2, dref + 3);
    unsigned hr[4]; CK(cudaMemcpy(hr, dref, 16, cudaMemcpyDeviceToHost));
    AttRec r; r.variant = variant; r.idx = idx; r.ev_ms = ms;
    r.maxerr = *reinterpret_cast<float*>(&hr[0]); r.maxref = *reinterpret_cast<float*>(&hr[1]); r.nbad = hr[2]; r.nsent = hr[3];
    if (variant == ATTV_K0) {  // counter accounting: every unit of every iteration released its row group's counter exactly once
      std::vector<unsigned> hc(NCNT * CNT_STRIDE); CK(cudaMemcpy(hc.data(), cnt, hc.size() * 4, cudaMemcpyDeviceToHost));
      for (int rg = 0; rg < ATT_NRG; ++rg) r.cnt_ok &= hc[(CNT_ATT + rg) * CNT_STRIDE] == (unsigned)NIT * ARR_ATT;
    }
#if ATT_STAMPS
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
    r.c_ns = r.stage_ns[2] + r.stage_ns[3];  // B->C barrier exit .. C3->D barrier exit
    std::vector<double> pe, pd, ge, gd;
    for (int q = 0; q < NCAL; ++q) {
      const int a0 = SL_CAL + 2 * q;
      pe.push_back((double)(s0[a0] - s0[a0 - 1])); pd.push_back((double)(s0[a0 + 1] - s0[a0]));
      if (q > 0) ge.push_back((double)(gt[2 + 2 * q] - gt[2 + 2 * q - 1]));
      gd.push_back((double)(gt[2 + 2 * q + 1] - gt[2 + 2 * q]));
    }
    r.calib_cyc = att_median(pd) - att_median(pe);
    r.calib_gt = (att_median(gd) - att_median(ge)) / ((double)ncal * r.nspc);  // canary: globaltimer readback / expected (1.0 = fine)
    r.ord_min_ns = 1e18;
    for (int c = 0; c < grid; ++c)
      for (int i = 1; i < ATT_NSLOT; ++i) {
        const long long d = (long long)(st[(size_t)c * ATT_NSLOT + i] - st[(size_t)c * ATT_NSLOT + i - 1]);
        if (d < 0) ++r.ord_viol;
        r.ord_min_ns = std::min(r.ord_min_ns, d * r.nspc);
      }
    char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%s_%03d.cta0.u64", rawdir.c_str(), name.c_str(), att_vname(variant), idx);
    FILE* f = fopen(fn, "wb"); fwrite(s0, 8, ATT_NSLOT, f); fclose(f);
    if (idx >= reps - 2) {  // all-CTA stamps (and marks) for the last two launches of each variant
      snprintf(fn, sizeof fn, "%s/%s_%s_%03d.allcta.u64", rawdir.c_str(), name.c_str(), att_vname(variant), idx);
      f = fopen(fn, "wb"); fwrite(st.data(), 8, st.size(), f); fclose(f);
#if ATT_MARKS
      std::vector<ull> mkh((size_t)grid * NIT * ATT_NMARK); CK(cudaMemcpy(mkh.data(), marks, mkh.size() * 8, cudaMemcpyDeviceToHost));
      snprintf(fn, sizeof fn, "%s/%s_%s_%03d.marks.u64", rawdir.c_str(), name.c_str(), att_vname(variant), idx);
      f = fopen(fn, "wb"); fwrite(mkh.data(), 8, mkh.size(), f); fclose(f);
#endif
    }
#endif
#ifdef ATT_HOOKS
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
  auto gate = [&](const AttRec& r, const char* g, bool pass, const char* valfmt, double a, double b = 0, double c = 0) {
    char vb[256]; snprintf(vb, sizeof vb, valfmt, a, b, c);
    JOUT("{\"launch\":%d,\"variant\":\"%s\",\"gate\":\"%s\",\"pass\":%s,\"val\":%s}\n", r.idx, att_vname(r.variant), g, pass ? "true" : "false", vb);
  };
  double nspc_med = 0; { std::vector<double> all; for (auto& r : recs) all.push_back(r.nspc); nspc_med = att_median(all); }
  auto var_nspc = [&](int v) { std::vector<double> x; for (auto& r : recs) if (r.variant == v) x.push_back(r.nspc); return att_median(x); };
  int nfail = 0;
  for (auto& r : recs) {
    JOUT("{\"rec\":\"launch\",\"name\":\"%s\",\"launch\":%d,\"variant\":\"%s\",\"event_ms\":%.6f,\"nspc\":%.6f,\"prologue_ns\":%.1f,"
         "\"stage_ns\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],\"C_ns\":%.1f,\"ctx\":%d,\"cwarps\":%d,\"acyc\":%lld,\"bcyc\":%lld,\"tcyc\":%lld,\"smem\":%zu,\"cluster\":%d}\n",
         name.c_str(), r.idx, att_vname(r.variant), r.ev_ms, r.nspc, r.prologue_ns, r.stage_ns[0], r.stage_ns[1], r.stage_ns[2], r.stage_ns[3],
         r.stage_ns[4], r.stage_ns[5], r.stage_ns[6], r.c_ns, ctx, cwarps, acyc, bcyc, tcyc, smem, cluster);
    const bool produces = r.variant != ATTV_NULL && r.variant != ATTV_CALIB;
    // sentinel coverage: a producing variant must have overwritten every attn element of the checked layer (attn is reset to the sentinel in the last
    // iteration's B phase); null / calib must leave all 400 x 256 untouched (the reset worked and nothing else writes attn).
    const bool sp = produces ? r.nsent == 0 : r.nsent == (unsigned)(QROWS * HD);
    gate(r, "sentinel", sp, "{\"nsent\":%.0f,\"expected\":%.0f}", (double)r.nsent, produces ? 0.0 : (double)(QROWS * HD)); nfail += !sp;
    // work: a producing variant matches the fp32 reference within the derived tolerance (NaN-poisoned stats / opart and unwritten attn fail it)
    const bool wp = produces ? (r.nbad == 0 && std::isfinite(r.maxerr)) : true;
    gate(r, "work", wp, "{\"maxerr\":%.6g,\"maxref\":%.6g,\"nbad\":%.0f}", r.maxerr, r.maxref, (double)r.nbad); nfail += !wp;
    if (r.variant == ATTV_K0) { gate(r, "counters", r.cnt_ok, "%.0f", r.cnt_ok ? 1.0 : 0.0); nfail += !r.cnt_ok; }
#if ATT_STAMPS
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
    if (v == ATTV_NULL) continue;
    std::vector<double> evv, evn, l1v, l1n, ev_pair, allv, alln;
    std::vector<double> st_v[NPH], st_n[NPH];
    for (auto& r : recs) {
      double tot = 0; for (int k = 0; k < NPH; ++k) tot += r.stage_ns[k];
      if (r.variant == v) { evv.push_back(r.ev_ms); l1v.push_back(r.c_ns); allv.push_back(tot); for (int k = 0; k < NPH; ++k) st_v[k].push_back(r.stage_ns[k]); }
      if (r.variant == ATTV_NULL) { evn.push_back(r.ev_ms); l1n.push_back(r.c_ns); alln.push_back(tot); for (int k = 0; k < NPH; ++k) st_n[k].push_back(r.stage_ns[k]); }
    }
    if (evn.empty()) continue;
    const double L0 = (att_median(evv) - att_median(evn)) * 1e3 / NIT;  // us per layer iteration
    for (size_t i = 0; i < std::min(evv.size(), evn.size()); ++i) ev_pair.push_back((evv[i] - evn[i]) * 1e3 / NIT);
#if ATT_STAMPS
    const double cdev = var_nspc(v) / var_nspc(ATTV_NULL) - 1.0;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"clock_cross\",\"pass\":%s,\"val\":{\"nspc\":%.6f,\"null_nspc\":%.6f,\"dev\":%.5f}}\n",
         att_vname(v), fabs(cdev) < 0.015 ? "true" : "false", var_nspc(v), var_nspc(ATTV_NULL), cdev);
    const double Lwin = (att_median(l1v) - att_median(l1n)) / 1e3;  // attention window only (B->C barrier exit .. C3->D barrier exit)
    const double Lall = (att_median(allv) - att_median(alln)) / 1e3;  // all seven stage periods of the iteration
    const bool g3 = fabs(L0 - Lall) < 0.2;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"delta_agreement\",\"pass\":%s,\"val\":{\"L0_us\":%.4f,\"L1_all_us\":%.4f,\"diff_us\":%.4f,"
         "\"L1_window_us\":%.4f,\"outside_window_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f,\"C_window_us\":%.4f,\"null_C_window_us\":%.4f,"
         "\"stage_delta_us\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]}}\n",
         att_vname(v), g3 ? "true" : "false", L0, Lall, L0 - Lall, Lwin, Lall - Lwin, att_pct(ev_pair, 0.1), att_pct(ev_pair, 0.9), att_median(l1v) / 1e3, att_median(l1n) / 1e3,
         (att_median(st_v[0]) - att_median(st_n[0])) / 1e3, (att_median(st_v[1]) - att_median(st_n[1])) / 1e3, (att_median(st_v[2]) - att_median(st_n[2])) / 1e3,
         (att_median(st_v[3]) - att_median(st_n[3])) / 1e3, (att_median(st_v[4]) - att_median(st_n[4])) / 1e3, (att_median(st_v[5]) - att_median(st_n[5])) / 1e3,
         (att_median(st_v[6]) - att_median(st_n[6])) / 1e3);
    nfail += !g3;
#else
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"L0_delta\",\"pass\":true,\"val\":{\"L0_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f}}\n",
         att_vname(v), L0, att_pct(ev_pair, 0.1), att_pct(ev_pair, 0.9));
#endif
  }
  for (int v : vs) {
    std::vector<double> ev; for (auto& r : recs) if (r.variant == v) ev.push_back(r.ev_ms);
    JOUT("{\"rec\":\"summary\",\"name\":\"%s\",\"variant\":\"%s\",\"stamps\":%d,\"n\":%zu,\"event_ms_med\":%.6f,\"event_ms_p10\":%.6f,\"event_ms_p90\":%.6f,"
         "\"per_layer_us_med\":%.4f}\n", name.c_str(), att_vname(v), ATT_STAMPS, ev.size(), att_median(ev), att_pct(ev, 0.1), att_pct(ev, 0.9), att_median(ev) * 1e3 / NIT);
  }
  JOUT("{\"rec\":\"done\",\"name\":\"%s\",\"failed_gates\":%d,\"nspc_median\":%.6f}\n", name.c_str(), nfail, nspc_med);
  fclose(jf);
  return 0;
}
