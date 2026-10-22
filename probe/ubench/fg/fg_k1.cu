// fg_k1.cu: K1 = full-tcgen05 fused F+G for the standalone harness (fg_common.cuh, built with -DFG_HOOKS for the extension points:
// variant names, variant-owned E, host prepare / post_run; -DFG_CLUSTER=2 additionally launches cooperative + cluster 2).
// Design (plan K1): swap-AB, weights on the MMA M side. Each CTA b < 128 owns 32 features x (gate, up) = 64 weight rows over the full K = 1024.
// The weight TMA box is 4D {64 k, 8 rows, 2 (gate|up via the stride between the two weight arrays), 4 blocks}: smem rows come out as
// [blk][gate 8 rows][up 8 rows], so in the M = 64 TMEM layout (row m at lane (m/16)*32 + m%16) thread (lane) holds gate row g and up row g+8 of the
// same feature for the same tokens: gelu*up is register-local. N = 56 tokens (50 + pad). Accumulator in TMEM (64 columns). No F->G barrier, no
// partials; act[50][4096] bf16 is written by the epilogue; G's next-layer Wq/Wk/Wv L2 prefetch is issued by the harness for every variant.
// Warp roles (256 threads): w0 lane0 TMA producer; w1 TMEM alloc/dealloc + elected MMA issuer; w2..w7 stage the adaRMS-normed activation tile
// into the UMMA SW128 K-major smem layout (non-NF) or w2,w3 compute rs (NF); w4..w7 are the epilogue (lane quadrant = warp%4) after staging.
// Per-stage tcgen05.commit -> empty mbarrier, final commit -> epilogue. First D weight stages are issued before the E->F barrier (PRE).
// Variants (runtime ids, see K1_LIST): ring stage KB x 8 KB, depth D, NF (norm folded: E writes A' = bf16(h*(1+sc)) + row 50 = shift, TMA
// brings A' straight to smem, rs applied in the epilogue), PRE.
// Build (login node):
//   nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DFG_HOOKS fg_k1.cu -o fg_k1 -lcuda
//   unstamped twin (gate 2): add -DFG_STAMPS=0 -> fg_k1_ns          cluster build: add -DK1_CL2 -DFG_CLUSTER=2 -> fg_k1c
// Run: fg_k1 --variants null,k0?,<names> ...  (k0 is not in this binary: the variant ids are null, calib, k1_*)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <string>
#include "fg_common.cuh"

// ------------------------------------------------------------------------------------------------ variant table
// X(idx, name, KB (K64 chunks per ring stage), D (ring depth), NF, PRE (first stages issued before the E->F barrier))
#define K1_LIST(X) \
  X(0, "k1_stage_s16_d3", 2, 3, 2) \
  X(1, "k1_nf_s16_d3", 2, 3, 3) \
  X(2, "k1_nf_ach_s16_d3", 2, 3, 7) \
  X(3, "k1_nf_e8_a51_s16_d3", 2, 3, 27) \
  X(4, "k1_nf_e8_a51_fast_ts_s16_d3", 2, 3, 3099) \
  X(5, "k1_nf_e8_a51_fast_ts_s32_d2", 4, 2, 3099) \
  X(6, "k1_nf_e8_a51_fast_ts_s16_d5", 2, 5, 3099) \
  X(7, "k1_nf_e8_a51_fast_ts_nopre", 2, 3, 3097) \
  X(8, "k1_nf_e8_a51_fast_ts_warm", 2, 3, 7195) \
  X(9, "cl2_s16_d3", 2, 3, 16384 + 1 + 2) \
  X(10, "cl2_s16_d3_fast_ts", 2, 3, 16384 + 1 + 2 + 1024 + 2048) \
  X(11, "cl2_s32_d4_fast_ts", 4, 4, 16384 + 1 + 2 + 1024 + 2048) \
  X(12, "cl2_s64_d2_fast_ts", 8, 2, 16384 + 1 + 2 + 1024 + 2048) \
  X(13, "cl2_s128_d1_fast_ts", 16, 1, 16384 + 1 + 2 + 1024 + 2048) \
  X(14, "cl2_s128_d1_fast_ts_nopre", 16, 1, 16384 + 1 + 1024 + 2048) \
  X(15, "cl2_s128_d1_fast_ts_warm", 16, 1, 16384 + 1 + 2 + 1024 + 2048 + 4096) \
  X(16, "cl2_epre_s32_d4_fast_ts", 4, 4, 16384 + 1 + 32768 + 1024 + 2048) \
  X(17, "cl2_s128_d1_ts", 16, 1, 16384 + 1 + 2 + 2048)
constexpr int FGV_K1 = 16;
#define X(i, n, KB, D, FL) +1
constexpr int K1_N = 0 K1_LIST(X);
#undef X

static const char* fg_ext_vname(int v) {
#define X(i, n, KB, D, FL) if (v == FGV_K1 + i) return n;
  K1_LIST(X)
#undef X
  return "var";
}
static int fg_ext_parse(const std::string& s) {
#define X(i, n, KB, D, FL) if (s == n) return FGV_K1 + i;
  K1_LIST(X)
#undef X
  return -1;
}

// ------------------------------------------------------------------------------------------------ PTX helpers (from u12.cu)
__constant__ __align__(64) CUtensorMap c_tmW;   // 4D weight map (gate|up interleaved boxes)
__constant__ __align__(64) CUtensorMap c_tmA;   // 2D map over g_Ap (NF activation tile), box 56 rows
__constant__ __align__(64) CUtensorMap c_tmA51; // same, box 51 rows
__constant__ __align__(64) CUtensorMap c_tmA32; // CL2: box 32 rows (CTA rank 0 half of the pair's B), dims {W, 64}
__constant__ __align__(64) CUtensorMap c_tmA20; // CL2: box 20 rows (rank 1: 19 tokens + shift)
__constant__ int c_gate_first;                  // 1 if the array at the lower address (map base) is the gate array
__device__ __align__(1024) bf16 g_Ap[64 * W];   // NF: A' = bf16(h*(1+sc)) rows 0..49, row 50 = shift, rows 51..55 zero
__device__ unsigned g_k1_to;                    // mbarrier wait timeouts
__device__ unsigned g_k1_cnt[148];              // F calls per CTA (K1 variants only; 180 per launch, so index = count % 180)
constexpr int K1_NSTEP = NSTEP, K1_NTS = 16;
__device__ ull g_k1_ts[148 * K1_NSTEP * K1_NTS];  // per-CTA stage stamps of the layer-17 iterations (stamped builds)

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
#ifdef K1_DBG
  for (ull i = 0; i < (1ull << 16) && !ok; ++i)
#else
  for (ull i = 0; i < (1ull << 22) && !ok; ++i)
#endif
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) {
    atomicAdd(&g_k1_to, 1u);
#ifdef K1_DBG
    if (atomicAdd(&g_k1_to, 0u) < 40) printf("TIMEOUT cta %d warp %d bar %x par %d\n", blockIdx.x, threadIdx.x >> 5, b, par);
#endif
  }
}
__device__ __forceinline__ void tma4d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory"); }
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void mma_tc(uint32_t d, ull da, ull db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc64(uint32_t slot_smem) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot_smem));
  // no relinquish_alloc_permit: the CTA allocates again every layer iteration
}
__device__ __forceinline__ void tmem_free64(uint32_t tb) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tb)); }
__device__ __forceinline__ ull sdesc(uint32_t addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ uint32_t idesc(int N, int M) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
// gelu_tanh(x) = 0.5 x (1 + tanh z) = x * sigmoid(2 z), z = sqrt(2/pi) (x + 0.044715 x^3): one ex2 + one rcp instead of the software tanhf
__device__ __forceinline__ float gelu_fast(float x) {
  const float z2 = 1.5957691216057308f * fmaf(0.044715f * x, x * x, x);
  return __fdividef(x, 1.0f + __expf(-z2));
}
__device__ __forceinline__ bool elect_one() { uint32_t e; asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(e)); return e != 0; }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void st_shared_v4(uint32_t a, uint4 v) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void st_shared_b16(uint32_t a, unsigned short v) { asm volatile("st.shared.b16 [%0], %1;" ::"r"(a), "h"(v) : "memory"); }
__device__ __forceinline__ float4 ld_shared_f4(uint32_t a) { float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); return v; }

// ------------------------------------------------------------------------------------------------ shared state (file scope: one copy for all variants)
__shared__ __align__(8) ull k1_bar[48];    // full[0..8) @0, empty[0..8) @8, bready[16] @16, acc @32, rsr @33, abar @34
__shared__ uint32_t k1_tslot;
__shared__ __align__(16) float k1_rs[64];
__shared__ __align__(16) float k1_nm[2 * W];   // non-NF: (1 + scale), shift for this layer
__shared__ ull k1_ts_s[K1_NTS];

constexpr int B_ATOM = 7168;                 // 56 rows x 128 B per K64 chunk
constexpr int B_BYTES = 16 * B_ATOM;         // 112 KB
constexpr int K1_NF_MAXCTA = 128;
#ifdef K1_CL2
constexpr size_t K1_RING_MAX = 131072;  // CL2: B tile is 64 KB, so the whole 128 KB weight tile of a CTA fits next to it
#else
constexpr size_t K1_RING_MAX = 81920;
#endif
#ifdef K1_CL2
constexpr size_t K1_SMEM = 16 * 4096 + K1_RING_MAX + 1024;
#else
constexpr size_t K1_SMEM = B_BYTES + K1_RING_MAX + 1024;
#endif

#if FG_STAMPS && !defined(K1_NOSTAMP)
#define K1_STAMP(i) do { k1_ts_s[(i)] = clock64(); } while (0)
#else
#define K1_STAMP(i) do {} while (0)
#endif
// stamp slots
enum { TS_ENTRY = 0, TS_PREISS, TS_BAR, TS_FULL0, TS_B0, TS_MMA0, TS_MMAEND, TS_EPI0, TS_LDDONE, TS_STDONE, TS_END, TS_BLAST, TS_ISSUED, TS_RS, TS_COMP, TS_COMP2 };

// NF variant of E: residual_tasks + A' = bf16(h * (1 + scale_F)) (+ row 50 = shift_F) for the TMA'd activation tile.
template <int S, bool WF, bool CL2 = false>
__device__ __forceinline__ void k1_E_nf(const Params& p, const float* gate, const float* modF) {
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
    *reinterpret_cast<__nv_bfloat162*>(g_Ap + (CL2 && t >= 31 ? t + 1 : t) * W + c) = __floats2bfloat162_rn(__fmul_rn(h0, __fadd_rn(1.0f, sc.x)), __fmul_rn(h1, __fadd_rn(1.0f, sc.y)));
  }
  if (blockIdx.x == 0)  // row 50 = shift (the MMA then produces W . shift in token column 50 for free)
    for (int i = threadIdx.x; i < W / 2; i += NTHREADS) {
      const float2 sh = __ldcg(reinterpret_cast<const float2*>(modF + W + 2 * i));
      if (CL2) {  // the shift row sits in both halves of the pair's B (row 31 of rank 0's tile, row 19 of rank 1's = g_Ap row 51)
        *reinterpret_cast<__nv_bfloat162*>(g_Ap + 31 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
        *reinterpret_cast<__nv_bfloat162*>(g_Ap + 51 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
      } else *reinterpret_cast<__nv_bfloat162*>(g_Ap + 50 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
    }
  if (WF) asm volatile("fence.proxy.async.global;" ::: "memory");  // generic-proxy writes -> visible to the TMA (async proxy) after the barrier
}

// ------------------------------------------------------------------------------------------------ the F phase
// FL bits: NF 1, PRE 2 (weights pre-issued before the E->F barrier), ACH 4 (NF: per-K64-chunk A' barriers so the MMA starts on chunk 0),
//          E8 8 (epilogue on all 8 warps, column split), A51 16 (A' box 51 rows: rows 51..55 of the tile are never read),
//          NOWF 256 (E: no writer-side proxy fence), CF 512 (consumer-side proxy fence before the A' TMA), FASTGELU 1024 (x*sigmoid(2z) via ex2/rcp),
//          TSTORE 2048 (epilogue results transposed through per-warp smem, one 16 B global store per lane)
constexpr int FL_NF = 1, FL_PRE = 2, FL_ACH = 4, FL_E8 = 8, FL_A51 = 16, FL_DBG_NOGELU = 64, FL_DBG_NOST = 128, FL_NOWF = 256, FL_CF = 512, FL_FASTGELU = 1024, FL_TSTORE = 2048, FL_WARM = 4096, FL_CL2 = 16384, FL_EPRE = 32768;  // EPRE (CL2): barrier init, TMEM alloc, cluster handshake and the weight TMAs are issued at the start of E (weights do not depend on E)  // DBG: timing experiments only (work gate fails)
template <int KB, int D, int FL>
__device__ __noinline__ void k1_F(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
  constexpr bool NF = FL & FL_NF, PRE = FL & FL_PRE, ACH = FL & FL_ACH, E8 = FL & FL_E8, A51 = FL & FL_A51;
  constexpr int NST = 16 / KB, SB = KB * 8192, PD = 8, ABOX = A51 ? 51 : 56;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  if (cta >= K1_NF_MAXCTA) { wpre_arrive(w); wpre_wait(w); return; }  // 20 of the 148 SMs have no tile in the 128-tile design
  const uint32_t raw = su32(smb), base = (raw + 1023) & ~1023u;
#if FG_STAMPS
  unsigned k1c = 0;  // stamp slot counter, loaded at entry (volatile asm) so its latency is off the critical path
  if (tid == 0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(k1c) : "l"(&g_k1_cnt[cta]) : "memory");
#endif
  const uint32_t bB = base, bW = base + B_BYTES;
  const uint32_t bf = su32(k1_bar), bfull = bf, bempty = bf + 64, bready = bf + 128, bacc = bf + 256, brs = bf + 264, babar = bf + 272;
  const CUtensorMap* tmW = reinterpret_cast<const CUtensorMap*>(&c_tmW);
  const CUtensorMap* tmA = reinterpret_cast<const CUtensorMap*>(A51 ? &c_tmA51 : &c_tmA);
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_ENTRY);
#endif
  if (tid == 0) {
    for (int i = 0; i < D; ++i) { mbar_init(bfull + 8 * i, 1); mbar_init(bempty + 8 * i, 1); }
    for (int i = 0; i < 16; ++i) mbar_init(bready + 8 * i, NF ? 1 : 6);
    mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) { tmem_alloc64(su32(&k1_tslot)); fence_before(); }
  wpre_arrive(w);  // CTA barrier (inits + tmem slot visible), then the release add
  auto issue_stage = [&](int g, int slot) {
    mbar_expect(bfull + 8 * slot, SB);
#pragma unroll
    for (int c = 0; c < KB; ++c) tma4d(bW + slot * SB + c * 8192, tmW, bfull + 8 * slot, (g * KB + c) * 64, 0, 0, l * 512 + 4 * cta);
  };
  if (PRE && tid == 0) {  // WPRE-equivalent: weights do not depend on E, so the first D stages go out before the E->F barrier completes
#pragma unroll
    for (int g = 0; g < (D < NST ? D : NST); ++g) issue_stage(g, g);
#if FG_STAMPS
    K1_STAMP(TS_PREISS);
#endif
  }
  wpre_wait(w);
  fence_after();
  const uint32_t tb = k1_tslot;
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_BAR);
#endif
  const float* modF = p.mod + (size_t)(2 * l + 1) * 3 * W;

  if (warp == 0) {
    if (lane == 0) {
      if (NF) {  // activation tile A': 16 boxes {64, ABOX}, one barrier for all (or one per chunk)
        if (FL & FL_CF) asm volatile("fence.proxy.async.global;" ::: "memory");
        if (!ACH) mbar_expect(babar, 16 * ABOX * 128);
        for (int c = 0; c < 16; ++c) {
          if (ACH) mbar_expect(bready + 8 * c, ABOX * 128);
          tma2d(bB + c * B_ATOM, tmA, ACH ? bready + 8 * c : babar, c * 64, 0);
        }
      }
      int slot = 0, lap = 0;
      int g = 0;
      if (PRE) { g = (D < NST ? D : NST); slot = g % D; lap = g / D; }
      for (; g < NST; ++g) {
        if (lap > 0) mbar_wait(bempty + 8 * slot, (lap - 1) & 1);
        issue_stage(g, slot);
        if (++slot == D) { slot = 0; ++lap; }
      }
#if FG_STAMPS
      K1_STAMP(TS_ISSUED);
#endif
    }
  } else if (warp == 1) {
    const bool el = elect_one();
    const ull dW0 = sdesc(bW), dB0 = sdesc(bB);
    const uint32_t id56 = idesc(56, 64);
    if (NF && !ACH) { mbar_wait(babar, 0);
#if FG_STAMPS
      if (el) K1_STAMP(TS_B0);
#endif
    }
    int slot = 0, lap = 0;
#pragma unroll 1
    for (int g = 0; g < NST; ++g) {
      mbar_wait(bfull + 8 * slot, lap & 1);
#if FG_STAMPS
      if (el && g == 0) K1_STAMP(TS_FULL0);
#endif
      fence_after();
#pragma unroll
      for (int c = 0; c < KB; ++c) {
        const int kc = g * KB + c;
        if (!NF || ACH) {
          mbar_wait(bready + 8 * kc, 0);
#if FG_STAMPS
          if (el && kc == 0) K1_STAMP(TS_B0);
#endif
        }
        if (el) {
          const ull dA = dW0 + (ull)((slot * SB + c * 8192) >> 4), dB = dB0 + (ull)((kc * B_ATOM) >> 4);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_tc(tb, dA + (ull)((kk * 32) >> 4), dB + (ull)((kk * 32) >> 4), id56, (kc | kk) != 0);
#if FG_STAMPS
          if (kc == 0) K1_STAMP(TS_MMA0);
#endif
        }
      }
      if (el) {
        if (g + 1 < NST) commit(bempty + 8 * slot);
        else {
          commit(bacc);
#if FG_STAMPS
          K1_STAMP(TS_MMAEND);
#endif
        }
      }
      if (++slot == D) { slot = 0; ++lap; }
    }
  } else if (!NF) {
    // ---- non-NF: warps 2..7 stage the adaRMS-normed tile from h
    {
      const int t2 = tid - 64, pc = t2 & 7, r0 = t2 >> 3;  // 8 lanes per row, rows r0 + 24 j
      for (int i = t2; i < 2 * W / 4; i += 192) {  // this layer's (1 + scale), shift -> smem
        float4 v = __ldcg(reinterpret_cast<const float4*>(modF) + i);
        if (i < W / 4) { v.x = __fadd_rn(1.0f, v.x); v.y = __fadd_rn(1.0f, v.y); v.z = __fadd_rn(1.0f, v.z); v.w = __fadd_rn(1.0f, v.w); }
        *reinterpret_cast<float4*>(&k1_nm[4 * i]) = v;
      }
      float rs[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int r = r0 + 24 * j;
        float s = 0.f;
        if (r < T) s = __ldcg(p.ssp + r * NSSP + pc) + __ldcg(p.ssp + r * NSSP + pc + 8);
        s += __shfl_xor_sync(0xffffffffu, s, 4); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 1);
        rs[j] = rsqrtf(__fadd_rn(s * (1.0f / W), 1e-6f));
      }
      uint4 hv[PD][3];
      auto load_chunk = [&](int kc, uint4 (&dst)[3]) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int r = r0 + 24 * j;
          dst[j] = r < T ? __ldcg(reinterpret_cast<const uint4*>(p.h + (size_t)r * W + kc * 64 + pc * 8)) : make_uint4(0, 0, 0, 0);
        }
      };
#pragma unroll
      for (int kc = 0; kc < PD; ++kc) load_chunk(kc, hv[kc]);
      named_bar(1, 192);  // k1_nm visible to all staging threads
#pragma unroll
      for (int kc = 0; kc < 16; ++kc) {
        const float4 s0 = ld_shared_f4(su32(&k1_nm[kc * 64 + pc * 8])), s1 = ld_shared_f4(su32(&k1_nm[kc * 64 + pc * 8 + 4]));
        const float4 h0 = ld_shared_f4(su32(&k1_nm[W + kc * 64 + pc * 8])), h1 = ld_shared_f4(su32(&k1_nm[W + kc * 64 + pc * 8 + 4]));
        const float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w}, sh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int r = r0 + 24 * j;
          uint4 u = hv[kc % PD][j];
          if (r < T) {
            uint32_t* wv = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              __nv_bfloat162 hv2 = *reinterpret_cast<__nv_bfloat162*>(&wv[e]);
              const float o0 = __fadd_rn(__fmul_rn(__fmul_rn(__low2float(hv2), rs[j]), sc[2 * e]), sh[2 * e]);
              const float o1 = __fadd_rn(__fmul_rn(__fmul_rn(__high2float(hv2), rs[j]), sc[2 * e + 1]), sh[2 * e + 1]);
              hv2 = __floats2bfloat162_rn(o0, o1);
              wv[e] = *reinterpret_cast<uint32_t*>(&hv2);
            }
          }
          if (r < 56) st_shared_v4(bB + kc * B_ATOM + (r >> 3) * 1024 + (r & 7) * 128 + ((pc ^ (r & 7)) << 4), u);
        }
        if (kc + PD < 16) load_chunk(kc + PD, hv[kc % PD]);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mbar_arrive(bready + 8 * kc);
      }
#if FG_STAMPS
      if (tid == 255) K1_STAMP(TS_BLAST);
#endif
    }
  } else if (warp < 4) {  // NF: rs[t] = rsqrt(sum(ssp[t][0..15]) / W + eps), serial order as the reference
    const int t = tid - 64;
    float rsv = 0.f;
    if (t < T) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < NSSP; ++j) s += __ldcg(p.ssp + t * NSSP + j);
      rsv = rsqrtf(__fadd_rn(s * (1.0f / W), 1e-6f));
    }
    k1_rs[t] = rsv;
    __syncwarp();
    if (lane == 0) mbar_arrive(brs);
#if FG_STAMPS
    if (tid == 64) K1_STAMP(TS_RS);
#endif
  }
  // ---- epilogue: lane quadrant q = warp & 3. E8: all 8 warps, warp w<4 takes columns 0..31, w>=4 columns 32..55; else warps 4..7 take all 56.
  // WARM: the epilogue code runs once per ~27 us layer iteration, so it is instruction-cache cold when the accumulator completes (measured: first
  // pass 0.7 us, second identical pass 0.06 us). With WARM each epilogue warp first runs the same loop body on dummy data (no TMEM access, no
  // stores) while it would otherwise be idle, so the real pass hits warm instructions.
  if (E8 || warp >= 4) {
    __syncwarp();
    const int q = warp & 3;
    const bool doA = !E8 || warp < 4, doBC = !E8 || warp >= 4;
    const int nrep = (FL & FL_WARM) ? 2 : 1;
    const uint32_t ta = tb + ((uint32_t)(q * 32) << 16);
    const int f = cta * 32 + q * 8 + (lane >> 2), tq = (lane & 3) * 2;
    const bool gf = c_gate_first != 0;
    const uint32_t scr = bW + warp * 1024;  // TSTORE: per-warp [token][8 features] bf16 scratch in the (now idle) weight ring
#pragma unroll 1
    for (int rr = 0; rr < nrep; ++rr) {
      int realv = (rr == nrep - 1);
      asm volatile("" : "+r"(realv));  // keep the warm and the real pass in the same code
      const bool real = realv != 0;
      if (real) {
        if (NF) mbar_wait(brs, 0);
        mbar_wait(bacc, 0);
#if FG_STAMPS
        if (tid == 128) K1_STAMP(TS_EPI0);
#endif
        fence_after();
      }
      uint32_t va[16], vb[8], vc[4];
#pragma unroll
      for (int i = 0; i < 16; ++i) va[i] = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) vb[i] = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) vc[i] = 0;
      if (real) {
        if (doA)
          asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                       : "=r"(va[0]), "=r"(va[1]), "=r"(va[2]), "=r"(va[3]), "=r"(va[4]), "=r"(va[5]), "=r"(va[6]), "=r"(va[7]), "=r"(va[8]), "=r"(va[9]), "=r"(va[10]),
                         "=r"(va[11]), "=r"(va[12]), "=r"(va[13]), "=r"(va[14]), "=r"(va[15]) : "r"(ta));
        if (doBC)
          asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                       : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3]), "=r"(vb[4]), "=r"(vb[5]), "=r"(vb[6]), "=r"(vb[7]) : "r"(ta + 32));
        if (doBC || NF)  // NF: every warp needs token column 50 (W . shift) from the last 8-column batch
          asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(vc[0]), "=r"(vc[1]), "=r"(vc[2]), "=r"(vc[3]) : "r"(ta + 48));
        wait_ld();
#if FG_STAMPS
        if (tid == 128) K1_STAMP(TS_LDDONE);
#endif
        fence_before();
      }
      float sh0 = 0.f, sh1 = 0.f;  // W . shift of the first / second array row of this thread's feature
      if (NF) {  // token column 50 (g = 6, lane&3 == 1, e = 0) holds W . shift
        const int src = (lane & ~3) | 1;
        sh0 = __shfl_sync(0xffffffffu, __uint_as_float(vc[0]), src);
        sh1 = __shfl_sync(0xffffffffu, __uint_as_float(vc[2]), src);
      }
      // All math first (register-only: smem rs values are preloaded so the compiler need not order them against the global stores), then the stores.
      float rsr[14];
#pragma unroll
      for (int g = 0; g < 7; ++g)
#pragma unroll
        for (int e = 0; e < 2; ++e) rsr[2 * g + e] = NF ? k1_rs[8 * g + tq + e] : 1.0f;  // k1_rs[t >= T] = 0
      unsigned short ob[14];
      auto compute = [&](int idx, uint32_t x0, uint32_t x1) {  // x0 / x1: first / second array row of feature f, token 8*(idx/2) + tq + idx%2
        float r0v = __uint_as_float(x0), r1v = __uint_as_float(x1);
        if (NF) { r0v = fmaf(rsr[idx], r0v, sh0); r1v = fmaf(rsr[idx], r1v, sh1); }
        const float gv = gf ? r0v : r1v, uv = gf ? r1v : r0v;
        const float o = (FL & FL_DBG_NOGELU) ? gv * uv : (FL & FL_FASTGELU) ? rbf(gelu_fast(rbf(gv))) * rbf(uv) : rbf(gelu_tanh_ref(rbf(gv))) * rbf(uv);
        ob[idx] = __bfloat16_as_ushort(__float2bfloat16_rn(o));
      };
#pragma unroll
      for (int i = 0; i < 14; ++i) ob[i] = 0;
      if (doA) {
#pragma unroll
        for (int idx = 0; idx < 8; ++idx) compute(idx, va[2 * (idx & ~1) + (idx & 1)], va[2 * (idx & ~1) + 2 + (idx & 1)]);
      }
      if (doBC) {
#pragma unroll
        for (int idx = 8; idx < 12; ++idx) compute(idx, vb[2 * ((idx - 8) & ~1) + (idx & 1)], vb[2 * ((idx - 8) & ~1) + 2 + (idx & 1)]);
#pragma unroll
        for (int idx = 12; idx < 14; ++idx) compute(idx, vc[idx & 1], vc[2 + (idx & 1)]);
      }
#if FG_STAMPS
      if (real && tid == 128) K1_STAMP(TS_COMP);
#endif
#pragma unroll
      for (int idx = 0; idx < 14; ++idx) {
        const int t = 8 * (idx >> 1) + tq + (idx & 1);
        const bool mine = idx < 8 ? doA : doBC;
        if (real && mine && t < T) {
          if (FL & FL_TSTORE) st_shared_b16(scr + t * 16 + (lane >> 2) * 2, ob[idx]);
          else if (!(FL & FL_DBG_NOST) || ob[idx] == 0x3f9e) p.act[(size_t)t * FF + f] = __ushort_as_bfloat16(ob[idx]);
        }
      }
      if ((FL & FL_TSTORE) && real) {
        __syncwarp();
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
          const int tt = 32 * pass + lane;
          if ((pass == 0 ? doA : doBC) && tt < T) {
            uint4 u; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(scr + tt * 16));
            *reinterpret_cast<uint4*>(p.act + (size_t)tt * FF + cta * 32 + q * 8) = u;
          }
        }
      }
#if FG_STAMPS
      if (real && tid == 128) K1_STAMP(TS_STDONE);
#endif
    }
  }
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64(tb); }
#if FG_STAMPS
  if (tid == 0) {
    K1_STAMP(TS_END);
    g_k1_cnt[cta] = k1c + 1;
    const int c = (int)(k1c % (unsigned)NIT);
    if (l == L - 1) {
      ull* dst = g_k1_ts + ((size_t)cta * K1_NSTEP + c / L) * K1_NTS;
      for (int i = 0; i < K1_NTS; ++i) dst[i] = k1_ts_s[i];
    }
  }
#endif
}


#ifdef K1_CL2
// ------------------------------------------------------------------------------------------------ CL2: cta_group::2 pair (M = 128, N = 64)
// Cluster of 2 (rank = cta & 1). Each CTA brings its own 64 weight rows (A, M half) and 32 token rows of the activation tile (B, N half): CTA0 tokens
// 0..30 + shift, CTA1 tokens 31..49 + shift (g_Ap rows 0..31 / 32..51). Rank 0's elected thread issues the pair MMA; TMA of both CTAs signal rank 0's
// full barrier (cta_group::2 form, peer bit cleared); tcgen05.commit multicasts to both CTAs. D layout (probe u13): CTA r holds its own 64 rows;
// lane m' (0..63) = row m', tokens = B-half 0 columns 0..31; lanes 64..127 = the same rows, B-half 1 columns (stored at columns 0..31).
__device__ __forceinline__ void tma4d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory"); }
__device__ __forceinline__ void tma2d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void mma_tc2(uint32_t d, ull da, ull db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit2(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"((unsigned short)3) : "memory"); }
__device__ __forceinline__ void tmem_alloc64c2(uint32_t slot) { asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot)); }
__device__ __forceinline__ void tmem_free64c2(uint32_t tb) { asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 64;" ::"r"(tb)); }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory"); }

extern __shared__ __align__(1024) unsigned char k1_dyn[];
__device__ __forceinline__ void k1_issue_w2(uint32_t bW, uint32_t bfull, int KB, int SB, int g, int slot, int rank, int l, int cta) {
  if (rank == 0) mbar_expect(bfull + 8 * slot, 2 * SB);
  for (int c = 0; c < KB; ++c) tma4d2(bW + slot * SB + c * 8192, reinterpret_cast<const CUtensorMap*>(&c_tmW), bfull + 8 * slot, (g * KB + c) * 64, 0, 0, l * 512 + 4 * cta);
}
// EPRE: everything that does not depend on E, issued at the start of E
template <int KB, int D>
__device__ __noinline__ void k1_pre2(int l) {
  const int tid = threadIdx.x, warp = tid >> 5, cta = blockIdx.x, rank = cta & 1;
  if (cta >= K1_NF_MAXCTA) return;
  constexpr int NST = 16 / KB, SB = KB * 8192, BA = 4096;
  const uint32_t base = (su32(k1_dyn) + 1023) & ~1023u, bW = base + 16 * BA;
  const uint32_t bf = su32(k1_bar), bfull = bf, bempty = bf + 64, bacc = bf + 256, brs = bf + 264, babar = bf + 272;
  if (tid == 0) {
#if FG_STAMPS
    K1_STAMP(TS_ENTRY);
#endif
    for (int i = 0; i < D; ++i) { mbar_init(bfull + 8 * i, 1); mbar_init(bempty + 8 * i, 1); }
    mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) { tmem_alloc64c2(su32(&k1_tslot)); fence_before(); }
#ifdef K1_DBG
  if (cta < 2 && l == 0 && (tid & 31) == 0) printf("pre2 cta %d warp %d before arrive\n", cta, warp);
#endif
  asm volatile("barrier.cluster.arrive.release;" ::: "memory");
#ifdef K1_DBG
  if (cta < 2 && l == 0 && (tid & 31) == 0) printf("pre2 cta %d warp %d after arrive\n", cta, warp);
#endif
  if (warp == 0) {  // the whole warp waits (a lone lane waiting on the cluster barrier never completes)
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
    if (tid == 0) {
      fence_after();
      for (int g = 0; g < (D < NST ? D : NST); ++g) k1_issue_w2(bW, bfull, KB, SB, g, g, rank, l, cta);
#if FG_STAMPS
      K1_STAMP(TS_PREISS);
#endif
    }
  }
}

template <int KB, int D, int FL>
__device__ __noinline__ void k1_F2(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
  constexpr bool PRE = FL & FL_PRE, EPRE = FL & FL_EPRE;
  constexpr int NST = 16 / KB, SB = KB * 8192, BA = 4096;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x, rank = cta & 1;
  if (cta >= K1_NF_MAXCTA) { wpre_arrive(w); wpre_wait(w); return; }
  const uint32_t raw = su32(smb), base = (raw + 1023) & ~1023u;
#if FG_STAMPS
  unsigned k1c = 0;  // stamp slot counter, loaded at entry (volatile asm) so its latency is off the critical path
  if (tid == 0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(k1c) : "l"(&g_k1_cnt[cta]) : "memory");
#endif
  const uint32_t bB = base, bW = base + 16 * BA;
  const uint32_t bf = su32(k1_bar), bfull = bf, bempty = bf + 64, bacc = bf + 256, brs = bf + 264, babar = bf + 272;
  const CUtensorMap* tmW = reinterpret_cast<const CUtensorMap*>(&c_tmW);
  const CUtensorMap* tmA = reinterpret_cast<const CUtensorMap*>(rank == 0 ? &c_tmA32 : &c_tmA20);
  if (EPRE) {  // barrier init / TMEM alloc / weight TMAs were done in k1_pre2 (start of E); warp 0 already passed the cluster wait
    if (warp != 0) asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
    fence_after();
    wpre_arrive(w);
  } else {
#if FG_STAMPS
    if (tid == 0) K1_STAMP(TS_ENTRY);
#endif
    if (tid == 0) {
      for (int i = 0; i < D; ++i) { mbar_init(bfull + 8 * i, 1); mbar_init(bempty + 8 * i, 1); }
      mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) { tmem_alloc64c2(su32(&k1_tslot)); fence_before(); }
    cluster_sync_all();  // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
    fence_after();
    wpre_arrive(w);
    if (PRE && tid == 0) {
      for (int g = 0; g < (D < NST ? D : NST); ++g) k1_issue_w2(bW, bfull, KB, SB, g, g, rank, l, cta);
#if FG_STAMPS
      K1_STAMP(TS_PREISS);
#endif
    }
  }
  wpre_wait(w);
  fence_after();
  const uint32_t tb = k1_tslot;
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_BAR);
#endif
  if (warp == 0) {
    if (lane == 0) {
      if (rank == 0) mbar_expect(babar, (32 + 20) * 128 * 16);
      for (int c = 0; c < 16; ++c) tma2d2(bB + c * BA, tmA, babar, c * 64, rank * 32);
      int slot = 0, lap = 0, g = 0;
      if (PRE || EPRE) { g = (D < NST ? D : NST); slot = g % D; lap = g / D; }
      for (; g < NST; ++g) {
        if (lap > 0) mbar_wait(bempty + 8 * slot, (lap - 1) & 1);
        k1_issue_w2(bW, bfull, KB, SB, g, slot, rank, l, cta);
        if (++slot == D) { slot = 0; ++lap; }
      }
#if FG_STAMPS
      K1_STAMP(TS_ISSUED);
#endif
    }
  } else if (warp == 1) {
    if (rank == 0) {
      const bool el = elect_one();
      const ull dW0 = sdesc(bW), dB0 = sdesc(bB);
      const uint32_t id = idesc(64, 128);
      mbar_wait(babar, 0);
#if FG_STAMPS
      if (el) K1_STAMP(TS_B0);
#endif
      int slot = 0, lap = 0;
#pragma unroll 1
      for (int g = 0; g < NST; ++g) {
        mbar_wait(bfull + 8 * slot, lap & 1);
#if FG_STAMPS
        if (el && g == 0) K1_STAMP(TS_FULL0);
#endif
        fence_after();
        if (el) {
#pragma unroll
          for (int c = 0; c < KB; ++c) {
            const int kc = g * KB + c;
            const ull dA = dW0 + (ull)((slot * SB + c * 8192) >> 4), dB = dB0 + (ull)((kc * BA) >> 4);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) mma_tc2(tb, dA + (ull)((kk * 32) >> 4), dB + (ull)((kk * 32) >> 4), id, (kc | kk) != 0);
#if FG_STAMPS
            if (kc == 0) K1_STAMP(TS_MMA0);
#endif
          }
          if (g + 1 < NST) commit2(bempty + 8 * slot);
          else {
            commit2(bacc);
#if FG_STAMPS
            K1_STAMP(TS_MMAEND);
#endif
          }
        }
        if (++slot == D) { slot = 0; ++lap; }
      }
    }
  } else if (warp < 4) {  // rs[t] = rsqrt(sum(ssp[t][0..15]) / W + eps)
    const int t = tid - 64;
    float rsv = 0.f;
    if (t < T) {
      float sx = 0.f;
#pragma unroll
      for (int j = 0; j < NSSP; ++j) sx += __ldcg(p.ssp + t * NSSP + j);
      rsv = rsqrtf(__fadd_rn(sx * (1.0f / W), 1e-6f));
    }
    k1_rs[t] = rsv;
    __syncwarp();
    if (lane == 0) mbar_arrive(brs);
#if FG_STAMPS
    if (tid == 64) K1_STAMP(TS_RS);
#endif
  }
  // ---- epilogue, all 8 warps: warp = (sub, q): TMEM lanes 32 q + 16 sub .. +15 = 16 rows of one 16-row weight block (blk), B-half h = q >> 1.
  // WARM: the epilogue code is instruction-cache cold when the accumulator completes (measured ~0.7 us first pass vs 0.06 us second). Every warp except the
  // MMA warp first runs the same code on zero data (no TMEM, no mbarrier, no stores) while it would otherwise idle, so the real pass hits warm instructions.
  {
    __syncwarp();
    const int q = warp & 3, sub = warp >> 2, h = q >> 1, blk = (q & 1) * 2 + sub;
    const uint32_t ta = tb + ((uint32_t)(q * 32 + sub * 16) << 16);
    const int f = cta * 32 + blk * 8 + (lane >> 2), tq = (lane & 3) * 2;
    const bool gf = c_gate_first != 0;
    const uint32_t scr = bW + warp * 1024;
    const int nrep = ((FL & FL_WARM) && warp != 1) ? 2 : 1;
#pragma unroll 1
    for (int rr = 0; rr < nrep; ++rr) {
      int realv = (rr == nrep - 1);
      asm volatile("" : "+r"(realv));
      const bool real = realv != 0;
      uint32_t va[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) va[i] = 0;
      if (real) {
        mbar_wait(brs, 0);
        mbar_wait(bacc, 0);
#if FG_STAMPS
        if (tid == 128) K1_STAMP(TS_EPI0);
#endif
        fence_after();
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                     : "=r"(va[0]), "=r"(va[1]), "=r"(va[2]), "=r"(va[3]), "=r"(va[4]), "=r"(va[5]), "=r"(va[6]), "=r"(va[7]), "=r"(va[8]), "=r"(va[9]), "=r"(va[10]),
                       "=r"(va[11]), "=r"(va[12]), "=r"(va[13]), "=r"(va[14]), "=r"(va[15]) : "r"(ta));
        wait_ld();
#if FG_STAMPS
        if (tid == 128) K1_STAMP(TS_LDDONE);
#endif
        fence_before();
      }
      // shift column: h = 0 -> local column 31 (g = 3, lane&3 == 3, e = 1); h = 1 -> column 19 (g = 2, lane&3 == 1, e = 1)
      const int src = (lane & ~3) | (h ? 1 : 3);
      const float sh0 = __shfl_sync(0xffffffffu, __uint_as_float(h ? va[9] : va[13]), src);
      const float sh1 = __shfl_sync(0xffffffffu, __uint_as_float(h ? va[11] : va[15]), src);
      float rsr[8];
#pragma unroll
      for (int idx = 0; idx < 8; ++idx) rsr[idx] = k1_rs[(h ? 31 : 0) + 8 * (idx >> 1) + tq + (idx & 1)];  // <= 62; zero for t >= T
      unsigned short ob[8];
#pragma unroll
      for (int idx = 0; idx < 8; ++idx) {
        const float r0v = fmaf(rsr[idx], __uint_as_float(va[2 * (idx & ~1) + (idx & 1)]), sh0), r1v = fmaf(rsr[idx], __uint_as_float(va[2 * (idx & ~1) + 2 + (idx & 1)]), sh1);
        const float gv = gf ? r0v : r1v, uv = gf ? r1v : r0v;
        const float o = (FL & FL_FASTGELU) ? rbf(gelu_fast(rbf(gv))) * rbf(uv) : rbf(gelu_tanh_ref(rbf(gv))) * rbf(uv);
        ob[idx] = __bfloat16_as_ushort(__float2bfloat16_rn(o));
      }
#if FG_STAMPS
      if (real && tid == 128) K1_STAMP(TS_COMP);
#endif
      const int nv = h ? 19 : 31;  // valid local columns
#pragma unroll
      for (int idx = 0; idx < 8; ++idx) {
        const int c = 8 * (idx >> 1) + tq + (idx & 1);
        if (real && c < nv) {
          if (FL & FL_TSTORE) st_shared_b16(scr + c * 16 + (lane >> 2) * 2, ob[idx]);
          else p.act[(size_t)((h ? 31 : 0) + c) * FF + f] = __ushort_as_bfloat16(ob[idx]);
        }
      }
      if (FL & FL_TSTORE) {
        __syncwarp();
        if (real && lane < nv) {
          uint4 u; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(scr + lane * 16));
          *reinterpret_cast<uint4*>(p.act + (size_t)((h ? 31 : 0) + lane) * FF + cta * 32 + blk * 8) = u;
        }
      }
#if FG_STAMPS
      if (real && tid == 128) K1_STAMP(TS_STDONE);
#endif
    }
  }
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64c2(tb); }
#if FG_STAMPS
  if (tid == 0) {
    K1_STAMP(TS_END);
    g_k1_cnt[cta] = k1c + 1;
    const int c = (int)(k1c % (unsigned)NIT);
    if (l == L - 1) {
      ull* dst = g_k1_ts + ((size_t)cta * K1_NSTEP + c / L) * K1_NTS;
      for (int i = 0; i < K1_NTS; ++i) dst[i] = k1_ts_s[i];
    }
  }
#endif
}
#endif  // K1_CL2

// a kernel cannot mix tcgen05 cta_group::1 and ::2 (ptxas), so the K1_CL2 binary only runs the CL2 variants (others fall back to the null F)
template <int KB, int D, int FL>
__device__ __forceinline__ void k1_dispatch(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
#ifdef K1_CL2
  if constexpr (FL & FL_CL2) k1_F2<KB, D, FL>(p, a, l, w, smb);
  else fg_null_F(w);
#else
  if constexpr (!(FL & FL_CL2)) k1_F<KB, D, FL>(p, a, l, w, smb);
  else fg_null_F(w);
#endif
}

// ------------------------------------------------------------------------------------------------ the VT
__device__ __forceinline__ int k1_fl(int v) {
#define X(i, n, KB, D, FL) if (v == FGV_K1 + i) return FL;
  K1_LIST(X)
#undef X
  return 0;
}
__device__ __forceinline__ bool k1_is_nf(int v) {
#define X(i, n, KB, D, FL) if (v == FGV_K1 + i) return (FL & 1) != 0;
  K1_LIST(X)
#undef X
  return false;
}

struct K1 {
#ifdef K1_CL2
  static constexpr const char* kName = "fg_k1c";
#else
  static constexpr const char* kName = "fg_k1";
#endif
  static constexpr bool kFgBarrier = false;
  static constexpr size_t kSmem = K1_SMEM;
  __device__ static void E(const Params& p, const FgArgs& a, int l, const float* nmodbase) {
#ifdef K1_CL2
    switch (a.variant) {
#define X(i, n, KB, D, FL) case FGV_K1 + i: if constexpr (FL & FL_EPRE) k1_pre2<KB, D>(l); break;
      K1_LIST(X)
#undef X
      default: break;
    }
#endif
    if (k1_is_nf(a.variant)) {
      if (k1_fl(a.variant) & FL_CL2) k1_E_nf<QD / KC, true, true>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W, nmodbase + (size_t)(2 * l + 1) * 3 * W);
      else if (k1_fl(a.variant) & FL_NOWF) k1_E_nf<QD / KC, false>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W, nmodbase + (size_t)(2 * l + 1) * 3 * W);
      else k1_E_nf<QD / KC, true>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W, nmodbase + (size_t)(2 * l + 1) * 3 * W);
    }
    else residual_tasks<QD / KC>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W);
  }
  __device__ static void F(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
    switch (a.variant) {
#define X(i, n, KB, D, FL) case FGV_K1 + i: k1_dispatch<KB, D, FL>(p, a, l, w, smb); return;
      K1_LIST(X)
#undef X
      default: break;
    }
    fg_null_F(w);
    if (a.variant == FGV_CALIB) fg_spin(a.ncal);
  }
  __device__ static void G(const Params&, const FgArgs&, int, unsigned, bool) {}

  // ---- host
  static void prepare(FgArgs& a, int grid) {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    const bool gate_first = (uintptr_t)a.wg < (uintptr_t)a.wu;
    const bf16* lo = gate_first ? a.wg : a.wu; const bf16* hi = gate_first ? a.wu : a.wg;
    CUtensorMap mw, ma;
    {
      cuuint64_t dims[4] = {W, 8, 2, (cuuint64_t)L * (FF / 8)}, strides[3] = {(cuuint64_t)W * 2, (cuuint64_t)((const char*)hi - (const char*)lo), (cuuint64_t)W * 2 * 8};
      cuuint32_t box[4] = {64, 8, 2, 4}, es[4] = {1, 1, 1, 1};
      CUresult r = enc(&mw, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)lo, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "weight tensor map encode failed %d (delta %zu)\n", (int)r, (size_t)((const char*)hi - (const char*)lo)); exit(1); }
    }
    void* ap; CK(cudaGetSymbolAddress(&ap, g_Ap)); CK(cudaMemset(ap, 0, 64 * W * 2));
    {
      cuuint64_t dims[2] = {W, 56}, strides[1] = {(cuuint64_t)W * 2};
      cuuint32_t box[2] = {64, 56}, es[2] = {1, 1};
      CUresult r = enc(&ma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ap, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "A tensor map encode failed %d\n", (int)r); exit(1); }
    }
    CUtensorMap ma51;
    {
      cuuint64_t dims[2] = {W, 56}, strides[1] = {(cuuint64_t)W * 2};
      cuuint32_t box[2] = {64, 51}, es[2] = {1, 1};
      CUresult r = enc(&ma51, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ap, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "A51 tensor map encode failed %d\n", (int)r); exit(1); }
    }
    CUtensorMap ma32, ma20;
    for (int k = 0; k < 2; ++k) {
      cuuint64_t dims[2] = {W, 64}, strides[1] = {(cuuint64_t)W * 2};
      cuuint32_t box[2] = {64, (cuuint32_t)(k == 0 ? 32 : 20)}, es[2] = {1, 1};
      CUresult r = enc(k == 0 ? &ma32 : &ma20, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ap, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "CL2 A tensor map encode failed %d\n", (int)r); exit(1); }
    }
    CK(cudaMemcpyToSymbol(c_tmA32, &ma32, sizeof ma32)); CK(cudaMemcpyToSymbol(c_tmA20, &ma20, sizeof ma20));
    CK(cudaMemcpyToSymbol(c_tmW, &mw, sizeof mw)); CK(cudaMemcpyToSymbol(c_tmA, &ma, sizeof ma)); CK(cudaMemcpyToSymbol(c_tmA51, &ma51, sizeof ma51));
    const int gfi = gate_first ? 1 : 0; CK(cudaMemcpyToSymbol(c_gate_first, &gfi, sizeof gfi));
    unsigned z = 0; CK(cudaMemcpyToSymbol(g_k1_to, &z, 4));
    fprintf(stderr, "k1 prepare: gate_first=%d delta=%zu grid=%d K1 variants=%d smem=%zu\n", gfi, (size_t)((const char*)hi - (const char*)lo), grid, K1_N, (size_t)kSmem);
  }
  static void post_run(int variant, int idx, int reps, const std::string& out, const std::string& name, double nspc) {
    unsigned to = 0; CK(cudaMemcpyFromSymbol(&to, g_k1_to, 4));
    if (to) { fprintf(stderr, "WARNING: %u mbarrier wait timeouts (variant %s launch %d)\n", to, fg_vname(variant), idx); unsigned z = 0; CK(cudaMemcpyToSymbol(g_k1_to, &z, 4)); }
#if FG_STAMPS
    if (variant >= FGV_K1 && idx >= reps - 2 && idx >= 0) {
      std::vector<ull> ts((size_t)148 * K1_NSTEP * K1_NTS);
      CK(cudaMemcpyFromSymbol(ts.data(), g_k1_ts, ts.size() * 8));
      char fn[512]; snprintf(fn, sizeof fn, "%s/raw/%s_%s_%03d.k1tl.u64", out.c_str(), name.c_str(), fg_vname(variant), idx);
      FILE* f = fopen(fn, "wb"); fwrite(&nspc, 8, 1, f); fwrite(ts.data(), 8, ts.size(), f); fclose(f);
    }
#endif
  }
};

int main(int argc, char** argv) { return fg_main<K1>(argc, argv); }
