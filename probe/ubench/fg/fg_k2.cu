// fg_k2.cu: K2 = redesigned F+G with HMMA (mma.sync m16n8k16) for the standalone harness (fg_common.cuh, built with -DFG_HOOKS like fg_k1.cu).
// Control for K1 (UMMA): the SAME restructured F+G, HMMA instead of tcgen05.
// Design: swap-AB (weights = m16 side, 50 tokens = 7 n8 tiles padded to 56); CTA c < 128 owns 32 features x (gate, up) = 64 weight rows over the
// FULL K = 1024 (CTAs 128..147 idle in F). Weights stream through a DEPTH-stage ring (stage = 64 rows x STAGEK; TMA boxes {64 K, 32 rows}
// SWIZZLE_128B, one box for the gate and one for the up rows). The m16 tile is 8 gate rows + 8 up rows of the same 8 features (ldmatrix row
// addresses are per lane, so no interleaved TMA is needed): c0/c1 = gate, c2/c3 = up of the same (feature, token) -> gelu*up in registers.
// Activation tile [56][1024] bf16 in smem as 16 swizzled 56x64 atoms (TMA SWIZZLE_128B layout), read with ldmatrix (conflict-free).
//   non-NF: TMA of raw h, then in-smem adaRMS norm (as gemm_phase's norm staging: (h*rs)*(1+sc)+sh, bf16), nmod / ssp by one bulk copy each.
//   NF    : E writes A' = bf16(h*(1+sc)) + row 50 = shift (same contract as K1's k1_E_nf); TMA brings A' straight in (per-chunk barriers, the
//           MMA starts on chunk 0); rs[t] and the shift column (W . shift, token column 50) are applied in the epilogue.
// Warp decomposition: MS m-groups x NS token halves x KS k16-step slices = 8 warps. No producer warp: the last warp to release a ring slot (smem
// counter) issues the refill TMA. First PRE stages are issued between the E->F arrive and wait (WPRE-equivalent). K-slice partials are reduced
// through smem (A region reused). No partials in global, no F->G barrier.
// Variant table K2_LIST below (runtime ids; one binary, ABAB-interleaved with null / calib).
// Build (login node): nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DFG_HOOKS fg_k2.cu -o fg_k2        [-DFG_STAMPS=0 -> fg_k2_ns]
//   attribution build with the in-F clock64 timeline: -DK2_TL=1 -> fg_k2_tl (NOT a gate-2 twin; used for the stage timeline only)
// Run: fg_k2 --variants null,calib,k2_... --reps 20 --out DIR
#include <cuda.h>
#include <cudaTypedefs.h>
#include <string>
#include "fg_common.cuh"

#ifndef K2_TL
#define K2_TL 0
#endif
// X(idx, name, MS, NS, KS, STAGEK, DEPTH, PRE, FL)
// FL bits: NF 1 (activation fold: E writes A'), WF 2 (writer-side proxy fence in E), ACCURATE_GELU 4 (tanhf), DBG_NOLD 64, DBG_NOW 128, DBG_NOA 256 (timing experiments; work gate fails)
#ifndef K2_LIST
#define K2_LIST(X) \
  X(0, "k2_m2k4_s128_d3", 2, 1, 4, 128, 3, 3, 0) \
  X(1, "k2_m4k2_s128_d3", 4, 1, 2, 128, 3, 3, 0) \
  X(2, "k2_m2n2k2_s128_d3", 2, 2, 2, 128, 3, 3, 0) \
  X(3, "k2_m1k8_s128_d3", 1, 1, 8, 128, 3, 3, 0) \
  X(4, "k2_m4k2_s128_d3_nf", 4, 1, 2, 128, 3, 3, 3) \
  X(5, "k2_m2k4_s128_d3_nf", 2, 1, 4, 128, 3, 3, 3) \
  X(6, "k2_m4k2_s128_d3_pre1_nf", 4, 1, 2, 128, 3, 1, 3) \
  X(7, "k2_m4k2_s64_d6_nf", 4, 1, 2, 64, 6, 6, 3) \
  X(8, "k2_m4k2_s128_d4_nf", 4, 1, 2, 128, 4, 4, 3) \
  X(9, "k2_m4k2_s128_d2_nf", 4, 1, 2, 128, 2, 2, 3) \
  X(10, "k2_m4k2_s128_d3_nf_wf0", 4, 1, 2, 128, 3, 3, 1) \
  X(11, "k2_m4k2_s128_d3_nf_accg", 4, 1, 2, 128, 3, 3, 7)
#endif
constexpr int FGV_K2 = 16;
#define X(i, n, MS, NS, KS, SK, D, PRE, FL) +1
constexpr int K2_N = 0 K2_LIST(X);
#undef X
static const char* fg_ext_vname(int v) {
#define X(i, n, MS, NS, KS, SK, D, PRE, FL) if (v == FGV_K2 + i) return n;
  K2_LIST(X)
#undef X
  return "var";
}
static int fg_ext_parse(const std::string& s) {
#define X(i, n, MS, NS, KS, SK, D, PRE, FL) if (s == n) return FGV_K2 + i;
  K2_LIST(X)
#undef X
  return -1;
}

__constant__ __align__(64) CUtensorMap c_mg;   // gate weights [L*4096][1024], box {64, 32}
__constant__ __align__(64) CUtensorMap c_mu;   // up weights
__constant__ __align__(64) CUtensorMap c_mh;   // h [50][1024], box {64, 56} (rows 50..55 zero-filled by TMA OOB)
__constant__ __align__(64) CUtensorMap c_mA;   // g_Ap [56][1024], box {64, 56}
__device__ __align__(1024) bf16 g_Ap[56 * W];  // NF: A' = bf16(h*(1+sc)) rows 0..49, row 50 = shift, rows 51..55 zero
__device__ unsigned long long g_tl[K2_N][128][16];  // in-F timeline sums (K2_TL)

__device__ __forceinline__ uint32_t sa_(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void kmbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void kmbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void kmbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok;
  do {
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void ktma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((unsigned long long)m), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void kbulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"((unsigned long long)src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma_k2(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool FAST>
__device__ __forceinline__ float gelu_k2(float x) {
  const float kBeta = 0.7978845608028654f, kKappa = 0.044715f;
  const float inner = kBeta * (x + kKappa * x * x * x);
  if (FAST) {
    float th;
    asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(inner));
    return 0.5f * x * (1.0f + th);
  }
  return 0.5f * x * (1.0f + tanhf(inner));
}
__device__ __forceinline__ float ld_sh_f32(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory"); return v; }

// NF variant of E (identical contract to K1's k1_E_nf): residual_tasks + A' = bf16(h * (1 + scale_F)) (+ row 50 = shift_F)
template <int S, bool WF>
__device__ __forceinline__ void k2_E_nf(const Params& p, const float* gate, const float* modF) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int task = warp * gridDim.x + (gridDim.x - 1 - blockIdx.x); task < T * NSSP; task += gridDim.x * NWARPS) {
    const int t = task / NSSP, c = (task % NSSP) * 64 + lane * 2;
    const float2 sc = __ldcg(reinterpret_cast<const float2*>(modF + c));
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
    *reinterpret_cast<__nv_bfloat162*>(g_Ap + t * W + c) = __floats2bfloat162_rn(__fmul_rn(h0, __fadd_rn(1.0f, sc.x)), __fmul_rn(h1, __fadd_rn(1.0f, sc.y)));
  }
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i < W / 2; i += NTHREADS) {
      const float2 sh = __ldcg(reinterpret_cast<const float2*>(modF + W + 2 * i));
      *reinterpret_cast<__nv_bfloat162*>(g_Ap + 50 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
    }
  if (WF) asm volatile("fence.proxy.async.global;" ::: "memory");
}

constexpr int NACT = FF / 32;                          // 128 active CTAs
constexpr int A_ROWS = 56, A_BYTES = A_ROWS * W * 2;   // 114688
constexpr int RING_MAX = 90112;
constexpr int SYNC_BYTES = 12288;                      // mbarriers, nmod (8192), ssp (3200), rs, shift column
constexpr size_t K2_SMEM = 1024 + A_BYTES + RING_MAX + SYNC_BYTES;
constexpr int FL_NF = 1, FL_WF = 2, FL_ACCG = 4, FL_DBG_NOLD = 64, FL_DBG_NOW = 128, FL_DBG_NOA = 256;

template <int MS, int NS, int KS, int STAGEK, int DEPTH, int PRE, int FL, int VIDX>
__device__ __noinline__ void k2_F(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
  constexpr bool NF = FL & FL_NF;
  static_assert(MS * NS * KS == 8, "8 warps");
  static_assert(MS == 1 || MS == 2 || MS == 4, "m groups");
  static_assert(NS == 1 || NS == 2, "token split");
  static_assert(!NF || NS == 1, "NF needs all token tiles (incl. the shift column) in every warp");
  constexpr int MT = 4 / MS;                 // m16 tiles per warp (each = 8 features x gate/up)
  constexpr int NT = (NS == 1) ? 7 : 4;      // n8 tiles per warp (NS=2: warp 1 computes a wasted 8th tile; its outputs are dropped)
  constexpr int SPS = STAGEK / 16, SPK = SPS / KS;
  static_assert(KS <= SPS && SPS % KS == 0, "k slices");
  constexpr int NSTG = W / STAGEK, SP = STAGEK / 64;
  static_assert(DEPTH <= NSTG && DEPTH <= 8 && PRE <= DEPTH, "ring");
  constexpr int STAGE_BYTES = STAGEK * 128, PAIRS = MT * NT;
  static_assert(DEPTH * STAGE_BYTES <= RING_MAX, "ring fits");
  static_assert(8 * PAIRS * 4 * 32 * 4 <= A_BYTES, "reduce scratch fits the A region");
  constexpr int ACH_K = NF ? STAGEK : 128;   // K extent of one activation-chunk barrier
  constexpr int NACH = W / ACH_K;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool act = (int)blockIdx.x < NACT;
  const uint32_t sbase0 = sa_(smb);
  const uint32_t A_u32 = (sbase0 + 1023u) & ~1023u;
  char* const Ag = reinterpret_cast<char*>(smb) + (A_u32 - sbase0);
  const uint32_t ring_u32 = A_u32 + A_BYTES, full0 = ring_u32 + RING_MAX, abar0 = full0 + 64, cnt0 = full0 + 192, pbar = full0 + 224,
                 nmod_u32 = full0 + 256, ssp_u32 = nmod_u32 + 8192, rs_u32 = ssp_u32 + 3200 + 64, shs_u32 = rs_u32 + 256;
#if K2_TL
  unsigned long long t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, tA0 = 0, tf0 = 0, wsum = 0;
  if (tid == 0) t0 = clock64();
#endif
  auto issue = [&](int s) {
    const int slot = s % DEPTH;
    const uint32_t bar = full0 + slot * 8, dst = ring_u32 + slot * STAGE_BYTES;
    if (FL & FL_DBG_NOW) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); return; }
    kmbar_expect(bar, STAGE_BYTES);
    const int rowc = l * FF + blockIdx.x * 32;
#pragma unroll
    for (int sp = 0; sp < SP; ++sp) {
      const int kc = s * STAGEK + sp * 64;
      ktma2d(dst + sp * 8192, &c_mg, bar, kc, rowc);
      ktma2d(dst + sp * 8192 + 4096, &c_mu, bar, kc, rowc);
    }
  };
  if (act && tid == 0) {
    for (int i = 0; i < NACH; ++i) kmbar_init(abar0 + i * 8, 1);
    kmbar_init(pbar, 1);
    for (int i = 0; i < DEPTH; ++i) { kmbar_init(full0 + i * 8, 1); asm volatile("st.shared.u32 [%0], %1;" ::"r"(cnt0 + i * 4), "r"(0u) : "memory"); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  wpre_arrive(w);
  if (PRE > 0 && act && tid == 0) {
#pragma unroll
    for (int s = 0; s < PRE; ++s) issue(s);
  }
  wpre_wait(w);
  if (!act) return;
#if K2_TL
  if (tid == 0) t1 = clock64();
#endif
  if (tid == 0) {
    // activation chunks (TMA), ssp (+ nmod for the in-F norm) by bulk copy; then the ring stages not yet issued
    if (!(FL & FL_DBG_NOA)) {
      if (NF) {
#pragma unroll
        for (int kc = 0; kc < NACH; ++kc) {
          kmbar_expect(abar0 + kc * 8, SP * 7168);
#pragma unroll
          for (int sp = 0; sp < SP; ++sp) ktma2d(A_u32 + (kc * SP + sp) * 7168, &c_mA, abar0 + kc * 8, kc * STAGEK + sp * 64, 0);
        }
      } else {
        asm volatile("fence.proxy.async.global;" ::: "memory");
#pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
          kmbar_expect(abar0 + kc * 8, 2 * 7168);
          ktma2d(A_u32 + (2 * kc) * 7168, &c_mh, abar0 + kc * 8, kc * 128, 0);
          ktma2d(A_u32 + (2 * kc + 1) * 7168, &c_mh, abar0 + kc * 8, kc * 128 + 64, 0);
        }
      }
    } else {
#pragma unroll
      for (int kc = 0; kc < NACH; ++kc) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(abar0 + kc * 8) : "memory");
    }
    if (NF) {
      kmbar_expect(pbar, T * NSSP * 4);
      kbulk(ssp_u32, p.ssp, T * NSSP * 4, pbar);
    } else {
      kmbar_expect(pbar, 2 * W * 4 + T * NSSP * 4);
      kbulk(nmod_u32, p.mod + (size_t)(2 * l + 1) * 3 * W, 2 * W * 4, pbar);
      kbulk(ssp_u32, p.ssp, T * NSSP * 4, pbar);
    }
#pragma unroll
    for (int s = PRE; s < DEPTH; ++s) issue(s);
  }
  if (!NF) {
    // ---------------- in-smem adaRMS norm of the raw h tile: (h*rs)*(1+sc)+sh, bf16 (thread: 16 B column group pc, rows q + 16x)
    const int pc = tid & 15, q = tid >> 4;
    kmbar_wait(pbar, 0);
    float rs[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int r = q + 16 * x;
      float v = 0.f;
      if (r < T) v = ld_sh_f32(ssp_u32 + (r * NSSP + pc) * 4);
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      v = __shfl_sync(0xffffffffu, v, lane & 16);
      rs[x] = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
    }
    float4 nv[2][4];
    auto ldn = [&](int kc, int b) {
      const uint32_t sp = nmod_u32 + (kc * 128 + pc * 8) * 4;
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(nv[b][0].x), "=f"(nv[b][0].y), "=f"(nv[b][0].z), "=f"(nv[b][0].w) : "r"(sp) : "memory");
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(nv[b][1].x), "=f"(nv[b][1].y), "=f"(nv[b][1].z), "=f"(nv[b][1].w) : "r"(sp + 16) : "memory");
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(nv[b][2].x), "=f"(nv[b][2].y), "=f"(nv[b][2].z), "=f"(nv[b][2].w) : "r"(sp + W * 4) : "memory");
      asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(nv[b][3].x), "=f"(nv[b][3].y), "=f"(nv[b][3].z), "=f"(nv[b][3].w) : "r"(sp + W * 4 + 16) : "memory");
    };
    ldn(0, 0);
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      if (kc + 1 < 8) ldn(kc + 1, (kc + 1) & 1);
      kmbar_wait(abar0 + kc * 8, 0);
#if K2_TL
      if (tid == 0 && kc == 0) tA0 = clock64();
#endif
      const float4 sc0 = nv[kc & 1][0], sc1 = nv[kc & 1][1], sh0 = nv[kc & 1][2], sh1 = nv[kc & 1][3];
      const float sc[8] = {__fadd_rn(1.0f, sc0.x), __fadd_rn(1.0f, sc0.y), __fadd_rn(1.0f, sc0.z), __fadd_rn(1.0f, sc0.w),
                           __fadd_rn(1.0f, sc1.x), __fadd_rn(1.0f, sc1.y), __fadd_rn(1.0f, sc1.z), __fadd_rn(1.0f, sc1.w)};
      const float sh[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        const int r = q + 16 * x;
        if (r >= T) continue;
        const uint32_t ad = A_u32 + (2 * kc + (pc >> 3)) * 7168 + r * 128 + (((pc & 7) ^ (r & 7)) << 4);
        uint4 hv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(hv.x), "=r"(hv.y), "=r"(hv.z), "=r"(hv.w) : "r"(ad) : "memory");
        uint32_t* wv = reinterpret_cast<uint32_t*>(&hv);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          __nv_bfloat162 hv2 = *reinterpret_cast<__nv_bfloat162*>(&wv[e]);
          const float o0 = __fadd_rn(__fmul_rn(__fmul_rn(__low2float(hv2), rs[x]), sc[2 * e]), sh[2 * e]);
          const float o1 = __fadd_rn(__fmul_rn(__fmul_rn(__high2float(hv2), rs[x]), sc[2 * e + 1]), sh[2 * e + 1]);
          hv2 = __floats2bfloat162_rn(o0, o1);
          wv[e] = *reinterpret_cast<uint32_t*>(&hv2);
        }
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ad), "r"(hv.x), "r"(hv.y), "r"(hv.z), "r"(hv.w) : "memory");
      }
    }
    __syncthreads();
  }
#if K2_TL
  if (tid == 0) t2 = clock64();
#endif
  // ---------------- MMA main loop
  const int kslice = warp % KS, ng = (warp / KS) % NS, mg = warp / (KS * NS);
  const int ntb = ng * 4, lm = lane >> 3, lr = lane & 7;
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  uint32_t fa[2][MT][4], fb[2][NT][2];
  constexpr int NL = MT + NT / 2 + (NT & 1);   // fragment-load micro-ops per k16 step
  auto ld_one = [&](int k, int st, int i, int slot, int buf) {
    if (FL & FL_DBG_NOLD) return;
    const int j = kslice + i * KS;
    if (k < MT) {
      const uint32_t sb = ring_u32 + slot * STAGE_BYTES;
      const int kk = j * 2 + (lm >> 1), span = kk >> 3, cc = kk & 7;
      const uint32_t ao = sb + span * 8192 + (lm & 1) * 4096 + (mg * MT * 8 + lr) * 128 + ((cc ^ lr) << 4);
      ldsm4(fa[buf][k], ao + k * 8 * 128);
    } else {
      const int np = k - MT;
      const int c = (st * SPS + j) * 2 + (lm & 1);
      const uint32_t abase = A_u32 + (c >> 3) * 7168, crow = lr * 128 + (((c & 7) ^ lr) << 4);
      if (np < NT / 2) {
        uint32_t r4[4];
        ldsm4(r4, abase + crow + (ntb + (lm >> 1) + np * 2) * 8 * 128);
        fb[buf][2 * np][0] = r4[0]; fb[buf][2 * np][1] = r4[1]; fb[buf][2 * np + 1][0] = r4[2]; fb[buf][2 * np + 1][1] = r4[3];
      } else {
        uint32_t r2[2];
        ldsm2(r2, abase + crow + (ntb + NT - 1) * 8 * 128);
        fb[buf][NT - 1][0] = r2[0]; fb[buf][NT - 1][1] = r2[1];
      }
    }
  };
  auto release = [&](int st) {
    __syncwarp();
    if (lane == 0) {
      const int slot = st % DEPTH;
      unsigned old;
      asm volatile("atom.acq_rel.cta.shared::cta.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(cnt0 + slot * 4) : "memory");
      if (old == 7u) {
        asm volatile("st.shared.u32 [%0], %1;" ::"r"(cnt0 + slot * 4), "r"(0u) : "memory");
        if (st + DEPTH < NSTG) issue(st + DEPTH);
      }
    }
  };
  // wait for everything stage st needs: its ring slot and (NF) its activation chunk
  auto wait_stage = [&](int st, int slot, uint32_t par) {
    kmbar_wait(full0 + slot * 8, par);
    if (NF) kmbar_wait(abar0 + (st * STAGEK / ACH_K) * 8, 0);
  };
  wait_stage(0, 0, 0);
#if K2_TL
  if (tid == 0) tf0 = clock64();
#endif
  {
#pragma unroll
    for (int k = 0; k < NL; ++k) ld_one(k, 0, 0, 0, 0);
  }
  int slot = 0; uint32_t par = 0;
#pragma unroll 1
  for (int st = 0; st < NSTG; ++st) {
    const int nslot = (slot + 1 == DEPTH) ? 0 : slot + 1;
    const uint32_t npar = par ^ (slot + 1 == DEPTH ? 1u : 0u);
#pragma unroll
    for (int i = 0; i < SPK; ++i) {
      const int cur = i & 1;
      const bool last = (i + 1 == SPK);
      const bool have_next = !last || (st + 1 < NSTG);
      const int nst = last ? st + 1 : st, ni = last ? 0 : i + 1, nsl = last ? nslot : slot;
      constexpr int MTNT = MT * NT;
#pragma unroll
      for (int q = 0; q < MTNT; ++q) {
        mma_k2(acc[q / NT][q % NT], fa[cur][q / NT], fb[cur][q % NT][0], fb[cur][q % NT][1]);
#pragma unroll
        for (int k = 0; k < NL; ++k)
          if (((k + 1) * MTNT) / (NL + 1) == q + 1) {
            if (have_next) {
              if (last && k == 0) {
#if K2_TL
                const unsigned long long tw0 = clock64();
#endif
                wait_stage(nst, nslot, npar);
#if K2_TL
                if (tid == 0) wsum += clock64() - tw0;
#endif
              }
              ld_one(k, nst, ni, nsl, cur ^ 1);
            }
          }
      }
      if (last) release(st);
    }
    if (SPK & 1) {  // buffers alternate; with odd SPK the parity flips per stage: swap by copying
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) { fa[0][mt][0] = fa[1][mt][0]; fa[0][mt][1] = fa[1][mt][1]; fa[0][mt][2] = fa[1][mt][2]; fa[0][mt][3] = fa[1][mt][3]; }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) { fb[0][nt][0] = fb[1][nt][0]; fb[0][nt][1] = fb[1][nt][1]; }
    }
    slot = nslot; par = npar;
  }
#if K2_TL
  if (tid == 0) t3 = clock64();
#endif
  // ---------------- K-slice reduce (smem) + epilogue: gelu(gate) * up in registers, act[t][f] bf16
  constexpr bool FASTG = !(FL & FL_ACCG);
  if (NF) {
    if (tid < T) {  // rs[t] = rsqrt(mean(h^2) + eps) from the 16 per-64-column sums of squares (fixed serial order)
      kmbar_wait(pbar, 0);
      float v = 0.f;
#pragma unroll
      for (int c = 0; c < NSSP; ++c) v += ld_sh_f32(ssp_u32 + (tid * NSSP + c) * 4);
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(rs_u32 + tid * 4), "f"(rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f))) : "memory");
    }
  }
  auto epi = [&](int mt, int nt, const float* c) {
    const int tile = ntb + nt;
    const int fl = (mg * MT + mt) * 8 + (lane >> 2);
    const int f = blockIdx.x * 32 + fl;
    const int t = tile * 8 + 2 * (lane & 3);
    float g0 = c[0], g1 = c[1], u0 = c[2], u1 = c[3];
    if (NF) {
      const float shg = ld_sh_f32(shs_u32 + fl * 8), shu = ld_sh_f32(shs_u32 + fl * 8 + 4);
      const float r0 = ld_sh_f32(rs_u32 + t * 4), r1 = ld_sh_f32(rs_u32 + (t + 1) * 4);
      g0 = fmaf(r0, g0, shg); u0 = fmaf(r0, u0, shu); g1 = fmaf(r1, g1, shg); u1 = fmaf(r1, u1, shu);
    }
    if (t < T) p.act[(size_t)t * FF + f] = __float2bfloat16_rn(rbf(gelu_k2<FASTG>(rbf(g0))) * rbf(u0));
    if (t + 1 < T) p.act[(size_t)(t + 1) * FF + f] = __float2bfloat16_rn(rbf(gelu_k2<FASTG>(rbf(g1))) * rbf(u1));
  };
  auto publish_shift = [&](int mt, const float* c) {  // token column 50 (tile 6, t4 == 1, c0 / c2) = W . shift for this feature's gate / up row
    if ((lane & 3) == 1) {
      const int fl = (mg * MT + mt) * 8 + (lane >> 2);
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(shs_u32 + fl * 8), "f"(c[0]) : "memory");
      asm volatile("st.shared.f32 [%0], %1;" ::"r"(shs_u32 + fl * 8 + 4), "f"(c[2]) : "memory");
    }
  };
  if (KS == 1) {
    if (NF) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) publish_shift(mt, acc[mt][6]);
      __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) epi(mt, nt, acc[mt][nt]);
#if K2_TL
    t4 = t3;
#endif
  } else {
    __syncthreads();
#if K2_TL
    if (tid == 0) t4 = clock64();
#endif
    float* scr = reinterpret_cast<float*>(Ag);
    const int gw = (warp / KS) * KS;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        if (((mt * NT + nt) % KS) != kslice) {
#pragma unroll
          for (int e = 0; e < 4; ++e) scr[((warp * PAIRS + mt * NT + nt) * 4 + e) * 32 + lane] = acc[mt][nt][e];
        }
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        if (((mt * NT + nt) % KS) == kslice) {
#pragma unroll
          for (int k = 0; k < KS; ++k)
            if (k != kslice) {
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[mt][nt][e] += scr[(((gw + k) * PAIRS + mt * NT + nt) * 4 + e) * 32 + lane];
            }
          if (NF && nt == 6) publish_shift(mt, acc[mt][nt]);
        }
    if (NF) __syncthreads();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        if (((mt * NT + nt) % KS) == kslice) epi(mt, nt, acc[mt][nt]);
  }
#if K2_TL
  if (tid == 0) {
    t5 = clock64();
    unsigned long long* g = g_tl[VIDX][blockIdx.x];
    g[0] += 1; g[1] += t1 - t0; g[2] += t2 - t1; g[3] += t3 - t2; g[4] += t4 - t3; g[5] += t5 - t4; g[6] += t5 - t1; g[7] += tA0 - t1; g[8] += tf0 - t1; g[9] += wsum;
  }
#endif
}

struct K2 {
  static constexpr const char* kName = "fg_k2";
  static constexpr bool kFgBarrier = false;
  static constexpr size_t kSmem = K2_SMEM;
  __device__ static void E(const Params& p, const FgArgs& a, int l, const float* nmodbase) {
    bool nf = false, wf = false;
#define X(i, n, MS, NS, KS, SK, D, PRE, FL) if (a.variant == FGV_K2 + i) { nf = (FL) & FL_NF; wf = ((FL) & FL_WF) != 0; }
    K2_LIST(X)
#undef X
    if (nf) {
      if (wf) k2_E_nf<QD / KC, true>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W, nmodbase + (size_t)(2 * l + 1) * 3 * W);
      else k2_E_nf<QD / KC, false>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W, nmodbase + (size_t)(2 * l + 1) * 3 * W);
    } else residual_tasks<QD / KC>(p, nmodbase + (size_t)(2 * l) * 3 * W + 2 * W);
  }
  __device__ static void F(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
    switch (a.variant) {
#define X(i, n, MS, NS, KS, SK, D, PRE, FL) case FGV_K2 + i: k2_F<MS, NS, KS, SK, D, PRE, FL, i>(p, a, l, w, smb); return;
      K2_LIST(X)
#undef X
      default: break;
    }
    fg_null_F(w);
    if (a.variant == FGV_CALIB) fg_spin(a.ncal);
  }
  __device__ static void G(const Params&, const FgArgs&, int, unsigned, bool) {}  // fused away

  static CUtensorMap mk2(PFN_cuTensorMapEncodeTiled_v12000 enc, const void* base, cuuint64_t rows, cuuint32_t boxrows) {
    CUtensorMap m; cuuint64_t dims[2] = {W, rows}, strides[1] = {(cuuint64_t)W * 2}; cuuint32_t box[2] = {64, boxrows}, es[2] = {1, 1};
    CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); exit(1); }
    return m;
  }
  static void prepare(FgArgs& a, int grid) {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    void* ap; CK(cudaGetSymbolAddress(&ap, g_Ap)); CK(cudaMemset(ap, 0, 56 * W * 2));
    CUtensorMap mg = mk2(enc, a.wg, (cuuint64_t)L * FF, 32), mu = mk2(enc, a.wu, (cuuint64_t)L * FF, 32), mh = mk2(enc, a.h, T, 56), mA = mk2(enc, ap, 56, 56);
    CK(cudaMemcpyToSymbol(c_mg, &mg, sizeof mg)); CK(cudaMemcpyToSymbol(c_mu, &mu, sizeof mu));
    CK(cudaMemcpyToSymbol(c_mh, &mh, sizeof mh)); CK(cudaMemcpyToSymbol(c_mA, &mA, sizeof mA));
    fprintf(stderr, "k2 prepare: grid=%d K2 variants=%d smem=%zu\n", grid, K2_N, (size_t)kSmem);
  }
  static void post_run(int, int, int, const std::string&, const std::string&, double) {}
};

int main(int argc, char** argv) {
  std::string out = ".", name = "fg_k2";
  for (int i = 1; i + 1 < argc; i += 2) { if (!strcmp(argv[i], "--out")) out = argv[i + 1]; if (!strcmp(argv[i], "--name")) name = argv[i + 1]; }
  const int rc = fg_main<K2>(argc, argv);
#if K2_TL
  static unsigned long long h[K2_N][128][16];
  CK(cudaMemcpyFromSymbol(h, g_tl, sizeof h));
  FILE* f = fopen((out + "/" + name + "_tl.jsonl").c_str(), "w");
  for (int v = 0; v < K2_N; ++v) {
    double s[10] = {0}, n = 0, mx = 0, mn = 1e18;
    for (int c = 0; c < NACT; ++c) { if (!h[v][c][0]) continue; n += h[v][c][0]; for (int k = 1; k < 10; ++k) s[k] += h[v][c][k]; const double x = (double)h[v][c][6] / h[v][c][0]; mx = std::max(mx, x); mn = std::min(mn, x); }
    if (n == 0) continue;
    const char* nm = fg_ext_vname(FGV_K2 + v);
    const char* fmt = "{\"rec\":\"k2_timeline_cycles\",\"variant\":\"%s\",\"arrive_to_barrier_exit\":%.0f,\"stage_A\":%.0f,\"mma_loop_w0\":%.0f,\"slowest_warp_wait\":%.0f,\"reduce_epilogue\":%.0f,\"after_barrier_total\":%.0f,\"after_barrier_total_max_cta\":%.0f,\"after_barrier_total_min_cta\":%.0f,\"A_chunk0_ready_seen\":%.0f,\"stage0_ready_seen\":%.0f,\"w0_stage_wait_total\":%.0f}\n";
    fprintf(f, fmt, nm, s[1] / n, s[2] / n, s[3] / n, s[4] / n, s[5] / n, s[6] / n, mx, mn, s[7] / n, s[8] / n, s[9] / n);
    printf(fmt, nm, s[1] / n, s[2] / n, s[3] / n, s[4] / n, s[5] / n, s[6] / n, mx, mn, s[7] / n, s[8] / n, s[9] / n);
  }
  fclose(f);
#endif
  return rc;
}
