// h_bench.cu: standalone H (MLP down projection, out[50][1024] = act[50][4096] . Wd[1024][4096]^T) + I (gated residual / next norm) study.
// Reuses the validity harness of probe/ubench/ab (ab_common.cuh: verbatim grid barrier / WPre / part accessors / tcgen05 helpers). One persistent
// cooperative cluster-2 kernel (148 CTAs x 256 thr, 197632 B dynamic smem as the megakernel) runs 180 layer iterations (10 steps x 18 layers, real-size
// distinct per-layer weights). Per iteration, barriers as in the megakernel (FUSE = 18: WPRE bit 4 = the F->H barrier runs inside H):
//   D  dummy: prefetch_l2(Wd[l]) (as mk phase D) + 4 MB Wo read                                     -> SYNC stamp S0
//   F  dummy: 16 MB Wg/Wu read + fresh act[50][4096] bf16 (hash of the iteration) + fence.proxy.async -> F->H WPRE split barrier, stamp S1 (wpre_wait)
//   H  variant: null (barrier only) | k0 (verbatim gemm_phase, 32 bf16 slabs) | k1_* (tcgen05 cta_group::2)  -> SYNC S2
//   I  variant consumer: fg_E form (h, ssp, A') over the variant's partial format                         -> SYNC S3
//   ABC dummy: 5 MB weight read + HMMA burn (--ccyc)                                                      -> SYNC S4
// then NCAL known-delay calibration rounds (same gates as ab_common.cuh). The last iteration snapshots h before I (one extra barrier, every variant)
// so the host can check the final h against an fp32 reference of layer 17's H + I.
// k1 variants (template <RPC, KS, RED>): a cluster pair owns 2 * RPC output features x one K slice of FF / KS; the pair's MMA is M = 2 * RPC (64 rows
// per CTA: the verified F layout; 128 rows per CTA: M = 256), N = 64 tokens (CTA r's B half = tokens 32 r .. 32 r + 31 of act, rows >= 50 zero).
// The CTA's weight slice is TMA'd before the F->H barrier wait (per-K64-chunk mbarriers), act chunks after it. K-split combine (RED):
//   0: bf16 slabs part[ks][t][n] (I sums KS slabs)   2: red.global.add.f32 into acc[t][n]   3: smem stage + cp.reduce.async.bulk.tensor add.f32 into acc
// (I reads acc once and zeroes it). Build: build.sh OUTDIR (stamped + -DFG_STAMPS=0 twins).
#include "../ab/ab_common.cuh"

#define PI05_GEMM_FRAG 2
#define PI05_LOAD_HOIST 1
#define PI05_GEMM_SPREAD 1
#define PI05_TILE_ROT 0
#define FIT_OPEN() do {} while (0)
#define FIT_EV(ev) do {} while (0)
#define FIT_CLOSE() do {} while (0)
#include "h_gemm.inc"  // VERBATIM gemm_phase + mma16816 / lds128 (extracted from ab_k0.cu by build.sh; identical to mk/pi05_mk.cu)

#ifndef H_MASK
#define H_MASK 0xffffffffu
#endif

// ------------------------------------------------------------- cta_group::2 TMA / MMA helpers (verbatim from mk/pi05_mk.cu)
__device__ __forceinline__ void tma2d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((unsigned long long)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void tma3d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               ::"r"(dst), "l"((unsigned long long)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2) : "memory"); }
__device__ __forceinline__ void mma_tc2(uint32_t d, unsigned long long da, unsigned long long db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit2(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"((unsigned short)3) : "memory"); }
__device__ __forceinline__ void tmem_alloc64c2(uint32_t slot) { asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot)); }
__device__ __forceinline__ void tmem_free64c2(uint32_t tb) { asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 64;" ::"r"(tb)); }
__device__ __forceinline__ void tld16x256(uint32_t ta, uint32_t (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]),
                 "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
}
__device__ __forceinline__ void tld16(uint32_t ta, uint32_t (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]),
                 "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
}

// ------------------------------------------------------------- variants
enum { HV_NULL = 0, HV_K0 = 1, HV_K1 = 16 };
// X(idx, RPC, KS, RED, name). RED: 0 bf16 slab (direct 2 B stores), 2 red.add.f32, 3 TMA reduce-add, 4 bf16 slab via smem transpose + 16 B stores,
// 5 red.add.v4.f32 via smem transpose, 6 bf16 slab via 16x256b fragments + stmatrix.trans + 16 B stores
// X(idx, RPC, KS, RED, WI, name); WI = K64 chunks per weight TMA instruction (row-major Wd, 2D / 3D map); WI >= 16: pre-tiled Wd (each CTA's slice
// contiguous, [l][n / RPC][ks][chunk][RPC rows][64]), WI / 16 chunks per 2D TMA of box {64, RPC * WI / 16}
#define HK1_LIST(X) \
  X(0, 64, 8, 0, 1, "k1_r64s8_bf") X(1, 64, 8, 2, 1, "k1_r64s8_at") X(2, 128, 16, 0, 1, "k1_r128s16_bf") X(3, 128, 16, 2, 1, "k1_r128s16_at") \
  X(4, 128, 16, 3, 1, "k1_r128s16_tr") X(5, 128, 16, 4, 1, "k1_r128s16_bx") X(6, 128, 16, 5, 1, "k1_r128s16_av") X(7, 128, 8, 4, 1, "k1_r128s8_bx") \
  X(8, 64, 8, 4, 1, "k1_r64s8_bx") X(9, 128, 16, 4, 4, "k1_r128s16_bx_w4") X(10, 128, 16, 4, 2, "k1_r128s16_bx_w2") X(11, 64, 8, 4, 8, "k1_r64s8_bx_w8") \
  X(12, 64, 8, 4, 2, "k1_r64s8_bx_w2") X(13, 128, 8, 4, 8, "k1_r128s8_bx_w8") X(14, 128, 16, 0, 4, "k1_r128s16_bf_w4") \
  X(15, 128, 16, 4, 16, "k1_r128s16_bx_t1") X(16, 128, 16, 4, 32, "k1_r128s16_bx_t2") X(17, 64, 8, 4, 16, "k1_r64s8_bx_t1") \
  X(18, 64, 8, 4, 64, "k1_r64s8_bx_t4") X(19, 128, 16, 6, 4, "k1_r128s16_sm_w4") X(20, 128, 16, 6, 1, "k1_r128s16_sm")
#define HK1_N(i, r, k, d, wi, n) +1
constexpr int NHK1 = 0 HK1_LIST(HK1_N);
__host__ __device__ constexpr int HK1_KS(int i) {
  int v = 0;
#define HK1_K(ii, r, k, d, wi, n) if (i == ii) v = k;
  HK1_LIST(HK1_K)
  return v;
}
__host__ __device__ constexpr int HK1_RED(int i) {
  int v = 0;
#define HK1_D(ii, r, k, d, wi, n) if (i == ii) v = d;
  HK1_LIST(HK1_D)
  return v;
}
#define HK1_S(ii, r, k, d, wi, n) n,
static const char* HK1_NAME[NHK1] = {HK1_LIST(HK1_S)};

struct HArgs {
  const bf16 *wd, *wo, *wgu, *wa;
  bf16* act; float* mod; bf16* h; bf16* hsnap; float* ssp; float* part; float* acc; bf16* ap;
  unsigned* bar; unsigned* sink;
  ull* stamps; ull* gtv;
  int variant, cwarps;
  long long ccyc, ncal;
  CUtensorMap mw64, mw128, mact, mr64, mr128;
  CUtensorMap mwx[2][4];  // 3D weight maps {64, L * W, FF / 64}, box {64, RPC = 64 << i, 1 << j}
  CUtensorMap mact3;      // act 3D {64, 64, FF / 64}, box {64, 32, 4} (KS 16 slices; ctx bit 3)
  CUtensorMap mwt[2][3];  // pre-tiled weights wt[i] (i = 0: RPC 64 KS 8, 1: RPC 128 KS 16) as 2D {64, L * W * FF / 64}, box {64, RPC << j}
  const bf16* wt[2];
  const bf16* wpf;        // the H weight buffer phase D prefetches (wd, or the variant's pre-tiled copy)
  int ctx;              // bit 0: D prefetches Wd (mk), bit 1: F streams 16 MB, bit 2: H always layer 0, bit 3: act as one 3D TMA (KS 16 only)
};

// k1 per-CTA timeline (one iteration, --tl): clock64 at these points
enum { TS_ENTRY, TS_BAR, TS_W0, TS_WALL, TS_A0, TS_AALL, TS_MMAEND, TS_LD, TS_ST, TS_END, TS_ISS, TS_CS, H_NTS };
__device__ ull g_h_ts[148 * H_NTS];
#define HTS(k) do { if (ts != nullptr && threadIdx.x == 0) ts[(k)] = clock64(); } while (0)
#define HTS1(k) do { if (ts != nullptr && (threadIdx.x & 31) == 0) ts[(k)] = clock64(); } while (0)  // MMA warp

__shared__ __align__(8) ull h_mbar[40];  // wbar[c] @ 8 c, abar[c] @ 128 + 8 c, acc @ 256
__shared__ uint32_t h_tslot;

struct HPtrs { float* part; float* acc; ull* ts; const CUtensorMap* act3; };  // act3 != null: act slice as one 3D box {64, 32, NCH}
template <int RPC, int KS, int RED, int WI>
__device__ __noinline__ void h_k1(const HPtrs hp, const CUtensorMap* tmW, const CUtensorMap* tmA, const CUtensorMap* tmR, int l, const WPre& w, bf16* smb) {
  constexpr int NRG = W / (2 * RPC), NCTA = 2 * NRG * KS, KSL = FF / KS, NCH = KSL / 64;
  constexpr uint32_t WCH = RPC * 128, ACH = 32 * 128;  // bytes of one K64 chunk: weights (RPC rows), act (32 tokens)
  static_assert(NCTA <= 148 && NCH <= 16 && NCH * (WCH + ACH) + 1024 <= AB_SMEM_DYN && T * RPC * 4 <= NCH * WCH, "h_k1 geometry");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x, rank = cta & 1;
  if (cta >= NCTA) { wpre_arrive(w); wpre_wait(w); return; }
  ull* ts = hp.ts != nullptr ? hp.ts + (size_t)cta * H_NTS : nullptr;
  HTS(TS_ENTRY);
  const int pi = cta >> 1, rg = pi % NRG, ks = pi / NRG;
  const int n0 = rg * 2 * RPC + rank * RPC, k0 = ks * KSL;
  const uint32_t base = (su32(smb) + 1023) & ~1023u, bW = base, bA = base + NCH * WCH;
  const uint32_t mb = su32(h_mbar), wbar = mb, abar = mb + 128, bacc = mb + 256;
  if (tid == 0) {
    for (int c = 0; c < NCH; ++c) { mbar_init(wbar + 8 * c, 1); mbar_init(abar + 8 * c, 1); }
    mbar_init(bacc, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // earlier phases used this smem through the generic proxy
  }
  if (warp == 1) { tmem_alloc64c2(su32(&h_tslot)); fence_before(); }
  cluster_sync_all();  // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
  HTS(TS_CS);
  fence_after();
  wpre_arrive(w);
  if (tid == 0) {  // PRE: the weight slice does not depend on F. WI = 1: one 2D box per K64 chunk; WI > 1: 3D box {64, RPC, WI} (map {64, rows, FF / 64})
    if constexpr (WI >= 16) {
      constexpr int WT = WI / 16;
      const int rb = ((l * (W / RPC) + n0 / RPC) * KS + ks) * NCH * RPC;
      for (int c = 0; c < NCH; c += WT) {
        if (rank == 0) mbar_expect(wbar + 8 * c, 2 * WT * WCH);
        tma2d2(bW + c * WCH, tmW, wbar + 8 * c, 0, rb + c * RPC);
      }
    } else if constexpr (WI == 1) {
      for (int c = 0; c < NCH; ++c) {
        if (rank == 0) mbar_expect(wbar + 8 * c, 2 * WCH);
        tma2d2(bW + c * WCH, tmW, wbar + 8 * c, k0 + c * 64, l * W + n0);
      }
    } else {
      for (int c = 0; c < NCH; c += WI) {
        if (rank == 0) mbar_expect(wbar + 8 * c, 2 * WI * WCH);
        tma3d2(bW + c * WCH, tmW, wbar + 8 * c, 0, l * W + n0, (k0 >> 6) + c);
      }
    }
  }
  HTS(TS_ISS);
  wpre_wait(w);
  HTS(TS_BAR);
  const uint32_t tb = h_tslot;
  if (warp == 0) {
    if (lane == 0) {
      if (hp.act3 != nullptr) {
        if (rank == 0) mbar_expect(abar, 2 * NCH * ACH);
        tma3d2(bA, hp.act3, abar, 0, rank * 32, k0 >> 6);
      } else
        for (int c = 0; c < NCH; ++c) {
          if (rank == 0) mbar_expect(abar + 8 * c, 2 * ACH);
          tma2d2(bA + c * ACH, tmA, abar + 8 * c, k0 + c * 64, rank * 32);
        }
    }
  } else if (warp == 1) {
    if (rank == 0) {
      const bool el = elect_one();
      const ull dW0 = sdesc(bW), dA0 = sdesc(bA);
      const uint32_t id = idesc(64, 2 * RPC);
      for (int c = 0; c < NCH; ++c) {
        if (c % (WI >= 16 ? WI / 16 : WI) == 0) mbar_wait(wbar + 8 * c, 0);
        if (c == 0) HTS1(TS_W0);
        if (c == NCH - 1) HTS1(TS_WALL);
        if (hp.act3 == nullptr || c == 0) mbar_wait(abar + 8 * c, 0);
        if (c == 0) HTS1(TS_A0);
        if (c == NCH - 1) HTS1(TS_AALL);
        fence_after();
        if (el) {
          const ull dW = dW0 + (ull)((c * WCH) >> 4), dA = dA0 + (ull)((c * ACH) >> 4);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_tc2(tb, dW + (ull)((kk * 32) >> 4), dA + (ull)((kk * 32) >> 4), id, (c | kk) != 0);
        }
        __syncwarp();
      }
      if (el) commit2(bacc);
    }
  }
  // ---- epilogue, all 8 warps. Warp q = warp & 3 reads TMEM lanes 32 q .. 32 q + 31; hs = warp >> 2 picks the column half.
  // RPC 128 (M = 256): lane = feature row 32 q + lane, columns 0..63 = tokens 0..63 (B halves of rank 0, 1).
  // RPC 64 (M = 128, F layout): lanes 0..63 = rows x tokens 0..31, lanes 64..127 = the same rows x tokens 32..63 (columns 0..31).
  __syncwarp();
  const int q = warp & 3, hs = warp >> 2;
  const int row = RPC == 128 ? 32 * q + lane : 32 * (q & 1) + lane;
  constexpr int NC = RPC == 128 ? 32 : 16;
  const int tok0 = RPC == 128 ? hs * 32 : (q >> 1) * 32 + hs * 16, tcol = RPC == 128 ? hs * 32 : hs * 16;
  uint32_t v[NC];
  mbar_wait(bacc, 0);
  HTS(TS_MMAEND);
  fence_after();
  if constexpr (RED == 6) {
    // M = 256 layout read as MMA fragments (16x256b: thread = rows l / 4, l / 4 + 8, tokens 2 (l % 4) + {0, 1} + 8 rep), packed to bf16x2 and stored
    // transposed with stmatrix: each 8x8 (8 features x 8 tokens) matrix lands as 8 token rows of 16 B in a [t][RPC] bf16 stage (row pitch 272 B:
    // conflict-free), then 16 B pieces to the slab. 4 stmatrix.x4 per warp instead of 32 scalar shared stores.
    static_assert(RPC == 128, "stmatrix epilogue: M = 256 layout");
    constexpr uint32_t SST = RPC * 2 + 16;
    uint32_t va[2][16];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) tld16x256(tb + ((uint32_t)(32 * q + 16 * sub) << 16) + hs * 32, va[sub]);
    wait_ld();
    fence_before();
    HTS(TS_LD);
    const int jm = lane >> 3, ri = lane & 7;
#pragma unroll
    for (int g4 = 0; g4 < 4; ++g4) {
      const int sub = g4 >> 1;
      uint32_t r[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int rep = (g4 & 1) * 2 + (j >> 1), hf = j & 1;
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(__uint_as_float(va[sub][4 * rep + 2 * hf]), __uint_as_float(va[sub][4 * rep + 2 * hf + 1]));
        r[j] = *reinterpret_cast<const uint32_t*>(&b2);
      }
      const int rep = (g4 & 1) * 2 + (jm >> 1), hf = jm & 1;
      const uint32_t sa = bW + (uint32_t)(hs * 32 + 8 * rep + ri) * SST + (uint32_t)(32 * q + 16 * sub + 8 * hf) * 2;
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(sa), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
    }
    __syncthreads();
    for (int i = tid; i < T * (RPC / 8); i += NTHREADS) {
      const int t = i / (RPC / 8), pc = i % (RPC / 8);
      uint4 u; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(bW + (uint32_t)t * SST + pc * 16));
      *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(hp.part) + ((size_t)ks * T + t) * W + n0 + pc * 8) = u;
    }
  } else {
    const uint32_t ta = tb + ((uint32_t)(32 * q) << 16) + tcol;
    tld16(ta, *reinterpret_cast<uint32_t(*)[16]>(&v[0]));
    if constexpr (NC == 32) tld16(ta + 16, *reinterpret_cast<uint32_t(*)[16]>(&v[16]));
    wait_ld();
    fence_before();
    HTS(TS_LD);
  }
  const int n = n0 + row;
  if constexpr (RED == 6) {
  } else if constexpr (RED == 0) {
    unsigned short* pb = reinterpret_cast<unsigned short*>(hp.part) + (size_t)ks * T * W + n;
#pragma unroll
    for (int j = 0; j < NC; ++j)
      if (tok0 + j < T) pb[(size_t)(tok0 + j) * W] = __bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float(v[j])));
  } else if constexpr (RED == 2) {
#pragma unroll
    for (int j = 0; j < NC; ++j)
      if (tok0 + j < T) asm volatile("red.global.add.f32 [%0], %1;" ::"l"(hp.acc + (size_t)(tok0 + j) * W + n), "f"(__uint_as_float(v[j])) : "memory");
  } else if constexpr (RED == 4 || RED == 5) {
    // transpose through smem (over the consumed weight tile): [t][RPC] bf16 (4) / fp32 (5), then 16 B row pieces to global
    constexpr int EB = RED == 4 ? 2 : 4, PR = RPC * EB / 16;  // 16 B pieces per token row
#pragma unroll
    for (int j = 0; j < NC; ++j)
      if (tok0 + j < T) {
        const uint32_t sa = bW + (uint32_t)(((tok0 + j) * RPC + row) * EB);
        if constexpr (RED == 4) st_shared_b16(sa, __bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float(v[j]))));
        else asm volatile("st.shared.f32 [%0], %1;" ::"r"(sa), "f"(__uint_as_float(v[j])) : "memory");
      }
    __syncthreads();
    for (int i = tid; i < T * PR; i += NTHREADS) {
      const int t = i / PR, pc = i % PR;
      uint4 u; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "r"(bW + (uint32_t)(i * 16)));
      if constexpr (RED == 4)
        *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(hp.part) + ((size_t)ks * T + t) * W + n0 + pc * 8) = u;
      else
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(hp.acc + (size_t)t * W + n0 + pc * 4), "f"(__uint_as_float(u.x)), "f"(__uint_as_float(u.y)),
                     "f"(__uint_as_float(u.z)), "f"(__uint_as_float(u.w)) : "memory");
    }
  } else {
    // stage [t][RPC] fp32 over the (consumed) weight tile, then one bulk tensor reduce-add of box {RPC, 50} at (n0, 0)
#pragma unroll
    for (int j = 0; j < NC; ++j)
      if (tok0 + j < T) asm volatile("st.shared.f32 [%0], %1;" ::"r"(bW + (uint32_t)(((tok0 + j) * RPC + row) * 4)), "f"(__uint_as_float(v[j])) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%1, %2}], [%3];"
                   ::"l"((unsigned long long)tmR), "r"(n0), "r"(0), "r"(bW) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
  }
  HTS(TS_ST);
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64c2(tb); }
  HTS(TS_END);
}

// I: gated residual of the H output + ssp + A' (fg_E of mk/pi05_mk.cu). MODE 0: the output is S bf16 slabs part[s][t][n]; MODE 1: one fp32
// accumulator acc[t][n], zeroed after the read for the next H (the zeroing stores are made visible to the async proxy by the closing fence).
template <int S, int MODE>
__device__ __forceinline__ void h_I(bf16* h, float* ssp, bf16* ap, const float* part, float* acc, const float* gate, const float* modF) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int task = warp * gridDim.x + (gridDim.x - 1 - blockIdx.x); task < T * NSSP; task += gridDim.x * NWARPS) {
    const int t = task / NSSP, c = (task % NSSP) * 64 + lane * 2;
    const float2 sc = __ldcg(reinterpret_cast<const float2*>(modF + c));
    float2 v = make_float2(0.f, 0.f);
    if constexpr (MODE == 0) {
      float2 x[S];
#pragma unroll
      for (int s = 0; s < S; ++s) x[s] = part_ld2(part, ((size_t)s * T + t) * W + c);
#pragma unroll
      for (int s = 0; s < S; ++s) { v.x += x[s].x; v.y += x[s].y; }
    } else {
      v = __ldcg(reinterpret_cast<const float2*>(acc + (size_t)t * W + c));
      __stcg(reinterpret_cast<float2*>(acc + (size_t)t * W + c), make_float2(0.f, 0.f));
    }
    const float2 gv = __ldcg(reinterpret_cast<const float2*>(gate + c));
    const unsigned hraw = __ldcg(reinterpret_cast<const unsigned*>(h + t * W + c));
    const __nv_bfloat162 hold = *reinterpret_cast<const __nv_bfloat162*>(&hraw);
    const float h0 = rbf(__fadd_rn(__low2float(hold), rbf(__fmul_rn(rbf(v.x), rbf(gv.x)))));
    const float h1 = rbf(__fadd_rn(__high2float(hold), rbf(__fmul_rn(rbf(v.y), rbf(gv.y)))));
    *reinterpret_cast<__nv_bfloat162*>(h + t * W + c) = __floats2bfloat162_rn(h0, h1);
    const float ss = warp_sum(fmaf(h1, h1, h0 * h0));
    if (lane == 0) ssp[task] = ss;
    *reinterpret_cast<__nv_bfloat162*>(ap + (t >= 31 ? t + 1 : t) * W + c) = __floats2bfloat162_rn(__fmul_rn(h0, __fadd_rn(1.0f, sc.x)), __fmul_rn(h1, __fadd_rn(1.0f, sc.y)));
  }
  if (blockIdx.x == 0)
    for (int i = threadIdx.x; i < W / 2; i += NTHREADS) {
      const float2 sh = __ldcg(reinterpret_cast<const float2*>(modF + W + 2 * i));
      *reinterpret_cast<__nv_bfloat162*>(ap + 31 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
      *reinterpret_cast<__nv_bfloat162*>(ap + 51 * W + 2 * i) = __floats2bfloat162_rn(sh.x, sh.y);
    }
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ uint32_t h_read(const bf16* w, size_t bytes, int it) {  // grid-strided streaming read (dummy phases)
  const size_t gt = (size_t)blockIdx.x * NTHREADS + threadIdx.x, nt = (size_t)gridDim.x * NTHREADS;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  uint32_t acc = it;
#pragma unroll 4
  for (size_t i = gt; i < bytes / 16; i += nt) { const uint4 v = __ldg(w4 + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  return acc;
}

template <int RPC, int KS, int RED, int WI>
__device__ __forceinline__ bool h_try(int idx, const HArgs& a, const HPtrs& hp, int l, const WPre& w, bf16* smb) {
  if (!((H_MASK >> idx) & 1) || a.variant != HV_K1 + idx) return false;
  h_k1<RPC, KS, RED, WI>(hp, WI >= 16 ? &a.mwt[RPC == 128][WI == 16 ? 0 : WI == 32 ? 1 : 2] : WI > 1 ? &a.mwx[RPC == 128][WI == 2 ? 1 : WI == 4 ? 2 : 3] : RPC == 64 ? &a.mw64 : &a.mw128, &a.mact, RPC == 64 ? &a.mr64 : &a.mr128, l, w, smb);
  return true;
}
__device__ __forceinline__ void h_phase(const HArgs& a, int l, const WPre& w, bf16* smb, ull* ts) {
  if (a.variant == HV_K0) {
    gemm_phase(a.act, FF, a.wd + (size_t)l * W * FF, W, nullptr, W, nullptr, W, a.part, smb, nullptr, 64, W / 64, nullptr, nullptr, w);
    return;
  }
  const HPtrs hp{a.part, a.acc, ts, (a.ctx & 8) ? &a.mact3 : nullptr};
#define HK1_T(ii, r, k, d, wi, n) if (h_try<r, k, d, wi>(ii, a, hp, l, w, smb)) return;
  HK1_LIST(HK1_T)
  wpre_arrive(w);  // null
  wpre_wait(w);
}
__device__ __forceinline__ void i_phase(const HArgs& a, const float* gate, const float* modF) {
  const int v = a.variant;
  if (v == HV_NULL || v == HV_K0) { h_I<FF / KC, 0>(a.h, a.ssp, a.ap, a.part, a.acc, gate, modF); return; }
  const int red = HK1_RED(v - HV_K1), ks = HK1_KS(v - HV_K1);
  if (red == 2 || red == 3 || red == 5) h_I<1, 1>(a.h, a.ssp, a.ap, a.part, a.acc, gate, modF);
  else if (ks == 8) h_I<8, 0>(a.h, a.ssp, a.ap, a.part, a.acc, gate, modF);
  else h_I<16, 0>(a.h, a.ssp, a.ap, a.part, a.acc, gate, modF);
}

constexpr int H_NPH = 5, H_SL_CAL = SL_IT + NIT * H_NPH, H_SL_END = H_SL_CAL + 2 * NCAL, H_NSLOT = H_SL_END + 1;
constexpr int H_TL_IT = NIT - 2;  // iteration whose k1 timeline is recorded

__global__ void __launch_bounds__(NTHREADS, 1) h_kernel(const __grid_constant__ HArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
#if FG_STAMPS
  __shared__ ull stm[H_NSLOT];
  __shared__ ull gtm[2 * NCAL];
#endif
  cg::grid_group grid = cg::this_grid();
  bf16* smb = reinterpret_cast<bf16*>(smem_raw);
  const int tid = threadIdx.x;
  unsigned bar_epoch = 0;
#if FG_STAMPS
#define FG_STAMP(sl) do { if (tid == 0) stm[(sl)] = clock64(); } while (0)
#else
#define FG_STAMP(sl) do {} while (0)
#endif
#define SYNC(sl) do { grid_barrier(a.bar, ++bar_epoch); FG_STAMP(sl); } while (0)
  FG_STAMP(SL_START);
  if (tid == 0) {
    a.bar[blockIdx.x * BAR_STRIDE] = 0u;
    if (blockIdx.x == 0) a.bar[160 * BAR_STRIDE] = 0u;
  }
  grid.sync();
  FG_STAMP(SL_PRO);
#if FG_STAMPS
  if (blockIdx.x == 0 && tid == 0) a.gtv[0] = globaltimer();
#endif
  uint32_t sk = 0;
  for (int it = 0; it < NIT; ++it) {
    const int l = it % L, ln = (l + 1) % L;
    const int sb = SL_IT + it * H_NPH;
    // ---- D dummy: warm the H weights (mk phase D), 4 MB Wo read
    if (a.ctx & 1) prefetch_l2(a.wpf + (size_t)l * W * FF, (size_t)W * FF * 2);
    sk ^= h_read(a.wo + (size_t)l * W * QD, (size_t)W * QD * 2, it);
    SYNC(sb + 0);
    // ---- F dummy: 16 MB gate/up weights, fresh act rows 0..49
    if (a.ctx & 2) sk ^= h_read(a.wgu + (size_t)l * 2 * FF * W, (size_t)2 * FF * W * 2, it);
    for (size_t i = (size_t)blockIdx.x * NTHREADS + tid; i < (size_t)T * FF / 2; i += (size_t)gridDim.x * NTHREADS) {
      const uint32_t hv = fg_hash((uint32_t)i * 2u + (uint32_t)it * 104729u);
      reinterpret_cast<__nv_bfloat162*>(a.act)[i] =
          __floats2bfloat162_rn(((int)(hv & 0xffff) - 32768) * (0.5f / 32768.f), ((int)(hv >> 16) - 32768) * (0.5f / 32768.f));
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");  // act (generic stores) -> TMA reads after the barrier
    {
      const WPre w{a.bar + 160 * BAR_STRIDE, (++bar_epoch) * gridDim.x,
#if FG_STAMPS
                   &stm[sb + 1]
#else
                   nullptr
#endif
      };
      h_phase(a, (a.ctx & 4) ? 0 : l, w, smb,
#if FG_STAMPS
              it == H_TL_IT ? g_h_ts : nullptr
#else
              nullptr
#endif
      );
    }
    SYNC(sb + 2);
    if (it == NIT - 1) {  // h snapshot for the host reference (every variant)
      for (int i = blockIdx.x * NTHREADS + tid; i < T * W / 2; i += gridDim.x * NTHREADS)
        reinterpret_cast<unsigned*>(a.hsnap)[i] = __ldcg(reinterpret_cast<const unsigned*>(a.h) + i);
      grid_barrier(a.bar, ++bar_epoch);
    }
    // ---- I: gated residual of layer l + next attention norm's A'
    i_phase(a, a.mod + (size_t)((2 * l + 1) * 3 + 2) * W, a.mod + (size_t)(2 * ln) * 3 * W);
    SYNC(sb + 3);
    // ---- A..C dummy
    sk ^= h_read(a.wa + (size_t)l * NQKV * W, (size_t)NQKV * W * 2, it);
    fg_burn(a.ccyc, a.cwarps);
    SYNC(sb + 4);
  }
  if (sk == 0x9e3779b9u) a.sink[0] = sk;
  for (int r = 0; r < NCAL; ++r) {
    SYNC(H_SL_CAL + 2 * r);
#if FG_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r] = globaltimer();
#endif
    fg_spin(a.ncal);
    SYNC(H_SL_CAL + 2 * r + 1);
#if FG_STAMPS
    if (blockIdx.x == 0 && tid == 0) gtm[2 * r + 1] = globaltimer();
#endif
  }
  FG_STAMP(H_SL_END);
#if FG_STAMPS
  if (blockIdx.x == 0 && tid == 0) {
    a.gtv[1] = globaltimer();
    for (int i = 0; i < 2 * NCAL; ++i) a.gtv[2 + i] = gtm[i];
  }
  __syncthreads();
  for (int i = tid; i < H_NSLOT; i += NTHREADS) a.stamps[(size_t)blockIdx.x * H_NSLOT + i] = stm[i];
#endif
#undef SYNC
#undef FG_STAMP
}

// fp32 reference of layer 17's H + I from the h snapshot: out = act . Wd^T, h_ref = rbf(hsnap + rbf(rbf(out) * rbf(gate))). Tolerance from the
// kernel's roundings (bf16 partials <= s / 512 with s = sum |act w|, the out rounding, the h rounding), x2 safety.
__global__ void h_ref(const bf16* act, const bf16* wd, const bf16* hsnap, const float* gate, const bf16* h, unsigned* maxerr, unsigned* maxref, unsigned* nbad, unsigned* maxratio) {
  const int t = blockIdx.x;
  for (int n = threadIdx.x; n < W; n += blockDim.x) {
    float o = 0.f, s = 0.f;
    for (int k = 0; k < FF; ++k) {
      const float p = __bfloat162float(act[(size_t)t * FF + k]) * __bfloat162float(wd[(size_t)n * FF + k]);
      o += p; s += fabsf(p);
    }
    const float g = gate[n];
    const float hr = rbf(__bfloat162float(hsnap[t * W + n]) + rbf(rbf(o) * rbf(g)));
    const float e = fabsf(__bfloat162float(h[t * W + n]) - hr);
    const float tol = 2.f * (fabsf(g) * (s / 256.f + fabsf(o) / 128.f) + fabsf(hr) / 128.f) + 1e-6f;
    atomicMax(maxerr, isfinite(e) ? __float_as_uint(e) : 0x7f800000u);
    atomicMax(maxref, __float_as_uint(fabsf(hr)));
    if (!(e <= tol)) atomicAdd(nbad, 1u);
    const float ratio = fminf(e / tol, 1e30f);
    atomicMax(maxratio, isfinite(ratio) ? __float_as_uint(ratio) : 0x7f800000u);
  }
}

// Wd [L * W][FF] -> pre-tiled [l][n / RPC][ks][chunk][r][64] (the H weight slice of each CTA contiguous)
__global__ void h_tile(const bf16* wd, bf16* wt, int rpc, int ks_n) {
  const int ksl = FF / ks_n, nch = ksl / 64;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)L * W * FF; i += (size_t)gridDim.x * blockDim.x) {
    const size_t row = i / FF; const int k = (int)(i % FF);
    const int l = (int)(row / W), n = (int)(row % W), blk = n / rpc, r = n % rpc, ks = k / ksl, c = (k % ksl) / 64, kk = k % 64;
    wt[((((size_t)(l * (W / rpc) + blk) * ks_n + ks) * nch + c) * rpc + r) * 64 + kk] = wd[i];
  }
}

// ------------------------------------------------------------- host
static const char* h_vname(int v) {
  if (v == HV_NULL) return "null";
  if (v == HV_K0) return "k0";
  if (v >= HV_K1 && v < HV_K1 + NHK1) return HK1_NAME[v - HV_K1];
  return "var";
}
static int h_parse(const std::string& s) {
  if (s == "null") return HV_NULL;
  if (s == "k0") return HV_K0;
  for (int i = 0; i < NHK1; ++i)
    if (s == HK1_NAME[i]) return ((H_MASK >> i) & 1) ? HV_K1 + i : -1;
  return -1;
}

int main(int argc, char** argv) {
  int reps = 20, cwarps = 0, ctx = 3; long long ncal = 35000, ccyc = 20000;
  std::string out = ".", name = std::string("h") + (FG_STAMPS ? "" : "_ns"), vlist = "null,k0";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i]; const char* v = argv[i + 1];
    if (k == "--reps") reps = atoi(v); else if (k == "--cwarps") cwarps = atoi(v); else if (k == "--ncal") ncal = atoll(v); else if (k == "--ccyc") ccyc = atoll(v); else if (k == "--ctx") ctx = atoi(v);
    else if (k == "--out") out = v; else if (k == "--name") name = v; else if (k == "--variants") vlist = v;
    else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  std::vector<int> vs;
  for (size_t p = 0; p < vlist.size();) {
    size_t q = vlist.find(',', p); if (q == std::string::npos) q = vlist.size();
    const std::string s = vlist.substr(p, q - p);
    vs.push_back(h_parse(s));
    if (vs.back() < 0) { fprintf(stderr, "bad variant %s (H_MASK=0x%x)\n", s.c_str(), H_MASK); return 2; }
    p = q + 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(h_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)AB_SMEM_DYN));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, h_kernel, NTHREADS, AB_SMEM_DYN));
  fprintf(stderr, "%s: %s grid=%d occ/SM=%d stamps=%d reps=%d ncal=%lld ccyc=%lld cwarps=%d ctx=%d mask=0x%x\n", name.c_str(), prop.name, grid, occ, FG_STAMPS, reps, ncal, ccyc, cwarps, ctx, H_MASK);
  if (occ < 1 || grid != 148) { fprintf(stderr, "kernel does not fit / grid != 148\n"); return 1; }
  std::string jpath = out + "/" + name + ".jsonl";
  FILE* jf = fopen(jpath.c_str(), "w"); if (!jf) { perror(jpath.c_str()); return 1; }
#define JOUT(...) do { fprintf(jf, __VA_ARGS__); fprintf(stdout, __VA_ARGS__); } while (0)

  const size_t nwd = (size_t)L * W * FF, nwo = (size_t)L * W * QD, nwgu = (size_t)L * 2 * FF * W, nwa = (size_t)L * NQKV * W;
  bf16 *wd, *wo, *wgu, *wa, *act, *h, *hsnap, *ap; float *mod, *ssp, *part, *acc; unsigned *bar, *sink, *dref; ull *stamps, *gtv; char* flush;
  CK(cudaMalloc(&wd, nwd * 2)); CK(cudaMalloc(&wo, nwo * 2)); CK(cudaMalloc(&wgu, nwgu * 2)); CK(cudaMalloc(&wa, nwa * 2));
  CK(cudaMalloc(&act, (size_t)64 * FF * 2)); CK(cudaMemset(act, 0, (size_t)64 * FF * 2));  // rows 50..63 stay zero (rank 1's B half)
  CK(cudaMalloc(&h, (size_t)T * W * 2)); CK(cudaMalloc(&hsnap, (size_t)T * W * 2));
  CK(cudaMalloc(&ap, (size_t)64 * W * 2)); CK(cudaMemset(ap, 0, (size_t)64 * W * 2));
  CK(cudaMalloc(&mod, (size_t)2 * L * 3 * W * 4)); CK(cudaMalloc(&ssp, (size_t)T * NSSP * 4));
  CK(cudaMalloc(&part, (size_t)(FF / KC) * T * W * 2)); CK(cudaMalloc(&acc, (size_t)T * W * 4));
  CK(cudaMalloc(&bar, BAR_WORDS * 4)); CK(cudaMalloc(&sink, 64)); CK(cudaMalloc(&dref, 32));
  CK(cudaMalloc(&stamps, (size_t)grid * H_NSLOT * 8)); CK(cudaMalloc(&gtv, (2 + 2 * NCAL) * 8));
  const size_t flush_bytes = (size_t)2 * prop.l2CacheSize; CK(cudaMalloc(&flush, flush_bytes));
  CK(cudaMemset(part, 0, (size_t)(FF / KC) * T * W * 2)); CK(cudaMemset(bar, 0, BAR_WORDS * 4)); CK(cudaMemset(sink, 0, 64));
  ab_init_bf16<<<1184, 256>>>(wd, nwd, 17, 0.05f); ab_init_bf16<<<1184, 256>>>(wo, nwo, 16, 0.05f);
  ab_init_bf16<<<1184, 256>>>(wgu, nwgu, 22, 0.05f); ab_init_bf16<<<1184, 256>>>(wa, nwa, 13, 0.05f);
  ab_init_f32<<<148, 256>>>(mod, (size_t)2 * L * 3 * W, 21, 0.3f, 0.1f);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  HArgs args{};
  args.wd = wd; args.wo = wo; args.wgu = wgu; args.wa = wa; args.act = act; args.mod = mod; args.h = h; args.hsnap = hsnap; args.ssp = ssp; args.part = part;
  args.acc = acc; args.ap = ap; args.bar = bar; args.sink = sink; args.stamps = stamps; args.gtv = gtv; args.cwarps = cwarps; args.ccyc = ccyc; args.ncal = ncal; args.ctx = ctx;
  {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    auto mk = [&](CUtensorMap* m, CUtensorMapDataType dt, void* base, cuuint64_t d0, cuuint64_t d1, cuuint64_t stride, cuuint32_t b0, cuuint32_t b1, CUtensorMapSwizzle sw) {
      cuuint64_t dims[2] = {d0, d1}, strides[1] = {stride};
      cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
      CUresult r = enc(m, dt, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); exit(1); }
    };
    mk(&args.mw64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, wd, FF, (cuuint64_t)L * W, (cuuint64_t)FF * 2, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    mk(&args.mw128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, wd, FF, (cuuint64_t)L * W, (cuuint64_t)FF * 2, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    for (int i = 0; i < 2; ++i)
      for (int j = 1; j < 4; ++j) {
        cuuint64_t dims[3] = {64, (cuuint64_t)L * W, FF / 64}, strides[2] = {(cuuint64_t)FF * 2, 128};
        cuuint32_t box[3] = {64, (cuuint32_t)(64 << i), (cuuint32_t)(1 << j)}, es[3] = {1, 1, 1};
        CUresult r = enc(&args.mwx[i][j], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, wd, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { fprintf(stderr, "3D weight map encode failed %d (i %d j %d)\n", (int)r, i, j); exit(1); }
      }
    for (int i = 0; i < 2; ++i) {
      bf16* t; CK(cudaMalloc(&t, nwd * 2));
      h_tile<<<1184, 256>>>(wd, t, i ? 128 : 64, i ? 16 : 8);
      args.wt[i] = t;
      for (int j = 0; j < 3; ++j)
        if ((64 << i) << j <= 256)
          mk(&args.mwt[i][j], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, t, 64, (cuuint64_t)L * W * FF / 64, 128, 64, (cuuint32_t)((64 << i) << j), CU_TENSOR_MAP_SWIZZLE_128B);
    }
    CK(cudaDeviceSynchronize());
    {
      cuuint64_t dims[3] = {64, 64, FF / 64}, strides[2] = {(cuuint64_t)FF * 2, 128};
      cuuint32_t box[3] = {64, 32, 4}, es[3] = {1, 1, 1};
      if (enc(&args.mact3, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, act, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
              CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) { fprintf(stderr, "act3 map\n"); exit(1); }
    }
    mk(&args.mact, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, act, FF, 64, (cuuint64_t)FF * 2, 64, 32, CU_TENSOR_MAP_SWIZZLE_128B);
    mk(&args.mr64, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, acc, W, T, (cuuint64_t)W * 4, 64, T, CU_TENSOR_MAP_SWIZZLE_NONE);
    mk(&args.mr128, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, acc, W, T, (cuuint64_t)W * 4, 128, T, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  struct Rec { int variant, idx; double ev_ms = 0, nspc = 0, hi_ns = 0, stage_ns[H_NPH] = {0}, calib_cyc = 0, calib_gt = 0, closure = 0, maxerr = 0, maxref = 0, maxratio = 0;
               unsigned nbad = 0, timeouts = 0; long ord_viol = 0; double ord_min_ns = 0; };
  std::vector<Rec> recs;
  std::vector<ull> st((size_t)grid * H_NSLOT), gt(2 + 2 * NCAL);
  const std::string rawdir = out + "/raw"; { std::string c = "mkdir -p " + rawdir; if (system(c.c_str())) return 1; }
  std::vector<char> dumped(HV_K1 + NHK1, 0);
  auto run = [&](int variant, int idx, bool record) {
    CK(cudaMemsetAsync(flush, 0, flush_bytes));
    ab_init_bf16<<<148, 256>>>(h, (size_t)T * W, 31, 1.0f);
    CK(cudaMemsetAsync(acc, 0, (size_t)T * W * 4));
    CK(cudaMemsetAsync(stamps, 0, (size_t)grid * H_NSLOT * 8));
    args.variant = variant;
    args.wpf = wd;
    if (variant >= HV_K1) {
      static const int wis[] = {
#define HK1_W(ii, r, k, d, wi, n) wi,
          HK1_LIST(HK1_W)};
      static const int rpcs[] = {
#define HK1_R(ii, r, k, d, wi, n) r,
          HK1_LIST(HK1_R)};
      if (wis[variant - HV_K1] >= 16) args.wpf = args.wt[rpcs[variant - HV_K1] == 128];
    }
    CK(cudaEventRecord(e0));
    {
      cudaLaunchConfig_t lc = {}; lc.gridDim = dim3(grid); lc.blockDim = dim3(NTHREADS); lc.dynamicSmemBytes = AB_SMEM_DYN; lc.stream = 0;
      cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
      at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
      lc.attrs = at; lc.numAttrs = 2;
      CK(cudaLaunchKernelEx(&lc, h_kernel, args));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaGetLastError());
    if (!record) return;
    unsigned to = 0; CK(cudaMemcpyFromSymbol(&to, g_ab_to, 4)); { unsigned z = 0; CK(cudaMemcpyToSymbol(g_ab_to, &z, 4)); }
    if (to) fprintf(stderr, "WARNING: %u mbarrier wait timeouts (variant %s launch %d)\n", to, h_vname(variant), idx);
    const int lr = (NIT - 1) % L;
    CK(cudaMemset(dref, 0, 32));
    h_ref<<<T, 256>>>(act, wd + (size_t)((ctx & 4) ? 0 : lr) * W * FF, hsnap, mod + (size_t)((2 * lr + 1) * 3 + 2) * W, h, dref, dref + 1, dref + 2, dref + 3);
    unsigned hr[4]; CK(cudaMemcpy(hr, dref, 16, cudaMemcpyDeviceToHost));
    Rec r; r.variant = variant; r.idx = idx; r.ev_ms = ms; r.timeouts = to;
    r.maxerr = *reinterpret_cast<float*>(&hr[0]); r.maxref = *reinterpret_cast<float*>(&hr[1]); r.nbad = hr[2]; r.maxratio = *reinterpret_cast<float*>(&hr[3]);
    if (idx == 0 && !dumped[variant]) {
      dumped[variant] = 1;
      std::vector<unsigned short> o((size_t)T * W);
      CK(cudaMemcpy(o.data(), h, o.size() * 2, cudaMemcpyDeviceToHost));
      char fn[512]; snprintf(fn, sizeof fn, "%s/%s.h.u16", rawdir.c_str(), h_vname(variant));
      FILE* f = fopen(fn, "wb"); fwrite(o.data(), 2, o.size(), f); fclose(f);
    }
#if FG_STAMPS
    CK(cudaMemcpy(st.data(), stamps, st.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gt.data(), gtv, gt.size() * 8, cudaMemcpyDeviceToHost));
    const ull* s0 = st.data();
    r.nspc = (double)(gt[1] - gt[0]) / (double)(s0[H_SL_END] - s0[SL_PRO]);
    const double total_ns = (double)(s0[H_SL_END] - s0[SL_START]) * r.nspc;
    r.closure = (total_ns - ms * 1e6) / (ms * 1e6);
    for (int k = 0; k < H_NPH; ++k) {
      double sum = 0;
      for (int i = 0; i < NIT; ++i) { const int sl = SL_IT + i * H_NPH + k; sum += (double)(s0[sl] - s0[sl - 1]); }
      r.stage_ns[k] = sum / NIT * r.nspc;
    }
    r.hi_ns = r.stage_ns[2] + r.stage_ns[3];  // F->H barrier exit .. I->A barrier exit
    std::vector<double> pe, pd, ge, gd;
    for (int qq = 0; qq < NCAL; ++qq) {
      const int a0 = H_SL_CAL + 2 * qq;
      pe.push_back((double)(s0[a0] - s0[a0 - 1])); pd.push_back((double)(s0[a0 + 1] - s0[a0]));
      if (qq > 0) ge.push_back((double)(gt[2 + 2 * qq] - gt[2 + 2 * qq - 1]));
      gd.push_back((double)(gt[2 + 2 * qq + 1] - gt[2 + 2 * qq]));
    }
    r.calib_cyc = fg_median(pd) - fg_median(pe);
    r.calib_gt = (fg_median(gd) - fg_median(ge)) / ((double)ncal * r.nspc);
    r.ord_min_ns = 1e18;
    for (int c = 0; c < grid; ++c)
      for (int i = 1; i < H_NSLOT; ++i) {
        const long long d = (long long)(st[(size_t)c * H_NSLOT + i] - st[(size_t)c * H_NSLOT + i - 1]);
        if (d < 0) ++r.ord_viol;
        r.ord_min_ns = std::min(r.ord_min_ns, d * r.nspc);
      }
    if (variant >= HV_K1 && idx >= reps - 2) {
      std::vector<ull> ts((size_t)148 * H_NTS);
      CK(cudaMemcpyFromSymbol(ts.data(), g_h_ts, ts.size() * 8));
      char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%s_%03d.tl.u64", rawdir.c_str(), name.c_str(), h_vname(variant), idx);
      FILE* f = fopen(fn, "wb"); fwrite(&r.nspc, 8, 1, f); fwrite(ts.data(), 8, ts.size(), f); fclose(f);
    }
#endif
    recs.push_back(r);
  };
  for (int v : vs) for (int i = 0; i < 3; ++i) run(v, -1, false);
  for (int r = 0; r < reps; ++r) {
    std::vector<int> ord = vs;
    if (r & 1) std::reverse(ord.begin(), ord.end());
    for (int v : ord) run(v, r, true);
  }
  auto gate = [&](const Rec& r, const char* g, bool pass, const char* valfmt, double x, double y = 0, double z = 0) {
    char vb[256]; snprintf(vb, sizeof vb, valfmt, x, y, z);
    JOUT("{\"launch\":%d,\"variant\":\"%s\",\"gate\":\"%s\",\"pass\":%s,\"val\":%s}\n", r.idx, h_vname(r.variant), g, pass ? "true" : "false", vb);
  };
  auto var_nspc = [&](int v) { std::vector<double> x; for (auto& r : recs) if (r.variant == v) x.push_back(r.nspc); return fg_median(x); };
  int nfail = 0;
  for (auto& r : recs) {
    JOUT("{\"rec\":\"launch\",\"name\":\"%s\",\"launch\":%d,\"variant\":\"%s\",\"event_ms\":%.6f,\"nspc\":%.6f,\"stage_ns\":[%.1f,%.1f,%.1f,%.1f,%.1f],\"HI_ns\":%.1f,"
         "\"timeouts\":%u,\"maxratio\":%.4f,\"maxerr\":%.6g,\"nbad\":%u}\n",
         name.c_str(), r.idx, h_vname(r.variant), r.ev_ms, r.nspc, r.stage_ns[0], r.stage_ns[1], r.stage_ns[2], r.stage_ns[3], r.stage_ns[4], r.hi_ns, r.timeouts,
         r.maxratio, r.maxerr, r.nbad);
    const bool wp = r.variant == HV_NULL ? r.timeouts == 0 : (r.nbad == 0 && std::isfinite(r.maxerr) && r.timeouts == 0);
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
  for (int v : vs) {
    if (v == HV_NULL) continue;
    std::vector<double> evv, evn, l1v, l1n, ev_pair, allv, alln, st_v[H_NPH], st_n[H_NPH];
    for (auto& r : recs) {
      double tot = 0; for (int k = 0; k < H_NPH; ++k) tot += r.stage_ns[k];
      if (r.variant == v) { evv.push_back(r.ev_ms); l1v.push_back(r.hi_ns); allv.push_back(tot); for (int k = 0; k < H_NPH; ++k) st_v[k].push_back(r.stage_ns[k]); }
      if (r.variant == HV_NULL) { evn.push_back(r.ev_ms); l1n.push_back(r.hi_ns); alln.push_back(tot); for (int k = 0; k < H_NPH; ++k) st_n[k].push_back(r.stage_ns[k]); }
    }
    if (evn.empty()) continue;
    const double L0 = (fg_median(evv) - fg_median(evn)) * 1e3 / NIT;
    for (size_t i = 0; i < std::min(evv.size(), evn.size()); ++i) ev_pair.push_back((evv[i] - evn[i]) * 1e3 / NIT);
#if FG_STAMPS
    const double cdev = var_nspc(v) / var_nspc(HV_NULL) - 1.0;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"clock_cross\",\"pass\":%s,\"val\":{\"nspc\":%.6f,\"null_nspc\":%.6f,\"dev\":%.5f}}\n",
         h_vname(v), fabs(cdev) < 0.015 ? "true" : "false", var_nspc(v), var_nspc(HV_NULL), cdev);
    const double Lwin = (fg_median(l1v) - fg_median(l1n)) / 1e3, Lall = (fg_median(allv) - fg_median(alln)) / 1e3;
    const bool g3 = fabs(L0 - Lall) < 0.2;
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"delta_agreement\",\"pass\":%s,\"val\":{\"L0_us\":%.4f,\"L1_all_us\":%.4f,\"diff_us\":%.4f,"
         "\"L1_window_us\":%.4f,\"outside_window_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f,\"stage_delta_us\":[%.3f,%.3f,%.3f,%.3f,%.3f]}}\n",
         h_vname(v), g3 ? "true" : "false", L0, Lall, L0 - Lall, Lwin, Lall - Lwin, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9),
         (fg_median(st_v[0]) - fg_median(st_n[0])) / 1e3, (fg_median(st_v[1]) - fg_median(st_n[1])) / 1e3, (fg_median(st_v[2]) - fg_median(st_n[2])) / 1e3,
         (fg_median(st_v[3]) - fg_median(st_n[3])) / 1e3, (fg_median(st_v[4]) - fg_median(st_n[4])) / 1e3);
    nfail += !g3;
#else
    JOUT("{\"launch\":-1,\"variant\":\"%s\",\"gate\":\"L0_delta\",\"pass\":true,\"val\":{\"L0_us\":%.4f,\"L0_pair_p10\":%.4f,\"L0_pair_p90\":%.4f}}\n",
         h_vname(v), L0, fg_pct(ev_pair, 0.1), fg_pct(ev_pair, 0.9));
#endif
  }
  for (int v : vs) {
    std::vector<double> ev, hi; for (auto& r : recs) if (r.variant == v) { ev.push_back(r.ev_ms); hi.push_back(r.hi_ns); }
    JOUT("{\"rec\":\"summary\",\"name\":\"%s\",\"variant\":\"%s\",\"stamps\":%d,\"n\":%zu,\"event_ms_med\":%.6f,\"event_ms_p10\":%.6f,\"event_ms_p90\":%.6f,"
         "\"per_layer_us_med\":%.4f,\"HI_ns_med\":%.1f}\n", name.c_str(), h_vname(v), FG_STAMPS, ev.size(), fg_median(ev), fg_pct(ev, 0.1), fg_pct(ev, 0.9),
         fg_median(ev) * 1e3 / NIT, fg_median(hi));
  }
  JOUT("{\"rec\":\"done\",\"name\":\"%s\",\"failed_gates\":%d}\n", name.c_str(), nfail);
  fclose(jf);
  return 0;
}
