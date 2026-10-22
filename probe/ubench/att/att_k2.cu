// att_k2.cu: K2 = HMMA (mma.sync m16n8k16) attention for the pi0.5 megakernel's phase C + C3, with the data movement / work decomposition / combine
// restructured (TMA bulk tensor loads + mbarrier, DSMEM pair combine, balanced decomposition). The math is K0's (same mma / ldmatrix fragments, same
// bf16 roundings: score = bf16(q.k)/16 masked, chunk-local max, p = bf16(exp(s - m)), l = fp32 sum p, partial O = P.V in fp32, combine with exp(m_c - M)).
// Variants run in one binary together with null / k0 / calib (K0 is included verbatim from att_k0.cu, its main() renamed away).
//
// Decomposition (all K2 variants): unit = (32 query rows, key chunk). 13 row groups (rows 0..415, the last has 16 valid) x 11 key chunks = 143 units on
// CTAs 0..142 (K0: 112 of 148 CTAs). Chunks: 10 prefix chunks of 96 keys (960 keys) + the tail chunk of 64 keys = prefix keys 960..967 + the 50 suffix keys
// + 6 pad. Tail chunks sit on the LOWEST block indices (the B tasks run on the highest ones). Warp tiling is not redundant (K0 computes QK^T twice):
//   QK^T: warp (mt = w & 1, kg = w >> 1) = 16 rows x (KC / 4) keys over the full HD -> scaled/masked score (bf16, exact) into smem S [32][104]
//   softmax: warp w owns rows 4w..4w+3; chunk max m, p = bf16(exp(s - m)) written back over S, l = sum p -> stats[row][kc] = (m, l)
//   P.V : warp w = output dims [32w, 32w + 32) for both 16-row m-tiles, A = P by ldmatrix from S, B = V by ldmatrix.trans.
// smem tiles are 128B-swizzled (the TMA SWIZZLE_128B layout, 4 column blocks of 64 elements, each [rows][128 B]) whether filled by TMA or cp.async.
// Partial O goes to opart[kc][row][256] fp32 in the XOR-permuted layout (float4 chunk index ^ (row & 7)); all 143 CTAs then combine 2-3 rows each
// (the rows of their own row group, row = rg * 32 + kc + 11 * s), after the row group's 11 units have released the counter.
//
// Variant flags (K2_LIST): TMA loads (else cp.async into the same swizzled layout), PRE (prefix K/V issued in B_pre, before the B tasks; else all in C),
// BULK (partial O staged in smem and written by one cp.async.bulk (1-D TMA store) instead of per-thread float2 stores).
// Build (login node): ./build.sh att_k2 -DATT_HOOKS      (stamped, _ns, _mk twins)       Run: att_k2 --variants null,k0,k2_...,calib --smem 197632 --cluster 2
#include <cudaTypedefs.h>
#include <cuda.h>
#include <string>
#include <type_traits>
#include "att_common.cuh"
template <class VT> int att_main_unused(int, char**) { return 0; }  // att_k0.cu's main() instantiates att_main<K0> (no hooks): route it here
#define main att_k0_main_unused
#define att_main att_main_unused
#include "att_k0.cu"
#undef att_main
#undef main

// ------------------------------------------------------------------------------------------------ variant table
constexpr int F_TMA = 1, F_PRE = 2, F_BULK = 4, F_LDQ = 8, F_FEXP = 16, F_V4 = 32, F_LL = 64, F_LLG = 128, F_PAIR = 256, F_PMAP = 512, F_BF = 1024, F_BFH = 2048, F_BFC = 4096;  // BFC: coalesced combine: stats [kc][row], combine warp = (octet, half), lane = row of the group (k1's mapping); else rows k = kc + 11 s, lane = octet  // PAIR: cluster-pair (CTA 2i, 2i+1 = same row group, adjacent key chunks) exchange half the partial O over DSMEM (st.async + mbarrier) and merge before the global partial store
// (LL: see below)  // LL: no counter / release - partial O and stats carry an 8-bit epoch tag in the low mantissa byte of every fp32, consumers poll the data (LLG: stats first)  // LDQ: q (and the tail's suffix K/V) by ld.global -> st.shared instead of TMA / cp.async
#ifndef K2_MASK
#define K2_MASK 0x1f0fff
#endif
#define K2_ON(i) (((K2_MASK) >> (i)) & 1)
#define K2_LIST(X) \
  X(0, "k2_cp", F_PRE) \
  X(1, "k2_cp_bulk", F_PRE | F_BULK) \
  X(2, "k2_tma", F_TMA | F_PRE) \
  X(3, "k2_tma_bulk", F_TMA | F_PRE | F_BULK) \
  X(4, "k2_tma_bulk_nopre", F_TMA | F_BULK) \
  X(5, "k2_ldq", F_TMA | F_PRE | F_LDQ) \
  X(6, "k2_cp_ldq", F_PRE | F_LDQ) \
  X(7, "k2_ldq_bulk", F_TMA | F_PRE | F_LDQ | F_BULK) \
  X(8, "k2_tma_fexp", F_TMA | F_PRE | F_FEXP) \
  X(9, "k2_tma_v4", F_TMA | F_PRE | F_V4) \
  X(10, "k2_tma_fexp_v4", F_TMA | F_PRE | F_FEXP | F_V4) \
  X(11, "k2_ldq_fexp", F_TMA | F_PRE | F_LDQ | F_FEXP) \
  X(12, "k2_tma_ll", F_TMA | F_PRE | F_LL) \
  X(13, "k2_tma_llg", F_TMA | F_PRE | F_LL | F_LLG) \
  X(14, "k2_cp_ll", F_PRE | F_LL) \
  X(15, "k2_tma_ll_fexp", F_TMA | F_PRE | F_LL | F_FEXP) \
  X(16, "k2_pair", F_TMA | F_PRE | F_PAIR) \
  X(17, "k2_pair_cp", F_PRE | F_PAIR) \
  X(18, "k2_tma_pmap", F_TMA | F_PRE | F_PMAP) \
  X(19, "k2_tma_bf", F_TMA | F_PRE | F_BF | F_BFC) \
  X(20, "k2_tma_bf_h", F_TMA | F_PRE | F_BF | F_BFC | F_BFH) \
  X(21, "k2_tma_bfr", F_TMA | F_PRE | F_BF) \
  X(22, "k2_tma_bfr_h", F_TMA | F_PRE | F_BF | F_BFH)
static const char* att_ext_vname(int v) {
#define X(i, n, fl) if (v == ATTV_EXT + i) return n;
  K2_LIST(X)
#undef X
  return "var";
}
static int att_ext_parse(const std::string& s) {
#define X(i, n, fl) if (s == n) return ATTV_EXT + i;
  K2_LIST(X)
#undef X
  return -1;
}

// ------------------------------------------------------------------------------------------------ geometry
constexpr int R2 = 32, NRG2 = (QROWS + R2 - 1) / R2;       // 13
constexpr int KP2 = 96, NPC2 = 10, KT2 = 64, NCH2 = NPC2 + 1;  // prefix chunks of 96, tail chunk of 64 (8 prefix + 50 suffix + pad)
constexpr int NUNIT2 = NRG2 * NCH2;                         // 143
static_assert(NUNIT2 <= 148 && NPC2 * KP2 + 8 == P, "chunking");
constexpr int SS_STRIDE = KP2 + 8;                          // S / P tile row stride (elements)
// smem map (bytes from the 1024-aligned base)
constexpr uint32_t OFF_K = 0, OFF_V = 4 * KP2 * 128, OFF_Q = OFF_V + 4 * KP2 * 128, OFF_S = OFF_Q + 4 * R2 * 128;
constexpr uint32_t OFF_MK = OFF_S + R2 * SS_STRIDE * 2, OFF_BAR = OFF_MK + 128, OFF_X = OFF_BAR + 64, OFF_END = OFF_X + 1024;
constexpr uint32_t OFF_RV = (OFF_END + 1023u) & ~1023u, OFF_RS = OFF_RV + 4 * 256 * 16, OFF_ML = OFF_RS + 256, OFF_RB = OFF_ML + 256, OFF_END2 = OFF_RB + 64;  // PAIR: partner half-O [nt][tid] float4, partner (m,l)[32], own (m,l)[32], mbarrier
constexpr size_t K2_SMEM = ATT_SMEM_UMMA;
static_assert(OFF_END2 + 1024 <= ATT_SMEM_UMMA, "smem");
constexpr int KBLK = KP2 * 128, QBLK = R2 * 128;            // bytes of one 64-column block of a K/V / Q tile

__constant__ __align__(64) CUtensorMap c_tmK, c_tmV, c_tmK8, c_tmV8, c_tmKs, c_tmVs, c_tmQ;
__device__ unsigned g_k2_to;                                // mbarrier wait timeouts

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
  for (ull i = 0; i < (1ull << 22) && !ok; ++i)
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) atomicAdd(&g_k2_to, 1u);
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a)); }
__device__ __forceinline__ void ldsm_x4a(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a)); }
__device__ __forceinline__ void ldsm_x4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a)); }
__device__ __forceinline__ void cp16s(uint32_t dst, const void* g, int nb) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(g), "r"(nb) : "memory"); }

// ---- LL protocol helpers: every fp32 of the partial O / stats carries tag = (it % 255) + 1 in its low byte (the consumer masks it off)
__device__ __forceinline__ unsigned k2_tag(int it) { return (unsigned)(it % 255) + 1u; }
__device__ __forceinline__ float k2_tagf(float x, unsigned tag) { return __uint_as_float(((__float_as_uint(x) + 0x80u) & ~0xffu) | tag); }
__device__ __forceinline__ uint4 ld_vol4(const void* p) {
  uint4 v; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint2 ld_vol2(const void* p) {
  uint2 v; asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory"); return v; }

// by-value argument block of the noinline unit / combine functions (no references into the caller's frame)
struct UA {
  const bf16 *pk, *pv, *ksuf, *vsuf, *q; const uint8_t* mask; bf16* attn; float *stats, *opart; unsigned* cnt; unsigned char* smem; int fuse;
};
template <int FL>
__device__ __forceinline__ void k2_map(int blk, int& rg, int& kc) {
  if (FL & (F_PAIR | F_PMAP)) {  // pair (2i, 2i+1): same row group, key chunks 2j, 2j+1 (j = 0..4); the 13 tail units (chunk 10) follow on blocks 130..142
    if (blk < NRG2) { rg = blk; kc = NCH2 - 1; }  // tail units on the lowest blocks (as K0 / k2_tma: the dummy B tasks run on the highest ones); block 13 idles
    else { const int p = (blk - 14) >> 1; rg = p % NRG2; kc = 2 * (p / NRG2) + (blk & 1); }
  } else { rg = blk % NRG2; kc = NCH2 - 1 - blk / NRG2; }
}
template <int FL>
__device__ __forceinline__ bool k2_idle(int blk) { return (FL & (F_PAIR | F_PMAP)) ? (blk == NRG2 || blk >= NRG2 + 1 + 2 * 65) : blk >= NUNIT2; }
__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, uint32_t rank) { uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(rank)); return o; }
__device__ __forceinline__ void st_async4(uint32_t a, float x, float y, float z, float w, uint32_t bar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w), "r"(bar) : "memory"); }
__device__ __forceinline__ void st_async2(uint32_t a, float x, float y, uint32_t bar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(a), "f"(x), "f"(y), "r"(bar) : "memory"); }
__device__ __forceinline__ unsigned char* k2_base(bf16* smb) {  // 1024-aligned base inside the dynamic smem
  const uint32_t s = su32(smb);
  return reinterpret_cast<unsigned char*>(smb) + (((s + 1023u) & ~1023u) - s);
}
// byte offset of (row, 16B chunk c of 32) in a tile of `rows` rows (4 column blocks of 64 elements, rows x 128 B, 128B-swizzled)
__device__ __forceinline__ uint32_t sw_off(int rows, int row, int c) { return (c >> 3) * rows * 128 + row * 128 + ((((c & 7) ^ row) & 7) << 4); }

// ---- tile fills
template <int FL>
__device__ __forceinline__ void k2_fill_prefix(const UA& u, uint32_t sb, int l, int kc, bool tail) {  // K and V prefix rows of chunk kc (thread 0 for TMA; all for cp.async)
  const int tid = threadIdx.x, j0 = kc * KP2, rows = tail ? 8 : KP2;
  if (FL & F_TMA) {
    if (tid == 0) {
      const uint32_t bK = sb + OFF_BAR, bV = bK + 8;
      mbar_expect(bK, 4 * rows * 128); mbar_expect(bV, 4 * rows * 128);
#pragma unroll
      for (int cb = 0; cb < 4; ++cb) {
        tma2d(sb + OFF_K + cb * KBLK, tail ? &c_tmK8 : &c_tmK, bK, cb * 64, l * P + j0);
        tma2d(sb + OFF_V + cb * KBLK, tail ? &c_tmV8 : &c_tmV, bV, cb * 64, l * P + j0);
      }
    }
  } else {
    const bf16* pk = u.pk + (size_t)l * P * HD; const bf16* pv = u.pv + (size_t)l * P * HD;
    for (int i = tid; i < rows * 32; i += NTHREADS) {
      const int r = i >> 5, c = i & 31;
      cp16s(sb + OFF_K + sw_off(KP2, r, c), pk + (size_t)(j0 + r) * HD + c * 8, 16);
      cp16s(sb + OFF_V + sw_off(KP2, r, c), pv + (size_t)(j0 + r) * HD + c * 8, 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
}
template <int FL>
__device__ __forceinline__ void k2_fill_post(const UA& u, uint32_t sb, int rg, bool tail) {  // q tile (all units) and the suffix K/V rows (tail unit)
  const int tid = threadIdx.x;
  if (FL & F_LDQ) {  // register-staged copies (no async proxy, no fence): all loads in flight, then the swizzled smem stores
    constexpr int NQ = R2 * 32 / NTHREADS;  // 4 uint4 per thread
    uint4 qv[NQ];
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
      const int i = tid + j * NTHREADS, r = i >> 5, c = i & 31, row = rg * R2 + r;
      qv[j] = row < QROWS ? __ldcg(reinterpret_cast<const uint4*>(u.q + (size_t)row * HD + c * 8)) : make_uint4(0, 0, 0, 0);
    }
    if (tail) {  // runtime branch: the suffix registers are only live inside it
      constexpr int NS = 56 * 32 / NTHREADS;  // 7
      uint4 sk[NS], sv[NS];
#pragma unroll
      for (int j = 0; j < NS; ++j) {
        const int i = tid + j * NTHREADS, r = i >> 5, c = i & 31;
        sk[j] = r < T ? __ldcg(reinterpret_cast<const uint4*>(u.ksuf + (size_t)r * HD + c * 8)) : make_uint4(0, 0, 0, 0);
        sv[j] = r < T ? __ldcg(reinterpret_cast<const uint4*>(u.vsuf + (size_t)r * HD + c * 8)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < NS; ++j) {
        const int i = tid + j * NTHREADS, r = i >> 5, c = i & 31;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(sb + OFF_K + sw_off(KP2, 8 + r, c)), "r"(sk[j].x), "r"(sk[j].y), "r"(sk[j].z), "r"(sk[j].w) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(sb + OFF_V + sw_off(KP2, 8 + r, c)), "r"(sv[j].x), "r"(sv[j].y), "r"(sv[j].z), "r"(sv[j].w) : "memory");
      }
    }
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
      const int i = tid + j * NTHREADS, r = i >> 5, c = i & 31;
      asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(sb + OFF_Q + sw_off(R2, r, c)), "r"(qv[j].x), "r"(qv[j].y), "r"(qv[j].z), "r"(qv[j].w) : "memory");
    }
    return;
  }
  if (FL & F_TMA) {
    if (tid == 0) {
#ifndef K2_NOFENCE
      asm volatile("fence.proxy.async.global;" ::: "memory");  // generic-proxy writes of the B tasks -> visible to the TMA after the barrier
#endif
      const uint32_t bK = sb + OFF_BAR, bV = bK + 8, bQ = bK + 16;
      mbar_expect(bQ, 4 * QBLK);
#pragma unroll
      for (int cb = 0; cb < 4; ++cb) tma2d(sb + OFF_Q + cb * QBLK, &c_tmQ, bQ, cb * 64, rg * R2);
      if (tail) {
        mbar_expect(bK, 4 * 56 * 128); mbar_expect(bV, 4 * 56 * 128);
#pragma unroll
        for (int cb = 0; cb < 4; ++cb) {
          tma2d(sb + OFF_K + cb * KBLK + 8 * 128, &c_tmKs, bK, cb * 64, 0);
          tma2d(sb + OFF_V + cb * KBLK + 8 * 128, &c_tmVs, bV, cb * 64, 0);
        }
      }
    }
  } else {
    for (int i = tid; i < R2 * 32; i += NTHREADS) {
      const int r = i >> 5, c = i & 31, row = rg * R2 + r;
      cp16s(sb + OFF_Q + sw_off(R2, r, c), u.q + (size_t)min(row, QROWS - 1) * HD + c * 8, row < QROWS ? 16 : 0);
    }
    if (tail)
      for (int i = tid; i < 56 * 32; i += NTHREADS) {
        const int r = i >> 5, c = i & 31;
        cp16s(sb + OFF_K + sw_off(KP2, 8 + r, c), u.ksuf + (size_t)min(r, T - 1) * HD + c * 8, r < T ? 16 : 0);
        cp16s(sb + OFF_V + sw_off(KP2, 8 + r, c), u.vsuf + (size_t)min(r, T - 1) * HD + c * 8, r < T ? 16 : 0);
      }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
}

// ---- QK^T for warp (mt, kg): 16 rows x NTW n-tiles, scaled / masked, bf16 into S
template <int FL, int NTW>
__device__ __forceinline__ void k2_qk(uint32_t sb, const uint8_t* Mk, float* smx, int warp, int lane, unsigned tag) {
  const int mt = warp & 1, kg = warp >> 1, g = lane >> 2, t4 = lane & 3, x = lane & 7;
  float sc[NTW][4];
#pragma unroll
  for (int n = 0; n < NTW; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) sc[n][e] = 0.f;
  const uint32_t aBase = sb + OFF_Q + (mt * 16 + x + ((lane >> 3) & 1) * 8) * 128;
  const uint32_t kBase = sb + OFF_K;
  const int bn0 = kg * NTW;
#pragma unroll
  for (int kk = 0; kk < HD / 16; ++kk) {
    uint32_t a0, a1, a2, a3;
    ldsm_x4a(a0, a1, a2, a3, aBase + (kk >> 2) * QBLK + (((((kk & 3) * 2 + (lane >> 4)) ^ x) & 7) << 4));
#pragma unroll
    for (int np = 0; np < NTW / 2; ++np) {
      const int krow = (bn0 + 2 * np) * 8 + (lane >> 4) * 8 + x;
      uint32_t b0, b1, b2, b3;
      ldsm_x4a(b0, b1, b2, b3, kBase + (kk >> 2) * KBLK + krow * 128 + (((((kk & 3) * 2 + ((lane >> 3) & 1)) ^ x) & 7) << 4));
      mma16816(sc[2 * np], a0, a1, a2, a3, b0, b1);
      mma16816(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
    }
    if (NTW & 1) {
      const int krow = (bn0 + NTW - 1) * 8 + x;
      uint32_t b0, b1;
      ldsm_x2(b0, b1, kBase + (kk >> 2) * KBLK + krow * 128 + (((((kk & 3) * 2 + ((lane >> 3) & 1)) ^ x) & 7) << 4));
      mma16816(sc[NTW - 1], a0, a1, a2, a3, b0, b1);
    }
  }
  // scale / mask in registers, then the fused softmax: row max over the quad (shuffle) and the 4 key-group warps (smem), p = bf16(exp(s - m)) kept in the
  // accumulator layout and written to S as bf16 (the PV A operand), row sums by the same two-level reduction
  float* xm = reinterpret_cast<float*>(smx);
  float* xs = xm + 4 * R2;
  float v[NTW][4];
  float lm[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < NTW; ++n) {
    const int col = (bn0 + n) * 8 + 2 * t4;
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[n][e] = Mk[col + (e & 1)] ? rbf(sc[n][e]) * 0.0625f : -INFINITY; lm[e >> 1] = fmaxf(lm[e >> 1], v[n][e]); }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    lm[h] = fmaxf(lm[h], __shfl_xor_sync(0xffffffffu, lm[h], 1));
    lm[h] = fmaxf(lm[h], __shfl_xor_sync(0xffffffffu, lm[h], 2));
  }
  const int ra = mt * 16 + g, rb = ra + 8;
  if (t4 == 0) { xm[kg * R2 + ra] = lm[0]; xm[kg * R2 + rb] = lm[1]; }
  __syncthreads();
  float mr[2];
  mr[0] = fmaxf(fmaxf(xm[ra], xm[R2 + ra]), fmaxf(xm[2 * R2 + ra], xm[3 * R2 + ra]));
  mr[1] = fmaxf(fmaxf(xm[rb], xm[R2 + rb]), fmaxf(xm[2 * R2 + rb], xm[3 * R2 + rb]));
  float ms[2] = {mr[0] == -INFINITY ? 0.f : mr[0], mr[1] == -INFINITY ? 0.f : mr[1]};
  if (FL & F_LL) {  // the shift is the tagged max itself (any consistent shift is valid); a fully masked chunk row uses -1e30 (p = 0, weight exp(-1e30 - M) = 0)
    ms[0] = k2_tagf(mr[0] == -INFINITY ? -1e30f : mr[0], tag); ms[1] = k2_tagf(mr[1] == -INFINITY ? -1e30f : mr[1], tag);
  }
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int n = 0; n < NTW; ++n) {
    const int col = (bn0 + n) * 8 + 2 * t4;
    float pr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (FL & F_FEXP) { float ee; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ee) : "f"((v[n][e] - ms[e >> 1]) * 1.4426950408889634f)); pr[e] = rbf(ee); }
      else pr[e] = rbf(expf(v[n][e] - ms[e >> 1]));
      ls[e >> 1] += pr[e];
    }
    const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]), hi = __floats2bfloat162_rn(pr[2], pr[3]);
    asm volatile("st.shared.b32 [%0], %1;" ::"r"(sb + OFF_S + (ra * SS_STRIDE + col) * 2), "r"(*reinterpret_cast<const uint32_t*>(&lo)) : "memory");
    asm volatile("st.shared.b32 [%0], %1;" ::"r"(sb + OFF_S + (rb * SS_STRIDE + col) * 2), "r"(*reinterpret_cast<const uint32_t*>(&hi)) : "memory");
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    ls[h] += __shfl_xor_sync(0xffffffffu, ls[h], 1);
    ls[h] += __shfl_xor_sync(0xffffffffu, ls[h], 2);
  }
  if (t4 == 0) { xs[kg * R2 + ra] = ls[0]; xs[kg * R2 + rb] = ls[1]; }
}

// ---- one unit
template <int FL>
__device__ __noinline__ void k2_unit(UA u, int l, int it, ull* mk) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3, x = lane & 7;
  if (k2_idle<FL>(blockIdx.x)) return;
  int rg, kc; k2_map<FL>(blockIdx.x, rg, kc);
  const bool tail = kc == NCH2 - 1;
  const int KC = tail ? KT2 : KP2, j0 = kc * KP2;
  unsigned char* base = u.smem;
  const uint32_t sb = su32(base);
  uint8_t* Mk = reinterpret_cast<uint8_t*>(base + OFF_MK);
  ATT_MARK(mk, 0);
#ifdef K2_PROBE
  { const uint4 pv = __ldcg(reinterpret_cast<const uint4*>(u.q) + (size_t)(rg * R2) * 32 + tid); if (pv.x == 0x12345u) u.stats[0] = 1.f; __syncthreads(); ATT_MARK(mk, 3); }
#endif
  if (!(FL & F_PRE)) {
    if (tid == 0) {
      const int cnt = (tail && !(FL & F_LDQ)) ? 2 : 1;  // tail: prefix part + suffix part each arrive with their expect_tx
      mbar_init(sb + OFF_BAR, cnt); mbar_init(sb + OFF_BAR + 8, cnt); mbar_init(sb + OFF_BAR + 16, 1);
      if (FL & F_PAIR) { mbar_init(sb + OFF_RB, 1); mbar_expect(sb + OFF_RB, 4 * 256 * 16 + 256); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    k2_fill_prefix<FL>(u, sb, l, kc, tail);
  }
  if (tid < KC) { const int j = j0 + tid; Mk[tid] = j < P ? (u.mask[j] != 0) : (j < NK); }
  k2_fill_post<FL>(u, sb, rg, tail);
#ifdef K2_QMARK
  if (FL & F_LDQ) { __syncthreads(); ATT_MARK(mk, 1); }
#endif
  if (FL & F_TMA) {
    if (!(FL & F_LDQ)) mbar_wait(sb + OFF_BAR + 16, 0);
#ifdef K2_QMARK
    __syncthreads(); ATT_MARK(mk, 1);
#endif
    mbar_wait(sb + OFF_BAR, 0);
    mbar_wait(sb + OFF_BAR + 8, 0);
  } else {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
  }
  __syncthreads();  // loads landed, mask visible
#ifndef K2_QMARK
  ATT_MARK(mk, 1);
#endif
  // ---- QK^T + fused softmax -> P (bf16, smem); row stats
  float* smx = reinterpret_cast<float*>(base + OFF_X);
  const unsigned tag = k2_tag(it);
  if (tail) k2_qk<FL, 2>(sb, Mk, smx, warp, lane, tag); else k2_qk<FL, 3>(sb, Mk, smx, warp, lane, tag);
  ATT_MARK(mk, 2);
  __syncthreads();
  if (tid < R2) {  // row stats (m, l): reduce the 4 key-group partials
    float m = fmaxf(fmaxf(smx[tid], smx[R2 + tid]), fmaxf(smx[2 * R2 + tid], smx[3 * R2 + tid]));
    const float* xs = smx + 4 * R2;
    float l = (xs[tid] + xs[R2 + tid]) + (xs[2 * R2 + tid] + xs[3 * R2 + tid]);
    if (FL & F_LL) { m = k2_tagf(m == -INFINITY ? -1e30f : m, tag); l = k2_tagf(l, tag); }
    const int row = rg * R2 + tid;
    if ((FL & F_PAIR) && !tail) {  // keep own (m, l) in smem, send them to the partner's smem (merged stats go to global after the O merge)
      reinterpret_cast<float2*>(base + OFF_ML)[tid] = make_float2(m, l);
      const uint32_t pr = (uint32_t)(blockIdx.x & 1) ^ 1u;
      st_async2(mapa_u32(sb + OFF_RS + tid * 8, pr), m, l, mapa_u32(sb + OFF_RB, pr));
    } else if (row < QROWS) *reinterpret_cast<float2*>(u.stats + ((FL & F_BFC) ? ((size_t)kc * QROWS + row) : ((size_t)row * NCH2 + kc)) * 2) = make_float2(m, l);
  }
#ifndef K2_PROBE
  ATT_MARK(mk, 3);
#endif
  // ---- P.V: warp w = dims [32w, 32w + 32), both m-tiles
  float acc[2][4][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  {
    const uint32_t pBase = sb + OFF_S + ((x + ((lane >> 3) & 1) * 8) * SS_STRIDE + (lane >> 4) * 8) * 2;
    const uint32_t vBase = sb + OFF_V;
    const int nks = KC / 16;
#pragma unroll
    for (int ks = 0; ks < KP2 / 16; ++ks) {
      if (ks < nks) {
        uint32_t pa[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4a(pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], pBase + (mt * 16 * SS_STRIDE + ks * 16) * 2);
        const int krow = ks * 16 + x + ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int ch = (warp & 1) * 4 + np * 2 + (lane >> 4);
          uint32_t b0, b1, b2, b3;
          ldsm_x4t(b0, b1, b2, b3, vBase + (warp >> 1) * KBLK + krow * 128 + (((ch ^ x) & 7) << 4));
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc[mt][2 * np], pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], b0, b1);
            mma16816(acc[mt][2 * np + 1], pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], b2, b3);
          }
        }
      }
    }
  }
#ifndef K2_DBGM
  ATT_MARK(mk, 4);
#endif
  if ((FL & F_PAIR) && !tail) {
    // ---- cluster pair: CTA s keeps the rows of m-tile s and merges the partner's partial for them; the other m-tile's accumulators go to the partner (st.async + mbarrier tx)
    const int s = blockIdx.x & 1;
    const uint32_t pr = (uint32_t)s ^ 1u, rv = mapa_u32(sb + OFF_RV, pr), rb = mapa_u32(sb + OFF_RB, pr);
    float* ml = reinterpret_cast<float*>(base + OFF_ML);
    const float* mr = reinterpret_cast<const float*>(base + OFF_RS);
    const float4* ro = reinterpret_cast<const float4*>(base + OFF_RV);
    auto go = [&](auto KEEPc) {
      constexpr int KEEP = decltype(KEEPc)::value;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) st_async4(rv + (nt * 256 + tid) * 16, acc[1 - KEEP][nt][0], acc[1 - KEEP][nt][1], acc[1 - KEEP][nt][2], acc[1 - KEEP][nt][3], rb);
      ATT_MARK(mk, 4);
      __syncthreads();  // own (m, l) table published
      mbar_wait(sb + OFF_RB, 0);
      const int r0 = KEEP * 16 + g, r1 = r0 + 8;
      const float m00 = ml[2 * r0], l00 = ml[2 * r0 + 1], m01 = mr[2 * r0], l01 = mr[2 * r0 + 1];
      const float m10 = ml[2 * r1], l10 = ml[2 * r1 + 1], m11 = mr[2 * r1], l11 = mr[2 * r1 + 1];
      const float M0 = fmaxf(m00, m01), M1 = fmaxf(m10, m11);
      const float a00 = m00 == -INFINITY ? 0.f : expf(m00 - M0), a01 = m01 == -INFINITY ? 0.f : expf(m01 - M0);
      const float a10 = m10 == -INFINITY ? 0.f : expf(m10 - M1), a11 = m11 == -INFINITY ? 0.f : expf(m11 - M1);
      const int slot = kc & ~1;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const float4 o = ro[nt * 256 + tid];
        const int col = warp * 32 + nt * 8 + 2 * t4;
        const int rw0 = rg * R2 + r0, rw1 = rg * R2 + r1;
        if (rw0 < QROWS)
          *reinterpret_cast<float2*>(u.opart + ((size_t)slot * QROWS + rw0) * HD + ((((col >> 2) ^ r0) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) =
              make_float2(a00 * acc[KEEP][nt][0] + a01 * o.x, a00 * acc[KEEP][nt][1] + a01 * o.y);
        if (rw1 < QROWS)
          *reinterpret_cast<float2*>(u.opart + ((size_t)slot * QROWS + rw1) * HD + ((((col >> 2) ^ r1) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) =
              make_float2(a10 * acc[KEEP][nt][2] + a11 * o.z, a10 * acc[KEEP][nt][3] + a11 * o.w);
      }
      if (tid < 16) {  // merged (M, l) of this CTA's 16 rows
        const int r = KEEP * 16 + tid, row = rg * R2 + r;
        const float ma = ml[2 * r], la = ml[2 * r + 1], mb = mr[2 * r], lb = mr[2 * r + 1], M = fmaxf(ma, mb);
        const float wa = ma == -INFINITY ? 0.f : expf(ma - M), wb = mb == -INFINITY ? 0.f : expf(mb - M);
        if (row < QROWS) *reinterpret_cast<float2*>(u.stats + ((size_t)row * NCH2 + slot) * 2) = make_float2(M, wa * la + wb * lb);
      }
    };
    if (s == 0) go(std::integral_constant<int, 0>()); else go(std::integral_constant<int, 1>());
    __syncthreads();
    if (tid == 0) red_release(u.cnt + (CNT_ATT + rg) * CNT_STRIDE);
    ATT_MARK(mk, 5);
    return;
  }
  // ---- partial O -> opart[kc][row][256] (XOR-permuted float4 chunks)
  const int nrow = min(R2, QROWS - rg * R2);
  if (FL & F_BF) {  // bf16 octet partials opart16[kc][col/8][row][8]: a warp store of one (octet, 8 rows) is 128 contiguous bytes
    uint32_t* o32 = reinterpret_cast<uint32_t*>(u.opart);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int oct = warp * 4 + nt;
        const int rw0 = rg * R2 + mt * 16 + g, rw1 = rw0 + 8;
        const __nv_bfloat162 lo = __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]), hi = __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
        if (rw0 < QROWS) o32[(((size_t)kc * 32 + oct) * QROWS + rw0) * 4 + t4] = *reinterpret_cast<const uint32_t*>(&lo);
        if (rw1 < QROWS) o32[(((size_t)kc * 32 + oct) * QROWS + rw1) * 4 + t4] = *reinterpret_cast<const uint32_t*>(&hi);
      }
    __syncthreads();
    if (tid == 0) red_release(u.cnt + (CNT_ATT + rg) * CNT_STRIDE);
  } else
  if (FL & (F_BULK | F_V4)) {
    __syncthreads();  // every warp is done with K / V / P (staging reuses the K region)
    float* st = reinterpret_cast<float*>(base + OFF_K);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = warp * 32 + nt * 8 + 2 * t4;
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        *reinterpret_cast<float2*>(st + r0 * HD + ((((col >> 2) ^ r0) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        *reinterpret_cast<float2*>(st + r1 * HD + ((((col >> 2) ^ r1) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      }
    if (FL & F_V4) {
      __syncthreads();
      float4* dst4 = reinterpret_cast<float4*>(u.opart + ((size_t)kc * QROWS + rg * R2) * HD);
      const float4* src4 = reinterpret_cast<const float4*>(st);
      const int n4 = nrow * HD / 4;
#pragma unroll
      for (int j = 0; j < 8; ++j) { const int i = tid + j * NTHREADS; if (i < n4) __stcg(dst4 + i, src4[i]); }
      __syncthreads();
      if (tid == 0) red_release(u.cnt + (CNT_ATT + rg) * CNT_STRIDE);
    } else {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      float* dst = u.opart + ((size_t)kc * QROWS + rg * R2) * HD;
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(sb + OFF_K), "r"(nrow * HD * 4) : "memory");
      asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group 0;" ::: "memory");
      red_release(u.cnt + (CNT_ATT + rg) * CNT_STRIDE);
    }
    }
  } else {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = warp * 32 + nt * 8 + 2 * t4;
        const int r0 = mt * 16 + g, r1 = r0 + 8;
        const int rw0 = rg * R2 + r0, rw1 = rg * R2 + r1;
        float4 a4 = make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
        if (FL & F_LL) { a4.x = k2_tagf(a4.x, tag); a4.y = k2_tagf(a4.y, tag); a4.z = k2_tagf(a4.z, tag); a4.w = k2_tagf(a4.w, tag); }
        if (rw0 < QROWS)
          *reinterpret_cast<float2*>(u.opart + ((size_t)kc * QROWS + rw0) * HD + ((((col >> 2) ^ r0) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) = make_float2(a4.x, a4.y);
        if (rw1 < QROWS)
          *reinterpret_cast<float2*>(u.opart + ((size_t)kc * QROWS + rw1) * HD + ((((col >> 2) ^ r1) & 7) | (col >> 2 & ~7)) * 4 + (col & 3)) = make_float2(a4.z, a4.w);
      }
    if (!(FL & F_LL)) {
      __syncthreads();
#ifdef K2_DBGM
      ATT_MARK(mk, 4);
#endif
      if (tid == 0) red_release(u.cnt + (CNT_ATT + rg) * CNT_STRIDE);
    }
  }
  ATT_MARK(mk, 5);
}

// ---- combine: CTA (rg, kc) takes rows k = kc + 11 s (s = 0..3) of its own row group once the row group's 11 units have released
template <int FL>
__device__ __noinline__ void k2_combine(UA u, int it, ull* mk) {
  const int tid = threadIdx.x;
  if (k2_idle<FL>(blockIdx.x)) return;
  int rg, kc; k2_map<FL>(blockIdx.x, rg, kc);
  const int slot = tid >> 6, cq = tid & 63, k = kc + NCH2 * slot, row = rg * R2 + k;
  if (FL & F_LL) {
    const unsigned tag = k2_tag(it);
    if (k < R2 && row < QROWS) {
      const float* sp = u.stats + (size_t)row * NCH2 * 2;
      uint4 o[NCH2];
      uint2 sv[NCH2];
      unsigned miss = (1u << NCH2) - 1u;
      unsigned smiss = (1u << NCH2) - 1u;
      if (FL & F_LLG) {  // stats first: the heavy loads start only once the producers' stats are visible
        while (smiss) {
#pragma unroll
          for (int c = 0; c < NCH2; ++c) if (smiss >> c & 1u) sv[c] = ld_vol2(sp + 2 * c);
#pragma unroll
          for (int c = 0; c < NCH2; ++c) if ((smiss >> c & 1u) && (sv[c].x & 0xffu) == tag && (sv[c].y & 0xffu) == tag) smiss &= ~(1u << c);
          if (smiss) __nanosleep(32);
        }
      }
      while (miss | smiss) {
#pragma unroll
        for (int c = 0; c < NCH2; ++c) {
          if (miss >> c & 1u) o[c] = ld_vol4(u.opart + ((size_t)c * QROWS + row) * HD + ((cq ^ row) & 7 | (cq & ~7)) * 4);
          if (smiss >> c & 1u) sv[c] = ld_vol2(sp + 2 * c);
        }
#pragma unroll
        for (int c = 0; c < NCH2; ++c) {
          if ((miss >> c & 1u) && (o[c].x & 0xffu) == tag && (o[c].y & 0xffu) == tag && (o[c].z & 0xffu) == tag && (o[c].w & 0xffu) == tag) miss &= ~(1u << c);
          if ((smiss >> c & 1u) && (sv[c].x & 0xffu) == tag && (sv[c].y & 0xffu) == tag) smiss &= ~(1u << c);
        }
        if (miss | smiss) __nanosleep(32);
      }
      ATT_MARK(mk, 6);
      float wgt[NCH2], ls[NCH2];
#pragma unroll
      for (int c = 0; c < NCH2; ++c) { wgt[c] = __uint_as_float(sv[c].x); ls[c] = __uint_as_float(sv[c].y); }
      float M = wgt[0];
#pragma unroll
      for (int c = 1; c < NCH2; ++c) M = fmaxf(M, wgt[c]);
      float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < NCH2; ++c) {
        const float w = expf(wgt[c] - M);
        const float4 ov = make_float4(__uint_as_float(o[c].x & ~0xffu), __uint_as_float(o[c].y & ~0xffu), __uint_as_float(o[c].z & ~0xffu), __uint_as_float(o[c].w & ~0xffu));
        num.x = fmaf(w, ov.x, num.x); num.y = fmaf(w, ov.y, num.y); num.z = fmaf(w, ov.z, num.z); num.w = fmaf(w, ov.w, num.w);
        den = fmaf(w, __uint_as_float(sv[c].y & ~0xffu), den);
      }
      *reinterpret_cast<uint2*>(u.attn + (size_t)row * HD + cq * 4) = pack_bf16x4(num.x / den, num.y / den, num.z / den, num.w / den);
    } else {
      ATT_MARK(mk, 6);
    }
    ATT_MARK(mk, 7);
    return;
  }
  constexpr int NC = (FL & F_PAIR) ? 6 : NCH2;  // PAIR: valid partial slots are the even chunks 0, 2, .., 10
#define K2_SL(c) ((FL & F_PAIR) ? 2 * (c) : (c))
  block_wait(u.cnt + (CNT_ATT + rg) * CNT_STRIDE, (unsigned)(it + 1) * NCH2, (u.fuse & FUSE_BACKOFF) != 0);
  ATT_MARK(mk, 6);
  if (FL & F_BFC) {
    constexpr int HC = (FL & F_BFH) ? 2 : 1;      // warps per octet
    constexpr int NW = 4 / HC;                    // 32-bit words (bf16 pairs) per thread
    const int wi = tid >> 5, lane = tid & 31, nO = (kc == NCH2 - 1) ? 2 : 3;
    const int oc = kc * 3 + wi / HC, hh = wi % HC, row2 = rg * R2 + lane;
    if (wi < nO * HC && row2 < QROWS) {
      uint32_t ow[NCH2][NW];
      float2 sv[NCH2];
#pragma unroll
      for (int c = 0; c < NCH2; ++c) {
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(u.opart) + (((size_t)c * 32 + oc) * QROWS + row2) * 4 + hh * NW;
        if constexpr (NW == 4) { const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p32)); ow[c][0] = v.x; ow[c][1] = v.y; ow[c][2] = v.z; ow[c][3] = v.w; }
        else { const uint2 v = __ldcg(reinterpret_cast<const uint2*>(p32)); ow[c][0] = v.x; ow[c][1] = v.y; }
        sv[c] = __ldcg(reinterpret_cast<const float2*>(u.stats) + (size_t)c * QROWS + row2);
      }
      float M = sv[0].x;
#pragma unroll
      for (int c = 1; c < NCH2; ++c) M = fmaxf(M, sv[c].x);
      float num[2 * NW];
#pragma unroll
      for (int e = 0; e < 2 * NW; ++e) num[e] = 0.f;
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < NCH2; ++c) {
        const float w = sv[c].x == -INFINITY ? 0.f : expf(sv[c].x - M);
#pragma unroll
        for (int e = 0; e < NW; ++e) {
          num[2 * e] = fmaf(w, __uint_as_float(ow[c][e] << 16), num[2 * e]);
          num[2 * e + 1] = fmaf(w, __uint_as_float(ow[c][e] & 0xffff0000u), num[2 * e + 1]);
        }
        den = fmaf(w, sv[c].y, den);
      }
      bf16* dst = u.attn + (size_t)row2 * HD + oc * 8 + hh * 2 * NW;
      if constexpr (NW == 4) {
        const uint2 lo = pack_bf16x4(num[0] / den, num[1] / den, num[2] / den, num[3] / den), hi = pack_bf16x4(num[4] / den, num[5] / den, num[6] / den, num[7] / den);
        *reinterpret_cast<uint4*>(dst) = make_uint4(lo.x, lo.y, hi.x, hi.y);
      } else {
        *reinterpret_cast<uint2*>(dst) = pack_bf16x4(num[0] / den, num[1] / den, num[2] / den, num[3] / den);
      }
    }
    ATT_MARK(mk, 7);
    return;
  }
  if (FL & F_BF) {
    constexpr int HC = (FL & F_BFH) ? 2 : 1;      // threads per octet
    constexpr int NW = 4 / HC;                    // 32-bit words (bf16 pairs) per thread
    const int s2 = tid / (32 * HC), oc = (tid / HC) & 31, hh = tid & (HC - 1);
    const int k2 = kc + NCH2 * s2, row2 = rg * R2 + k2;
    if (s2 < 8 / HC && k2 < R2 && row2 < QROWS) {
      const float2* st = reinterpret_cast<const float2*>(u.stats + (size_t)row2 * NCH2 * 2);
      uint32_t ow[NCH2][NW];
      float wgt[NCH2], ls[NCH2];
#pragma unroll
      for (int c = 0; c < NCH2; ++c) {
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(u.opart) + (((size_t)c * 32 + oc) * QROWS + row2) * 4 + hh * NW;
        if constexpr (NW == 4) { const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p32)); ow[c][0] = v.x; ow[c][1] = v.y; ow[c][2] = v.z; ow[c][3] = v.w; }
        else { const uint2 v = __ldcg(reinterpret_cast<const uint2*>(p32)); ow[c][0] = v.x; ow[c][1] = v.y; }
      }
#pragma unroll
      for (int c = 0; c < NCH2; ++c) { const float2 v = __ldcg(st + c); wgt[c] = v.x; ls[c] = v.y; }
      float M = wgt[0];
#pragma unroll
      for (int c = 1; c < NCH2; ++c) M = fmaxf(M, wgt[c]);
      float num[2 * NW];
#pragma unroll
      for (int e = 0; e < 2 * NW; ++e) num[e] = 0.f;
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < NCH2; ++c) {
        const float w = wgt[c] == -INFINITY ? 0.f : expf(wgt[c] - M);
#pragma unroll
        for (int e = 0; e < NW; ++e) {
          num[2 * e] = fmaf(w, __uint_as_float(ow[c][e] << 16), num[2 * e]);
          num[2 * e + 1] = fmaf(w, __uint_as_float(ow[c][e] & 0xffff0000u), num[2 * e + 1]);
        }
        den = fmaf(w, ls[c], den);
      }
      bf16* dst = u.attn + (size_t)row2 * HD + oc * 8 + hh * 2 * NW;
      if constexpr (NW == 4) {
        const uint2 lo = pack_bf16x4(num[0] / den, num[1] / den, num[2] / den, num[3] / den), hi = pack_bf16x4(num[4] / den, num[5] / den, num[6] / den, num[7] / den);
        *reinterpret_cast<uint4*>(dst) = make_uint4(lo.x, lo.y, hi.x, hi.y);
      } else {
        *reinterpret_cast<uint2*>(dst) = pack_bf16x4(num[0] / den, num[1] / den, num[2] / den, num[3] / den);
      }
    }
    ATT_MARK(mk, 7);
    return;
  }
  if (k < R2 && row < QROWS) {
    const float2* st = reinterpret_cast<const float2*>(u.stats + (size_t)row * NCH2 * 2);
    float4 o[NC];
    float wgt[NC], ls[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) o[c] = __ldcg(reinterpret_cast<const float4*>(u.opart + ((size_t)K2_SL(c) * QROWS + row) * HD + ((cq ^ row) & 7 | (cq & ~7)) * 4));
#pragma unroll
    for (int c = 0; c < NC; ++c) { const float2 v = __ldcg(st + K2_SL(c)); wgt[c] = v.x; ls[c] = v.y; }
    float M = wgt[0];
#pragma unroll
    for (int c = 1; c < NC; ++c) M = fmaxf(M, wgt[c]);
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = 0.f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const float w = wgt[c] == -INFINITY ? 0.f : expf(wgt[c] - M);
      num.x = fmaf(w, o[c].x, num.x); num.y = fmaf(w, o[c].y, num.y); num.z = fmaf(w, o[c].z, num.z); num.w = fmaf(w, o[c].w, num.w);
      den = fmaf(w, ls[c], den);
    }
    *reinterpret_cast<uint2*>(u.attn + (size_t)row * HD + cq * 4) = pack_bf16x4(num.x / den, num.y / den, num.z / den, num.w / den);
  }
  ATT_MARK(mk, 7);
}

struct K2 {
  static constexpr const char* kName = "att_k2";
  static constexpr bool kCCBarrier = false;
  static constexpr size_t kSmem = K2_SMEM;
  static constexpr int kCluster = 0;
  __device__ static UA mkua(const Params& p, bf16* smb) {
    UA u; u.pk = p.prefix_k; u.pv = p.prefix_v; u.ksuf = p.ksuf; u.vsuf = p.vsuf; u.q = p.q; u.mask = p.prefix_mask; u.attn = p.attn;
    u.stats = p.stats; u.opart = p.opart; u.cnt = p.cnt; u.smem = k2_base(smb); u.fuse = p.fuse;
    return u;
  }
  template <int FL>
  __device__ static void pre_one(const UA& u, int l) {
    if (k2_idle<FL>(blockIdx.x)) return;
    int rg, kc; k2_map<FL>(blockIdx.x, rg, kc);
    const bool tail = kc == NCH2 - 1;
    const uint32_t sb = su32(u.smem);
    __syncthreads();
    if (threadIdx.x == 0) {
      const int cnt = (tail && !(FL & F_LDQ)) ? 2 : 1;
      mbar_init(sb + OFF_BAR, cnt); mbar_init(sb + OFF_BAR + 8, cnt); mbar_init(sb + OFF_BAR + 16, 1);
      if (FL & F_PAIR) { mbar_init(sb + OFF_RB, 1); mbar_expect(sb + OFF_RB, 4 * 256 * 16 + 256); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // earlier phases used this smem through the generic proxy
    }
    k2_fill_prefix<FL>(u, sb, l, kc, tail);
#ifndef K2_NOTMPF
    if ((FL & F_TMA) && threadIdx.x == 0) {  // warm the tensor-map descriptors used after the barrier (q; suffix K/V on the tail unit)
      asm volatile("prefetch.tensormap [%0];" ::"l"((ull)&c_tmQ) : "memory");
      if (tail) { asm volatile("prefetch.tensormap [%0];" ::"l"((ull)&c_tmKs) : "memory"); asm volatile("prefetch.tensormap [%0];" ::"l"((ull)&c_tmVs) : "memory"); }
    }
#endif
  }
  // LL variants: at the first iteration of a launch every unit zeroes its own partial-O / stats slots (stale tags from the previous launch); the B -> C barrier orders it
  __device__ static void ll_reset(const UA& u, const AttArgs& a) {
    if (k2_idle<F_LL>(blockIdx.x)) return;
    if (ld_relaxed(a.bar + 160 * BAR_STRIDE) / gridDim.x != 1u) return;
    int rg, kc; k2_map<F_LL>(blockIdx.x, rg, kc);
    const int nrow = min(R2, QROWS - rg * R2);
    uint4* o4 = reinterpret_cast<uint4*>(u.opart + ((size_t)kc * QROWS + rg * R2) * HD);
    for (int i = threadIdx.x; i < nrow * HD / 4; i += NTHREADS) o4[i] = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < nrow) *reinterpret_cast<uint2*>(u.stats + ((size_t)(rg * R2 + threadIdx.x) * NCH2 + kc) * 2) = make_uint2(0, 0);
  }
  __device__ static void B_pre(const Params& p, const AttArgs& a, int l, bf16* smb) {
    K0::B_pre(p, a, l, smb);
    if (a.variant < ATTV_EXT) return;
    const UA u = mkua(p, smb);
    switch (a.variant - ATTV_EXT) {
#define X(i, n, fl) case i: if (!K2_ON(i)) break; if ((fl) & F_PRE) pre_one<(fl)>(u, l); if ((fl) & F_LL) ll_reset(u, a); break;
      K2_LIST(X)
#undef X
    }
  }
  __device__ static void C(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    if (a.variant < ATTV_EXT) { K0::C(p, a, l, it, smb, mk); return; }
    const UA u = mkua(p, smb);
    switch (a.variant - ATTV_EXT) {
#define X(i, n, fl) case i: if (K2_ON(i)) k2_unit<(fl)>(u, l, it, mk); break;
      K2_LIST(X)
#undef X
    }
  }
  __device__ static void C3(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    if (a.variant < ATTV_EXT) { K0::C3(p, a, l, it, smb, mk); return; }
    const UA u = mkua(p, smb);
    switch (a.variant - ATTV_EXT) {
#define X(i, n, fl) case i: if (K2_ON(i)) k2_combine<(fl)>(u, it, mk); break;
      K2_LIST(X)
#undef X
    }
  }
  // ---- host
  static void prepare(AttArgs& a, int grid) {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    auto mk = [&](const void* base, int rows, int box_rows, CUtensorMap& sym, const char* what) {
      CUtensorMap m;
      cuuint64_t dims[2] = {(cuuint64_t)HD, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)HD * 2};
      cuuint32_t box[2] = {64, (cuuint32_t)box_rows}, es[2] = {1, 1};
      CUresult r = enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                       CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map %s failed %d\n", what, (int)r); exit(1); }
      CK(cudaMemcpyToSymbol(sym, &m, sizeof m));
    };
    mk(a.prefix_k, L * P, KP2, c_tmK, "K");
    mk(a.prefix_v, L * P, KP2, c_tmV, "V");
    mk(a.prefix_k, L * P, 8, c_tmK8, "K8");
    mk(a.prefix_v, L * P, 8, c_tmV8, "V8");
    mk(a.ksuf, T, 56, c_tmKs, "Ksuf");
    mk(a.vsuf, T, 56, c_tmVs, "Vsuf");
    mk(a.q, QROWS, R2, c_tmQ, "Q");
    unsigned z = 0; CK(cudaMemcpyToSymbol(g_k2_to, &z, 4));
    fprintf(stderr, "k2 prepare: grid=%d units=%d smem=%zu\n", grid, NUNIT2, (size_t)kSmem);
  }
  static void post_run(int variant, int idx, int reps, const std::string& out, const std::string& name, double nspc) {
    unsigned to = 0; CK(cudaMemcpyFromSymbol(&to, g_k2_to, 4));
    if (to) { fprintf(stderr, "WARNING: %u mbarrier wait timeouts (variant %s launch %d)\n", to, att_vname(variant), idx); unsigned z = 0; CK(cudaMemcpyToSymbol(g_k2_to, &z, 4)); }
  }
};

int main(int argc, char** argv) { return att_main<K2>(argc, argv); }
