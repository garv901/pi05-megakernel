// ab_k1.cu: K1 = fused tcgen05 A + B (QKV projection + reduce + RoPE + store) for the standalone A+B harness (ab_common.cuh).
// Design = the F design of mk/pi05_mk.cu (fg_F / fg_E, FG_UMMA) applied to phase A: swap-AB (weights on the MMA M side), cta_group::2 pair (M = 128, N = 64), full K
// (or K split across two pairs), E-side norm fold (A' = bf16(h*(1+scale)) + shift rows from ab_E), epilogue = rs * acc + shift (the MMA produces W.shift in token column
// 31 / 19), bf16 round, RoPE, transposed bf16 store through per-warp smem. For A the two weight "halves" of a CTA's 64 rows are rows j and j+128 of one head (the RoPE
// pair), so thread (lane) holds x1 (row g) and x2 (row g+8) for the same j: RoPE is register-local. Weight map: 4D {W, 8 rows, 2 (the +128-row half), R/8 blocks},
// box {64, 8, 2, 4}, SW128 -> smem rows [blk][j rows 0..7][j+128 rows 0..7]. 40 CTAs (ABA0 = 108 .. 147): a = cta - 108, head = a / 4 (0..7 q, 8 k, 9 v), j0 = (a % 4) * 32.
// Variants (AB_MASK bits):
//   1  k1_cl2          full-K cta_group::2 on CTAs 108..147, RoPE epilogue                 (the port candidate)
//   2  k1_cl2_norope   same, epilogue stores x1 / x2 without RoPE                          (Q2 by diff + stamps)
//   4  k1_ks2_c2       80 CTAs (68..147): K split across two cta_group::2 pairs (pair p: K 0..511, pair p + 20: K 512..1023) with a global fp32 partial + per-warp
//                      counter (the portable K-split)
//   8  k1_ks2_c1       80 CTAs, cta_group::1 M = 64 / N = 64 in cluster pairs (rank = K half) with a DSMEM register exchange (harness-only upper bound; own binary)
// Build (login node):
//   nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DAB_MASK=7 ab_k1.cu -o ab_k1c -lcuda      (unstamped twin: -DFG_STAMPS=0 -o ab_k1c_ns)
//   nvcc ... -DAB_MASK=8 ab_k1.cu -o ab_k1s -lcuda                                                              (cta_group::1 binary: ab_k1s / ab_k1s_ns)
#ifndef AB_MASK
#define AB_MASK 7
#endif
#include <string>
#include "ab_common.cuh"
static_assert(!((AB_MASK & 8) && (AB_MASK & 7)), "cta_group::1 and ::2 cannot be mixed in one kernel");

enum { V_CL2 = 0, V_NOROPE = 1, V_KS2C2 = 2, V_KS2C1 = 3 };
static const char* K1_NAMES[4] = {"k1_cl2", "k1_cl2_norope", "k1_ks2_c2", "k1_ks2_c1"};

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ void tma4d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory"); }
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void tma4d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory"); }
__device__ __forceinline__ void tma2d2(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory"); }
#if AB_MASK & 7
__device__ __forceinline__ void mma_tc2(uint32_t d, ull da, ull db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit2(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"((unsigned short)3) : "memory"); }
__device__ __forceinline__ void tmem_alloc64c2(uint32_t slot) { asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot)); }
__device__ __forceinline__ void tmem_free64c2(uint32_t tb) { asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 64;" ::"r"(tb)); }
#endif
#if AB_MASK & 8
__device__ __forceinline__ void mma_tc1(uint32_t d, ull da, ull db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit1(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void tmem_alloc64c1(uint32_t slot) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(slot)); }
__device__ __forceinline__ void tmem_free64c1(uint32_t tb) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" ::"r"(tb)); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void st_cluster_v4(uint32_t a, uint4 v) { asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory"); }
__device__ __forceinline__ void mbar_arrive_remote(uint32_t a) { asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait_cluster(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
  for (ull i = 0; i < (1ull << 22) && !ok; ++i)
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) atomicAdd(&g_ab_to, 1u);
}
#endif
__device__ __forceinline__ void ld_tmem16(uint32_t (&va)[16], uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(va[0]), "=r"(va[1]), "=r"(va[2]), "=r"(va[3]), "=r"(va[4]), "=r"(va[5]), "=r"(va[6]), "=r"(va[7]), "=r"(va[8]), "=r"(va[9]), "=r"(va[10]),
                 "=r"(va[11]), "=r"(va[12]), "=r"(va[13]), "=r"(va[14]), "=r"(va[15]) : "r"(ta));
}

// ------------------------------------------------------------------------------------------------ shared state
__shared__ __align__(8) ull ab_bar[48];    // full @0, acc @32 (byte 256), rsr @33 (264), abar @34 (272), xbar[8] @36 (288)
__shared__ uint32_t ab_tslot;
__shared__ __align__(16) float ab_rs[64];
constexpr int K1_NSTEP = NSTEP, K1_NTS = 16;
__shared__ ull ab_ts_s[K1_NTS];
__device__ unsigned g_ab_cnt[148];
__device__ ull g_ab_ts[148 * K1_NSTEP * K1_NTS];
#if FG_STAMPS
#define K1_STAMP(i) do { ab_ts_s[(i)] = clock64(); } while (0)
#else
#define K1_STAMP(i) do {} while (0)
#endif
enum { TS_ENTRY = 0, TS_PREISS, TS_BAR, TS_FULL0, TS_B0, TS_MMA0, TS_MMAEND, TS_EPI0, TS_LDDONE, TS_STDONE, TS_END, TS_ISSUED, TS_COMP, TS_X0, TS_X1, TS_X2 };
constexpr int ABA2_N = 40, KS_A0 = 68, KS_NPAIR = 40;   // A CTAs of the full-K designs / first CTA of the 80-CTA K-split designs

// ------------------------------------------------------------------------------------------------ the shared epilogue math + store
// va: this thread's 16 accumulator words (row g: x1 of head column j, row g + 8: x2 of column j + 128; token columns 8*(i>>1) + tq + (i&1) of its 32-column half h; the shift
// column is local column 31 (h = 0) / 19 (h = 1)). rs[t] from ab_rs. Math as phase B except the rounding point: x = rbf(fmaf(rs, acc, shift_acc)).
template <bool ROPE>
__device__ __forceinline__ void ab_epi(const Params& p, const uint32_t (&va)[16], int head, int j0, int blk, int h, uint32_t scr, const float2 (&rp)[8], int lane) {
  const int tq = (lane & 3) * 2;
  const int src = (lane & ~3) | (h ? 1 : 3);
  const float sh0 = __shfl_sync(0xffffffffu, __uint_as_float(h ? va[9] : va[13]), src);
  const float sh1 = __shfl_sync(0xffffffffu, __uint_as_float(h ? va[11] : va[15]), src);
  float rsr[8];
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) rsr[idx] = ab_rs[(h ? 31 : 0) + 8 * (idx >> 1) + tq + (idx & 1)];  // <= 62; zero for t >= T
  const bool dorope = ROPE && head < 9;
  const int nv = h ? 19 : 31;  // valid local columns
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    const float x1 = rbf(fmaf(rsr[idx], __uint_as_float(va[2 * (idx & ~1) + (idx & 1)]), sh0));
    const float x2 = rbf(fmaf(rsr[idx], __uint_as_float(va[2 * (idx & ~1) + 2 + (idx & 1)]), sh1));
    float o1 = x1, o2 = x2;
    if (dorope) {
      const float cs = rp[idx].x, sn = rp[idx].y;
      o1 = rbf(rbf(x1 * cs) + rbf(-x2 * sn));
      o2 = rbf(rbf(x2 * cs) + rbf(x1 * sn));
    }
    const int c = 8 * (idx >> 1) + tq + (idx & 1);
    if (c < nv) {
      st_shared_b16(scr + c * 32 + (lane >> 2) * 2, __bfloat16_as_ushort(__float2bfloat16_rn(o1)));
      st_shared_b16(scr + c * 32 + 16 + (lane >> 2) * 2, __bfloat16_as_ushort(__float2bfloat16_rn(o2)));
    }
  }
  __syncwarp();
  if (lane < nv) {
    uint4 u1, u2;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u1.x), "=r"(u1.y), "=r"(u1.z), "=r"(u1.w) : "r"(scr + lane * 32));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u2.x), "=r"(u2.y), "=r"(u2.z), "=r"(u2.w) : "r"(scr + lane * 32 + 16));
    const int t = (h ? 31 : 0) + lane;
    bf16* dst = (head < 8 ? p.q + (size_t)t * QD + head * HD : head == 8 ? p.ksuf + (size_t)t * KVD : p.vsuf + (size_t)t * KVD) + j0 + blk * 8;
    *reinterpret_cast<uint4*>(dst) = u1;
    *reinterpret_cast<uint4*>(dst + HD / 2) = u2;
  }
}
// RoPE cos / sin of this thread's 8 (token, j) pairs, loaded before the accumulator is ready
__device__ __forceinline__ void ab_load_rope(const float* rope, int head, int j, int h, int lane, float2 (&rp)[8]) {
  const int tq = (lane & 3) * 2, nv = h ? 19 : 31;
#pragma unroll
  for (int idx = 0; idx < 8; ++idx) {
    const int c = 8 * (idx >> 1) + tq + (idx & 1), t = (h ? 31 : 0) + c;
    rp[idx] = (head < 9 && c < nv) ? __ldg(reinterpret_cast<const float2*>(rope + ((size_t)t * 128 + j) * 2)) : make_float2(1.f, 0.f);
  }
}
__device__ __forceinline__ void ab_rs_warps(const Params& p, int tid, int lane) {  // warps 2, 3: rs[t] = rsqrt(sum(ssp[t][0..15]) / W + eps)
  const int t = tid - 64;
  float rsv = 0.f;
  if (t < T) {
    float sx = 0.f;
#pragma unroll
    for (int j = 0; j < NSSP; ++j) sx += __ldcg(p.ssp + t * NSSP + j);
    rsv = rsqrtf(__fadd_rn(sx * (1.0f / W), 1e-6f));
  }
  ab_rs[t] = rsv;
  __syncwarp();
  if (lane == 0) mbar_arrive(su32(&ab_bar[33]));
#if FG_STAMPS
  (void)0;
#endif
}
#define AB_STAMP_SAVE(tbase, cta, k1c, l) do { \
  if (tid == 0) { K1_STAMP(TS_END); g_ab_cnt[cta] = k1c + 1; const int cc_ = (int)(k1c % (unsigned)NIT); \
    if ((l) == L - 1) { ull* dst_ = g_ab_ts + ((size_t)(cta) * K1_NSTEP + cc_ / L) * K1_NTS; for (int i_ = 0; i_ < K1_NTS; ++i_) dst_[i_] = ab_ts_s[i_]; } } } while (0)

#if AB_MASK & 3
// ------------------------------------------------------------------------------------------------ full-K cta_group::2 (40 CTAs, 108..147)
// Map pointers are passed by value (a const& gave an invalid __local__ read in fg_F).
template <bool ROPE>
__device__ __noinline__ void ab_A2(const Params& p, const CUtensorMap* tmWq, const CUtensorMap* tmWk, const CUtensorMap* tmWv, const CUtensorMap* tmA32, const CUtensorMap* tmA20,
                                   int l, const WPre& w, bf16* smb) {
  constexpr int KB = 16, SB = KB * 8192, BA = 4096;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x, rank = cta & 1;
  const int an = cta - ABA0, head = an >> 2, j0 = (an & 3) * 32;
  const uint32_t raw = su32(smb), base = (raw + 1023) & ~1023u;
#if FG_STAMPS
  unsigned k1c = 0;
  if (tid == 0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(k1c) : "l"(&g_ab_cnt[cta]) : "memory");
  if (tid == 0) K1_STAMP(TS_ENTRY);
#endif
  const uint32_t bB = base, bW = base + 16 * BA;
  const uint32_t bf = su32(ab_bar), bfull = bf, bacc = bf + 256, brs = bf + 264, babar = bf + 272;
  const CUtensorMap* tmA = rank == 0 ? tmA32 : tmA20;
  const CUtensorMap* tmW = head < 8 ? tmWq : (head == 8 ? tmWk : tmWv);
  const int c3 = ((head < 8 ? l * QD + head * HD : l * KVD) + j0) >> 3;
  if (tid == 0) {
    mbar_init(bfull, 1);
    mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // earlier phases used this smem through the generic proxy
  }
  if (warp == 1) { tmem_alloc64c2(su32(&ab_tslot)); fence_before(); }
  cluster_sync_all();  // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
  fence_after();
  wpre_arrive(w);
  if (tid == 0) {  // PRE: the whole weight tile does not depend on I
    if (rank == 0) mbar_expect(bfull, 2 * SB);
    for (int c = 0; c < KB; ++c) tma4d2(bW + c * 8192, tmW, bfull, c * 64, 0, 0, c3);
#if FG_STAMPS
    K1_STAMP(TS_PREISS);
#endif
  }
  float2 rp[8];  // RoPE table entries, loaded while the I->A barrier / TMAs / MMAs are in flight
  const int qq = warp & 3, sub = warp >> 2, h = qq >> 1, blk = (qq & 1) * 2 + sub;
  if (ROPE) ab_load_rope(p.rope, head, j0 + blk * 8 + (lane >> 2), h, lane, rp);
  wpre_wait(w);
  fence_after();
  const uint32_t tb = ab_tslot;
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_BAR);
#endif
  if (warp == 0) {
    if (lane == 0) {
      if (rank == 0) mbar_expect(babar, (32 + 20) * 128 * 16);
      for (int c = 0; c < 16; ++c) tma2d2(bB + c * BA, tmA, babar, c * 64, rank * 32);
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
      mbar_wait(bfull, 0);
#if FG_STAMPS
      if (el) K1_STAMP(TS_FULL0);
#endif
      fence_after();
      if (el) {
#if FG_STAMPS
        K1_STAMP(TS_MMA0);
#endif
#pragma unroll
        for (int c = 0; c < KB; ++c) {
          const ull dA = dW0 + (ull)((c * 8192) >> 4), dB = dB0 + (ull)((c * BA) >> 4);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_tc2(tb, dA + (ull)((kk * 32) >> 4), dB + (ull)((kk * 32) >> 4), id, (c | kk) != 0);
        }
        commit2(bacc);
#if FG_STAMPS
        K1_STAMP(TS_MMAEND);
#endif
      }
    }
  } else if (warp < 4) {
    ab_rs_warps(p, tid, lane);
  }
  {
    __syncwarp();
    const uint32_t ta = tb + ((uint32_t)(qq * 32 + sub * 16) << 16);
    const uint32_t scr = bW + warp * 1024;  // per-warp transpose scratch (reuses the weight tile after the MMAs are done)
    uint32_t va[16];
    mbar_wait(brs, 0);
    mbar_wait(bacc, 0);
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_EPI0);
#endif
    fence_after();
    ld_tmem16(va, ta);
    wait_ld();
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_LDDONE);
#endif
    fence_before();
    ab_epi<ROPE>(p, va, head, j0, blk, h, scr, rp, lane);
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_STDONE);
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");  // q / ksuf / vsuf (generic proxy) are read by the K2 attention TMA after the A->C barrier
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64c2(tb); }
#if FG_STAMPS
  AB_STAMP_SAVE(0, cta, k1c, l);
#endif
}
#endif

#if AB_MASK & 4
// ------------------------------------------------------------------------------------------------ K split across two cta_group::2 pairs (80 CTAs, 68..147)
// Pair pr = (cta - 68) / 2: kh = pr / 20 (K half), tile = pr % 20; CTA rank r of the pair owns 64 weight rows (tile * 2 + r). K half 1 writes its raw fp32 accumulators to
// a global partial (layout [an][warp][4][lane] float4) and releases a per-warp counter; K half 0 waits for the same warp's counter, adds, and runs the epilogue.
template <bool ROPE>
__device__ __noinline__ void ab_A2k(const Params& p, const CUtensorMap* tmWq, const CUtensorMap* tmWk, const CUtensorMap* tmWv, const CUtensorMap* tmA32, const CUtensorMap* tmA20,
                                    int l, int it, float* part2, unsigned* kscnt, const WPre& w, bf16* smb) {
  constexpr int KB = 8, SB = KB * 8192, BA = 4096;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x, rank = cta & 1;
  const int pr = (cta - KS_A0) >> 1, kh = pr / 20, an = (pr % 20) * 2 + rank, head = an >> 2, j0 = (an & 3) * 32;
  const uint32_t raw = su32(smb), base = (raw + 1023) & ~1023u;
#if FG_STAMPS
  unsigned k1c = 0;
  if (tid == 0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(k1c) : "l"(&g_ab_cnt[cta]) : "memory");
  if (tid == 0) K1_STAMP(TS_ENTRY);
#endif
  const uint32_t bB = base, bW = base + KB * BA;
  const uint32_t bf = su32(ab_bar), bfull = bf, bacc = bf + 256, brs = bf + 264, babar = bf + 272;
  const CUtensorMap* tmA = rank == 0 ? tmA32 : tmA20;
  const CUtensorMap* tmW = head < 8 ? tmWq : (head == 8 ? tmWk : tmWv);
  const int c3 = ((head < 8 ? l * QD + head * HD : l * KVD) + j0) >> 3;
  if (tid == 0) {
    mbar_init(bfull, 1);
    mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (warp == 1) { tmem_alloc64c2(su32(&ab_tslot)); fence_before(); }
  cluster_sync_all();
  fence_after();
  wpre_arrive(w);
  if (tid == 0) {
    if (rank == 0) mbar_expect(bfull, 2 * SB);
    for (int c = 0; c < KB; ++c) tma4d2(bW + c * 8192, tmW, bfull, kh * 512 + c * 64, 0, 0, c3);
#if FG_STAMPS
    K1_STAMP(TS_PREISS);
#endif
  }
  float2 rp[8];
  const int qq = warp & 3, sub = warp >> 2, h = qq >> 1, blk = (qq & 1) * 2 + sub;
  if (ROPE && kh == 0) ab_load_rope(p.rope, head, j0 + blk * 8 + (lane >> 2), h, lane, rp);
  wpre_wait(w);
  fence_after();
  const uint32_t tb = ab_tslot;
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_BAR);
#endif
  if (warp == 0) {
    if (lane == 0) {
      if (rank == 0) mbar_expect(babar, (32 + 20) * 128 * KB);
      for (int c = 0; c < KB; ++c) tma2d2(bB + c * BA, tmA, babar, kh * 512 + c * 64, rank * 32);
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
      mbar_wait(bfull, 0);
#if FG_STAMPS
      if (el) K1_STAMP(TS_FULL0);
#endif
      fence_after();
      if (el) {
#if FG_STAMPS
        K1_STAMP(TS_MMA0);
#endif
#pragma unroll
        for (int c = 0; c < KB; ++c) {
          const ull dA = dW0 + (ull)((c * 8192) >> 4), dB = dB0 + (ull)((c * BA) >> 4);
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) mma_tc2(tb, dA + (ull)((kk * 32) >> 4), dB + (ull)((kk * 32) >> 4), id, (c | kk) != 0);
        }
        commit2(bacc);
#if FG_STAMPS
        K1_STAMP(TS_MMAEND);
#endif
      }
    }
  } else if (warp < 4 && kh == 0) {
    ab_rs_warps(p, tid, lane);
  }
  {
    __syncwarp();
    const uint32_t ta = tb + ((uint32_t)(qq * 32 + sub * 16) << 16);
    const uint32_t scr = bW + warp * 1024;
    uint32_t va[16];
    if (kh == 0) mbar_wait(brs, 0);
    mbar_wait(bacc, 0);
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_EPI0);
#endif
    fence_after();
    ld_tmem16(va, ta);
    wait_ld();
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_LDDONE);
#endif
    fence_before();
    float4* pp = reinterpret_cast<float4*>(part2) + (size_t)(an * 8 + warp) * 4 * 32 + lane;
    unsigned* cnt = kscnt + (size_t)(an * 8 + warp) * CNT_STRIDE;
    if (kh == 1) {
#pragma unroll
      for (int j = 0; j < 4; ++j) __stcg(pp + j * 32, make_float4(__uint_as_float(va[4 * j]), __uint_as_float(va[4 * j + 1]), __uint_as_float(va[4 * j + 2]), __uint_as_float(va[4 * j + 3])));
      __syncwarp();
      if (lane == 0) red_release(cnt);
    } else {
      if (lane == 0) spin_acquire(cnt, (unsigned)it + 1u);
      __syncwarp();
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_X0);
#endif
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 o = __ldcg(pp + j * 32);
        va[4 * j] = __float_as_uint(__uint_as_float(va[4 * j]) + o.x); va[4 * j + 1] = __float_as_uint(__uint_as_float(va[4 * j + 1]) + o.y);
        va[4 * j + 2] = __float_as_uint(__uint_as_float(va[4 * j + 2]) + o.z); va[4 * j + 3] = __float_as_uint(__uint_as_float(va[4 * j + 3]) + o.w);
      }
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_X1);
#endif
      ab_epi<ROPE>(p, va, head, j0, blk, h, scr, rp, lane);
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_STDONE);
#endif
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64c2(tb); }
#if FG_STAMPS
  AB_STAMP_SAVE(0, cta, k1c, l);
#endif
}
#endif

#if AB_MASK & 8
// ------------------------------------------------------------------------------------------------ K split, cta_group::1 M = 64 / N = 64 with a DSMEM exchange (harness only)
// CTA pair (68 + 2t, 69 + 2t) = tile t (64 weight rows), rank = K half. B tile: A' rows 0..51 (the I layout: rows 0..31 and 32..51 of Ap) in an N = 64 atom (8 KB per K64
// chunk, rows 52..63 never read). Epilogue warp (q = lane quadrant = 16-row block, h = column half): same layout as the cl2 epilogue. Rank 1 pushes its 16 accumulator words
// per thread into rank 0's smem scratch and arrives on rank 0's per-warp mbarrier; rank 0 adds and runs the epilogue.
template <bool ROPE>
__device__ __noinline__ void ab_A1(const Params& p, const CUtensorMap* tmWq, const CUtensorMap* tmWk, const CUtensorMap* tmWv, const CUtensorMap* tmA32, const CUtensorMap* tmA20,
                                   int l, const WPre& w, bf16* smb) {
  constexpr int KB = 8, BA = 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  const int ix = cta - KS_A0, an = ix >> 1, kh = ix & 1, head = an >> 2, j0 = (an & 3) * 32;
  const uint32_t raw = su32(smb), base = (raw + 1023) & ~1023u;
#if FG_STAMPS
  unsigned k1c = 0;
  if (tid == 0) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(k1c) : "l"(&g_ab_cnt[cta]) : "memory");
  if (tid == 0) K1_STAMP(TS_ENTRY);
#endif
  const uint32_t bB = base, bW = base + KB * BA, bX = bW + KB * 8192;
  const uint32_t bf = su32(ab_bar), bfull = bf, bacc = bf + 256, brs = bf + 264, babar = bf + 272, bx = bf + 288;
  const CUtensorMap* tmW = head < 8 ? tmWq : (head == 8 ? tmWk : tmWv);
  const int c3 = ((head < 8 ? l * QD + head * HD : l * KVD) + j0) >> 3;
  if (tid == 0) {
    mbar_init(bfull, 1);
    mbar_init(bacc, 1); mbar_init(brs, 2); mbar_init(babar, 1);
    for (int i = 0; i < 8; ++i) mbar_init(bx + 8 * i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (warp == 1) { tmem_alloc64c1(su32(&ab_tslot)); fence_before(); }
  cluster_sync_all();  // the partner's exchange barriers are initialised before any remote arrive
  fence_after();
  wpre_arrive(w);
  if (tid == 0) {
    mbar_expect(bfull, KB * 8192);
    for (int c = 0; c < KB; ++c) tma4d(bW + c * 8192, tmW, bfull, kh * 512 + c * 64, 0, 0, c3);
#if FG_STAMPS
    K1_STAMP(TS_PREISS);
#endif
  }
  float2 rp[8];
  const int qq = warp & 3, h = warp >> 2, blk = qq;
  if (ROPE && kh == 0) ab_load_rope(p.rope, head, j0 + blk * 8 + (lane >> 2), h, lane, rp);
  wpre_wait(w);
  fence_after();
  const uint32_t tb = ab_tslot;
#if FG_STAMPS
  if (tid == 0) K1_STAMP(TS_BAR);
#endif
  if (warp == 0) {
    if (lane == 0) {
      mbar_expect(babar, KB * (32 + 20) * 128);
      for (int c = 0; c < KB; ++c) {
        tma2d(bB + c * BA, tmA32, babar, kh * 512 + c * 64, 0);
        tma2d(bB + c * BA + 4096, tmA20, babar, kh * 512 + c * 64, 32);
      }
#if FG_STAMPS
      K1_STAMP(TS_ISSUED);
#endif
    }
  } else if (warp == 1) {
    const bool el = elect_one();
    const ull dW0 = sdesc(bW), dB0 = sdesc(bB);
    const uint32_t id = idesc(64, 64);
    mbar_wait(babar, 0);
#if FG_STAMPS
    if (el) K1_STAMP(TS_B0);
#endif
    mbar_wait(bfull, 0);
#if FG_STAMPS
    if (el) K1_STAMP(TS_FULL0);
#endif
    fence_after();
    if (el) {
#if FG_STAMPS
      K1_STAMP(TS_MMA0);
#endif
#pragma unroll
      for (int c = 0; c < KB; ++c) {
        const ull dA = dW0 + (ull)((c * 8192) >> 4), dB = dB0 + (ull)((c * BA) >> 4);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) mma_tc1(tb, dA + (ull)((kk * 32) >> 4), dB + (ull)((kk * 32) >> 4), id, (c | kk) != 0);
      }
      commit1(bacc);
#if FG_STAMPS
      K1_STAMP(TS_MMAEND);
#endif
    }
  } else if (warp < 4 && kh == 0) {
    ab_rs_warps(p, tid, lane);
  }
  {
    __syncwarp();
    const uint32_t ta = tb + ((uint32_t)(qq * 32) << 16) + (uint32_t)(h * 32);
    const uint32_t scr = bW + warp * 1024;
    uint32_t va[16];
    if (kh == 0) mbar_wait(brs, 0);
    mbar_wait(bacc, 0);
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_EPI0);
#endif
    fence_after();
    ld_tmem16(va, ta);
    wait_ld();
#if FG_STAMPS
    if (tid == 128) K1_STAMP(TS_LDDONE);
#endif
    fence_before();
    const uint32_t xs = bX + (warp * 32 + lane) * 64;  // this thread's 64 B exchange slot (rank 0's smem)
    if (kh == 1) {
      const uint32_t rx = mapa(xs, 0);
#pragma unroll
      for (int j = 0; j < 4; ++j) st_cluster_v4(rx + 16 * j, make_uint4(va[4 * j], va[4 * j + 1], va[4 * j + 2], va[4 * j + 3]));
      __syncwarp();
      if (lane == 0) mbar_arrive_remote(mapa(bx + 8 * warp, 0));
    } else {
      mbar_wait_cluster(bx + 8 * warp, 0);
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_X0);
#endif
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float4 o; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(o.x), "=f"(o.y), "=f"(o.z), "=f"(o.w) : "r"(xs + 16 * j));
        va[4 * j] = __float_as_uint(__uint_as_float(va[4 * j]) + o.x); va[4 * j + 1] = __float_as_uint(__uint_as_float(va[4 * j + 1]) + o.y);
        va[4 * j + 2] = __float_as_uint(__uint_as_float(va[4 * j + 2]) + o.z); va[4 * j + 3] = __float_as_uint(__uint_as_float(va[4 * j + 3]) + o.w);
      }
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_X1);
#endif
      ab_epi<ROPE>(p, va, head, j0, blk, h, scr, rp, lane);
#if FG_STAMPS
      if (tid == 128) K1_STAMP(TS_STDONE);
#endif
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free64c1(tb); }
  cluster_sync_all();  // rank 1 must not leave (and free its smem barrier peers) before rank 0 consumed the exchange
#if FG_STAMPS
  AB_STAMP_SAVE(0, cta, k1c, l);
#endif
}
#endif

// ------------------------------------------------------------------------------------------------ the VT
struct K1 {
#if AB_MASK & 8
  static constexpr const char* kName = "ab_k1s";
#else
  static constexpr const char* kName = "ab_k1c";
#endif
  static constexpr const char* kBin = kName;
  static constexpr bool kHasK0 = false;
  static const char* vname(int v) { return (v >= ABV_K1 && v < ABV_K1 + 4) ? K1_NAMES[v - ABV_K1] : nullptr; }
  static int parse(const std::string& s) {
    for (int i = 0; i < 4; ++i) if (s == K1_NAMES[i] && (AB_MASK & (1 << i))) return ABV_K1 + i;
    return -1;
  }
  static bool norope(int v) { return v == ABV_K1 + V_NOROPE; }
  __device__ static void A(const Params& p, const AbArgs& a, int l, int it, const WPre& w, bf16* smb) {
    const int v = a.variant - ABV_K1, cta = blockIdx.x;
    const CUtensorMap* mq = &a.maps[0], *mk = &a.maps[1], *mv = &a.maps[2], *m32 = &a.maps[3], *m20 = &a.maps[4];
    bool did = false;
#if AB_MASK & 3
    if (v == V_CL2 || v == V_NOROPE) {
      if (cta >= ABA0) {
        if (v == V_CL2) ab_A2<true>(p, mq, mk, mv, m32, m20, l, w, smb); else ab_A2<false>(p, mq, mk, mv, m32, m20, l, w, smb);
        if (a.ctx & CTX_KV) ab_kv_issue(a.pk, a.pv, l, smb);  // ak_pre after the epilogue (CTAs 108..142; >= 143 return)
      } else ab_null_A(a, l, w, smb, ABA0);
      did = true;
    }
#endif
#if AB_MASK & 4
    if (v == V_KS2C2) {
      if (cta >= KS_A0) {
        ab_A2k<true>(p, mq, mk, mv, m32, m20, l, it, a.kspart, a.kscnt, w, smb);
        if (a.ctx & CTX_KV) ab_kv_issue(a.pk, a.pv, l, smb);
      } else ab_null_A(a, l, w, smb, KS_A0);
      did = true;
    }
#endif
#if AB_MASK & 8
    if (v == V_KS2C1) {
      if (cta >= KS_A0) {
        ab_A1<true>(p, mq, mk, mv, m32, m20, l, w, smb);
        if (a.ctx & CTX_KV) ab_kv_issue(a.pk, a.pv, l, smb);
      } else ab_null_A(a, l, w, smb, KS_A0);
      did = true;
    }
#endif
    if (!did) ab_null_A(a, l, w, smb, ABA0);  // null / null_apw
  }
  __device__ static void B(const Params&, const AbArgs&, int, bf16*) {}

  // ---- host
  static void prepare(AbArgs& a, int grid, bf16* ap) {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    auto wmap = [&](CUtensorMap* m, const bf16* base, size_t rows) {
      cuuint64_t dims[4] = {W, 8, 2, (cuuint64_t)(rows / 8)}, strides[3] = {(cuuint64_t)W * 2, (cuuint64_t)128 * W * 2, (cuuint64_t)W * 2 * 8};
      cuuint32_t box[4] = {64, 8, 2, 4}, es[4] = {1, 1, 1, 1};
      CUresult r = enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "weight tensor map encode failed %d\n", (int)r); exit(1); }
    };
    wmap(&a.maps[0], a.wq, (size_t)L * QD); wmap(&a.maps[1], a.wk, (size_t)L * KVD); wmap(&a.maps[2], a.wv, (size_t)L * KVD);
    for (int k = 0; k < 2; ++k) {
      cuuint64_t dims[2] = {W, 64}, strides[1] = {(cuuint64_t)W * 2};
      cuuint32_t box[2] = {64, (cuuint32_t)(k == 0 ? 32 : 20)}, es[2] = {1, 1};
      CUresult r = enc(&a.maps[3 + k], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ap, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "A tensor map encode failed %d\n", (int)r); exit(1); }
    }
    CK(cudaMalloc(&a.kspart, (size_t)ABA2_N * 8 * 16 * 32 * 4));
    unsigned z = 0; CK(cudaMemcpyToSymbol(g_ab_to, &z, 4));
    fprintf(stderr, "k1 prepare: grid=%d mask=%d\n", grid, AB_MASK);
  }
  static void post_run(int variant, int idx, int reps, const std::string& out, const std::string& name, double nspc) {
#if FG_STAMPS
    if (variant >= ABV_K1 && idx >= reps - 2 && idx >= 0) {
      std::vector<ull> ts((size_t)148 * K1_NSTEP * K1_NTS);
      CK(cudaMemcpyFromSymbol(ts.data(), g_ab_ts, ts.size() * 8));
      char fn[512]; snprintf(fn, sizeof fn, "%s/raw/%s_%s_%03d.k1tl.u64", out.c_str(), name.c_str(), K1_NAMES[variant - ABV_K1], idx);
      FILE* f = fopen(fn, "wb"); fwrite(&nspc, 8, 1, f); fwrite(ts.data(), 8, ts.size(), f); fclose(f);
    }
#endif
  }
};

int main(int argc, char** argv) { return ab_main<K1>(argc, argv); }
