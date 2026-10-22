// att_k1.cu: K1 = tcgen05 (UMMA, cta_group::2) + FA4-style attention variants for the standalone attention harness (att_common.cuh, -DATT_HOOKS).
// Also contains K0 verbatim (ATTV_K0, as att_k0.cu) so null / k0 / K1 variants are interleaved (ABAB) in one binary.
//
// Unit = one cluster pair (2 CTAs, rank = blockIdx.x & 1), cta_group::2 M = 256: pair-tile t in {0,1}, CTA rank r owns 100 query rows [100 (2t + r), +100) padded to the
// 128 TMEM lanes (400 rows = 4 x 100; rows >= 100 hold garbage and are never stored). Keys: 1024 slots (968 prefix, 50 suffix, 6 pad) in chunks of NCH keys; a unit
// handles CPC consecutive chunks, S = (1024 / NCH) / CPC key splits, 2 S pairs = 4 S CTAs (CTA ids 0 .. 4S - 1; the others only run combine tasks).
//   S = Q K^T : A = Q smem (4 x [128 rows][64 hd] SW128 K-major, TMA box {64, 100}), B = K smem (each CTA stages NCH/2 keys), N = NCH, D = TMEM cols 256 + i NCH
//   P = bf16(exp2(t - m)) (one thread per row, tcgen05.ld/st 32x32b), P overwrites S in TMEM (A-from-TMEM) or goes to smem (PSMEM)
//   O = P V   : B = V smem MN-major SW128 (each CTA stages its 128-wide hd half as 2 x [NCH keys][64 hd]), N = 256, D = TMEM cols 0..255
// K/V prefix chunks are TMA-prefetched in B_pre (before the B tasks, like K0's PREFILL=2); Q and the suffix keys (written by the B tasks) are TMA'd after the B->C barrier.
// Partials: fp32 quad-major opart[split][col/4][row][4] and stats[split][row] = (m_log2, l); the combine (C3) reads them in ONE round trip after the row-group counter.
// Modes (FL bits): PP = S/P ping-pong with lazy rescale (online softmax, tau), else exact unit-wide max (two passes over S in TMEM), PSMEM, TAU0, ...
// Build (login node):   ./build.sh att_k1 -DATT_HOOKS      (needs att_k1_ptx.cuh)
// Run: att_k1 --smem 197632 --cluster 2 --variants null,k0,<names>   (K1 variants need the cluster launch; they are cta_group::2 only)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <string>
#include "att_common.cuh"
#include "att_k1_ptx.cuh"

// ------------------------------------------------------------------------------------------------ variant table
// X(idx, name, NCH, CPC, FL)
constexpr int FL_PP = 1, FL_PSMEM = 2, FL_TAU0 = 4, FL_BF16P = 8, FL_POLY = 16, FL_CMB1 = 32, FL_SM2 = 64, FL_STRM = 128;
// FL_BF16P: partials as bf16 octets [split][col/8][row][8]; FL_CMB1: bf16 combine with 128 tasks (one octet per thread, overlaps unit CTAs);
// FL_STRM (bf16 only): per-split completion flags (word 1 + split of the group counter line) and a streaming combine that accumulates splits in arrival order;
// FL_SM2: softmax over two warps per TMEM lane quadrant (column halves), CPC == 1 only; FL_POLY: ex2 of a fraction of the elements on the FMA pipe.
#ifdef K1_SEL_H
#include K1_SEL_H
#define K1_LIST(X) K1_SELECT(X)
#else
#define K1_ALL(X) \
  X(0, "u1_c1", 64, 1, 0) \
  X(1, "u1_c1_ps", 64, 1, FL_PSMEM) \
  X(2, "u1_c2", 64, 2, 0) \
  X(3, "u1_c4", 64, 4, 0) \
  X(4, "u2_pp_c2", 64, 2, FL_PP) \
  X(5, "u2_pp_c4", 64, 4, FL_PP) \
  X(6, "u3_pp_c2_tau0", 64, 2, FL_PP | FL_TAU0) \
  X(7, "u3_pp_c4_tau0", 64, 4, FL_PP | FL_TAU0) \
  X(9, "u1_n32_c2", 32, 2, 0) \
  X(10, "u2_pp_n32_c4", 32, 4, FL_PP) \
  X(11, "u1_n128_c1", 128, 1, 0) \
  X(12, "u1_n128_c2", 128, 2, 0) \
  X(20, "r2_bf", 64, 1, FL_BF16P) \
  X(21, "r2_bf_c128", 64, 1, FL_BF16P | FL_CMB1) \
  X(22, "r2_sm2", 64, 1, FL_SM2) \
  X(23, "r2_bf_sm2", 64, 1, FL_BF16P | FL_SM2) \
  X(24, "r2_bf_c2", 64, 2, FL_BF16P) \
  X(25, "r2_bf_sm2_poly", 64, 1, FL_BF16P | FL_SM2 | FL_POLY) \
  X(26, "r2_bf_poly", 64, 1, FL_BF16P | FL_POLY) \
  X(27, "r2_bf_strm", 64, 1, FL_BF16P | FL_STRM) \
  X(28, "r2_bf_c128_strm", 64, 1, FL_BF16P | FL_CMB1 | FL_STRM) \
  X(29, "r2_bf_sm2_c128", 64, 1, FL_BF16P | FL_SM2 | FL_CMB1) \
  X(30, "r2_bf_sm2_c128_strm", 64, 1, FL_BF16P | FL_SM2 | FL_CMB1 | FL_STRM)
#define K1_LIST(X) K1_ALL(X)
#endif
constexpr int ATTV_K1 = ATTV_EXT + 1;
#define X(i, n, NCH, CPC, FL) +1
constexpr int K1_N = 0 K1_LIST(X);
#undef X
static const char* att_ext_vname(int v) {
#define X(i, n, NCH, CPC, FL) if (v == ATTV_K1 + i) return n;
  K1_LIST(X)
#undef X
  return "var";
}
static int att_ext_parse(const std::string& s) {
#define X(i, n, NCH, CPC, FL) if (s == n) return ATTV_K1 + i;
  K1_LIST(X)
#undef X
  return -1;
}

// ------------------------------------------------------------------------------------------------ shared state / constants
constexpr int K1_NSLOT = 1024;          // key slots: 968 prefix, 50 suffix, 6 pad
constexpr int K1_RPC = 100;             // query rows per CTA
constexpr int K1_NGRP = 4;              // row groups (counters CNT_ATT + g)
constexpr int K1_QBYTES = 4 * 16384;    // Q tile region
__constant__ __align__(64) CUtensorMap c_tm[16];
__device__ unsigned g_k1_to;
__shared__ __align__(8) ull k1_bar[32];  // qfull[4] @0, kfull[4] @4, vfull[4] @8, sfull[4] @12, pfull[4] @16, pvdone[4] @20
__shared__ uint32_t k1_tslot;
__shared__ float k1_redm[2][128];  // SM2: per-half partial row max / partial row sum
__shared__ float k1_redl[128];
#define KB_Q(k) (su32(k1_bar) + 8 * (k))
#define KB_K(c) (su32(k1_bar) + 8 * (4 + (c)))
#define KB_V(c) (su32(k1_bar) + 8 * (8 + (c)))
#define KB_S(c) (su32(k1_bar) + 8 * (12 + (c)))
#define KB_P(c) (su32(k1_bar) + 8 * (16 + (c)))
#define KB_D(c) (su32(k1_bar) + 8 * (20 + (c)))
#ifdef K1_DBG
#define K1_SPIN (1ull << 18)
#else
#define K1_SPIN (1ull << 24)
#endif
__device__ __forceinline__ void k1_wait(uint32_t b, uint32_t par) {
  ull i = 0;
  while (!mbar_try(b, par)) {
    if (++i >= K1_SPIN) {
      atomicAdd(&g_k1_to, 1u);
#ifdef K1_DBG
      if (atomicAdd(&g_k1_to, 0u) < 30) printf("TIMEOUT cta %d warp %d bar %x par %d\n", blockIdx.x, threadIdx.x >> 5, b, par);
#endif
      return;
    }
  }
}
#if ATT_MARKS
#define K1_MARK(mk, k) do { if ((mk) != nullptr) (mk)[(k)] = clock64(); } while (0)
#else
#define K1_MARK(mk, k) do {} while (0)
#endif
__device__ __forceinline__ float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// 2^x on the FMA pipe (x <= 0 after the max subtraction): degree-3 minimax of 2^f on [0,1), relative error 7.5e-5 (well under the bf16 rounding of p)
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -126.f);
  const float j = floorf(x), f = x - j;
  const float p = fmaf(fmaf(fmaf(0.07802412464f, f, 0.22606275817f), f, 0.69583763411f), f, 0.99992463609f);
  return __int_as_float(__float_as_int(p) + ((int)j << 23));
}
constexpr float K1_SC = 0.0625f * 1.4426950408889634f;  // score scale (1/16) in the log2 domain

template <int NCH> struct K1G {
  static constexpr int NCHUNK = K1_NSLOT / NCH;
  static constexpr int KHALF = NCH * 256;       // bytes of K per CTA per chunk (NCH/2 rows x 512 B)
  static constexpr int VHALF = NCH * 256;       // bytes of V per CTA per chunk (NCH keys x 256 B)
  static constexpr int CSTRIDE = NCH * 512;     // smem bytes per chunk (K then V)
  static constexpr int MK = NCH == 32 ? 0 : NCH == 64 ? 1 : 2;
  static constexpr int M_KP = 1 + MK, M_VP = 4 + MK, M_KP8 = 7, M_KS8 = 8, M_VP8 = 9, M_VS8 = 10;
};

// ---- K0 verbatim (att_k0.cu), dispatch-guarded by a.variant == ATTV_K0
constexpr int ATT_PS_ELEMS = 0, ATT_WRED_BYTES = 0;
struct K0Code {
  // ---- VERBATIM pre-B prefill (mk/pi05_mk.cu, PI05_ATT_PREFILL == 2): issued before the B tasks, after Wo / prefix K/V prefetch
  __device__ static void B_pre(const Params& p, const AttArgs& a, int l, bf16* smb) {
    const int tid = threadIdx.x;
    if (a.variant != ATTV_K0) return;
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
  }
  // ---- VERBATIM phase C (split-key attention units); the unit's cp.async-filled smem, red.release on CNT_ATT + rg
  __device__ static void C(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    if (a.variant != ATTV_K0) {
      if (a.variant == ATTV_CALIB) att_spin(a.ncal);
      return;
    }
      // ---- C: split-key attention. Unit = (ATT_ROWS query rows, ATT_KEYS keys):
      // scores = bf16(q.k)/16 (masked), chunk-local max m_c, p = bf16(exp(s - m_c)),
      // l_c = sum p, o_c = p.V. The combine phase rescales by exp(m_c - M).
      {
        bf16* Ks = smb;
        bf16* Qs = Ks + ATT_KEYS * KS_STRIDE;
        bf16* Vs = Qs + ATT_ROWS * KS_STRIDE;
        uint8_t* Mk = reinterpret_cast<uint8_t*>(Vs + ATT_KEYS * VS_STRIDE);
        const bf16* pk = p.prefix_k + (size_t)l * P * HD;
        const bf16* pv = p.prefix_v + (size_t)l * P * HD;
        for (int u = blockIdx.x; u < ATT_NRG * ATT_NCHUNK; u += gridDim.x) {
          const int rg = u % ATT_NRG, kc = u / ATT_NRG;
          const int r0 = rg * ATT_ROWS, j0 = kc * ATT_KEYS;
          // Prefilled unit: only rows P <= j < NK remain (empty unless the chunk holds suffix keys).
          const bool pre = u == blockIdx.x;
          const int jlo = pre ? min(max(P - j0, 0), ATT_KEYS) : 0;
          const int jhi = pre ? min(max(NK - j0, 0), ATT_KEYS) : ATT_KEYS;
          if (!pre && tid < ATT_KEYS) {
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
              const int r = r0 + rr;  // rows >= QROWS (last row group) are zero-filled
              cp_async16(Qs + rr * KS_STRIDE + pc * 8, p.q + (size_t)min(r, QROWS - 1) * HD + pc * 8, r < QROWS ? 16 : 0);
            }
            cp_async_wait_all();
          }
          __syncthreads();
          ATT_MARK(mk, 1);
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
            ATT_MARK(mk, 2);
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
              rsum[0] += pr[0]; rsum[0] += pr[1];
              rsum[1] += pr[2]; rsum[1] += pr[3];
              const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]);
              const __nv_bfloat162 hi = __floats2bfloat162_rn(pr[2], pr[3]);
              pa[nt][0] = *reinterpret_cast<const uint32_t*>(&lo);
              pa[nt][1] = *reinterpret_cast<const uint32_t*>(&hi);
            }
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              rsum[hf] += __shfl_xor_sync(0xffffffffu, rsum[hf], 1);
              rsum[hf] += __shfl_xor_sync(0xffffffffu, rsum[hf], 2);
            }
            ATT_MARK(mk, 3);
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
            ATT_MARK(mk, 4);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
              const int c = dh * 128 + nt * 8 + 2 * t4;
              if (ra < QROWS) *reinterpret_cast<float2*>(p.opart + ((size_t)kc * QROWS + ra) * HD + c) = make_float2(acc[nt][0], acc[nt][1]);
              if (rb < QROWS) *reinterpret_cast<float2*>(p.opart + ((size_t)kc * QROWS + rb) * HD + c) = make_float2(acc[nt][2], acc[nt][3]);
            }
          }
          __syncthreads();  // publish stats/opart of this (row group, key chunk) unit
          if (tid == 0 && (p.fuse & FUSE_CC3)) red_release(p.cnt + (CNT_ATT + rg) * CNT_STRIDE);
          ATT_MARK(mk, 5);
        }
      }
  }
  // ---- VERBATIM phase C3 (combine; counter wait per 4-row task)
  __device__ static void C3(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    const int tid = threadIdx.x;
    const unsigned it1 = (unsigned)(it + 1);  // counter epoch of this layer
    const bool backoff = (p.fuse & FUSE_BACKOFF) != 0;
    if (a.variant != ATTV_K0) return;
    {
      constexpr int CR = NTHREADS * 4 / HD;  // 4 rows per task, one column quad per thread
      for (int task = gridDim.x - 1 - blockIdx.x; task < QROWS / CR; task += gridDim.x) {
        if (p.fuse & FUSE_CC3) block_wait(p.cnt + (CNT_ATT + task * CR / ATT_ROWS) * CNT_STRIDE, it1 * ARR_ATT, backoff);
        ATT_MARK(mk, 6);
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
        ATT_MARK(mk, 7);
      }
    }
  }
};


// ------------------------------------------------------------------------------------------------ K/V loads of one chunk (prefix part in B_pre, suffix part after the barrier)
// li = local chunk index in the unit (barrier slot), c = absolute chunk. Normal chunks (all slots < P) are loaded in the pre phase only; the special chunk(s)
// that hold suffix keys load their prefix 8-row groups in the pre phase and the suffix groups in the post phase (the B tasks write ksuf / vsuf this layer).
template <int NCH>
__device__ __forceinline__ void k1_load_chunk(int c, int li, int l, uint32_t rank, uint32_t Kb, uint32_t Vb, bool want_k, bool want_v, bool post) {
  using G = K1G<NCH>;
  const bool special = (c + 1) * NCH > P;
  const int s0c = c * NCH;
  if (want_k) {
    const uint32_t bar = KB_K(li);
    if (!post && rank == 0) mbar_expect(bar, 2 * G::KHALF);
    if (!special) {
      if (!post)
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) tma2d2(Kb + kc * (NCH * 64), &c_tm[G::M_KP], bar, kc * 64, l * P + s0c + (int)rank * (NCH / 2));
    } else {
#pragma unroll
      for (int i = 0; i < NCH / 16; ++i) {
        const int s0 = s0c + (int)rank * (NCH / 2) + 8 * i;
        const bool suf = s0 >= P;
        if (suf == post)
#pragma unroll
          for (int kc = 0; kc < 4; ++kc)
            tma2d2(Kb + kc * (NCH * 64) + i * 1024, suf ? &c_tm[G::M_KS8] : &c_tm[G::M_KP8], bar, kc * 64, suf ? s0 - P : l * P + s0);
      }
    }
  }
  if (want_v) {
    const uint32_t bar = KB_V(li);
    if (!post && rank == 0) mbar_expect(bar, 2 * G::VHALF);
    if (!special) {
      if (!post)
#pragma unroll
        for (int h = 0; h < 2; ++h) tma2d2(Vb + h * (NCH * 128), &c_tm[G::M_VP], bar, (int)rank * 128 + h * 64, l * P + s0c);
    } else {
#pragma unroll
      for (int i = 0; i < NCH / 8; ++i) {
        const int s0 = s0c + 8 * i;
        const bool suf = s0 >= P;
        if (suf == post)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            tma2d2(Vb + h * (NCH * 128) + i * 1024, suf ? &c_tm[G::M_VS8] : &c_tm[G::M_VP8], bar, (int)rank * 128 + h * 64, suf ? s0 - P : l * P + s0);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ pre-B: barrier init, TMEM alloc, cluster sync, K/V prefix TMAs
template <int NCH, int CPC, int FL>
__device__ __noinline__ void k1_prefill(int l, bf16* smb) {
  using G = K1G<NCH>;
  constexpr int S = G::NCHUNK / CPC;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int pi = blockIdx.x >> 1;
  if (pi >= 2 * S) return;
  const uint32_t rank = blockIdx.x & 1;
  const int kg = pi >> 1;
  const uint32_t base = (su32(smb) + 1023) & ~1023u;
  const uint32_t KVb = base + K1_QBYTES;
  if (tid == 0) {
    for (int i = 0; i < 4; ++i) { mbar_init(KB_Q(i), 1); mbar_init(KB_K(i), 1); mbar_init(KB_V(i), 1); mbar_init(KB_S(i), 1); mbar_init(KB_P(i), (FL & FL_SM2) ? 16 : 8); mbar_init(KB_D(i), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) { tmem_alloc2(su32(&k1_tslot), 512); fence_before(); }
  cluster_sync_all();  // both CTAs' barriers initialised and TMEM allocated before any cross-CTA TMA / MMA traffic
  if (tid == 0) {
    // FA4 load order K0, K1, V0, K2, V1, ...
    const int c0 = kg * CPC;
    k1_load_chunk<NCH>(c0, 0, l, rank, KVb, KVb + G::KHALF, true, false, false);
#pragma unroll
    for (int i = 0; i < CPC; ++i) {
      if (i + 1 < CPC) k1_load_chunk<NCH>(c0 + i + 1, i + 1, l, rank, KVb + (i + 1) * G::CSTRIDE, KVb + (i + 1) * G::CSTRIDE + G::KHALF, true, false, false);
      k1_load_chunk<NCH>(c0 + i, i, l, rank, KVb + i * G::CSTRIDE, KVb + i * G::CSTRIDE + G::KHALF, false, true, false);
    }
  }
}

// ------------------------------------------------------------------------------------------------ softmax helpers (one thread = one query row = one TMEM lane)
template <int N> __device__ __forceinline__ void k1_ld_cols(uint32_t ta, uint32_t* v) {
  if constexpr (N == 32) tmem_ld32_x32(ta, v);
  else if constexpr (N == 64) tmem_ld32_x64(ta, v);
  else { tmem_ld32_x64(ta, v); tmem_ld32_x64(ta + 64, v + 64); }
}
template <int N> __device__ __forceinline__ void k1_st_words(uint32_t ta, const uint32_t* w) {  // N / 2 words
  if constexpr (N == 32) tmem_st32_x16(ta, w);
  else if constexpr (N == 64) tmem_st32_x32(ta, w);
  else tmem_st32_x64(ta, w);
}
template <int NCH>
__device__ __forceinline__ float k1_rowmax(const uint32_t* v, const uint32_t* mw) {
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < NCH; ++j) {
    const float t = ((mw[j >> 5] >> (j & 31)) & 1u) ? __uint_as_float(v[j]) * K1_SC : -INFINITY;
    m = fmaxf(m, t);
  }
  return m;
}
// p = bf16(exp2(t - m)) packed in pairs (low = even key), lsum += sum of the rounded p
template <int NCH, bool POLY = false>
__device__ __forceinline__ void k1_pack(const uint32_t* v, const uint32_t* mw, float msafe, uint32_t* pw, float& lsum) {
#pragma unroll
  for (int j2 = 0; j2 < NCH / 2; ++j2) {
    float p0, p1;
    {
      const int j = 2 * j2;
      const float t0 = ((mw[j >> 5] >> (j & 31)) & 1u) ? __uint_as_float(v[j]) * K1_SC : -INFINITY;
      const float t1 = ((mw[(j + 1) >> 5] >> ((j + 1) & 31)) & 1u) ? __uint_as_float(v[j + 1]) * K1_SC : -INFINITY;
      p0 = ex2f(t0 - msafe);
      if constexpr (POLY) p1 = ex2_poly(t1 - msafe); else p1 = ex2f(t1 - msafe);
    }
    const __nv_bfloat162 b2 = __floats2bfloat162_rn(p0, p1);
    lsum += __low2float(b2); lsum += __high2float(b2);
    pw[j2] = *reinterpret_cast<const uint32_t*>(&b2);
  }
}

// ------------------------------------------------------------------------------------------------ the unit (phase C of a K1 variant)
template <int NCH, int CPC, int FL>
__device__ __noinline__ void k1_unit(const uint8_t* pmask, float2* stats, float* opart, unsigned* cnt, bf16* smb, int l, int it, ull* mk) {
  using G = K1G<NCH>;
  constexpr int S = G::NCHUNK / CPC;
  constexpr bool PP = (FL & FL_PP) != 0, PSM = (FL & FL_PSMEM) != 0, BF = (FL & FL_BF16P) != 0, SM2 = (FL & FL_SM2) != 0, POLY = (FL & FL_POLY) != 0;
  static_assert(!SM2 || (CPC == 1 && NCH == 64 && !PP && !PSM), "SM2: serial, one 64-key chunk");
  constexpr int SCOL = 256, NW = NCH / 32;
  static_assert(256 + CPC * NCH <= 512 && CPC <= 4 && NCH >= 32 && NCH <= 128, "TMEM plan");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int pi = blockIdx.x >> 1;
  if (pi >= 2 * S) return;
  const uint32_t rank = blockIdx.x & 1;
  const int t = pi & 1, kg = pi >> 1, grp = 2 * t + (int)rank, c0 = kg * CPC;
  const uint32_t base = (su32(smb) + 1023) & ~1023u;
  const uint32_t Qb = base, KVb = Qb + K1_QBYTES, Pb = KVb + CPC * G::CSTRIDE;
  fence_after();
  const uint32_t tb = k1_tslot;
  const uint32_t lq = ((uint32_t)((warp & 3) * 32)) << 16;

  if (warp == 0) {
    if (lane == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");  // the B tasks' generic-proxy writes of q / ksuf / vsuf, observed through the grid barrier
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        if (rank == 0) mbar_expect(KB_Q(kc), 2 * K1_RPC * 128);
        tma2d2(Qb + kc * 16384, &c_tm[0], KB_Q(kc), kc * 64, K1_RPC * grp);
      }
#pragma unroll
      for (int i = 0; i < CPC; ++i) k1_load_chunk<NCH>(c0 + i, i, l, rank, KVb + i * G::CSTRIDE, KVb + i * G::CSTRIDE + G::KHALF, true, true, true);
    }
  }
  const bool mma = (warp == 1 && rank == 0);  // the MMA warp of the pair
  bool el = false;
  if (mma) el = elect_one();
  const uint32_t idQK = idesc(NCH, 256, 0), idPV = idesc(256, 256, 1);
  auto QK = [&](int i) {
    k1_wait(KB_K(i), 0);
    const uint32_t Kb = KVb + i * G::CSTRIDE;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      if (i == 0) k1_wait(KB_Q(kc), 0);
      fence_after();
      if (el)
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_ss2(tb + SCOL + i * NCH, sdesc(Qb + kc * 16384) + (ull)((kk * 32) >> 4), sdesc(Kb + kc * (NCH * 64)) + (ull)((kk * 32) >> 4), idQK, (kc | kk) != 0);
    }
    if (el) commit2(KB_S(i));
    if (i == 0) K1_MARK(mk, 1);
  };
  auto PV = [&](int i) {
    k1_wait(KB_P(i), 0);
    k1_wait(KB_V(i), 0);
    fence_after();
    const uint32_t Vb = KVb + i * G::CSTRIDE + G::KHALF;
    if (el) {
#pragma unroll
      for (int kk = 0; kk < NCH / 16; ++kk) {
        const ull db = sdesc_mn(Vb + kk * 2048, NCH * 128, 1024);
        if constexpr (PSM) mma_ss2(tb, sdesc(Pb + i * 16384) + (ull)((kk * 32) >> 4), db, idPV, (i | kk) != 0);
        else mma_ts2(tb, tb + SCOL + i * NCH + 8 * kk, db, idPV, (i | kk) != 0);
      }
      commit2(KB_D(i));
    }
  };
  if (mma) {
    if constexpr (PP) {
      QK(0);
#pragma unroll
      for (int i = 0; i < CPC; ++i) { if (i + 1 < CPC) QK(i + 1); PV(i); }
    } else if constexpr (SM2) {
      QK(0);  // PV(0) is issued after this warp's own softmax half
    } else {
#pragma unroll
      for (int i = 0; i < CPC; ++i) QK(i);
#pragma unroll
      for (int i = 0; i < CPC; ++i) PV(i);
    }
  }
  float m_ref = -INFINITY, lsum = 0.f;
  if (SM2 || warp >= 4) {  // softmax warps: TMEM lane quadrant warp & 3, row = 32 (warp & 3) + lane
    const int row = (warp & 3) * 32 + lane;
    uint32_t mw[CPC][NW];
#pragma unroll
    for (int i = 0; i < CPC; ++i)
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const int slot = (c0 + i) * NCH + 32 * w + lane;
        const bool ok = slot < P ? (pmask[slot] != 0) : (slot < NK);
        mw[i][w] = __ballot_sync(0xffffffffu, ok);
      }
    auto write_p = [&](int i, const uint32_t* pw) {
      if constexpr (PSM) {
        const uint32_t pr = Pb + i * 16384 + row * 128;
#pragma unroll
        for (int cch = 0; cch < NCH / 8; ++cch) {
          const uint32_t a = pr + ((cch ^ (row & 7)) << 4);
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(pw[4 * cch]), "r"(pw[4 * cch + 1]), "r"(pw[4 * cch + 2]), "r"(pw[4 * cch + 3]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      } else {
        k1_st_words<NCH>(tb + lq + SCOL + i * NCH, pw);
        wait_st();
        fence_before();
      }
      __syncwarp();
      if (lane == 0) mbar_arrive_remote(KB_P(i), 0);
    };
    uint32_t v[NCH], pw[NCH / 2];
    if constexpr (PP) {
      constexpr float TAU = (FL & FL_TAU0) ? 0.f : 8.f;
#pragma unroll
      for (int i = 0; i < CPC; ++i) {
        k1_wait(KB_S(i), 0);
        fence_after();
        k1_ld_cols<NCH>(tb + lq + SCOL + i * NCH, v);
        wait_ld();
        if (i == CPC - 1) K1_MARK(mk, 2);
        const float mi = k1_rowmax<NCH>(v, mw[i]);
        float muse = mi;
        if (i > 0) {
          const bool need = mi > m_ref + TAU;
          if (__any_sync(0xffffffffu, need)) {
            k1_wait(KB_D(i - 1), 0);  // PV(i-1) finished: O is stable
            fence_after();
            const float sc = need ? ex2f(m_ref - mi) : 1.f;
#pragma unroll
            for (int cb = 0; cb < 4; ++cb) {
              uint32_t o[64];
              tmem_ld32_x64(tb + lq + cb * 64, o);
              wait_ld();
#pragma unroll
              for (int e = 0; e < 64; ++e) o[e] = __float_as_uint(__uint_as_float(o[e]) * sc);
              tmem_st32_x64(tb + lq + cb * 64, o);
            }
            wait_st();
            fence_before();
            lsum *= sc;
          }
          muse = need ? mi : m_ref;
        }
        m_ref = muse;
        const float msafe = muse == -INFINITY ? 0.f : muse;
        k1_pack<NCH, POLY>(v, mw[i], msafe, pw, lsum);
        write_p(i, pw);
        if (i == CPC - 1) K1_MARK(mk, 3);
      }
    } else {
      if constexpr (SM2) {
        // two warps per lane quadrant: warps 4..7 take key columns 0..31, warps 0..3 columns 32..63; row max / sum exchanged through smem
        const int hh = warp < 4 ? 1 : 0, pq = warp & 3;
        k1_wait(KB_S(0), 0);
        fence_after();
        uint32_t vh[32], pwh[16];
        tmem_ld32_x32(tb + lq + SCOL + 32 * hh, vh);
        wait_ld();
        if (hh == 0) K1_MARK(mk, 2);
        const uint32_t mwh = hh ? mw[0][1] : mw[0][0];
        float mh = k1_rowmax<32>(vh, &mwh);
        k1_redm[hh][row] = mh;
        asm volatile("bar.sync %0, 64;" ::"r"(1 + pq) : "memory");
        m_ref = fmaxf(mh, k1_redm[hh ^ 1][row]);
        k1_pack<32, POLY>(vh, &mwh, m_ref == -INFINITY ? 0.f : m_ref, pwh, lsum);
        if (hh == 1) k1_redl[row] = lsum;
        k1_st_words<32>(tb + lq + SCOL + 16 * hh, pwh);
        wait_st();
        fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive_remote(KB_P(0), 0);
        if (hh == 0) K1_MARK(mk, 3);
      } else if constexpr (CPC == 1) {
        k1_wait(KB_S(0), 0);
        fence_after();
        k1_ld_cols<NCH>(tb + lq + SCOL, v);
        wait_ld();
        K1_MARK(mk, 2);
        m_ref = k1_rowmax<NCH>(v, mw[0]);
        k1_pack<NCH, POLY>(v, mw[0], m_ref == -INFINITY ? 0.f : m_ref, pw, lsum);
        write_p(0, pw);
        K1_MARK(mk, 3);
      } else {
        float m = -INFINITY;
#pragma unroll
        for (int i = 0; i < CPC; ++i) {
          k1_wait(KB_S(i), 0);
          fence_after();
          k1_ld_cols<NCH>(tb + lq + SCOL + i * NCH, v);
          wait_ld();
          m = fmaxf(m, k1_rowmax<NCH>(v, mw[i]));
        }
        K1_MARK(mk, 2);
        m_ref = m;
        const float msafe = m == -INFINITY ? 0.f : m;
#pragma unroll
        for (int i = 0; i < CPC; ++i) {
          k1_ld_cols<NCH>(tb + lq + SCOL + i * NCH, v);
          wait_ld();
          k1_pack<NCH, POLY>(v, mw[i], msafe, pw, lsum);
          write_p(i, pw);
        }
        K1_MARK(mk, 3);
      }
    }
  }
  if constexpr (SM2) { if (mma) PV(0); }
  // ---- epilogue (all 8 warps): warp -> lane quadrant warp & 3, column half warp >> 2 of O (128 columns); partials out in quad-major layout
  {
    const int row = (warp & 3) * 32 + lane, half = warp >> 2;
    const int grow = K1_RPC * grp + row;
    const bool valid = row < K1_RPC;
    k1_wait(KB_D(CPC - 1), 0);
    fence_after();
    if (tid == 128) K1_MARK(mk, 4);
    if constexpr (SM2) {
      asm volatile("bar.sync %0, 64;" ::"r"(5 + (warp & 3)) : "memory");  // the lane-quadrant pair: other half's row sum is in smem
      if (half == 1) lsum += k1_redl[row];
    }
    if (half == 1 && valid) stats[kg * QROWS + grow] = make_float2(m_ref, lsum);
#pragma unroll
    for (int cb = 0; cb < 2; ++cb) {
      uint32_t o[64];
      tmem_ld32_x64(tb + lq + half * 128 + cb * 64, o);
      wait_ld();
      if constexpr (BF) {
        if (valid) {
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            uint2 lo = pack_bf16x4(__uint_as_float(o[8 * q]), __uint_as_float(o[8 * q + 1]), __uint_as_float(o[8 * q + 2]), __uint_as_float(o[8 * q + 3]));
            uint2 hi = pack_bf16x4(__uint_as_float(o[8 * q + 4]), __uint_as_float(o[8 * q + 5]), __uint_as_float(o[8 * q + 6]), __uint_as_float(o[8 * q + 7]));
            *reinterpret_cast<uint4*>(opart + ((size_t)(kg * 32 + half * 16 + cb * 8 + q) * QROWS + grow) * 4) = make_uint4(lo.x, lo.y, hi.x, hi.y);
          }
        }
      } else if (valid) {
#pragma unroll
        for (int qd = 0; qd < 16; ++qd)
          *reinterpret_cast<float4*>(opart + ((size_t)(kg * 64 + half * 32 + cb * 16 + qd) * QROWS + grow) * 4) =
              make_float4(__uint_as_float(o[4 * qd]), __uint_as_float(o[4 * qd + 1]), __uint_as_float(o[4 * qd + 2]), __uint_as_float(o[4 * qd + 3]));
      }
    }
  }
  fence_before();
  __syncthreads();
  if (warp == 1) { fence_after(); tmem_free2(tb, 512); }
  if (tid == 0) {
    if constexpr ((FL & FL_STRM) != 0)
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(cnt + (CNT_ATT + grp) * CNT_STRIDE + 1 + kg), "r"((unsigned)(it + 1)) : "memory");
    else red_release(cnt + (CNT_ATT + grp) * CNT_STRIDE);
  }
  K1_MARK(mk, 5);
}

// ------------------------------------------------------------------------------------------------ combine: 128 tasks (4 row groups x 32 column-quad pairs), one round trip after the counter
template <int S, int FL>
__device__ __noinline__ void k1_combine(const float2* stats, const float* opart, bf16* attn, unsigned* cnt, int it, bool backoff, ull* mk) {
  constexpr bool BF = (FL & FL_BF16P) != 0, C1 = (FL & FL_CMB1) != 0;
  constexpr int OPT = BF ? (C1 ? 1 : 2) : 2;           // column units (quads / octets) per task, per row
  constexpr int NTASK = K1_NGRP * (BF ? 32 / OPT : 32);
  const int task = gridDim.x - 1 - blockIdx.x;
  if (task >= NTASK) return;
  const int g = task / (NTASK / K1_NGRP), cp = task % (NTASK / K1_NGRP), tid = threadIdx.x;
  constexpr bool STRM = (FL & FL_STRM) != 0;
  static_assert(!STRM || BF, "streaming combine is bf16-only");
  if constexpr (STRM) {
    // streaming combine: per-split flags, online (max, sum) accumulation in arrival order; one warp-uniform poll round = one ballot over the S flags
    const int warp = tid >> 5, lane = tid & 31;
    if (warp * 32 >= OPT * K1_RPC) return;
    const bool act = tid < OPT * K1_RPC;
    const int rl = tid % K1_RPC, co = cp * OPT + tid / K1_RPC, grow = K1_RPC * g + rl;
    const unsigned* fl = cnt + (CNT_ATT + g) * CNT_STRIDE + 1;
    const unsigned tgt = (unsigned)(it + 1);
    uint32_t pend = (1u << S) - 1u;
    float M = -INFINITY, den = 0.f, num[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) num[e] = 0.f;
    bool first = true;
    while (pend) {
      bool rdy = false;
      if (lane < S) rdy = ((pend >> lane) & 1u) && ld_acquire(fl + lane) >= tgt;
      uint32_t nb = __ballot_sync(0xffffffffu, rdy);
      if (nb == 0) { __nanosleep(16); continue; }
      pend &= ~nb;
      __syncwarp();
      if (act) {
        float2 st[S];
        uint4 o[S];
#pragma unroll
        for (int c = 0; c < S; ++c)
          if ((nb >> c) & 1u) {
            st[c] = __ldcg(stats + (size_t)c * QROWS + grow);
            o[c] = __ldcg(reinterpret_cast<const uint4*>(opart) + ((size_t)(c * 32 + co) * QROWS + grow));
          }
#pragma unroll
        for (int c = 0; c < S; ++c)
          if ((nb >> c) & 1u) {
            const float ms = st[c].x, Mn = fmaxf(M, ms);
            if (Mn == -INFINITY) continue;  // nothing unmasked so far
            const float a = ex2f(M - Mn), w = ex2f(ms - Mn);  // ex2(-inf) = 0 for the empty side
            const uint32_t ow[4] = {o[c].x, o[c].y, o[c].z, o[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              num[2 * e] = fmaf(w, __uint_as_float(ow[e] << 16), num[2 * e] * a);
              num[2 * e + 1] = fmaf(w, __uint_as_float(ow[e] & 0xffff0000u), num[2 * e + 1] * a);
            }
            den = fmaf(w, st[c].y, den * a);
            M = Mn;
          }
      }
      if (first) { K1_MARK(mk, 6); first = false; }
    }
    if (act) {
      const float rd = 1.f / den;
      const uint2 lo = pack_bf16x4(num[0] * rd, num[1] * rd, num[2] * rd, num[3] * rd), hi = pack_bf16x4(num[4] * rd, num[5] * rd, num[6] * rd, num[7] * rd);
      *reinterpret_cast<uint4*>(attn + (size_t)grow * HD + co * 8) = make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
    K1_MARK(mk, 7);
    return;
  }
  block_wait(cnt + (CNT_ATT + g) * CNT_STRIDE, (unsigned)(it + 1) * S, backoff);
  K1_MARK(mk, 6);
  if constexpr (BF) {
    if (tid < OPT * K1_RPC) {
      const int rl = tid % K1_RPC, co = cp * OPT + tid / K1_RPC, grow = K1_RPC * g + rl;
      float2 st[S];
      uint4 o[S];
#pragma unroll
      for (int c = 0; c < S; ++c) {
        st[c] = __ldcg(stats + (size_t)c * QROWS + grow);
        o[c] = __ldcg(reinterpret_cast<const uint4*>(opart) + ((size_t)(c * 32 + co) * QROWS + grow));
      }
      float M = st[0].x;
#pragma unroll
      for (int c = 1; c < S; ++c) M = fmaxf(M, st[c].x);
      float num[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) num[e] = 0.f;
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < S; ++c) {
        const float w = st[c].x == -INFINITY ? 0.f : ex2f(st[c].x - M);
        const uint32_t ow[4] = {o[c].x, o[c].y, o[c].z, o[c].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          num[2 * e] = fmaf(w, __uint_as_float(ow[e] << 16), num[2 * e]);
          num[2 * e + 1] = fmaf(w, __uint_as_float(ow[e] & 0xffff0000u), num[2 * e + 1]);
        }
        den = fmaf(w, st[c].y, den);
      }
      const float rd = 1.f / den;
      const uint2 lo = pack_bf16x4(num[0] * rd, num[1] * rd, num[2] * rd, num[3] * rd), hi = pack_bf16x4(num[4] * rd, num[5] * rd, num[6] * rd, num[7] * rd);
      *reinterpret_cast<uint4*>(attn + (size_t)grow * HD + co * 8) = make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
  } else if (tid < 2 * K1_RPC) {
    const int rl = tid % K1_RPC, cq = cp * 2 + tid / K1_RPC, grow = K1_RPC * g + rl;
    float2 st[S];
    float4 o[S];
#pragma unroll
    for (int c = 0; c < S; ++c) {
      st[c] = __ldcg(stats + (size_t)c * QROWS + grow);
      o[c] = __ldcg(reinterpret_cast<const float4*>(opart) + ((size_t)(c * 64 + cq) * QROWS + grow));
    }
    float M = st[0].x;
#pragma unroll
    for (int c = 1; c < S; ++c) M = fmaxf(M, st[c].x);
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = 0.f;
#pragma unroll
    for (int c = 0; c < S; ++c) {
      const float w = st[c].x == -INFINITY ? 0.f : ex2f(st[c].x - M);
      num.x = fmaf(w, o[c].x, num.x); num.y = fmaf(w, o[c].y, num.y); num.z = fmaf(w, o[c].z, num.z); num.w = fmaf(w, o[c].w, num.w);
      den = fmaf(w, st[c].y, den);
    }
    *reinterpret_cast<uint2*>(attn + (size_t)grow * HD + cq * 4) = pack_bf16x4(num.x / den, num.y / den, num.z / den, num.w / den);
  }
  K1_MARK(mk, 7);
}

// K0 bodies behind a call boundary (arguments by value): keeps the register allocation of the surrounding phases independent of the K0 body and of the stamps
__device__ __noinline__ void k0_B_pre_ni(Params p, AttArgs a, int l, bf16* smb) { K0Code::B_pre(p, a, l, smb); }
__device__ __noinline__ void k0_C_ni(Params p, AttArgs a, int l, int it, bf16* smb, ull* mk) { K0Code::C(p, a, l, it, smb, mk); }
__device__ __noinline__ void k0_C3_ni(Params p, AttArgs a, int l, int it, bf16* smb, ull* mk) { K0Code::C3(p, a, l, it, smb, mk); }

// ------------------------------------------------------------------------------------------------ VT
struct K1 {
  static constexpr const char* kName = "att_k1";
  static constexpr bool kCCBarrier = false;
  static constexpr size_t kSmem = ATT_SMEM_UMMA;
  static constexpr int kCluster = 2;
  __device__ static void B_pre(const Params& p, const AttArgs& a, int l, bf16* smb) {
    if (a.variant == ATTV_K0) { k0_B_pre_ni(p, a, l, smb); return; }
    switch (a.variant) {
#define X(i, n, NCH, CPC, FL) case ATTV_K1 + i: k1_prefill<NCH, CPC, FL>(l, smb); return;
      K1_LIST(X)
#undef X
      default: return;
    }
  }
  __device__ static void C(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    if (a.variant == ATTV_K0) { k0_C_ni(p, a, l, it, smb, mk); return; }
    switch (a.variant) {
#define X(i, n, NCH, CPC, FL) case ATTV_K1 + i: k1_unit<NCH, CPC, FL>(p.prefix_mask, reinterpret_cast<float2*>(p.stats), p.opart, p.cnt, smb, l, it, mk); return;
      K1_LIST(X)
#undef X
      default: return;
    }
  }
  __device__ static void C3(const Params& p, const AttArgs& a, int l, int it, bf16* smb, ull* mk) {
    if (a.variant == ATTV_K0) { k0_C3_ni(p, a, l, it, smb, mk); return; }
    const bool backoff = (p.fuse & FUSE_BACKOFF) != 0;
    switch (a.variant) {
#define X(i, n, NCH, CPC, FL) case ATTV_K1 + i: k1_combine<K1G<NCH>::NCHUNK / CPC, FL>(reinterpret_cast<const float2*>(p.stats), p.opart, p.attn, p.cnt, it, backoff, mk); return;
      K1_LIST(X)
#undef X
      default: return;
    }
  }
  // ---- host
  static void prepare(AttArgs& a, int grid) {
    PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
    CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
    CUtensorMap maps[16]; memset(maps, 0, sizeof maps);
    auto mk2d = [&](CUtensorMap* m, const void* base, uint64_t rows, uint32_t box_rows, const char* nm) {
      cuuint64_t dims[2] = {HD, rows}, strides[1] = {(cuuint64_t)HD * 2};
      cuuint32_t box[2] = {64, box_rows}, es[2] = {1, 1};
      CUresult r = enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map %s encode failed %d\n", nm, (int)r); exit(1); }
    };
    mk2d(&maps[0], a.q, QROWS, K1_RPC, "q");
    for (int k = 0; k < 3; ++k) {
      const int nch = 32 << k;
      mk2d(&maps[1 + k], a.prefix_k, (uint64_t)L * P, nch / 2, "kp");
      mk2d(&maps[4 + k], a.prefix_v, (uint64_t)L * P, nch, "vp");
    }
    mk2d(&maps[7], a.prefix_k, (uint64_t)L * P, 8, "kp8");
    mk2d(&maps[8], a.ksuf, T, 8, "ks8");
    mk2d(&maps[9], a.prefix_v, (uint64_t)L * P, 8, "vp8");
    mk2d(&maps[10], a.vsuf, T, 8, "vs8");
    CK(cudaMemcpyToSymbol(c_tm, maps, sizeof maps));
    unsigned z = 0; CK(cudaMemcpyToSymbol(g_k1_to, &z, 4));
    fprintf(stderr, "k1 prepare: grid=%d K1 variants=%d smem=%zu\n", grid, K1_N, (size_t)kSmem);
  }
  static void post_run(int variant, int idx, int reps, const std::string& out, const std::string& name, double nspc) {
    (void)reps; (void)out; (void)name; (void)nspc;
    unsigned to = 0; CK(cudaMemcpyFromSymbol(&to, g_k1_to, 4));
    if (to) { fprintf(stderr, "WARNING: %u mbarrier wait timeouts (variant %s launch %d)\n", to, att_vname(variant), idx); unsigned z = 0; CK(cudaMemcpyToSymbol(g_k1_to, &z, 4)); }
  }
};

int main(int argc, char** argv) { return att_main<K1>(argc, argv); }
