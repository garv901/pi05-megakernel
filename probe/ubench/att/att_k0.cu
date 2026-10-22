// att_k0.cu: K0 = the megakernel's attention phase verbatim for the standalone harness (att_common.cuh). The pre-B K/V + mask prefill (B_pre), the
// split-key unit loop (C) and the combine (C3) are copies of the code in mk/pi05_mk.cu (default build: PI05_ATT_FA=1 layout, 64-row units x 64 keys =
// 7 x 16 = 112 units on 112 of 148 CTAs, PI05_ATT_PREFILL=2, FUSE_CC3 counter hand-off, k64) with the preprocessor conditionals resolved for those
// defaults and the megakernel's CST(k) debug stamps replaced by ATT_MARK(mk, k) (no-ops unless -DATT_MARKS=1). Only the harness glue differs:
// tid / lane / warp / g / t4 / it1 / backoff are defined at the top of each function as the megakernel's layer loop defines them.
// Variants in one binary (runtime --variants null,k0,calib; ABAB-interleaved launches): null (every barrier kept, C + C3 + prefill removed; the harness'
// L2 prefetches of Wo / prefix K/V / Wg / Wu are kept), k0 (the real thing), calib (C slot spins --ncal cycles).
// Build (login node):
//   nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a att_k0.cu -o att_k0
//   unstamped twin (gate 2): add -DATT_STAMPS=0 -> att_k0_ns       sub-phase timeline binary: add -DATT_MARKS=1 -> att_k0_mk (attribution only)
// Run: att_k0 [--reps 20] [--ncal 35000] [--acyc 5500] [--bcyc 1500] [--tcyc 13700] [--ctx 15] [--cwarps 0] [--smem 117248] [--cluster 1]
//      [--variants null,k0,calib] [--out DIR] [--name TAG]   (FG_UMMA-build context: --smem 197632 --cluster 2)
#include "att_common.cuh"

constexpr int ATT_PS_ELEMS = 0, ATT_WRED_BYTES = 0;  // layout 1 (PI05_ATT_FA = 1) has no P tile / row-reduction scratch

struct K0 {
  static constexpr const char* kName = "att_k0";
  static constexpr bool kCCBarrier = false;
  static constexpr size_t kSmem = ATT_SMEM_DYN;
  static constexpr int kCluster = 0;
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

int main(int argc, char** argv) { return att_main<K0>(argc, argv); }
