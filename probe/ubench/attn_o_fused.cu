// U7: fused attention (phases C + C3) + o_proj (phase D) of the pi0.5 megakernel, P = 128, vs the split production C + D (11.84 us/layer in situ).
// One persistent kernel, 148 CTAs x 256 thr, 1 CTA/SM. Per iteration: unmeasured pre-phase (copy q/k/v, stream cold pool, L2 prefetch, optional PF cp.async)
// -> grid barrier (S0) -> measured phase -> grid barrier (S1).  interval = S1 - S0 (block 0).  Variants:
//   ctl: measured phase empty.  F1: CTA = (head h = b/16, Wo column slice j = b%16 of 64 cols), full 256 dims, FA2 online softmax over KB-key blocks,
//   part1[h][50][1024].  F2: CTA = (chunk c = b/8 = head x 128-dim half, slice j = b%8 of 128 cols), all 178 keys resident in smem, two-pass softmax in
//   registers, part2[c][50][1024] (production phase-D layout).  Each CTA recomputes its head's attention (redundant across CTAs); no communication.
//   F3 (U8, probe/specs/U8_attn_o_rowsplit.md): CTA = (head h = b/16, row block rb = (b/4)%4 of 12/13 tokens, Wo column slice j = b%4 of 256 cols); the 8 warps split the 178 keys
//   (padded 192 = 8 x 24) for S, merge row max / sum through smem (fixed warp order), split the 256 output dims for P.V, then each warp does o_proj over 32 columns with Wo taken
//   straight from global/L2 into registers (gemm_phase load_w layout, ldg.nc; half 0 issued right after the barrier before the Q/K/V wait, half 1 at the start of o_proj, no spills
//   required; see build.txt).  part1[h][50][1024] as F1.  Redundant: each (h, rb) attention is done by 4 slice CTAs, each Wo element is read by 4 row-block CTAs.  Smem 218,624 B.
// Spec: probe/specs/U7_attn_o_fused.md.  Build: nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v attn_o_fused.cu -o attn_o_fused
// Notes / deviations (see report): one templated kernel per variant (ctl, F1<32>, F1<64>, F2) so register counts are per variant; fill/L2P/pre-phase run on all
// CTAs also with --solo 1 (solo only gates PF issue + measured work); F2 keys are padded to 192 (rows 178..191 zero-filled, masked by key index, no Mk array);
// F1 online softmax uses running max m_new for p = bf16(exp(s - m_new)) (production uses chunk-final max), rounding-level difference only;
// the 16 F1/F2 working-CTA stamps are written by thread 0 after a CTA sync (t_q: Q + block-0 / all keys landed; t_att: O in smem; t_o: partial stored).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;
constexpr int NT = 256, T = 50, NQH = 8, HD = 256, P = 128, NK = P + T, W = 1024, QD = 2048, L = 18, NKP = 192, NWORK = 128, NCTA = 148;
constexpr int XS = HD + 8;  // smem row stride (elements) of 256-wide tiles: conflict-free ldmatrix
constexpr size_t POOL = 512ull << 20, L2REG = 16800000;

// ---------------------------------------------------------------- helpers (copied from mk/pi05_mk.cu:224-330)
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* sp, const void* gp, int nb) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(sp);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gp), "r"(nb) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ void grid_barrier(unsigned* cnt, unsigned epoch) {  // PI05_BARRIER 3 path
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory"); (void)old;
    while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
  }
  __syncthreads();
}

struct Args {
  const bf16 *pk, *pv, *qsrc, *ksrc, *vsrc, *wo; bf16 *qbuf, *ksuf, *vsuf; float* part;
  const char *pool, *l2reg; unsigned* cnt; unsigned *iv, *tq, *ta, *to, *sink;
  int nit, it0, pf, solo, l2p; size_t fill;
};

// rows [0,nrows) of a [rows][chunks*8] tile -> smem; row j0+jj comes from `pref` (j < P), `suf` (P <= j < NK) or is zero-filled (j >= NK).
// part 0: only rows independent of phase B (j < P or j >= NK); 1: only suffix rows (P <= j < NK); 2: all.
__device__ __forceinline__ void load_rows(bf16* dst, int dstride, int nrows, int j0, int chunks, int coloff, const bf16* pref, const bf16* suf, int part) {
  for (int i = threadIdx.x; i < nrows * chunks; i += NT) {
    const int jj = i / chunks, pc = i % chunks, j = j0 + jj;
    const bool dep = j >= P && j < NK;
    if ((part == 0 && dep) || (part == 1 && !dep)) continue;
    const bf16* s = j < P ? pref + (size_t)j * HD : (j < NK ? suf + (size_t)(j - P) * HD : pref);
    cp_async16(dst + jj * dstride + pc * 8, s + coloff + pc * 8, j < NK ? 16 : 0);
  }
}
__device__ __forceinline__ void load_wo(bf16* dst, int dstride, int rows, int chunks, const bf16* src) {  // src: first element of the slice, row stride QD
  for (int i = threadIdx.x; i < rows * chunks; i += NT) { const int r = i / chunks, c = i % chunks; cp_async16(dst + r * dstride + c * 8, src + (size_t)r * QD + c * 8, 16); }
}
__device__ __forceinline__ void load_q(bf16* dst, const bf16* qh) {  // 64 rows, rows >= T zero-filled
  for (int i = threadIdx.x; i < 64 * 32; i += NT) { const int r = i >> 5, c = i & 31; cp_async16(dst + r * XS + c * 8, qh + (size_t)min(r, T - 1) * HD + c * 8, r < T ? 16 : 0); }
}

// S = Q K^T for the m-tile (16 rows), NTB key n-tiles; Ks = first key row of the block (stride XS).
template <int NTB> __device__ __forceinline__ void s_block(float (*sc)[4], const bf16* Qs, const bf16* Ks, int mt, int lane) {
#pragma unroll
  for (int nt = 0; nt < NTB; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) sc[nt][e] = 0.f;
  const bf16* qa = Qs + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * XS + (lane >> 4) * 8;
  const bf16* kb = Ks + ((lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 8;
#pragma unroll 4
  for (int kk = 0; kk < HD / 16; ++kk) {
    uint32_t a0, a1, a2, a3;
    ldmatrix_x4(a0, a1, a2, a3, qa + kk * 16);
#pragma unroll
    for (int np = 0; np < NTB / 2; ++np) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, kb + np * 16 * XS + kk * 16);
      mma16816(sc[2 * np], a0, a1, a2, a3, b0, b1);
      mma16816(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
    }
  }
}
// o_proj: Os [64][KD+8] bf16 x Wo smem [n][KD+8] -> fp32 rows < T at outbase[row][col], warp covers NCW columns dg*NCW...
template <int KD, int NCW> __device__ __forceinline__ void oproj(const bf16* Os, const bf16* Wo, float* outbase, int mt, int dg, int lane) {
  constexpr int ST = KD + 8;
  float c[NCW / 8][4];
#pragma unroll
  for (int nt = 0; nt < NCW / 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) c[nt][e] = 0.f;
  const bf16* qa = Os + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * ST + (lane >> 4) * 8;
  const bf16* kb = Wo + (dg * NCW + (lane >> 4) * 8 + (lane & 7)) * ST + ((lane >> 3) & 1) * 8;
#pragma unroll 4
  for (int kk = 0; kk < KD / 16; ++kk) {
    uint32_t a0, a1, a2, a3;
    ldmatrix_x4(a0, a1, a2, a3, qa + kk * 16);
#pragma unroll
    for (int np = 0; np < NCW / 16; ++np) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, kb + np * 16 * ST + kk * 16);
      mma16816(c[2 * np], a0, a1, a2, a3, b0, b1);
      mma16816(c[2 * np + 1], a0, a1, a2, a3, b2, b3);
    }
  }
  const int g = lane >> 2, t4 = lane & 3, ra = mt * 16 + g, rb = ra + 8;
#pragma unroll
  for (int nt = 0; nt < NCW / 8; ++nt) {
    const int col = dg * NCW + nt * 8 + 2 * t4;
    if (ra < T) *reinterpret_cast<float2*>(outbase + (size_t)ra * W + col) = make_float2(c[nt][0], c[nt][1]);
    if (rb < T) *reinterpret_cast<float2*>(outbase + (size_t)rb * W + col) = make_float2(c[nt][2], c[nt][3]);
  }
}
__device__ __forceinline__ void st_bf2(bf16* p, float a, float b) { *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b); }

// ---------------------------------------------------------------- F1 smem: Wo[64][XS] | Q/O[64][XS] | K[2][KB][XS] | V[2][KB][XS]
template <int KB> constexpr size_t smem_f1() { return (size_t)(64 + 64 + 4 * KB) * XS * 2; }
// F2 smem: Wo[128][136] | Q/O[64][XS] | K[192][XS] | V[192][136]
constexpr size_t smem_f2() { return (size_t)(128 * 136 + 64 * XS + NKP * XS + NKP * 136) * 2; }

template <int KB> __device__ __forceinline__ void f1_issue_pre(const Args& a, int l, bf16* sm) {  // PF: Wo slice + first prefix key block
  const int h = blockIdx.x / 16, j = blockIdx.x % 16;
  bf16 *Wo = sm, *Ks = sm + 128 * XS, *Vs = Ks + 2 * KB * XS;
  load_wo(Wo, XS, 64, 32, a.wo + (size_t)l * W * QD + (size_t)(64 * j) * QD + 256 * h);
  load_rows(Ks, XS, KB, 0, 32, 0, a.pk + (size_t)l * P * HD, a.ksuf, 0);
  load_rows(Vs, XS, KB, 0, 32, 0, a.pv + (size_t)l * P * HD, a.vsuf, 0);
  cp_commit();
}
template <int KB> __device__ __forceinline__ void f1_work(const Args& a, int l, bf16* sm, unsigned long long t0, unsigned* tq, unsigned* ta) {
  constexpr int NTB = KB / 8, NBLK = NKP / KB;
  const int h = blockIdx.x / 16, j = blockIdx.x % 16, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int mt = warp & 3, dh = warp >> 2;
  bf16 *Wo = sm, *Qs = sm + 64 * XS, *Ks = sm + 128 * XS, *Vs = Ks + 2 * KB * XS;
  const bf16* pk = a.pk + (size_t)l * P * HD; const bf16* pv = a.pv + (size_t)l * P * HD;
  load_q(Qs, a.qbuf + (size_t)h * T * HD);
  if (!a.pf) load_wo(Wo, XS, 64, 32, a.wo + (size_t)l * W * QD + (size_t)(64 * j) * QD + 256 * h);
  load_rows(Ks, XS, KB, 0, 32, 0, pk, a.ksuf, a.pf ? 1 : 2);
  load_rows(Vs, XS, KB, 0, 32, 0, pv, a.vsuf, a.pf ? 1 : 2);
  cp_commit();
  float acc[16][4], m[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
  for (int blk = 0; blk < NBLK; ++blk) {
    const int cur = blk & 1;
    if (blk + 1 < NBLK) {
      load_rows(Ks + (cur ^ 1) * KB * XS, XS, KB, (blk + 1) * KB, 32, 0, pk, a.ksuf, 2);
      load_rows(Vs + (cur ^ 1) * KB * XS, XS, KB, (blk + 1) * KB, 32, 0, pv, a.vsuf, 2);
    }
    cp_commit();
    cp_wait<1>();
    __syncthreads();
    if (blk == 0 && tid == 0) tq[blockIdx.x] = (unsigned)(gt() - t0);
    const bf16 *Kb = Ks + cur * KB * XS, *Vb = Vs + cur * KB * XS;
    float sc[NTB][4];
    s_block<NTB>(sc, Qs, Kb, mt, lane);
    float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < NTB; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int cc = blk * KB + nt * 8 + 2 * t4 + (e & 1);
        const float v = cc < NK ? rbf(sc[nt][e]) * 0.0625f : -INFINITY;
        sc[nt][e] = v;
        rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
      }
    float scl[2];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 1));
      rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 2));
      const float mn = fmaxf(m[hf], rmax[hf]);  // finite: every block holds a valid key
      scl[hf] = expf(m[hf] - mn); m[hf] = mn; ls[hf] *= scl[hf];
    }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) { acc[nt][0] *= scl[0]; acc[nt][1] *= scl[0]; acc[nt][2] *= scl[1]; acc[nt][3] *= scl[1]; }
    uint32_t pa[NTB][2];
#pragma unroll
    for (int nt = 0; nt < NTB; ++nt) {
      float pr[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) pr[e] = rbf(expf(sc[nt][e] - m[e >> 1]));
      ls[0] += pr[0]; ls[0] += pr[1]; ls[1] += pr[2]; ls[1] += pr[3];
      const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]), hi = __floats2bfloat162_rn(pr[2], pr[3]);
      pa[nt][0] = *reinterpret_cast<const uint32_t*>(&lo); pa[nt][1] = *reinterpret_cast<const uint32_t*>(&hi);
    }
#pragma unroll
    for (int ks = 0; ks < KB / 16; ++ks) {
      const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4_trans(b0, b1, b2, b3, Vb + krow * XS + dh * 128 + np * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * np], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b0, b1);
        mma16816(acc[2 * np + 1], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b2, b3);
      }
    }
    __syncthreads();  // buffer `cur` consumed before block blk+2 is issued into it; at the end Q is dead
  }
  // O /= l, bf16, into the Q buffer
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) { ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 1); ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 2); ls[hf] = 1.f / ls[hf]; }
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int c = dh * 128 + nt * 8 + 2 * t4;
    st_bf2(Qs + (mt * 16 + g) * XS + c, acc[nt][0] * ls[0], acc[nt][1] * ls[0]);
    st_bf2(Qs + (mt * 16 + g + 8) * XS + c, acc[nt][2] * ls[1], acc[nt][3] * ls[1]);
  }
  __syncthreads();
  if (tid == 0) ta[blockIdx.x] = (unsigned)(gt() - t0);
  oproj<256, 32>(Qs, Wo, a.part + ((size_t)h * T) * W + 64 * j, mt, dh, lane);
}

constexpr int VS2 = 136;
__device__ __forceinline__ void f2_issue_pre(const Args& a, int l, bf16* sm) {  // PF: Wo slice + all prefix K/V rows (+ zero pad rows)
  const int c = blockIdx.x / 8, j = blockIdx.x % 8, dh = c & 1;
  bf16 *Wo = sm, *Ks = sm + 128 * VS2 + 64 * XS, *Vs = Ks + NKP * XS;
  load_wo(Wo, VS2, 128, 16, a.wo + (size_t)l * W * QD + (size_t)(128 * j) * QD + 128 * c);
  load_rows(Ks, XS, NKP, 0, 32, 0, a.pk + (size_t)l * P * HD, a.ksuf, 0);
  load_rows(Vs, VS2, NKP, 0, 16, 128 * dh, a.pv + (size_t)l * P * HD, a.vsuf, 0);
  cp_commit();
}
__device__ __forceinline__ void f2_work(const Args& a, int l, bf16* sm, unsigned long long t0, unsigned* tq, unsigned* ta) {
  constexpr int NTB = NKP / 8;
  const int c = blockIdx.x / 8, j = blockIdx.x % 8, h = c >> 1, dh = c & 1, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int mt = warp & 3, dg = warp >> 2;
  bf16 *Wo = sm, *Qs = sm + 128 * VS2, *Ks = Qs + 64 * XS, *Vs = Ks + NKP * XS;
  const int part = a.pf ? 1 : 2;
  load_q(Qs, a.qbuf + (size_t)h * T * HD);
  if (!a.pf) load_wo(Wo, VS2, 128, 16, a.wo + (size_t)l * W * QD + (size_t)(128 * j) * QD + 128 * c);
  load_rows(Ks, XS, NKP, 0, 32, 0, a.pk + (size_t)l * P * HD, a.ksuf, part);
  load_rows(Vs, VS2, NKP, 0, 16, 128 * dh, a.pv + (size_t)l * P * HD, a.vsuf, part);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  if (tid == 0) tq[blockIdx.x] = (unsigned)(gt() - t0);
  float sc[NTB][4];
  s_block<NTB>(sc, Qs, Ks, mt, lane);
  float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int nt = 0; nt < NTB; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v = nt * 8 + 2 * t4 + (e & 1) < NK ? rbf(sc[nt][e]) * 0.0625f : -INFINITY;
      sc[nt][e] = v;
      rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
    }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 1));
    rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 2));
  }
  uint32_t pa[NTB][2];
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int nt = 0; nt < NTB; ++nt) {
    float pr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) pr[e] = rbf(expf(sc[nt][e] - rmax[e >> 1]));
    ls[0] += pr[0]; ls[0] += pr[1]; ls[1] += pr[2]; ls[1] += pr[3];
    const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]), hi = __floats2bfloat162_rn(pr[2], pr[3]);
    pa[nt][0] = *reinterpret_cast<const uint32_t*>(&lo); pa[nt][1] = *reinterpret_cast<const uint32_t*>(&hi);
  }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) { ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 1); ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 2); ls[hf] = 1.f / ls[hf]; }
  float acc[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < NKP / 16; ++ks) {
    const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4_trans(b0, b1, b2, b3, Vs + krow * VS2 + dg * 64 + np * 16 + (lane >> 4) * 8);
      mma16816(acc[2 * np], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b0, b1);
      mma16816(acc[2 * np + 1], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b2, b3);
    }
  }
  __syncthreads();  // all warps done reading Q before it is overwritten by O
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int cc = dg * 64 + nt * 8 + 2 * t4;
    st_bf2(Qs + (mt * 16 + g) * VS2 + cc, acc[nt][0] * ls[0], acc[nt][1] * ls[0]);
    st_bf2(Qs + (mt * 16 + g + 8) * VS2 + cc, acc[nt][2] * ls[1], acc[nt][3] * ls[1]);
  }
  __syncthreads();
  if (tid == 0) ta[blockIdx.x] = (unsigned)(gt() - t0);
  oproj<128, 64>(Qs, Wo, a.part + ((size_t)c * T) * W + 128 * j, mt, dg, lane);
}

// ---------------------------------------------------------------- F3 smem: K[192][XS] | V[192][XS] | Q/O[16][XS] | P[16][200] | stats fp32 [2][8][16] (max, sum)
constexpr int PS3 = NKP + 8;
constexpr size_t smem_f3() { return (size_t)(2 * NKP * XS + 16 * XS + 16 * PS3) * 2 + 2 * 8 * 16 * 4; }
static_assert(smem_f3() <= 232448, "F3 smem");
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void f3_issue_pre(const Args& a, int l, bf16* sm) {  // PF: prefix K/V rows + zero pad rows
  bf16 *Ks = sm, *Vs = sm + NKP * XS;
  load_rows(Ks, XS, NKP, 0, 32, 0, a.pk + (size_t)l * P * HD, a.ksuf, 0);
  load_rows(Vs, XS, NKP, 0, 32, 0, a.pv + (size_t)l * P * HD, a.vsuf, 0);
  cp_commit();
}
__device__ __forceinline__ void f3_work(const Args& a, int l, bf16* sm, unsigned long long t0, unsigned* tq, unsigned* ta) {
  const int h = blockIdx.x / 16, rb = (blockIdx.x / 4) % 4, j = blockIdx.x % 4, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int tb = rb * T / 4, ntok = (rb + 1) * T / 4 - tb;
  bf16 *Ks = sm, *Vs = sm + NKP * XS, *Qs = Vs + NKP * XS, *Ps = Qs + 16 * XS;
  float* stm = reinterpret_cast<float*>(Ps + 16 * PS3); float* sts = stm + 8 * 16;
  // Wo registers (gemm_phase load_w layout): rows n = 256j + 32w + 8nt + g, k = 256h + 128hf + 32sl + 8t4 .. +8
  const bf16* wb = a.wo + (size_t)l * W * QD + (size_t)(256 * j + 32 * w + g) * QD + 256 * h + 8 * t4;
  uint4 bw0[4][4], bw1[4][4];
#pragma unroll
  for (int sl = 0; sl < 4; ++sl)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) bw0[sl][nt] = ldg_nc(reinterpret_cast<const uint4*>(wb + (size_t)(nt * 8) * QD + 32 * sl));
  for (int i = tid; i < 16 * 32; i += NT) { const int r = i >> 5, c = i & 31; cp_async16(Qs + r * XS + c * 8, a.qbuf + ((size_t)h * T + tb + min(r, ntok - 1)) * HD + c * 8, r < ntok ? 16 : 0); }
  const int part = a.pf ? 1 : 2;
  load_rows(Ks, XS, NKP, 0, 32, 0, a.pk + (size_t)l * P * HD, a.ksuf, part);
  load_rows(Vs, XS, NKP, 0, 32, 0, a.pv + (size_t)l * P * HD, a.vsuf, part);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  if (tid == 0) tq[blockIdx.x] = (unsigned)(gt() - t0);
  // S = Q K^T over this warp's 24 keys (3 n-tiles)
  float sc[3][4];
#pragma unroll
  for (int nt = 0; nt < 3; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) sc[nt][e] = 0.f;
  {
    const bf16* qa = Qs + ((lane & 7) + ((lane >> 3) & 1) * 8) * XS + (lane >> 4) * 8;
    const bf16* kb = Ks + (24 * w + (lane >> 4) * 8 + (lane & 7)) * XS + ((lane >> 3) & 1) * 8;
    const bf16* k2 = Ks + (24 * w + 16 + (lane & 7)) * XS + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int kk = 0; kk < HD / 16; ++kk) {
      uint32_t a0, a1, a2, a3, b0, b1, b2, b3, c0, c1;
      ldmatrix_x4(a0, a1, a2, a3, qa + kk * 16);
      ldmatrix_x4(b0, b1, b2, b3, kb + kk * 16);
      ldmatrix_x2(c0, c1, k2 + kk * 16);
      mma16816(sc[0], a0, a1, a2, a3, b0, b1);
      mma16816(sc[1], a0, a1, a2, a3, b2, b3);
      mma16816(sc[2], a0, a1, a2, a3, c0, c1);
    }
  }
  float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int nt = 0; nt < 3; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v = 24 * w + nt * 8 + 2 * t4 + (e & 1) < NK ? rbf(sc[nt][e]) * 0.0625f : -INFINITY;
      sc[nt][e] = v;
      rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
    }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 1));
    rmax[hf] = fmaxf(rmax[hf], __shfl_xor_sync(0xffffffffu, rmax[hf], 2));
    if (t4 == 0) stm[w * 16 + g + 8 * hf] = rmax[hf];
  }
  __syncthreads();  // all maxima visible; Q dead
  float M[2];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) { M[hf] = stm[g + 8 * hf]; for (int x = 1; x < 8; ++x) M[hf] = fmaxf(M[hf], stm[x * 16 + g + 8 * hf]); }
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int nt = 0; nt < 3; ++nt) {
    float pr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) pr[e] = rbf(expf(sc[nt][e] - M[e >> 1]));
    ls[0] += pr[0]; ls[0] += pr[1]; ls[1] += pr[2]; ls[1] += pr[3];
    st_bf2(Ps + g * PS3 + 24 * w + nt * 8 + 2 * t4, pr[0], pr[1]);
    st_bf2(Ps + (g + 8) * PS3 + 24 * w + nt * 8 + 2 * t4, pr[2], pr[3]);
  }
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 1); ls[hf] += __shfl_xor_sync(0xffffffffu, ls[hf], 2);
    if (t4 == 0) sts[w * 16 + g + 8 * hf] = ls[hf];
  }
  __syncthreads();  // P and sums visible
  float inv[2];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) { float s = 0.f; for (int x = 0; x < 8; ++x) s += sts[x * 16 + g + 8 * hf]; inv[hf] = 1.f / s; }
  // O[16][32w .. +32) = P V
  float acc[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
  {
    const bf16* pa = Ps + ((lane & 7) + ((lane >> 3) & 1) * 8) * PS3 + (lane >> 4) * 8;
#pragma unroll
    for (int ks = 0; ks < NKP / 16; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_x4(a0, a1, a2, a3, pa + ks * 16);
      const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4_trans(b0, b1, b2, b3, Vs + krow * XS + 32 * w + np * 16 + (lane >> 4) * 8);
        mma16816(acc[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(acc[2 * np + 1], a0, a1, a2, a3, b2, b3);
      }
    }
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int cc = 32 * w + nt * 8 + 2 * t4;
    st_bf2(Qs + g * XS + cc, acc[nt][0] * inv[0], acc[nt][1] * inv[0]);
    st_bf2(Qs + (g + 8) * XS + cc, acc[nt][2] * inv[1], acc[nt][3] * inv[1]);
  }
  __syncthreads();
  if (tid == 0) ta[blockIdx.x] = (unsigned)(gt() - t0);
  // o_proj: second Wo half loads run under the first half's mma
#pragma unroll
  for (int sl = 0; sl < 4; ++sl)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) bw1[sl][nt] = ldg_nc(reinterpret_cast<const uint4*>(wb + (size_t)(nt * 8) * QD + 128 + 32 * sl));
  float c[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) c[nt][e] = 0.f;
  auto half = [&](const uint4 (&b)[4][4], int hf) {
#pragma unroll
    for (int sl = 0; sl < 4; ++sl) {
      const uint4 x0 = *reinterpret_cast<const uint4*>(Qs + g * XS + hf * 128 + sl * 32 + t4 * 8);
      const uint4 x1 = *reinterpret_cast<const uint4*>(Qs + (g + 8) * XS + hf * 128 + sl * 32 + t4 * 8);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(c[nt], x0.x, x1.x, x0.y, x1.y, b[sl][nt].x, b[sl][nt].y);
        mma16816(c[nt], x0.z, x1.z, x0.w, x1.w, b[sl][nt].z, b[sl][nt].w);
      }
    }
  };
  half(bw0, 0);
  half(bw1, 1);
  float* ob = a.part + ((size_t)h * T + tb) * W + 256 * j + 32 * w;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = nt * 8 + 2 * t4;
    if (g < ntok) *reinterpret_cast<float2*>(ob + (size_t)g * W + col) = make_float2(c[nt][0], c[nt][1]);
    if (g + 8 < ntok) *reinterpret_cast<float2*>(ob + (size_t)(g + 8) * W + col) = make_float2(c[nt][2], c[nt][3]);
  }
}

// V: 0 ctl, 1 F1<KB>, 2 F2, 3 F3
template <int V, int KB> __global__ void __launch_bounds__(NT, 1) k_main(Args a) {
  extern __shared__ __align__(16) char smraw[];
  bf16* sm = reinterpret_cast<bf16*>(smraw);
  const int tid = threadIdx.x;
  const bool work = V != 0 && blockIdx.x < NWORK && (!a.solo || blockIdx.x == 0);
  uint32_t acc = 0;
  const size_t wcta = (a.fill / NCTA / 16) * 16;
  for (int it = 0; it < a.nit; ++it) {
    const int git = a.it0 + it, l = git % L;
    // ---- pre-phase (not measured)
    {
      const size_t gtid = (size_t)blockIdx.x * NT + tid, nth = (size_t)gridDim.x * NT;
      const uint4* qs = reinterpret_cast<const uint4*>(a.qsrc + (size_t)l * NQH * T * HD); uint4* qd = reinterpret_cast<uint4*>(a.qbuf);
      for (size_t i = gtid; i < (size_t)NQH * T * HD / 8; i += nth) qd[i] = qs[i];
      const uint4* ks = reinterpret_cast<const uint4*>(a.ksrc + (size_t)l * T * HD); uint4* kd = reinterpret_cast<uint4*>(a.ksuf);
      const uint4* vs = reinterpret_cast<const uint4*>(a.vsrc + (size_t)l * T * HD); uint4* vd = reinterpret_cast<uint4*>(a.vsuf);
      for (size_t i = gtid; i < (size_t)T * HD / 8; i += nth) { kd[i] = ks[i]; vd[i] = vs[i]; }
      if (wcta) {
        const char* base = a.pool + ((size_t)git * a.fill) % (POOL - a.fill);
        const uint4* q = reinterpret_cast<const uint4*>(base + (size_t)blockIdx.x * wcta);
        const size_t vec = wcta / 16;
        for (size_t i = tid; i < vec; i += NT * 8) {
          uint4 v[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) { size_t idx = i + (size_t)u * NT; v[u] = idx < vec ? ldg_nc(q + idx) : make_uint4(0, 0, 0, 0); }
#pragma unroll
          for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].w;
        }
      }
      const char* pw = reinterpret_cast<const char*>(a.wo + (size_t)l * W * QD);
      for (size_t i = gtid; i < (size_t)W * QD * 2 / 128; i += nth) asm volatile("prefetch.global.L2 [%0];" ::"l"(pw + i * 128));
      const char* pkk = reinterpret_cast<const char*>(a.pk + (size_t)l * P * HD); const char* pvv = reinterpret_cast<const char*>(a.pv + (size_t)l * P * HD);
      for (size_t i = gtid; i < (size_t)P * HD * 2 / 128; i += nth) { asm volatile("prefetch.global.L2 [%0];" ::"l"(pkk + i * 128)); asm volatile("prefetch.global.L2 [%0];" ::"l"(pvv + i * 128)); }
      if (work && a.pf) { if constexpr (V == 1) f1_issue_pre<KB>(a, l, sm); else if constexpr (V == 2) f2_issue_pre(a, l, sm); else if constexpr (V == 3) f3_issue_pre(a, l, sm); }
    }
    grid_barrier(a.cnt, 2 * it + 1);
    unsigned long long t0 = 0;
    if (tid == 0) t0 = gt();
    // ---- measured phase
    if (a.l2p) {
      const size_t gtid = (size_t)blockIdx.x * NT + tid, nth = (size_t)gridDim.x * NT;
      for (size_t i = gtid; i < L2REG / 128; i += nth) asm volatile("prefetch.global.L2 [%0];" ::"l"(a.l2reg + i * 128));
    }
    if (work) {
      if constexpr (V == 1) f1_work<KB>(a, l, sm, t0, a.tq + (size_t)it * NWORK, a.ta + (size_t)it * NWORK);
      else if constexpr (V == 2) f2_work(a, l, sm, t0, a.tq + (size_t)it * NWORK, a.ta + (size_t)it * NWORK);
      else if constexpr (V == 3) f3_work(a, l, sm, t0, a.tq + (size_t)it * NWORK, a.ta + (size_t)it * NWORK);
      __syncthreads();
      if (tid == 0) a.to[(size_t)it * NWORK + blockIdx.x] = (unsigned)(gt() - t0);
    }
    grid_barrier(a.cnt, 2 * it + 2);
    if (tid == 0 && blockIdx.x == 0) a.iv[it] = (unsigned)(gt() - t0);
  }
  if (acc == 0x9e3779b9u) a.sink[0] = acc;
}

// ---------------------------------------------------------------- host
struct Opts { int variant = 0, kb = 32, pf = 0, solo = 0, l2p = 1, check = 0; double fill = 34; };
static bf16 hb(float x) { return __float2bfloat16(x); }
static float bf(bf16 x) { return __bfloat162float(x); }
static uint64_t rs = 0x1234abcdULL;
static double urand() { rs += 0x9e3779b97f4a7c15ULL; uint64_t z = rs; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31; return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static void fill_normal(std::vector<bf16>& v, double sd) { for (size_t i = 0; i < v.size(); i += 2) { double r = sqrt(-2 * log(urand())) * sd, th = 6.283185307179586 * urand(); v[i] = hb((float)(r * cos(th))); if (i + 1 < v.size()) v[i + 1] = hb((float)(r * sin(th))); } }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }

struct Data {
  std::vector<bf16> pk, pv, qsrc, ksrc, vsrc, wo;
  bf16 *dpk, *dpv, *dq, *dk, *dv, *dwo, *dqbuf, *dksuf, *dvsuf; float* dpart; char *pool, *l2reg; unsigned *cnt, *iv, *tq, *ta, *to, *sink;
};
static Data D;
static void init_data(int nit) {
  D.pk.resize((size_t)L * P * HD); D.pv.resize(D.pk.size()); D.qsrc.resize((size_t)L * NQH * T * HD); D.ksrc.resize((size_t)L * T * HD); D.vsrc.resize(D.ksrc.size()); D.wo.resize((size_t)L * W * QD);
  fill_normal(D.pk, 1); fill_normal(D.pv, 1); fill_normal(D.qsrc, 1); fill_normal(D.ksrc, 1); fill_normal(D.vsrc, 1); fill_normal(D.wo, 1.0 / sqrt(2048.0));
  auto up = [](bf16** d, const std::vector<bf16>& h) { CK(cudaMalloc(d, h.size() * 2)); CK(cudaMemcpy(*d, h.data(), h.size() * 2, cudaMemcpyHostToDevice)); };
  up(&D.dpk, D.pk); up(&D.dpv, D.pv); up(&D.dq, D.qsrc); up(&D.dk, D.ksrc); up(&D.dv, D.vsrc); up(&D.dwo, D.wo);
  CK(cudaMalloc(&D.dqbuf, (size_t)NQH * T * HD * 2)); CK(cudaMalloc(&D.dksuf, (size_t)T * HD * 2)); CK(cudaMalloc(&D.dvsuf, (size_t)T * HD * 2));
  CK(cudaMalloc(&D.dpart, (size_t)16 * T * W * 4)); CK(cudaMalloc(&D.pool, POOL)); CK(cudaMemset(D.pool, 1, POOL)); CK(cudaMalloc(&D.l2reg, L2REG)); CK(cudaMemset(D.l2reg, 1, L2REG));
  CK(cudaMalloc(&D.cnt, 256)); CK(cudaMalloc(&D.iv, nit * 4)); CK(cudaMalloc(&D.tq, (size_t)nit * NWORK * 4)); CK(cudaMalloc(&D.ta, (size_t)nit * NWORK * 4));
  CK(cudaMalloc(&D.to, (size_t)nit * NWORK * 4)); CK(cudaMalloc(&D.sink, 4096));
  CK(cudaMemset(D.tq, 0, (size_t)nit * NWORK * 4)); CK(cudaMemset(D.ta, 0, (size_t)nit * NWORK * 4)); CK(cudaMemset(D.to, 0, (size_t)nit * NWORK * 4));
}
static Args mkargs(const Opts& o, int nit, int it0) {
  Args a; a.pk = D.dpk; a.pv = D.dpv; a.qsrc = D.dq; a.ksrc = D.dk; a.vsrc = D.dv; a.wo = D.dwo; a.qbuf = D.dqbuf; a.ksuf = D.dksuf; a.vsuf = D.dvsuf; a.part = D.dpart;
  a.pool = D.pool; a.l2reg = D.l2reg; a.cnt = D.cnt; a.iv = D.iv; a.tq = D.tq; a.ta = D.ta; a.to = D.to; a.sink = D.sink;
  a.nit = nit; a.it0 = it0; a.pf = o.pf; a.solo = o.solo; a.l2p = o.l2p; a.fill = (size_t)(o.fill * 1e6); return a;
}
// host fp64 reference of layer l: out[t][n]
static void reference(int l, std::vector<double>& out) {
  std::vector<double> att((size_t)T * QD);
  const bf16* pk = &D.pk[(size_t)l * P * HD]; const bf16* pv = &D.pv[(size_t)l * P * HD]; const bf16* ks = &D.ksrc[(size_t)l * T * HD]; const bf16* vs = &D.vsrc[(size_t)l * T * HD];
  for (int h = 0; h < NQH; ++h)
    for (int t = 0; t < T; ++t) {
      const bf16* q = &D.qsrc[(((size_t)l * NQH + h) * T + t) * HD];
      double s[NK], mx = -1e300;
      for (int j = 0; j < NK; ++j) { const bf16* k = j < P ? pk + (size_t)j * HD : ks + (size_t)(j - P) * HD; double d = 0; for (int e = 0; e < HD; ++e) d += (double)bf(q[e]) * bf(k[e]); s[j] = d / 16; mx = std::max(mx, s[j]); }
      double sum = 0; for (int j = 0; j < NK; ++j) { s[j] = exp(s[j] - mx); sum += s[j]; }
      for (int e = 0; e < HD; ++e) { double o = 0; for (int j = 0; j < NK; ++j) o += s[j] * bf(j < P ? pv[(size_t)j * HD + e] : vs[(size_t)(j - P) * HD + e]); att[(size_t)t * QD + h * HD + e] = bf(hb((float)(o / sum))); }
    }
  out.assign((size_t)T * W, 0);
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < W; ++n) { const bf16* w = &D.wo[((size_t)l * W + n) * QD]; double d = 0; for (int k = 0; k < QD; ++k) d += att[(size_t)t * QD + k] * bf(w[k]); out[(size_t)t * W + n] = d; }
}

template <int V, int KB> static size_t smem_of() { return V == 1 ? smem_f1<KB>() : V == 2 ? smem_f2() : V == 3 ? smem_f3() : 0; }
template <int V, int KB> static int go(const Opts& o) {
  constexpr int NIT = 200, NL = 5, DISC = 20;
  auto kfn = &k_main<V, KB>;
  const size_t smem = smem_of<V, KB>();
  CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kfn));
  const char* vn = V == 0 ? "ctl" : V == 1 ? "F1" : V == 2 ? "F2" : "F3";
  if (o.check) {
    const int nsl = V == 1 || V == 3 ? 8 : 16;
    int bad = 0;
    for (int l : {0, 17}) {
      Args a = mkargs(o, 1, l);
      CK(cudaMemset(D.cnt, 0, 256)); CK(cudaMemset(D.dpart, 0, (size_t)16 * T * W * 4));
      kfn<<<NCTA, NT, smem>>>(a); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<float> part((size_t)nsl * T * W); CK(cudaMemcpy(part.data(), D.dpart, part.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<double> ref; reference(l, ref);
      double e2 = 0, r2 = 0, mx = 0; bool nf = false;
      for (int i = 0; i < T * W; ++i) {
        double v = 0; for (int s = 0; s < nsl; ++s) v += part[(size_t)s * T * W + i];
        if (!std::isfinite(v)) nf = true;
        e2 += (v - ref[i]) * (v - ref[i]); r2 += ref[i] * ref[i]; mx = std::max(mx, fabs(v - ref[i]));
      }
      const double rel = sqrt(e2 / r2); const bool fail = nf || !(rel <= 2e-2); bad |= fail;
      printf("check variant=%s kb=%d pf=%d layer=%d rel_l2=%.3e max_abs=%.3e nonfinite=%d %s\n", vn, KB, o.pf, l, rel, mx, (int)nf, fail ? "FAIL" : "PASS");
    }
    return bad;
  }
  std::vector<double> iv, wm, tqm, tam, tom;
  for (int r = 0; r < NL; ++r) {
    Args a = mkargs(o, NIT, 0);
    CK(cudaMemset(D.cnt, 0, 256));
    kfn<<<NCTA, NT, smem>>>(a); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<unsigned> hiv(NIT), tq((size_t)NIT * NWORK), ta(tq.size()), to(tq.size());
    CK(cudaMemcpy(hiv.data(), D.iv, 4 * NIT, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(tq.data(), D.tq, tq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ta.data(), D.ta, ta.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(to.data(), D.to, to.size() * 4, cudaMemcpyDeviceToHost));
    const int nw = V == 0 ? 0 : (o.solo ? 1 : NWORK);
    for (int it = DISC; it < NIT; ++it) {
      iv.push_back(hiv[it] / 1e3);  // block 0: S1 - S0 (ns)
      unsigned mq = 0, ma = 0, mo = 0;
      for (int b = 0; b < nw; ++b) { mq = std::max(mq, tq[(size_t)it * NWORK + b]); ma = std::max(ma, ta[(size_t)it * NWORK + b]); mo = std::max(mo, to[(size_t)it * NWORK + b]); }
      tqm.push_back(mq / 1e3); tam.push_back(ma / 1e3); tom.push_back(mo / 1e3);
    }
  }
  char buf[512];
  snprintf(buf, sizeof buf, "variant=%s kb=%d pf=%d solo=%d l2p=%d fill=%g interval_med=%.3f interval_p10=%.3f interval_p90=%.3f work_max_med=%.3f t_q_med=%.3f t_att_med=%.3f t_o_med=%.3f smem=%zu regs=%d local=%zu",
           vn, V == 1 ? KB : 0, o.pf, o.solo, o.l2p, o.fill, median(iv), pct(iv, 0.1), pct(iv, 0.9), median(tom), median(tqm), median(tam), median(tom), smem, fa.numRegs, fa.localSizeBytes);
  puts(buf); fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--variant")) { ++i; o.variant = !strcmp(argv[i], "ctl") ? 0 : !strcmp(argv[i], "F1") ? 1 : !strcmp(argv[i], "F2") ? 2 : !strcmp(argv[i], "F3") ? 3 : -1; }
    else if (is("--kb")) o.kb = atoi(argv[++i]); else if (is("--pf")) o.pf = atoi(argv[++i]); else if (is("--solo")) o.solo = atoi(argv[++i]);
    else if (is("--l2p")) o.l2p = atoi(argv[++i]); else if (is("--fill")) o.fill = atof(argv[++i]); else if (!strcmp(argv[i], "--check")) o.check = 1;
    else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
  }
  if (o.variant < 0 || (o.kb != 32 && o.kb != 64)) { fprintf(stderr, "bad variant/kb\n"); return 2; }
  init_data(200);
  if (o.variant == 0) return go<0, 0>(o);
  if (o.variant == 2) return go<2, 0>(o);
  if (o.variant == 3) return go<3, 0>(o);
  return o.kb == 32 ? go<1, 32>(o) : go<1, 64>(o);
}
