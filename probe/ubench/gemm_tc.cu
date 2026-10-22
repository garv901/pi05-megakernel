// U10: tcgen05 GEMM task vs gemm_phase's per-warp mma.sync task (go / no-go for a tcgen05 rewrite of the GEMM phases).
// Same launch shape / timing as U9 (gemm_task.cu): 148 CTAs x 256 thr, counter grid barrier per phase, t0 = after the
// barrier, t1 = after a warp fence (stores performed), critical path = (max t1 - min t0) over all busy warps, globaltimer,
// median over phases 2..nph-1 x reps. Cold weights: the same 4 GB pool walk, [rows][1024] bf16 per phase (K-major B).
// Per CTA task (chunk s, columns col0..col0+ncols): A[0:64, k0:k0+128] bf16 (rows >= 50 zero) staged from global by all
// threads into a 128B-swizzled smem tile (2 k-atoms of 64x64); B = ncols x 128 k via TMA (cp.async.bulk.tensor, SWIZZLE_128B,
// 2 k-atoms; box rows 128/32/8 so ncols is any multiple of 8); thread 0 issues tcgen05.mma.cta_group::1.kind::f16 (M=64,
// bf16 in, fp32 accum in TMEM, 8 x K=16 steps per <=256-col group; ncols > 256 -> 2 groups with separate mbarriers),
// tcgen05.commit -> mbarrier, all 8 warps tcgen05.ld.16x256b (warp w: lane quadrant w%4, column blocks w/4) and write
// part[s][t][n] (t<50; row stride Ntot) as float2 (fp32) or bf16x2, then warp fence + t1. TMEM (512 cols) alloc'd once.
// fl flags: 1 = TMA issued before the grid barrier (U9's pre=1 / WPRE); 2 = bf16 partials; 4 = B and A resident in smem
// (staged once before the loop; no TMA / staging in the timed path); 8 = no TMEM ld, no stores (MMA + commit wait only);
// 16 = tcgen05.ld but no stores; 32 = A pre-staged (not in the timed path); 64 = empty task (barrier + fence + t1 only: floor);
// 128 = t1 taken before the __threadfence (stores issued, not yet performed).
// shapes: n256 = 8 chunks x 256 cols/CTA (b%8, like gemm_phase's 8 warps x 32 cols), A/F/H = balanced phases: chunk =
// b*S/148, the chunk's columns split evenly (multiple of 8) over its CTAs (A: S=8,N=2560; F: S=8,N=8192; H: S=32,N=1024).
// H note: weights come from the same [rows][1024] pool: chunk s reads rows (s/8)*1024+col, k0=(s%8)*128 (same bytes).
// Also: a single-CTA MMA latency microbench (N=32/256, 1 MMA vs 8-MMA chain + commit + mbarrier wait, + tcgen05.ld latency).
// Correctness: every storing config is verified on the last phase of a 3-phase run vs a CUDA-core reference kernel.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a gemm_tc.cu -o gemm_tc      Run: gemm_tc [nph=100] [reps=5]
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
constexpr int NT = 256, NW = NT / 32, GRID = 148, T = 50, ROWS = 9728;
constexpr size_t PHASE_BYTES = (size_t)ROWS * 1024 * 2;
constexpr int A_BYTES = 16384, B_BYTES = 512 * 256, SMEM = 1024 + A_BYTES + B_BYTES + 256;
constexpr int ASTRIDE = 4096;  // A global [50][4096] bf16

__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT;\n}" ::"r"(b), "r"(par) : "memory"); }
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
__device__ __forceinline__ void ld1(uint32_t (&v)[8], uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(ta)); }
__device__ __forceinline__ void ld2(uint32_t (&v)[8], uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(ta)); }
__device__ __forceinline__ void ld4(uint32_t (&v)[16], uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
                 "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta)); }
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B (8 rows x 128 B), LBO unused, sm100 version bit
__device__ __forceinline__ ull sdesc(uint32_t addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ uint32_t idesc(int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | (4u << 24); }  // f32 acc, bf16 x bf16, K-major, M=64

__device__ __forceinline__ void gbar(unsigned* cnt, unsigned epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
  }
  __syncthreads();
}

struct CtaCfg { int chunk, col0, ncols, pad; };

// lane (2g+a) issues the TMA boxes of group g, k-atom a (boxes of 128/32/8 rows) after lane 0 posted the expect_tx.
__device__ __forceinline__ void issue_tma(const CUtensorMap* m128, const CUtensorMap* m32, const CUtensorMap* m8, uint32_t bB, uint32_t bar0,
                                           int lane, int s, int col0, int ng0, int ng1, size_t prow) {
  if (lane == 0) { mbar_expect(bar0, ng0 * 256); if (ng1) mbar_expect(bar0 + 8, ng1 * 256); }
  __syncwarp();
  const int g = lane >> 1, a = lane & 1;
  if (g < (ng1 ? 2 : 1)) {
    const int ng = g ? ng1 : ng0;
    uint32_t dst = bB + (g ? 2 * ng0 * 128 : 0) + a * ng * 128;
    int row = (int)prow + (s >> 3) * 1024 + col0 + (g ? ng0 : 0);
    const int kc = (s & 7) * 128 + a * 64;
    const uint32_t bar = bar0 + g * 8;
    int rem = ng;
    for (; rem >= 128; rem -= 128, dst += 128 * 128, row += 128) tma2d(dst, m128, bar, kc, row);
    for (; rem >= 32; rem -= 32, dst += 32 * 128, row += 32) tma2d(dst, m32, bar, kc, row);
    for (; rem >= 8; rem -= 8, dst += 8 * 128, row += 8) tma2d(dst, m8, bar, kc, row);
  }
}

__device__ __forceinline__ void stage_a(uint8_t* sA, const bf16* Asrc, int s, int tid) {
  uint4 v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = tid + 256 * j, r = c >> 4, kc = c & 15;
    if (r < T) v[j] = *reinterpret_cast<const uint4*>(Asrc + (size_t)r * ASTRIDE + s * 128 + kc * 8);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = tid + 256 * j, r = c >> 4, kc = c & 15;
    if (r < T) *reinterpret_cast<uint4*>(sA + (kc >> 3) * 8192 + r * 128 + (((kc & 7) ^ (r & 7)) << 4)) = v[j];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void put(float* part, bf16* pbf, size_t idx, uint32_t a, uint32_t b, int bf) {
  if (bf) *reinterpret_cast<__nv_bfloat162*>(pbf + idx) = __floats2bfloat162_rn(__uint_as_float(a), __uint_as_float(b));
  else *reinterpret_cast<float2*>(part + idx) = make_float2(__uint_as_float(a), __uint_as_float(b));
}

__global__ void __launch_bounds__(NT, 1) k_tc(const __grid_constant__ CUtensorMap m128, const __grid_constant__ CUtensorMap m32,
                                               const __grid_constant__ CUtensorMap m8, const bf16* __restrict__ Asrc, int nph, int phase0,
                                               size_t pool_bytes, const CtaCfg* cfgs, int Ntot, int fl, unsigned* cnt, float* part,
                                               bf16* pbf, ull* tm) {
  extern __shared__ uint8_t smraw[];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  uint8_t* sA = sm;
  const uint32_t bA = base, bB = base + A_BYTES, bar0 = base + A_BYTES + B_BYTES;  // tma[0..1] at bar0, mma[0..1] at bar0+16
  volatile uint32_t* tslot = reinterpret_cast<volatile uint32_t*>(sm + A_BYTES + B_BYTES + 64);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const CtaCfg c = cfgs[blockIdx.x];
  const int s = c.chunk, col0 = c.col0, ncols = c.ncols;
  const bool busy = ncols > 0;
  const int ng0 = ncols > 256 ? (ncols / 16) * 8 : ncols, ng1 = ncols - ng0;
  const int ngr = ng1 ? 2 : 1;
  if (warp == 0 && !(fl & 256)) {  // 256: skip TMEM alloc (only with 64, to test whether holding TMEM slows the fence)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(su32((void*)tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (tid == 0) { for (int i = 0; i < 4; ++i) mbar_init(bar0 + 8 * i, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  for (int i = tid; i < A_BYTES / 16; i += NT) reinterpret_cast<uint4*>(sA)[i] = make_uint4(0, 0, 0, 0);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  fence_after();
  const uint32_t tb = *tslot;
  const size_t pb0 = ((size_t)phase0 * PHASE_BYTES) % (pool_bytes - PHASE_BYTES);
  if (busy && (fl & (4 | 32))) stage_a(sA, Asrc, s, tid);
  if (busy && (fl & 4)) {
    if (warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, pb0 / 2048);
    for (int g = 0; g < ngr; ++g) mbar_wait(bar0 + 8 * g, 0);
  }
  __syncthreads();
  const int q = warp & 3, half = warp >> 2;
  const uint32_t id0 = idesc(ng0), id1 = idesc(ng1 ? ng1 : 8);
  for (int p = 0; p < nph; ++p) {
    const size_t prow = (((size_t)(phase0 + p) * PHASE_BYTES) % (pool_bytes - PHASE_BYTES)) / 2048;
    if (busy && !(fl & 4) && (fl & 1) && warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, prow);
    gbar(cnt, p + 1);
    const ull t0 = gt();
    if (busy && (fl & 64)) {
      if (lane == 0) { if (fl & 512) asm volatile("fence.acq_rel.gpu;" ::: "memory"); else __threadfence(); const ull t1 = gt(); tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2] = t0; tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2 + 1] = t1; }
    } else if (busy) {
      if (!(fl & 4) && !(fl & 1) && warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, prow);
      if (!(fl & (4 | 32))) stage_a(sA, Asrc, s, tid);
      __syncthreads();
      if (tid == 0) {
        for (int g = 0; g < ngr; ++g) {
          const int ng = g ? ng1 : ng0;
          if (!(fl & 4)) mbar_wait(bar0 + 8 * g, p & 1);
          fence_after();
          const uint32_t bBg = bB + (g ? 2 * ng0 * 128 : 0);
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            const int a = kk >> 2, off = (kk & 3) * 32;
            mma_tc(tb + g * 256, sdesc(bA + a * 8192 + off), sdesc(bBg + a * ng * 128 + off), g ? id1 : id0, kk > 0);
          }
          commit(bar0 + 16 + 8 * g);
        }
      }
      __syncwarp();
      for (int g = 0; g < ngr; ++g) {
        mbar_wait(bar0 + 16 + 8 * g, p & 1);
        fence_after();
        if (fl & 8) continue;
        const int ng = g ? ng1 : ng0, gcol = col0 + (g ? ng0 : 0), nb = ng / 8, nblk = (nb + 1) / 2;
        for (int b0 = half; b0 < nblk; b0 += 8) {
          uint32_t v[4][8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int bi = b0 + 2 * j, gi = bi * 2;
            if (bi < nblk) {
              const uint32_t ta = tb + ((uint32_t)(q * 32) << 16) + g * 256 + gi * 8;
              if (gi + 1 < nb) ld2(v[j], ta); else ld1(v[j], ta);
            }
          }
          wait_ld();
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int bi = b0 + 2 * j, gi = bi * 2;
            if (bi < nblk) {
              const int nrep = (gi + 1 < nb) ? 2 : 1;
              for (int rp = 0; rp < nrep; ++rp) {
                const int col = gcol + (gi + rp) * 8 + 2 * (lane & 3), r0 = q * 16 + (lane >> 2);
                if (fl & 16) { if (__uint_as_float(v[j][rp * 4]) == -1.2345e30f) part[lane] = __uint_as_float(v[j][rp * 4 + 3]); continue; }
                if (r0 < T) put(part, pbf, ((size_t)(s * 64 + r0)) * Ntot + col, v[j][rp * 4], v[j][rp * 4 + 1], fl & 2);
                if (r0 + 8 < T) put(part, pbf, ((size_t)(s * 64 + r0 + 8)) * Ntot + col, v[j][rp * 4 + 2], v[j][rp * 4 + 3], fl & 2);
              }
            }
          }
        }
      }
      fence_before();
      __syncwarp();
      if (lane == 0) {
        ull t1 = gt();
        __threadfence();
        if (!(fl & 128)) t1 = gt();
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt + 32 + ((blockIdx.x * NW + warp) & 63) * 32) : "memory");
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2] = t0;
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2 + 1] = t1;
      }
    }
  }
  __syncthreads();
  fence_after();
  if (warp == 0 && !(fl & 256)) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tb));
}

// single-CTA latency: kind 0 = 1 MMA (K=16), 1 = 8-MMA chain (K=128); both + commit + mbarrier wait; kind 2 = tcgen05.ld
// 16x256b.x4 (64 cols) + wait::ld on the finished accumulator. out[0] = cycles/iter, out[1] = ns/cycle.
__global__ void k_lat(int N, int kind, int iters, double* out) {
  extern __shared__ uint8_t smraw[];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u;
  uint8_t* sm = smraw + (base - raw);
  const uint32_t bA = base, bB = base + 16384, bar = base + 16384 + 65536;
  volatile uint32_t* tslot = reinterpret_cast<volatile uint32_t*>(sm + 16384 + 65536 + 64);
  const int tid = threadIdx.x;
  if (tid < 32) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(su32((void*)tslot))); asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  for (int i = tid; i < (16384 + 65536) / 16; i += blockDim.x) reinterpret_cast<uint4*>(sm)[i] = make_uint4(0, 0, 0, 0);
  if (tid == 0) { mbar_init(bar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); fence_after();
  const uint32_t tb = *tslot;
  const int nm = kind == 0 ? 1 : 8;
  double tot = 0; ull g0 = 0, c0 = 0;
  uint32_t sink = 0;
  for (int it = -20; it < iters; ++it) {
    if (it == 0) { g0 = gt(); c0 = clock64(); }
    const ull a = clock64();
    if (tid == 0) {
      for (int kk = 0; kk < nm; ++kk) { const int at = kk >> 2, off = (kk & 3) * 32; mma_tc(tb, sdesc(bA + at * 8192 + off), sdesc(bB + at * N * 128 + off), idesc(N), kk > 0); }
      commit(bar);
    }
    __syncwarp();
    mbar_wait(bar, it & 1);
    fence_after();
    ull b = clock64();
    if (kind == 2) {
      uint32_t v[16];
      const ull a2 = clock64();
      for (int r = 0; r < 20; ++r) { ld4(v, tb); wait_ld(); sink += v[r & 15]; }
      b = clock64();
      if (it >= 0) tot += (double)(b - a2) / 20.0;
    } else if (it >= 0) tot += (double)(b - a);
    __syncwarp();
  }
  const ull g1 = gt(), c1 = clock64();
  if (tid == 0) { out[0] = tot / iters; out[1] = (double)(g1 - g0) / (double)(c1 - c0); if (sink == 0x12345u) out[2] = 1; }
  __syncthreads();
  if (tid < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tb));
}

__device__ __forceinline__ float hashf(uint32_t i) { uint32_t h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 65536.f); }
__global__ void k_init(bf16* p, size_t n, uint32_t seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((uint32_t)i + seed)); }

__global__ void k_ref(const bf16* pool, size_t prow, const bf16* A, const CtaCfg* cfgs, int Ntot, const float* part, const bf16* pbf, int bf,
                      unsigned* maxerr, unsigned* maxref, unsigned* nbad) {
  const CtaCfg c = cfgs[blockIdx.x];
  for (int j = threadIdx.x; j < c.ncols; j += blockDim.x) {
    const int col = c.col0 + j;
    const bf16* w = pool + (prow + (c.chunk >> 3) * 1024 + col) * 1024 + (c.chunk & 7) * 128;
    for (int t = 0; t < T; ++t) {
      float acc = 0.f;
      for (int k = 0; k < 128; ++k) acc += __bfloat162float(A[(size_t)t * ASTRIDE + c.chunk * 128 + k]) * __bfloat162float(w[k]);
      const size_t idx = ((size_t)(c.chunk * 64 + t)) * Ntot + col;
      const float got = bf ? __bfloat162float(pbf[idx]) : part[idx];
      const float err = fabsf(got - acc);
      atomicMax(maxerr, __float_as_uint(err)); atomicMax(maxref, __float_as_uint(fabsf(acc)));
      if (!(err <= (bf ? 0.005f * fabsf(acc) + 1e-3f : 1e-4f))) atomicAdd(nbad, 1u);
    }
  }
}

static double pctl(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }

static std::vector<CtaCfg> build(int shape, int& Ntot) {
  std::vector<CtaCfg> v(GRID);
  if (shape == 0) {
    Ntot = 8192;
    for (int b = 0; b < GRID; ++b) v[b] = {b % 8, (b / 8) * 256, 256, 0};
    return v;
  }
  const int S = shape == 3 ? 32 : 8;
  Ntot = shape == 1 ? 2560 : shape == 2 ? 8192 : 1024;
  std::vector<int> m(S, 0), j(S, 0);
  for (int b = 0; b < GRID; ++b) m[b * S / GRID]++;
  const int G = Ntot / 8;
  for (int b = 0; b < GRID; ++b) {
    const int c = b * S / GRID, k = j[c]++;
    const int c0 = (k * G) / m[c], c1 = ((k + 1) * G) / m[c];
    v[b] = {c, c0 * 8, (c1 - c0) * 8, 0};
  }
  return v;
}

struct TC { const char* name; int shape; int fl; };

int main(int argc, char** argv) {
  const int nph = argc > 1 ? atoi(argv[1]) : 100, reps = argc > 2 ? atoi(argv[2]) : 5;
  const size_t POOL = 4ull << 30;
  bf16* pool; bf16* A; unsigned* cnt; float* part; bf16* pbf; ull* tm; CtaCfg* dcfg; unsigned* ver;
  const size_t PARTN = 4ull << 20;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&A, 64 * ASTRIDE * 2)); CK(cudaMalloc(&cnt, 64 * 1024));
  CK(cudaMalloc(&part, PARTN * 4 * 2)); CK(cudaMalloc(&pbf, PARTN * 2 * 2)); CK(cudaMalloc(&tm, (size_t)nph * GRID * NW * 16));
  CK(cudaMalloc(&dcfg, GRID * sizeof(CtaCfg))); CK(cudaMalloc(&ver, 64));
  k_init<<<4096, 256>>>(pool, POOL / 2, 12345u); k_init<<<64, 256>>>(A, 64 * ASTRIDE, 777u); CK(cudaDeviceSynchronize());

  PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  CUtensorMap maps[3]; const int boxes[3] = {128, 32, 8};
  for (int i = 0; i < 3; ++i) {
    cuuint64_t dims[2] = {1024, (cuuint64_t)(POOL / 2048)}, strides[1] = {2048};
    cuuint32_t box[2] = {64, (cuuint32_t)boxes[i]}, es[2] = {1, 1};
    CUresult r = enc(&maps[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pool, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); return 1; }
  }
  CK(cudaFuncSetAttribute(k_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute(k_lat, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));

  // MMA latency microbench (one CTA)
  {
    double* d; CK(cudaMalloc(&d, 32));
    for (int kind = 0; kind < 3; ++kind)
      for (int N : {32, 64, 128, 256}) {
        if (kind == 2 && N != 256) continue;
        k_lat<<<1, 128, 100 * 1024>>>(N, kind, 2000, d); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        double h[3]; CK(cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost));
        printf("{\"u\":\"U10\",\"lat\":\"%s\",\"N\":%d,\"cycles\":%.0f,\"ns_per_cycle\":%.4f,\"ns\":%.0f}\n",
               kind == 0 ? "mma_k16_commit_wait" : kind == 1 ? "mma8_chain_commit_wait" : "tcgen05_ld_x4_64col_each_of_20_serial", N, h[0], h[1], h[0] * h[1]);
      }
    fflush(stdout);
  }

  std::vector<TC> cfgs = {
    {"tc_f32_n256", 0, 1}, {"tc_f32_n256_pre0", 0, 0}, {"tc_bf16_n256", 0, 3}, {"tc_bf16_n256_pre0", 0, 2},
    {"tc_nost_n256", 0, 1 | 16}, {"tc_mmaonly_n256", 0, 4 | 8}, {"tc_bres_nost_n256", 0, 4 | 16}, {"tc_bres_f32_n256", 0, 4},
    {"tc_f32_n256_Apre", 0, 1 | 32}, {"tc_empty", 0, 64}, {"tc_empty_notmem", 0, 64 | 256}, {"tc_empty_acqrel", 0, 64 | 512}, {"tc_f32_n256_nofence", 0, 1 | 128},
    {"A_bal_f32", 1, 1}, {"A_bal_f32_pre0", 1, 0}, {"A_bal_bf16", 1, 3}, {"A_bal_nost", 1, 1 | 16}, {"A_bal_mmaonly", 1, 4 | 8},
    {"F_bal_f32", 2, 1}, {"F_bal_f32_pre0", 2, 0}, {"F_bal_bf16", 2, 3}, {"F_bal_nost", 2, 1 | 16}, {"F_bal_mmaonly", 2, 4 | 8},
    {"H_bal_f32", 3, 1}, {"H_bal_f32_pre0", 3, 0}, {"H_bal_bf16", 3, 3}, {"H_bal_nost", 3, 1 | 16}, {"H_bal_mmaonly", 3, 4 | 8},
  };
  std::vector<ull> h((size_t)nph * GRID * NW * 2);
  int phase0 = 0;
  for (auto& c : cfgs) {
    int Ntot; auto cfg = build(c.shape, Ntot);
    CK(cudaMemcpy(dcfg, cfg.data(), GRID * sizeof(CtaCfg), cudaMemcpyHostToDevice));
    int ncmax = 0; for (auto& x : cfg) ncmax = std::max(ncmax, x.ncols);
    auto run = [&](int np, int p0) {
      CK(cudaMemset(cnt, 0, 64 * 1024)); CK(cudaMemset(tm, 0, (size_t)nph * GRID * NW * 16));
      k_tc<<<GRID, NT, SMEM>>>(maps[0], maps[1], maps[2], A, np, p0, POOL, dcfg, Ntot, c.fl, cnt, part, pbf, tm);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    if (!(c.fl & (4 | 8 | 16 | 64))) {  // verify the last phase of a 3-phase run
      CK(cudaMemset(part, 0, PARTN * 8)); CK(cudaMemset(pbf, 0, PARTN * 4)); CK(cudaMemset(ver, 0, 64));
      run(3, phase0);
      const size_t prow = (((size_t)(phase0 + 2) * PHASE_BYTES) % (POOL - PHASE_BYTES)) / 2048;
      k_ref<<<GRID, 256>>>(pool, prow, A, dcfg, Ntot, part, pbf, c.fl & 2, ver, ver + 1, ver + 2); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      unsigned hv[3]; CK(cudaMemcpy(hv, ver, 12, cudaMemcpyDeviceToHost));
      float me, mr; memcpy(&me, &hv[0], 4); memcpy(&mr, &hv[1], 4);
      printf("{\"u\":\"U10\",\"verify\":\"%s\",\"max_abs_err\":%.3e,\"max_abs_ref\":%.3f,\"err_over_maxref\":%.3e,\"bad\":%u,\"of\":%d}\n",
             c.name, me, mr, me / mr, hv[2], 50 * [&] { int n = 0; for (auto& x : cfg) n += x.ncols; return n; }());
      fflush(stdout);
    }
    std::vector<double> dur, crit;
    for (int r = 0; r < reps; ++r) {
      run(nph, phase0); phase0 += nph;
      CK(cudaMemcpy(h.data(), tm, h.size() * 8, cudaMemcpyDeviceToHost));
      for (int p = 2; p < nph; ++p) {
        ull lo = ~0ull, hi = 0;
        for (int w = 0; w < GRID * NW; ++w) {
          const ull t0 = h[((size_t)p * GRID * NW + w) * 2], t1 = h[((size_t)p * GRID * NW + w) * 2 + 1];
          if (!t1) continue;
          dur.push_back((double)(t1 - t0)); lo = std::min(lo, t0); hi = std::max(hi, t1);
        }
        crit.push_back((double)(hi - lo));
      }
    }
    printf("{\"u\":\"U10\",\"cfg\":\"%s\",\"fl\":%d,\"ncols_max\":%d,\"warp_ns_p50\":%.0f,\"warp_ns_p90\":%.0f,\"phase_crit_ns_p50\":%.0f,\"phase_crit_ns_p90\":%.0f}\n",
           c.name, c.fl, ncmax, pctl(dur, .5), pctl(dur, .9), pctl(crit, .5), pctl(crit, .9));
    fflush(stdout);
  }
  return 0;
}
