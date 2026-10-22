// U11: stage timeline of U10's tcgen05 GEMM task (gemm_tc.cu). Same task, shapes, cold-weight pool walk, grid barrier and
// globaltimer critical path as U10; adds per-CTA clock64 stamps at every stage boundary so stage costs are measured directly
// instead of inferred by removing stages. Stamps live in registers and are written after the warp fence (after t1).
// Per CTA, relative to T0 = min over worker warps of the clock at grid-barrier exit:
//   tid 0 (TMA + MMA issuer): A staged (after the worker named barrier), per group g: TMA mbarrier passed, first MMA issued,
//     commit issued; with PERMMA (2048) a stamp after every MMA.
//   observer warp 8 (OBS, 1024; lane 0 spins only, no work): TMA landed (mbarrier phase completes) and MMA done (commit
//     mbarrier arrives), per group. Without OBS warp 8 just takes part in the grid barrier.
//   each worker warp (lane 0): barrier exit, commit seen (group 0), first / last tcgen05.wait::ld returned, stores issued,
//     fence done (= t1).
// fl flags as in U10: 1 = TMA pre-issued before the barrier; 2 = bf16 partials; 4 = A and B resident in smem; 8 = MMA +
// commit wait only; 16 = tcgen05.ld, no stores; 32 = A pre-staged; 64 = empty task; plus 1024 = OBS, 2048 = PERMMA.
// Output: JSON lines per cfg: U10-style warp/crit globaltimer stats, then "ev" lines (event time after T0) and "d" lines
// (per-CTA / per-warp stage durations; warp_period = one phase incl. the grid barrier), all in ns (clock64 converted with the measured ns/cycle). argv[3] = dir: raw
// stamps of the last rep per cfg as <cfg>.bin ([nph][148][96] u64 clocks) + <cfg>.tm.bin ([nph][148*8][2] globaltimer).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a gemm_tc2.cu -o gemm_tc2      Run: gemm_tc2 [nph=100] [reps=5] [rawdir]
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
constexpr int NW = 8, NT = (NW + 1) * 32, GRID = 148, T = 50, ROWS = 9728;  // warps 0..7 work, warp 8 observes
constexpr size_t PHASE_BYTES = (size_t)ROWS * 1024 * 2;
constexpr int A_BYTES = 16384, B_BYTES = 512 * 256, SMEM = 1024 + A_BYTES + B_BYTES + 256;
constexpr int ASTRIDE = 4096;  // A global [50][4096] bf16
constexpr int OBS = 1024, PERMMA = 2048;
// stamp slots per (phase, CTA)
constexpr int E = 96, S_ASTAGED = 1, S_TMAW = 2, S_MMA0 = 4, S_COMMIT = 6, S_OTMA = 8, S_ODONE = 10, S_W = 12, WS = 6, S_MMA = 64;
enum { W_T0, W_SEEN, W_LD0, W_LD1, W_ST, W_FENCE };

__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ ull clk() { ull t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t) :: "memory"); return t; }
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
__device__ __forceinline__ void wbar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }  // worker warps only
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
                                               bf16* pbf, ull* tm, ull* st) {
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
  if (warp == 0) {
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
  if (busy && (fl & (4 | 32)) && warp < NW) stage_a(sA, Asrc, s, tid);
  if (busy && (fl & 4)) {
    if (warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, pb0 / 2048);
    for (int g = 0; g < ngr; ++g) mbar_wait(bar0 + 8 * g, 0);
  }
  __syncthreads();
  const int q = warp & 3, half = warp >> 2;
  const uint32_t id0 = idesc(ng0), id1 = idesc(ng1 ? ng1 : 8);
  for (int p = 0; p < nph; ++p) {
    const size_t prow = (((size_t)(phase0 + p) * PHASE_BYTES) % (pool_bytes - PHASE_BYTES)) / 2048;
    ull* S = st + ((size_t)p * GRID + blockIdx.x) * E;
    if (busy && !(fl & 4) && (fl & 1) && warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, prow);
    gbar(cnt, p + 1);
    const ull t0 = gt(), c0 = clk();
    if (warp == NW) {  // observer
      if (busy && (fl & OBS) && !(fl & 64) && lane == 0) {
        ull ot[2] = {0, 0}, od[2] = {0, 0};
#pragma unroll
        for (int g = 0; g < 2; ++g) if (g < ngr && !(fl & 4)) { mbar_wait(bar0 + 8 * g, p & 1); ot[g] = clk(); }
#pragma unroll
        for (int g = 0; g < 2; ++g) if (g < ngr) { mbar_wait(bar0 + 16 + 8 * g, p & 1); od[g] = clk(); }
#pragma unroll
        for (int g = 0; g < 2; ++g) { S[S_OTMA + g] = ot[g]; S[S_ODONE + g] = od[g]; }
      }
      continue;
    }
    ull w[WS] = {c0, 0, 0, 0, 0, 0};
    if (busy && (fl & 64)) {
      if (lane == 0) {
        __threadfence();
        const ull t1 = gt();
        w[W_FENCE] = clk();
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2] = t0;
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2 + 1] = t1;
#pragma unroll
        for (int i = 0; i < WS; ++i) S[S_W + warp * WS + i] = w[i];
      }
    } else if (busy) {
      if (!(fl & 4) && !(fl & 1) && warp == 0) issue_tma(&m128, &m32, &m8, bB, bar0, lane, s, col0, ng0, ng1, prow);
      if (!(fl & (4 | 32))) stage_a(sA, Asrc, s, tid);
      wbar();
      ull is[8] = {0, 0, 0, 0, 0, 0, 0, 0}, mm[16];
      if (tid == 0) {
        is[0] = clk();
#pragma unroll
        for (int g = 0; g < 2; ++g) {
          if (g < ngr) {
            const int ng = g ? ng1 : ng0;
            if (!(fl & 4)) mbar_wait(bar0 + 8 * g, p & 1);
            is[1 + g] = clk();
            fence_after();
            const uint32_t bBg = bB + (g ? 2 * ng0 * 128 : 0);
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
              const int a = kk >> 2, off = (kk & 3) * 32;
              mma_tc(tb + g * 256, sdesc(bA + a * 8192 + off), sdesc(bBg + a * ng * 128 + off), g ? id1 : id0, kk > 0);
              if (fl & PERMMA) mm[g * 8 + kk] = clk();
              else if (kk == 0) is[3 + g] = clk();
            }
            commit(bar0 + 16 + 8 * g);
            is[5 + g] = clk();
          } else if (fl & PERMMA) {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) mm[g * 8 + kk] = 0;
          }
        }
      }
      __syncwarp();
      bool first = true;
      for (int g = 0; g < ngr; ++g) {
        mbar_wait(bar0 + 16 + 8 * g, p & 1);
        if (g == 0) w[W_SEEN] = clk();
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
          w[W_LD1] = clk();
          if (first) { w[W_LD0] = w[W_LD1]; first = false; }
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
      w[W_ST] = clk();
      fence_before();
      __syncwarp();
      if (lane == 0) {
        __threadfence();
        const ull t1 = gt();
        w[W_FENCE] = clk();
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt + 32 + ((blockIdx.x * NW + warp) & 63) * 32) : "memory");
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2] = t0;
        tm[((size_t)p * GRID * NW + blockIdx.x * NW + warp) * 2 + 1] = t1;
#pragma unroll
        for (int i = 0; i < WS; ++i) S[S_W + warp * WS + i] = w[i];
        if (tid == 0) {
          S[S_ASTAGED] = is[0];
#pragma unroll
          for (int g = 0; g < 2; ++g) { S[S_TMAW + g] = is[1 + g]; S[S_MMA0 + g] = is[3 + g]; S[S_COMMIT + g] = is[5 + g]; }
          if (fl & PERMMA) {
#pragma unroll
            for (int i = 0; i < 16; ++i) S[S_MMA + i] = mm[i];
            S[S_MMA0] = mm[0]; S[S_MMA0 + 1] = mm[8];
          }
        }
      }
    }
  }
  __syncthreads();
  fence_after();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tb));
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

// accumulates samples (cycles) per named series; printed as ns
struct Series { std::string name; std::vector<double> v; };
static void add(std::vector<Series>& ss, const std::string& n, double x) {
  for (auto& s : ss) if (s.name == n) { s.v.push_back(x); return; }
  ss.push_back({n, {x}});
}

int main(int argc, char** argv) {
  const int nph = argc > 1 ? atoi(argv[1]) : 100, reps = argc > 2 ? atoi(argv[2]) : 5;
  const char* rawdir = argc > 3 ? argv[3] : nullptr;
  const size_t POOL = 4ull << 30;
  bf16* pool; bf16* A; unsigned* cnt; float* part; bf16* pbf; ull* tm; ull* st; CtaCfg* dcfg; unsigned* ver;
  const size_t PARTN = 4ull << 20, STN = (size_t)nph * GRID * E, TMN = (size_t)nph * GRID * NW * 2;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&A, 64 * ASTRIDE * 2)); CK(cudaMalloc(&cnt, 64 * 1024));
  CK(cudaMalloc(&part, PARTN * 4 * 2)); CK(cudaMalloc(&pbf, PARTN * 2 * 2)); CK(cudaMalloc(&tm, TMN * 8)); CK(cudaMalloc(&st, STN * 8));
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

  std::vector<TC> cfgs = {
    {"tc_f32_n256", 0, 1 | OBS}, {"tc_f32_n256_noobs", 0, 1}, {"tc_f32_n256_permma", 0, 1 | OBS | PERMMA},
    {"tc_f32_n256_Apre", 0, 1 | 32 | OBS}, {"tc_f32_n256_pre0", 0, OBS}, {"tc_bf16_n256", 0, 3 | OBS},
    {"tc_bres_f32_n256", 0, 4 | OBS}, {"tc_bres_nost_n256", 0, 4 | 16 | OBS}, {"tc_mmaonly_n256", 0, 4 | 8 | OBS},
    {"tc_mmaonly_n256_permma", 0, 4 | 8 | OBS | PERMMA}, {"tc_empty", 0, 64},
    {"A_bal_f32", 1, 1 | OBS}, {"F_bal_f32", 2, 1 | OBS}, {"F_bal_f32_permma", 2, 1 | OBS | PERMMA}, {"H_bal_f32", 3, 1 | OBS},
  };
  std::vector<ull> h(TMN), hs(STN);
  int phase0 = 0;
  for (auto& c : cfgs) {
    int Ntot; auto cfg = build(c.shape, Ntot);
    CK(cudaMemcpy(dcfg, cfg.data(), GRID * sizeof(CtaCfg), cudaMemcpyHostToDevice));
    int ncmax = 0; for (auto& x : cfg) ncmax = std::max(ncmax, x.ncols);
    auto run = [&](int np, int p0) {
      CK(cudaMemset(cnt, 0, 64 * 1024)); CK(cudaMemset(tm, 0, TMN * 8)); CK(cudaMemset(st, 0, STN * 8));
      k_tc<<<GRID, NT, SMEM>>>(maps[0], maps[1], maps[2], A, np, p0, POOL, dcfg, Ntot, c.fl, cnt, part, pbf, tm, st);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    if (!(c.fl & (4 | 8 | 16 | 64))) {  // verify the last phase of a 3-phase run
      CK(cudaMemset(part, 0, PARTN * 8)); CK(cudaMemset(pbf, 0, PARTN * 4)); CK(cudaMemset(ver, 0, 64));
      run(3, phase0);
      const size_t prow = (((size_t)(phase0 + 2) * PHASE_BYTES) % (POOL - PHASE_BYTES)) / 2048;
      k_ref<<<GRID, 256>>>(pool, prow, A, dcfg, Ntot, part, pbf, c.fl & 2, ver, ver + 1, ver + 2); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      unsigned hv[3]; CK(cudaMemcpy(hv, ver, 12, cudaMemcpyDeviceToHost));
      float me, mr; memcpy(&me, &hv[0], 4); memcpy(&mr, &hv[1], 4);
      printf("{\"u\":\"U11\",\"verify\":\"%s\",\"max_abs_err\":%.3e,\"err_over_maxref\":%.3e,\"bad\":%u}\n", c.name, me, me / mr, hv[2]);
      fflush(stdout);
    }
    std::vector<double> dur, crit, nspc;
    std::vector<Series> ev, d;
    for (int r = 0; r < reps; ++r) {
      run(nph, phase0); phase0 += nph;
      CK(cudaMemcpy(h.data(), tm, TMN * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hs.data(), st, STN * 8, cudaMemcpyDeviceToHost));
      // ns per cycle from CTA 0 warp 0: globaltimer t0 vs clock at barrier exit, phase 2 .. nph-1
      {
        const ull g0 = h[((size_t)2 * GRID * NW) * 2], g1 = h[((size_t)(nph - 1) * GRID * NW) * 2];
        const ull k0 = hs[((size_t)2 * GRID) * E + S_W], k1 = hs[((size_t)(nph - 1) * GRID) * E + S_W];
        if (g1 > g0 && k1 > k0) nspc.push_back((double)(g1 - g0) / (double)(k1 - k0));
      }
      for (int p = 2; p < nph; ++p) {
        ull lo = ~0ull, hi = 0;
        for (int w = 0; w < GRID * NW; ++w) {
          const ull t0 = h[((size_t)p * GRID * NW + w) * 2], t1 = h[((size_t)p * GRID * NW + w) * 2 + 1];
          if (!t1) continue;
          dur.push_back((double)(t1 - t0)); lo = std::min(lo, t0); hi = std::max(hi, t1);
        }
        if (hi) crit.push_back((double)(hi - lo));
        for (int b = 0; b < GRID; ++b) {
          if (!cfg[b].ncols) continue;
          const ull* S = &hs[((size_t)p * GRID + b) * E];
          ull T0 = ~0ull, fmax = 0;
          for (int w = 0; w < NW; ++w) { if (S[S_W + w * WS + W_T0]) T0 = std::min(T0, S[S_W + w * WS + W_T0]); fmax = std::max(fmax, S[S_W + w * WS + W_FENCE]); }
          if (T0 == ~0ull) continue;
          auto rel = [&](ull x) { return (double)((long long)(x - T0)); };
          const char* gn[2] = {"g0", "g1"};
          if (S[S_ASTAGED]) add(ev, "tid0_A_staged", rel(S[S_ASTAGED]));
          for (int g = 0; g < 2; ++g) {
            std::string sfx = std::string("_") + gn[g];
            if (S[S_OTMA + g]) add(ev, "obs_tma_landed" + sfx, rel(S[S_OTMA + g]));
            if (S[S_TMAW + g]) add(ev, "tid0_tma_waited" + sfx, rel(S[S_TMAW + g]));
            if (S[S_MMA0 + g]) add(ev, "tid0_mma_first_issued" + sfx, rel(S[S_MMA0 + g]));
            if (S[S_COMMIT + g]) add(ev, "tid0_commit_issued" + sfx, rel(S[S_COMMIT + g]));
            if (S[S_ODONE + g]) add(ev, "obs_mma_done" + sfx, rel(S[S_ODONE + g]));
            // per-CTA durations
            if (S[S_TMAW + g] && S[S_MMA0 + g]) add(d, "issue_first_mma" + sfx, (double)(S[S_MMA0 + g] - S[S_TMAW + g]));
            if (S[S_TMAW + g] && S[S_COMMIT + g]) add(d, "issue_8mma_commit" + sfx, (double)(S[S_COMMIT + g] - S[S_TMAW + g]));
            if (S[S_ODONE + g] && S[S_COMMIT + g]) add(d, "mma_tail_after_commit" + sfx, (double)((long long)(S[S_ODONE + g] - S[S_COMMIT + g])));
            if (S[S_ODONE + g] && S[S_TMAW + g]) add(d, "mma_exec_from_tma_waited" + sfx, (double)((long long)(S[S_ODONE + g] - S[S_TMAW + g])));
            if (S[S_OTMA + g] && S[S_TMAW + g]) add(d, "tma_wait_late_vs_land" + sfx, (double)((long long)(S[S_TMAW + g] - S[S_OTMA + g])));
          }
          if (S[S_ASTAGED] && S[S_OTMA]) add(d, "tma_land_minus_A_staged_g0", (double)((long long)(S[S_OTMA] - S[S_ASTAGED])));
          if (c.fl & PERMMA) for (int i = 1; i < 16; ++i) if (i != 8 && S[S_MMA + i] && S[S_MMA + i - 1]) add(d, "mma_issue_gap", (double)(S[S_MMA + i] - S[S_MMA + i - 1]));
          for (int w = 0; w < NW; ++w) {
            const ull* W = S + S_W + w * WS;
            if (W[W_SEEN]) add(ev, "warp_commit_seen_g0", rel(W[W_SEEN]));
            if (W[W_LD0]) add(ev, "warp_first_ld_done", rel(W[W_LD0]));
            if (W[W_LD1]) add(ev, "warp_last_ld_done", rel(W[W_LD1]));
            if (W[W_ST]) add(ev, "warp_stores_issued", rel(W[W_ST]));
            if (W[W_FENCE]) add(ev, "warp_fence_done", rel(W[W_FENCE]));
            if (W[W_T0] && W[W_FENCE]) add(d, "warp_total", (double)(W[W_FENCE] - W[W_T0]));
            if (p + 1 < nph && W[W_T0]) { const ull nx = hs[((size_t)(p + 1) * GRID + b) * E + S_W + w * WS + W_T0]; if (nx) add(d, "warp_period", (double)(nx - W[W_T0])); }
            if (W[W_SEEN] && S[S_ODONE]) add(d, "warp_wake_after_mma_done_g0", (double)((long long)(W[W_SEEN] - S[S_ODONE])));
            if (W[W_LD0] && W[W_SEEN] && cfg[b].ncols <= 256) add(d, "warp_first_ld_after_seen", (double)(W[W_LD0] - W[W_SEEN]));
            if (W[W_LD1] && W[W_SEEN] && cfg[b].ncols <= 256) add(d, "warp_ld_store_loop_until_last_ld", (double)(W[W_LD1] - W[W_SEEN]));
            if (W[W_ST] && W[W_LD1]) add(d, "warp_last_ld_to_stores_issued", (double)(W[W_ST] - W[W_LD1]));
            if (W[W_ST] && W[W_SEEN] && !W[W_LD1]) add(d, "warp_seen_to_end", (double)(W[W_ST] - W[W_SEEN]));
            if (W[W_FENCE] && W[W_ST]) add(d, "warp_fence", (double)(W[W_FENCE] - W[W_ST]));
          }
          if (fmax) add(d, "cta_last_fence_done", (double)(fmax - T0));
        }
      }
    }
    if (rawdir) {
      std::string base = std::string(rawdir) + "/" + c.name;
      FILE* f = fopen((base + ".bin").c_str(), "wb"); if (f) { fwrite(hs.data(), 8, STN, f); fclose(f); }
      f = fopen((base + ".tm.bin").c_str(), "wb"); if (f) { fwrite(h.data(), 8, TMN, f); fclose(f); }
    }
    const double k = pctl(nspc, .5);
    printf("{\"u\":\"U11\",\"cfg\":\"%s\",\"fl\":%d,\"ncols_max\":%d,\"ns_per_cycle\":%.4f,\"warp_ns_p50\":%.0f,\"warp_ns_p90\":%.0f,\"phase_crit_ns_p50\":%.0f,\"phase_crit_ns_p90\":%.0f}\n",
           c.name, c.fl, ncmax, k, pctl(dur, .5), pctl(dur, .9), pctl(crit, .5), pctl(crit, .9));
    for (auto& s : ev)
      printf("{\"u\":\"U11\",\"cfg\":\"%s\",\"ev\":\"%s\",\"n\":%zu,\"ns_p10\":%.0f,\"ns_p50\":%.0f,\"ns_p90\":%.0f}\n", c.name, s.name.c_str(), s.v.size(),
             pctl(s.v, .1) * k, pctl(s.v, .5) * k, pctl(s.v, .9) * k);
    for (auto& s : d)
      printf("{\"u\":\"U11\",\"cfg\":\"%s\",\"d\":\"%s\",\"n\":%zu,\"ns_p10\":%.0f,\"ns_p50\":%.0f,\"ns_p90\":%.0f}\n", c.name, s.name.c_str(), s.v.size(),
             pctl(s.v, .1) * k, pctl(s.v, .5) * k, pctl(s.v, .9) * k);
    fflush(stdout);
  }
  return 0;
}
