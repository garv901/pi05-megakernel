// U1: HBM ramp / Little's law.  Time to stream S bytes (L2-cold) as a function of CTAs and bytes in
// flight per SM, for ldg.128->regs, cp.async, TMA bulk.  In-kernel time = first CTA start .. last CTA end
// (globaltimer), so launch latency is excluded; event time is also reported.
// Also: pointer-chase latency (HBM cold, L2 hit).
// Build: nvcc -O3 -std=c++17 -arch=sm_100 hbm_ramp.cu -o hbm_ramp
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }

constexpr int NT = 256;          // threads per CTA
constexpr size_t STAGE = 4096;   // bytes per 'stage' = NT * 16 B

template <int UNR>
__global__ void __launch_bounds__(NT, 1) k_ldg(const uint4* __restrict__ base, size_t vec_per_cta, unsigned* sink, unsigned long long* tm) {
  if (threadIdx.x == 0) atomicMin(&tm[0], gt());
  const uint4* p = base + blockIdx.x * vec_per_cta;
  uint32_t acc = 0;
  for (size_t i = threadIdx.x; i < vec_per_cta; i += (size_t)NT * UNR) {
    uint4 v[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) { size_t idx = i + (size_t)u * NT; v[u] = idx < vec_per_cta ? ldg_nc(p + idx) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < UNR; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tm[1], gt());
}

__device__ __forceinline__ void cp16(void* s, const void* g) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g) : "memory");
}
template <int D>
__global__ void __launch_bounds__(NT, 1) k_cpasync(const char* __restrict__ base, size_t bytes_per_cta, unsigned long long* tm) {
  extern __shared__ __align__(128) char sm[];
  if (threadIdx.x == 0) atomicMin(&tm[0], gt());
  const char* p = base + blockIdx.x * bytes_per_cta;
  const int ns = (int)(bytes_per_cta / STAGE);
  for (int j = 0; j < D && j < ns; ++j) { cp16(sm + j * STAGE + threadIdx.x * 16, p + j * STAGE + threadIdx.x * 16); asm volatile("cp.async.commit_group;" ::: "memory"); }
  for (int j = 0; j < ns; ++j) {
    asm volatile("cp.async.wait_group %0;" ::"n"(D - 1) : "memory");
    __syncthreads();
    const int nj = j + D;
    if (nj < ns) cp16(sm + (nj % D) * STAGE + threadIdx.x * 16, p + (size_t)nj * STAGE + threadIdx.x * 16);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tm[1], gt());
}

__device__ __forceinline__ void mb_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, int parity) {
  asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes), "r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
}
template <int D>
__global__ void __launch_bounds__(NT, 1) k_tma(const char* __restrict__ base, size_t bytes_per_cta, unsigned long long* tm) {
  extern __shared__ __align__(128) char sm[];
  __shared__ uint64_t mb[64];
  if (threadIdx.x == 0) {
    atomicMin(&tm[0], gt());
    for (int d = 0; d < D; ++d) mb_init(&mb[d], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const char* p = base + blockIdx.x * bytes_per_cta;
  const int ns = (int)(bytes_per_cta / STAGE);
  if (threadIdx.x == 0) {
    for (int j = 0; j < D && j < ns; ++j) { mb_expect(&mb[j], STAGE); bulk(sm + j * STAGE, p + (size_t)j * STAGE, STAGE, &mb[j]); }
    for (int j = 0; j < ns; ++j) {
      const int s = j % D;
      mb_wait(&mb[s], (j / D) & 1);
      if (j + D < ns) { mb_expect(&mb[s], STAGE); bulk(sm + s * STAGE, p + (size_t)(j + D) * STAGE, STAGE, &mb[s]); }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tm[1], gt());
}

// pointer chase: one thread, dependent loads
__global__ void k_chase(const unsigned long long* __restrict__ nxt, int steps, unsigned long long* out) {
  unsigned long long idx = 0;
  const unsigned long long t0 = gt();
  asm volatile("" ::: "memory");
  for (int i = 0; i < steps; ++i) { unsigned long long v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(nxt + idx) : "memory"); idx = v; }
  const unsigned long long t1 = gt();
  out[0] = t1 - t0; out[1] = idx;
}

__global__ void k_fill(unsigned long long* p, size_t n, unsigned long long v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v; }

__global__ void k_flush(uint4* p, size_t n, unsigned v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = make_uint4(v, v, v, v); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 15;
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
  fprintf(stderr, "%s SMs=%d L2=%d\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize);
  const size_t POOL = 3ull << 30, FLUSH = 512ull << 20;
  char *pool, *flush; unsigned long long* tm; unsigned* sink;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&flush, FLUSH)); CK(cudaMalloc(&tm, 16)); CK(cudaMalloc(&sink, 16));
  CK(cudaMemset(pool, 1, POOL));
  #define SETA(K, D) CK(cudaFuncSetAttribute(K<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, D * STAGE));
  SETA(k_cpasync, 2) SETA(k_cpasync, 4) SETA(k_cpasync, 8) SETA(k_cpasync, 16) SETA(k_cpasync, 32) SETA(k_cpasync, 48)
  SETA(k_tma, 2) SETA(k_tma, 4) SETA(k_tma, 8) SETA(k_tma, 16) SETA(k_tma, 32) SETA(k_tma, 48)
  const double Smb[] = {0.18, 4, 5, 8, 16.8, 34.6};
  const int ctas[] = {21, 50, 80, 128, 148};
  const int infl[] = {2, 4, 8, 16, 32, 48};  // x 4 KB per SM
  const char* paths[] = {"ldg", "cpasync", "tma"};
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int pi = 0; pi < 3; ++pi)
    for (double s : Smb)
      for (int nc : ctas)
        for (int in : infl) {
          size_t per = (size_t)(s * 1e6 / nc / STAGE + 0.999) * STAGE; if (per < STAGE) per = STAGE;
          const size_t real = per * nc;
          std::vector<double> tin, tev;
          for (int r = 0; r < reps + 2; ++r) {
            size_t off = ((size_t)(r * 97 + pi * 13 + in) * (63ull << 20)) % (POOL - real - 4096); off &= ~4095ull;
            CK((k_flush<<<2048, 256>>>((uint4*)flush, FLUSH / 16, r & 0xff), cudaGetLastError()));
            unsigned long long init[2] = {~0ull, 0ull};
            CK(cudaMemcpyAsync(tm, init, 16, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e0));
            const char* b = pool + off;
            if (pi == 0) {
              const size_t vec = per / 16;
#define L(U) case U: k_ldg<U><<<nc, NT>>>((const uint4*)b, vec, sink, tm); break;
              switch (in) { L(2) L(4) L(8) L(16) L(32) L(48) }
            } else if (pi == 1) {
#define C(D) case D: k_cpasync<D><<<nc, NT, D * STAGE>>>(b, per, tm); break;
              switch (in) { C(2) C(4) C(8) C(16) C(32) C(48) }
            } else {
#define M(D) case D: k_tma<D><<<nc, NT, D * STAGE>>>(b, per, tm); break;
              switch (in) { M(2) M(4) M(8) M(16) M(32) M(48) }
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            unsigned long long res[2]; CK(cudaMemcpy(res, tm, 16, cudaMemcpyDeviceToHost));
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (r >= 2) { tin.push_back((res[1] - res[0]) / 1e3); tev.push_back(ms * 1e3); }
          }
          const double mi = median(tin), me = median(tev);
          printf("{\"u\":\"U1\",\"path\":\"%s\",\"S_req_mb\":%.2f,\"S_real_mb\":%.3f,\"ctas\":%d,\"inflight_kb_per_sm\":%d,\"inflight_total_mb\":%.2f,"
                 "\"kernel_us_median\":%.3f,\"kernel_us_min\":%.3f,\"event_us_median\":%.3f,\"TBps_inkernel\":%.3f}\n",
                 paths[pi], s, real / 1e6, nc, in * 4, in * 4.0 * nc / 1024, mi, *std::min_element(tin.begin(), tin.end()), me, real / (mi * 1e-6) / 1e12);
          fflush(stdout);
        }
  // latency: pointer chase over 1 GB (cold) and a 1 MB loop (L2 hit)
  {
    const size_t N = 1ull << 27;  // 128M entries * 8 B = 1 GB
    unsigned long long* nxt; CK(cudaMalloc(&nxt, N * 8));
    std::vector<unsigned long long> h(1 << 20);
    // cold: stride 4096+ entries random-ish, one chain of 256 steps (each step a distinct 128B line far apart)
    const int steps = 256;
    std::vector<unsigned long long> chain(N, 0);  // 1 GB host
    std::vector<unsigned long long> lines(steps);
    { unsigned long long x = 88172645463325252ull; for (int i = 0; i < steps; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; lines[i] = (x % (N / 16)) * 16; } }
    unsigned long long cur = 0;
    for (int i = 0; i < steps; ++i) { chain[cur] = lines[i]; cur = lines[i]; }
    chain[cur] = 0;
    CK(cudaMemcpy(nxt, chain.data(), N * 8, cudaMemcpyHostToDevice));
    unsigned long long* out; CK(cudaMalloc(&out, 16));
    std::vector<double> cold, warm;
    for (int r = 0; r < 12; ++r) {
      CK((k_flush<<<2048, 256>>>((uint4*)flush, FLUSH / 16, r), cudaGetLastError()));
      k_chase<<<1, 1>>>(nxt, steps, out); CK(cudaDeviceSynchronize());
      unsigned long long o[2]; CK(cudaMemcpy(o, out, 16, cudaMemcpyDeviceToHost)); if (r >= 2) cold.push_back((double)o[0] / steps);
      k_chase<<<1, 1>>>(nxt, steps, out); CK(cudaDeviceSynchronize());  // same lines now in L2
      CK(cudaMemcpy(o, out, 16, cudaMemcpyDeviceToHost)); if (r >= 2) warm.push_back((double)o[0] / steps);
    }
    printf("{\"u\":\"U1-latency\",\"hbm_cold_ns_per_load\":%.1f,\"l2_hit_ns_per_load\":%.1f}\n", median(cold), median(warm));
  }
  return 0;
}
