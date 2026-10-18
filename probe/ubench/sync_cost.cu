// U3: barrier cost in context.  148 CTAs x 256 thr (1 CTA/SM, 117 KB smem like the megakernel).
// Each "phase" = [stream W bytes per CTA from a cold pool] [issue prefetch.global.L2 of PF bytes spread over the grid]
// [optional skew spin on CTA 0] [barrier].  MODE 0 replaces the grid barrier with __syncthreads (control);
// MODE 1 = production counter barrier (atom.add.release + ld.acquire poll).  Barrier overhead = T(mode1) - T(mode0).
// Build: nvcc -O3 -std=c++17 -arch=sm_100 sync_cost.cu -o sync_cost
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int NT = 256;
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ __forceinline__ void gbar(unsigned* cnt, unsigned epoch, int mode) {
  __syncthreads();
  if (mode == 1) {
    if (threadIdx.x == 0) {
      unsigned old;
      asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
      while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
    }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(NT, 1) k_phases(const char* __restrict__ pool, size_t pool_bytes, int nph, size_t w_per_cta, size_t pf_total,
                                                  unsigned skew_ns, int mode, unsigned* cnt, unsigned long long* tm, unsigned* sink) {
  extern __shared__ char sm[];
  if (threadIdx.x == 0) atomicMin(&tm[0], gt());
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * w_per_cta + pf_total;
  for (int p = 0; p < nph; ++p) {
    const char* base = pool + (stride * p) % (pool_bytes - stride);
    if (w_per_cta) {
      const uint4* q = (const uint4*)(base + (size_t)blockIdx.x * w_per_cta);
      const size_t vec = w_per_cta / 16;
      for (size_t i = threadIdx.x; i < vec; i += NT * 8) {
        uint4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { size_t idx = i + (size_t)u * NT; v[u] = idx < vec ? ldg_nc(q + idx) : make_uint4(0, 0, 0, 0); }
#pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].w;
      }
    }
    if (pf_total) {
      const char* pb = base + (size_t)gridDim.x * w_per_cta;
      const size_t n = pf_total / 128, gt_ = (size_t)blockIdx.x * NT + threadIdx.x, nt = (size_t)gridDim.x * NT;
      for (size_t i = gt_; i < n; i += nt) asm volatile("prefetch.global.L2 [%0];" ::"l"(pb + i * 128));
    }
    if (skew_ns && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t0 = gt(); while (gt() - t0 < skew_ns) {} }
    gbar(cnt, p + 1, mode);
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tm[1], gt());
}

__global__ void k_flush(uint4* p, size_t n, unsigned v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = make_uint4(v, v, v, v); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int nph = argc > 1 ? atoi(argv[1]) : 200, reps = argc > 2 ? atoi(argv[2]) : 11;
  const size_t POOL = 3ull << 30, FLUSH = 512ull << 20;
  char *pool, *flush; unsigned* cnt; unsigned long long* tm; unsigned* sink;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&flush, FLUSH)); CK(cudaMalloc(&cnt, 256)); CK(cudaMalloc(&tm, 16)); CK(cudaMalloc(&sink, 16));
  CK(cudaMemset(pool, 1, POOL));
  const int smem = 117 * 1024;
  CK(cudaFuncSetAttribute(k_phases, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  struct Sc { const char* name; size_t w_total; size_t pf; unsigned skew; };
  const size_t MB = 1000000;
  std::vector<Sc> scs = {
    {"empty", 0, 0, 0}, {"skew_500ns", 0, 0, 500}, {"skew_2us", 0, 0, 2000},
    {"prefetch_4MB", 0, 4 * MB, 0}, {"prefetch_16MB", 0, 16 * MB, 0},
    {"stream_5MB", 5 * MB, 0, 0}, {"stream_16.8MB", 16800000, 0, 0},
    {"stream_5MB_pf_16MB", 5 * MB, 16 * MB, 0}, {"stream_16.8MB_pf_8MB", 16800000, 8 * MB, 0},
  };
  for (auto& sc : scs) {
    const size_t wcta = (sc.w_total / 148 / 16) * 16;
    double t[2];
    for (int mode = 0; mode < 2; ++mode) {
      std::vector<double> v;
      for (int r = 0; r < reps; ++r) {
        CK((k_flush<<<2048, 256>>>((uint4*)flush, FLUSH / 16, r), cudaGetLastError())); CK(cudaMemsetAsync(cnt, 0, 256));
        unsigned long long init[2] = {~0ull, 0ull}; CK(cudaMemcpyAsync(tm, init, 16, cudaMemcpyHostToDevice));
        k_phases<<<148, NT, smem>>>(pool, POOL, nph, wcta, sc.pf, sc.skew, mode, cnt, tm, sink);
        CK(cudaDeviceSynchronize());
        unsigned long long res[2]; CK(cudaMemcpy(res, tm, 16, cudaMemcpyDeviceToHost));
        if (r >= 1) v.push_back((res[1] - res[0]) / 1e3 / nph);
      }
      t[mode] = median(v);
    }
    printf("{\"u\":\"U3\",\"scenario\":\"%s\",\"nphases\":%d,\"us_per_phase_syncthreads_only\":%.3f,\"us_per_phase_counter_barrier\":%.3f,\"barrier_overhead_us\":%.3f,\"x1634_ms\":%.3f}\n",
           sc.name, nph, t[0], t[1], t[1] - t[0], (t[1] - t[0]) * 1634 / 1e3);
    fflush(stdout);
  }
  return 0;
}
