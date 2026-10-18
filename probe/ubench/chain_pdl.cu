// U4: kernel-boundary cost in a CUDA graph, with and without PDL.
// A chain of N synthetic "weight-stream GEMV" kernels, each reading its own cold S-byte weight slice and a small
// vector written by the previous kernel (true data dependency).  Forms:
//   0 plain launch, no PDL attribute
//   1 PDL attribute; griddepcontrol.wait at entry (before weight loads); implicit trigger at completion
//   2 PDL; wait at entry; launch_dependents right after the wait (what the Triton kernels do)
//   3 PDL; L2-prefetch of the whole weight slice BEFORE the wait; trigger after the wait
//   4 PDL; prefetch before wait; no explicit trigger (implicit at completion)
//   5 PDL; first register batch of weight loads BEFORE the wait; trigger after the wait
// Build: nvcc -O3 -std=c++17 -arch=sm_100 chain_pdl.cu -o chain_pdl
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int NT = 256;
__device__ __forceinline__ uint4 ldg_nc(const uint4* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }

template <int FORM>
__global__ void __launch_bounds__(NT) k_chain(const uint4* __restrict__ W, size_t vec_per_cta, const unsigned* x_in, unsigned* x_out) {
  const uint4* q = W + blockIdx.x * vec_per_cta;
  uint32_t acc = 0;
  uint4 v0[8];
  if (FORM == 3 || FORM == 4) {
    const char* pb = (const char*)q;
    for (size_t i = threadIdx.x; i < vec_per_cta * 16 / 128; i += NT) asm volatile("prefetch.global.L2 [%0];" ::"l"(pb + i * 128));
  }
  if (FORM == 5) {
#pragma unroll
    for (int u = 0; u < 8; ++u) { size_t idx = threadIdx.x + (size_t)u * NT; v0[u] = idx < vec_per_cta ? ldg_nc(q + idx) : make_uint4(0, 0, 0, 0); }
  }
  if (FORM >= 1) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (FORM == 2 || FORM == 3 || FORM == 5) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  size_t start = 0;
  if (FORM == 5) {
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v0[u].x ^ v0[u].w;
    start = (size_t)NT * 8;
  }
  for (size_t i = threadIdx.x + start; i < vec_per_cta; i += NT * 8) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) { size_t idx = i + (size_t)u * NT; v[u] = idx < vec_per_cta ? ldg_nc(q + idx) : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].w;
  }
  const unsigned xin = x_in[(blockIdx.x * 7) % gridDim.x];
  acc ^= xin;
  __shared__ unsigned red[NT];
  red[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { unsigned s = 0; for (int i = 0; i < NT; ++i) s ^= red[i]; x_out[blockIdx.x] = s + 1; }
}

__global__ void k_flush(uint4* p, size_t n, unsigned v) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = make_uint4(v, v, v, v); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <int F>
static void launch_form(cudaStream_t st, int ctas, const uint4* w, size_t vec, const unsigned* xi, unsigned* xo) {
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(ctas); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = (F >= 1);
  cfg.attrs = at; cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, k_chain<F>, w, vec, xi, xo));
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 600;
  const size_t POOL = 3ull << 30, FLUSH = 512ull << 20;
  char *pool, *flush; unsigned* x;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&flush, FLUSH)); CK(cudaMalloc(&x, 2 * 1024 * 4));
  CK(cudaMemset(pool, 1, POOL)); CK(cudaMemset(x, 0, 2 * 1024 * 4));
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const double smb[] = {0, 4, 5, 8, 16.8};
  const int ctas_list[] = {128, 148};
  for (int ctas : ctas_list)
  for (double s : smb) {
    const size_t per = s == 0 ? 0 : (size_t)(s * 1e6 / ctas / 4096 + 0.999) * 4096;
    const size_t real = per * ctas;
    for (int form = 0; form <= 5; ++form) {
      cudaGraph_t g; cudaGraphExec_t ge;
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < N; ++i) {
        const uint4* w = (const uint4*)(pool + ((size_t)i * (real ? real : 1)) % (POOL - real - 4096));
        const unsigned* xi = x + (i & 1) * 1024; unsigned* xo = x + ((i + 1) & 1) * 1024;
        const size_t vec = per / 16;
        switch (form) {
          case 0: launch_form<0>(st, ctas, w, vec, xi, xo); break; case 1: launch_form<1>(st, ctas, w, vec, xi, xo); break;
          case 2: launch_form<2>(st, ctas, w, vec, xi, xo); break; case 3: launch_form<3>(st, ctas, w, vec, xi, xo); break;
          case 4: launch_form<4>(st, ctas, w, vec, xi, xo); break; case 5: launch_form<5>(st, ctas, w, vec, xi, xo); break;
        }
      }
      CK(cudaStreamEndCapture(st, &g));
      CK(cudaGraphInstantiate(&ge, g, 0));
      std::vector<double> t;
      cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
      for (int r = 0; r < 8; ++r) {
        CK((k_flush<<<2048, 256, 0, st>>>((uint4*)flush, FLUSH / 16, r), cudaGetLastError()));
        CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (r >= 2) t.push_back(ms * 1e3 / N);
      }
      printf("{\"u\":\"U4\",\"ctas\":%d,\"S_mb\":%.3f,\"form\":%d,\"N\":%d,\"us_per_kernel_median\":%.3f,\"us_per_kernel_min\":%.3f}\n", ctas, real / 1e6, form, N, median(t), *std::min_element(t.begin(), t.end()));
      fflush(stdout);
      cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
  }
  return 0;
}
