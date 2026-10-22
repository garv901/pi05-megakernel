// u13: cta_group::2 M=128 pair MMA D layout probe. A row m = (m+1) (bf16), B all ones, K=16 -> D[m][n] = 16 (m+1).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
typedef unsigned long long ull;
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ ull sdesc(uint32_t addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ uint32_t idesc(int N, int M) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
__global__ void __cluster_dims__(2, 1, 1) k(float* out) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) ull bar; __shared__ uint32_t slot;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t rank; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(rank));
  const uint32_t base = (su32(sm) + 1023) & ~1023u; unsigned char* gp = sm + (base - su32(sm));
  // A: 64 rows x 128B; B: 32 rows x 128B at +8192
  for (int i = tid; i < 64 * 64; i += blockDim.x) { __nv_bfloat16 v = __float2bfloat16(1.0f); reinterpret_cast<__nv_bfloat16*>(gp)[i] = v; }
  for (int i = tid; i < 32 * 64; i += blockDim.x) reinterpret_cast<__nv_bfloat16*>(gp + 8192)[i] = __float2bfloat16((float)(rank * 32 + i / 64 + 1));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(su32(&bar))); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (warp == 1) asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;" ::"r"(su32(&slot)));
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tb = slot;
  if (rank == 0 && warp == 1 && lane == 0) {
    const ull da = sdesc(base), db = sdesc(base + 8192);
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(tb), "l"(da), "l"(db), "r"(idesc(64, 128)), "r"(0) : "memory");
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(su32(&bar)), "h"((unsigned short)3) : "memory");
  }
  { uint32_t ok = 0; while (!ok) asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], 0;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(su32(&bar)) : "memory"); }
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp < 4) {  // read all 128 lanes (4 quadrants), columns 0..1 and 40..41 (N=64)
    uint32_t a, b; uint32_t cc[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(a) : "r"(tb + ((uint32_t)(warp * 32) << 16)));
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(b) : "r"(tb + ((uint32_t)(warp * 32) << 16) + 40));
    { const int cl[8] = {1, 16, 31, 32, 33, 48, 56, 63}; (void)cl;
#define LDC(i, c) asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(cc[i]) : "r"(tb + ((uint32_t)(warp * 32) << 16) + c));
      LDC(0, 1) LDC(1, 16) LDC(2, 31) LDC(3, 32) LDC(4, 33) LDC(5, 48) LDC(6, 56) LDC(7, 63) }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if (lane == 3) { printf("rank%d warp%d lane%d: c0=%g c1=%g c16=%g c31=%g c32=%g c33=%g c48=%g c56=%g c63=%g c40=%g\n", (int)rank, warp, lane, __uint_as_float(a)/16, __uint_as_float(cc[0])/16, __uint_as_float(cc[1])/16, __uint_as_float(cc[2])/16, __uint_as_float(cc[3])/16, __uint_as_float(cc[4])/16, __uint_as_float(cc[5])/16, __uint_as_float(cc[6])/16, __uint_as_float(cc[7])/16, __uint_as_float(b)/16); }
    out[(rank * 128 + warp * 32 + lane) * 2] = __uint_as_float(a);
    out[(rank * 128 + warp * 32 + lane) * 2 + 1] = __uint_as_float(b);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 64;" ::"r"(tb));
}
int main() {
  float* d; cudaMalloc(&d, 2 * 128 * 2 * 4); cudaMemset(d, 0, 2 * 128 * 2 * 4);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 20000);
  k<<<2, 128, 20000>>>(d); cudaError_t e = cudaDeviceSynchronize(); printf("err=%s\n", cudaGetErrorString(e));
  float h[512]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  for (int r = 0; r < 2; ++r) { printf("CTA%d: lane:D/16 (col0 | col40)\n", r); for (int l = 0; l < 128; ++l) printf("%d:%g|%g%s", l, h[(r * 128 + l) * 2] / 16, h[(r * 128 + l) * 2 + 1] / 16, (l % 8 == 7) ? "\n" : "  "); }
}
