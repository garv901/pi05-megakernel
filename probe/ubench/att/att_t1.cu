// att_t1.cu: unit test of the tcgen05 primitives used by att_k1.cu on one cluster-of-2 pair (cta_group::2, M = 256):
//   (1) S = Q K^T: A = Q (128 rows per CTA, K-major SW128, 4 x 16 KB chunks), B = K (32 keys per CTA, K-major, 4 x 4 KB), N = 64, D at TMEM col 256
//   (2) O = P V: A = P from TMEM (bf16 packed, written with tcgen05.st), B = V MN-major SW128 (hd half per CTA: 2 x [64 keys][64 hd] tiles, LBO = 8 KB), N = 256
// All data are small integers (exact in bf16 / fp32); prints mismatch counts. Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a att_t1.cu -o att_t1
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "att_k1_ptx.cuh"
typedef __nv_bfloat16 bf16;
typedef unsigned long long ull;
__device__ __host__ inline float qv(int m, int k) { return (float)(((m * 3 + k * 5) % 7) - 3); }
__device__ __host__ inline float kv(int n, int k) { return (float)(((n * 7 + k * 3) % 5) - 2); }
__device__ __host__ inline float vv(int key, int hd) { return (float)(((key * 5 + hd * 3) % 7) - 3); }
__device__ __host__ inline float pv(int m, int key) { return (float)((m * 5 + key * 3) % 4); }
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) tk(float* S, float* O, int amode) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) ull bar_s, bar_o;
  __shared__ uint32_t slot;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cluster_rank();
  const uint32_t base = (su32(sm) + 1023) & ~1023u; unsigned char* gp = sm + (base - su32(sm));
  const uint32_t Qb = base, Kb = base + 65536, Vb = base + 65536 + 16384;
  auto put = [&](unsigned char* t, int row, int k, float v) {  // K-major SW128: row = 128 B, 16 B chunk ^ (row & 7)
    *reinterpret_cast<bf16*>(t + row * 128 + ((((k * 2) >> 4) ^ (row & 7)) << 4) + ((k * 2) & 15)) = __float2bfloat16(v); };
  for (int i = tid; i < 128 * 256; i += 128) { const int m = i >> 8, k = i & 255; put(gp + (k >> 6) * 16384, m, k & 63, qv(rank * 128 + m, k)); }
  for (int i = tid; i < 32 * 256; i += 128) { const int n = i >> 8, k = i & 255; put(gp + 65536 + (k >> 6) * 4096, n, k & 63, kv(rank * 32 + n, k)); }
  for (int i = tid; i < 64 * 128; i += 128) { const int key = i >> 7, c = i & 127; put(gp + 65536 + 16384 + (c >> 6) * 8192, key, c & 63, vv(key, rank * 128 + c)); }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (tid == 0) { mbar_init(su32(&bar_s), 1); mbar_init(su32(&bar_o), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (warp == 1) tmem_alloc2(su32(&slot), 512);
  fence_before();
  cluster_sync_all();
  fence_after();
  const uint32_t tb = slot;
  if (rank == 0 && warp == 1 && elect_one()) {
    for (int kc = 0; kc < 4; ++kc)
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) mma_ss2(tb + 256, sdesc(Qb + kc * 16384) + (ull_t)((kk * 32) >> 4), sdesc(Kb + kc * 4096) + (ull_t)((kk * 32) >> 4), idesc(64, 256), (kc | kk) != 0);
    commit2(su32(&bar_s));
  }
  { uint32_t par = 0; uint32_t b = su32(&bar_s); long it = 0; while (!mbar_try(b, par) && ++it < (1l << 24)) {} if (it >= (1l << 24)) printf("timeout bar_s cta %d warp %d\n", (int)rank, warp); }
  fence_after();
  const uint32_t lq = ((uint32_t)(warp * 32) << 16);
  {
    uint32_t v[64];
    tmem_ld32_x64(tb + 256 + lq, v); wait_ld();
    for (int c = 0; c < 64; ++c) S[((size_t)(rank * 128 + warp * 32 + lane)) * 64 + c] = __uint_as_float(v[c]);
  }
  // P
  {
    uint32_t pw[32];
    const int m = rank * 128 + warp * 32 + lane;
    for (int j = 0; j < 32; ++j) { __nv_bfloat162 b2 = __floats2bfloat162_rn(pv(m, 2 * j), pv(m, 2 * j + 1)); pw[j] = *reinterpret_cast<uint32_t*>(&b2); }
    tmem_st32_x32(tb + 256 + lq, pw); wait_st(); fence_before();
  }
  cluster_sync_all();
  fence_after();
  if (rank == 0 && warp == 1 && elect_one()) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) mma_ts2(tb, tb + 256 + 8 * kk, sdesc_mn(Vb + kk * 2048, 8192, 1024), idesc(256, 256, 1), kk != 0);
    commit2(su32(&bar_o));
  }
  { uint32_t b = su32(&bar_o); long it = 0; while (!mbar_try(b, 0) && ++it < (1l << 24)) {} if (it >= (1l << 24)) printf("timeout bar_o cta %d warp %d\n", (int)rank, warp); }
  fence_after();
  for (int h = 0; h < 4; ++h) {
    uint32_t v[64];
    tmem_ld32_x64(tb + h * 64 + lq, v); wait_ld();
    for (int c = 0; c < 64; ++c) O[((size_t)(rank * 128 + warp * 32 + lane)) * 256 + h * 64 + c] = __uint_as_float(v[c]);
  }
  fence_before();
  cluster_sync_all();
  if (warp == 1) tmem_free2(tb, 512);
}
int main() {
  float *dS, *dO; cudaMalloc(&dS, 256 * 64 * 4); cudaMalloc(&dO, 256 * 256 * 4);
  cudaMemset(dS, 0xff, 256 * 64 * 4); cudaMemset(dO, 0xff, 256 * 256 * 4);
  const int sm = 65536 + 16384 + 16384 + 1024;
  cudaFuncSetAttribute(tk, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  tk<<<2, 128, sm>>>(dS, dO, 0);
  cudaError_t e = cudaDeviceSynchronize(); printf("err=%s\n", cudaGetErrorString(e));
  std::vector<float> S(256 * 64), O(256 * 256);
  cudaMemcpy(S.data(), dS, S.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(O.data(), dO, O.size() * 4, cudaMemcpyDeviceToHost);
  int badS = 0, badO = 0, shown = 0;
  for (int m = 0; m < 256; ++m) for (int n = 0; n < 64; ++n) {
    float r = 0; for (int k = 0; k < 256; ++k) r += qv(m, k) * kv(n, k);
    if (S[m * 64 + n] != r) { if (shown++ < 6) printf("S[%d][%d] got %g want %g\n", m, n, S[m * 64 + n], r); ++badS; } }
  shown = 0;
  for (int m = 0; m < 256; ++m) for (int d = 0; d < 256; ++d) {
    float r = 0; for (int key = 0; key < 64; ++key) r += pv(m, key) * vv(key, d);
    if (O[m * 256 + d] != r) { if (shown++ < 6) printf("O[%d][%d] got %g want %g\n", m, d, O[m * 256 + d], r); ++badO; } }
  printf("S mismatches %d / %d, O mismatches %d / %d\n", badS, 256 * 64, badO, 256 * 256);
}
