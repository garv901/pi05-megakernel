// tma_ing.cu: TMA ingress of the H weight slice in isolation. NCTA CTAs (cluster 2, 1 CTA/SM) each pull BYTES of Wd ([L*1024][4096] bf16, row-major)
// into smem with one of the H load shapes, after a grid-wide start flag; clock64 from issue to mbarrier completion per CTA (median / max over CTAs,
// ns via a globaltimer span). Weights L2-warm (the same slices are read by a warm-up launch) or cold (L2 flushed before the launch).
//   mode 0: 2D box {64, R} per K64 chunk, cta_group::1 (each CTA its own barrier)
//   mode 1: same, cta_group::2 (both CTAs signal rank 0's barrier; rank 0 times)
//   mode 2: 3D box {64, R, NCH} one instruction, cta_group::1
//   mode 3: LDG: 256 threads read the slice with 16 B loads (all loads in flight, consumed by an xor)
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a tma_ing.cu -o tma_ing -lcuda
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
constexpr int W = 1024, FF = 4096, L = 18;

__device__ unsigned g_to;
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
  for (ull i = 0; i < (1ull << 24) && !ok; ++i)
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) atomicAdd(&g_to, 1u);
}

struct Args { CUtensorMap m2[2], m3[2]; const uint4* wd; ull* out; unsigned* flag; int mode, rpc, nch, ncta, layer; };

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256, 1) k_ing(const __grid_constant__ Args a) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) ull bar;
  const int cta = blockIdx.x, rank = cta & 1, tid = threadIdx.x;
  const uint32_t base = (su32(sm) + 1023) & ~1023u, b = su32(&bar);
  const int ri = a.rpc == 128, nrg = W / (2 * a.rpc), pi = cta >> 1, rg = pi % nrg, ks = pi / nrg;
  const int n0 = rg * 2 * a.rpc + rank * a.rpc, k0 = ks * a.nch * 64, row0 = a.layer * W + n0;
  const uint32_t bytes = (uint32_t)a.rpc * 128 * a.nch;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
  // grid start flag: every CTA spins until the host-side counter of arrivals reaches ncta (then all issue together)
  if (tid == 0) {
    atomicAdd(a.flag, 1u);
    while (*(volatile unsigned*)a.flag < (unsigned)gridDim.x) {}
  }
  __syncthreads();
  if (cta >= a.ncta) return;
  ull t0 = 0, t1 = 0;
  if (a.mode <= 2) {
    if (tid == 0) {
      t0 = clock64();
      const bool c2 = a.mode == 1;
      if (!c2 || rank == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(c2 ? 2 * bytes : bytes) : "memory");
      if (a.mode == 2) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                     ::"r"(base), "l"((ull)&a.m3[ri]), "r"(b), "r"(0), "r"(row0), "r"(k0 >> 6) : "memory");
      } else {
        for (int c = 0; c < a.nch; ++c) {
          if (c2)
            asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                         ::"r"(base + c * a.rpc * 128), "l"((ull)&a.m2[ri]), "r"(b & 0xFEFFFFFFu), "r"(k0 + c * 64), "r"(row0) : "memory");
          else
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                         ::"r"(base + c * a.rpc * 128), "l"((ull)&a.m2[ri]), "r"(b), "r"(k0 + c * 64), "r"(row0) : "memory");
        }
      }
      if (!c2 || rank == 0) { mbar_wait(b, 0); t1 = clock64(); }
    }
  } else {
    // LDG: rows n0.. n0 + rpc, columns k0 .. k0 + 64 nch; 16 B per load, all in flight per thread (up to 32), then xor
    const int per_row = a.nch * 8, n = a.rpc * per_row;  // uint4 pieces
    if (tid == 0) t0 = clock64();
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int i0 = tid; i0 < n; i0 += 256 * 16) {
      uint4 v[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int i = i0 + j * 256;
        v[j] = i < n ? __ldg(a.wd + ((size_t)(row0 + i / per_row) * FF + k0) / 8 + i % per_row) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) { acc.x ^= v[j].x; acc.y ^= v[j].y; acc.z ^= v[j].z; acc.w ^= v[j].w; }
    }
    if (acc.x == 0x12345678u) a.out[1023] = acc.y;
    __syncthreads();
    if (tid == 0) t1 = clock64();
  }
  if (tid == 0) { a.out[cta * 2] = t0; a.out[cta * 2 + 1] = t1; }
}
__global__ void k_gt(ull* o, long long spin) {  // ns per cycle: globaltimer over a long clock64 spin
  const ull g0 = ({ ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); t; });
  const long long c0 = clock64();
  while (clock64() - c0 < spin) {}
  const long long c1 = clock64();
  const ull g1 = ({ ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); t; });
  o[0] = g1 - g0; o[1] = (ull)(c1 - c0);
}

int main() {
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  const size_t nwd = (size_t)L * W * FF;
  void* wd; CK(cudaMalloc(&wd, nwd * 2)); CK(cudaMemset(wd, 1, nwd * 2));
  char* flush; const size_t fb = (size_t)2 * prop.l2CacheSize; CK(cudaMalloc(&flush, fb));
  ull* out; CK(cudaMalloc(&out, 4096 * 8)); unsigned* flag; CK(cudaMalloc(&flag, 4));
  PFN_cuTensorMapEncodeTiled_v12000 enc; cudaDriverEntryPointQueryResult qr;
  CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&enc, 12000, cudaEnableDefault, &qr));
  Args a{};
  for (int i = 0; i < 2; ++i) {
    const cuuint32_t r = 64 << i;
    { cuuint64_t d[2] = {FF, (cuuint64_t)L * W}, s[1] = {(cuuint64_t)FF * 2}; cuuint32_t bx[2] = {64, r}, es[2] = {1, 1};
      if (enc(&a.m2[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wd, d, s, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) return 1; }
  }
  ull* gto; CK(cudaMalloc(&gto, 16)); k_gt<<<1, 1>>>(gto, 200000000LL); ull g[2]; CK(cudaMemcpy(g, gto, 16, cudaMemcpyDeviceToHost));
  const double nspc = (double)g[0] / (double)g[1];
  printf("nspc %.4f\n", nspc);
  CK(cudaFuncSetAttribute(k_ing, cudaFuncAttributeMaxDynamicSharedMemorySize, 197632));
  struct Cfg { int mode, rpc, nch, ncta; };
  std::vector<Cfg> cfgs;
  for (int mode = 0; mode < 4; ++mode)
    for (int rpc : {64, 128})
      for (int ncta : {2, 16, 64, 128, 148}) {
        const int nch = rpc == 128 ? 4 : 8;  // 64 KB per CTA (the H r128s16 / r64s8 slices)
        cfgs.push_back({mode, rpc, nch, ncta});
      }
  for (int rpc : {64, 128}) for (int nch : {1, 2, 8}) for (int mode : {0, 2}) cfgs.push_back({mode, rpc, nch, 128});  // slice size sweep at 128 CTAs
  printf("mode rpc nch KB ncta | cold: med_ns max_ns GBps/SM(med) agg_TBps(max) | warm: med_ns max_ns GBps/SM agg_TBps\n");
  for (auto c : cfgs) {
    // 3D map for this nch
    { cuuint64_t d[3] = {64, (cuuint64_t)L * W, FF / 64}, s[2] = {(cuuint64_t)FF * 2, 128}; cuuint32_t bx[3] = {64, (cuuint32_t)c.rpc, (cuuint32_t)c.nch}, es[3] = {1, 1, 1};
      if (enc(&a.m3[c.rpc == 128], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, wd, d, s, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) return 1; }
    if ((W / (2 * c.rpc)) * (FF / (c.nch * 64)) * 2 < c.ncta) continue;  // not enough distinct slices
    a.wd = (const uint4*)wd; a.out = out; a.flag = flag; a.mode = c.mode; a.rpc = c.rpc; a.nch = c.nch; a.ncta = c.ncta; a.layer = 3;
    double res[2][2];
    for (int warm = 0; warm < 2; ++warm) {
      std::vector<double> med, mx;
      for (int rep = 0; rep < 7; ++rep) {
        CK(cudaMemset(flush, rep, fb));
        if (warm) {  // warm-up launch reads the same slices
          CK(cudaMemset(flag, 0, 4));
          k_ing<<<148, 256, 197632>>>(a);
        }
        CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(out, 0, 4096 * 8));
        k_ing<<<148, 256, 197632>>>(a);
        CK(cudaDeviceSynchronize());
        std::vector<ull> o(2 * 148); CK(cudaMemcpy(o.data(), out, o.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<double> d;
        for (int i = 0; i < c.ncta; ++i) if (o[2 * i + 1]) d.push_back((double)(o[2 * i + 1] - o[2 * i]) * nspc);
        std::sort(d.begin(), d.end());
        med.push_back(d[d.size() / 2]); mx.push_back(d.back());
      }
      std::sort(med.begin(), med.end()); std::sort(mx.begin(), mx.end());
      res[warm][0] = med[3]; res[warm][1] = mx[3];
    }
    const double kb = c.rpc * 128.0 * c.nch / 1024;
    printf("%d %3d %d %4.0f %3d | %7.0f %7.0f %6.1f %5.2f | %7.0f %7.0f %6.1f %5.2f\n", c.mode, c.rpc, c.nch, kb, c.ncta, res[0][0], res[0][1], kb * 1024 / res[0][0],
           kb * 1024 * c.ncta / res[0][1] / 1e3, res[1][0], res[1][1], kb * 1024 / res[1][0], kb * 1024 * c.ncta / res[1][1] / 1e3);
  }
  unsigned to; CK(cudaMemcpyFromSymbol(&to, g_to, 4)); printf("timeouts %u\n", to);
  return 0;
}
