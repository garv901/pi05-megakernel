// U9: gemm_phase warp-task latency vs k-depth, back-to-back tasks and the stream-K fixup (go / no-go for hybrid stream-K).
// 148 CTAs x 256 thr (1 CTA/SM, 117 KB smem like the megakernel); per phase: counter grid barrier, then each busy warp runs
// gemm_phase's tile body on cold weights: 16 x LDG.128 per 32-k slab (b[sl][nt], register-resident), 32 MMAs m16n8k16 per
// slab against a fragment-interleaved A tile in smem, partial stores of rows < 50 (or red.add of them: stream-K fixup), a
// warp fence (stores performed) and the counter publish. Task 1's weights are issued before the barrier when PRE (WPRE);
// a second task's after task 1's last MMA (LOAD_HOIST). Per warp: t0 = after the barrier, t1 = after the fence.
// Weights: [9728 rows][1024] bf16 per phase (19.9 MB, F-sized), block b reads k chunk (b % 8) like gemm_phase; the pool
// (4 GB) is walked so every phase is cold.
// clk_*: the same intervals in clock64 (U11 found globaltimer t0 reads stale by 1-2 us on most warps); clk_period = one
// phase (barrier + task) on one warp, clock64 between successive barrier exits.
// pre (flags): 32 = empty task (floor; U10 addition); 1 = task 1's weights issued before the barrier (WPRE); 2 = busy warps spread warp-major over all CTAs
// (warp * 148 + block < nbusy) instead of filling CTAs 0.. in order; 4 = no weight loads (b constant: MMA + stores);
// 8 = no partial stores (MMAs kept: asm volatile); 16 = t1 before the fence (stores issued, not yet performed).
// Build: nvcc -O3 -std=c++17 -arch=sm_100a gemm_task.cu -o gemm_task      Run: gemm_task [nph=100] [reps=5]
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int NT = 256, NW = NT / 32, GRID = 148, K = 1024, ROWS = 9728, T = 50;
constexpr size_t PHASE_BYTES = (size_t)ROWS * K * 2;
__device__ __forceinline__ unsigned long long clk() { unsigned long long t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((unsigned)__cvta_generic_to_shared(p))); return v; }
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }

__device__ __forceinline__ void gbar(unsigned* cnt, unsigned epoch) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    while ((int)(ld_acquire(cnt) - epoch * gridDim.x) < 0) {}
  }
  __syncthreads();
}

template <int SL>
__device__ __forceinline__ void load_w(uint4 (&b)[4][4], const char* wb, int row0, int k0, int g, int t4) {
#pragma unroll
  for (int sl = 0; sl < SL; ++sl)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      b[sl][nt] = __ldg(reinterpret_cast<const uint4*>(wb + ((size_t)(row0 + nt * 8 + g) * K + k0 + sl * 32 + t4 * 8) * 2));
}

template <int SL>
__device__ __forceinline__ void mmas(float (&acc)[4][4][4], const uint4 (&b)[4][4], const uint4* A4, int lane) {
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
  for (int sl = 0; sl < SL; ++sl)
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const uint4* af = A4 + (mt * 4 + sl) * 2 * 32 + lane;
      const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[sl][nt].x, b[sl][nt].y);
        mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[sl][nt].z, b[sl][nt].w);
      }
    }
}

// MODE 0: float2 stores (gemm_phase); 1: atomicAdd(float2) into a slot shared by `share` warps (stream-K fixup);
// 2: bf16x2 stores (half the partial bytes)
template <int MODE>
__device__ __forceinline__ void store(const float (&acc)[4][4][4], float* dst, int g, int t4) {
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = mt * 16 + g, c = nt * 8 + 2 * t4;
      float2* p0 = reinterpret_cast<float2*>(dst + r0 * 32 + c);
      float2* p1 = reinterpret_cast<float2*>(dst + (r0 + 8) * 32 + c);
      if (MODE == 2) {
        __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(dst);
        if (r0 < T) *reinterpret_cast<__nv_bfloat162*>(q + r0 * 32 + c) = __floats2bfloat162_rn(acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T)
          *reinterpret_cast<__nv_bfloat162*>(q + (r0 + 8) * 32 + c) = __floats2bfloat162_rn(acc[mt][nt][2], acc[mt][nt][3]);
      } else if (MODE == 0) {
        if (r0 < T) *p0 = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T) *p1 = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
      } else {
        if (r0 < T) atomicAdd(p0, make_float2(acc[mt][nt][0], acc[mt][nt][1]));
        if (r0 + 8 < T) atomicAdd(p1, make_float2(acc[mt][nt][2], acc[mt][nt][3]));
      }
    }
}

// SL1: 32-k slabs of task 1 (1..4); SL2: of task 2 (0 = none); MODE: how the last task's partials land (task 1 of two
// always stores: it is the data-parallel tile)
template <int SL1, int SL2, int MODE>
__global__ void __launch_bounds__(NT, 1) k_task(const char* __restrict__ pool, size_t pool_bytes, int nph, int phase0,
                                                 int nbusy, int share, int pre, unsigned* cnt, float* part,
                                                 unsigned long long* tm, unsigned long long* tc) {
  extern __shared__ uint4 A4[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t4 = lane & 3;
  const int gw = blockIdx.x * NW + warp, s = blockIdx.x % 8;
  const bool busy = (pre & 2) ? warp * GRID + (int)blockIdx.x < nbusy : gw < nbusy;
  const bool ld = !(pre & 4), st = !(pre & 8);
  for (int i = threadIdx.x; i < 4 * 4 * 2 * 32; i += NT) A4[i] = make_uint4(0x3f803f80u + i, 0x3f003f00u, 0x3e803e80u, i);
  __syncthreads();
  const int row1 = ((blockIdx.x / 8) * NW + warp) * 32, row2 = row1 + (GRID / 8 + 1) * NW * 32;
  const int k0 = s * 128;
  float* dst1 = part + (size_t)gw * T * 32 * 2;
  float* dst2 = part + (size_t)(gw / share) * T * 32 * 2 + T * 32;
  for (int p = 0; p < nph; ++p) {
    const char* wb = pool + ((size_t)(phase0 + p) * PHASE_BYTES) % (pool_bytes - PHASE_BYTES);
    uint4 b[4][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) b[i / 4][i % 4] = make_uint4(0x3f803f80u + lane, 0x3f803f80u, 0x3f003f00u, i);
    if (busy && ld && (pre & 1)) load_w<SL1>(b, wb, row1, k0, g, t4);
    gbar(cnt, p + 1);
    const unsigned long long t0 = gt(), c0 = clk();
    if (busy && (pre & 32)) {  // empty task (floor): barrier + fence + t1 only
      if (lane == 0) { __threadfence(); const unsigned long long t1 = gt(), c1 = clk(); tm[((size_t)p * GRID * NW + gw) * 2] = t0; tm[((size_t)p * GRID * NW + gw) * 2 + 1] = t1;
                       tc[((size_t)p * GRID * NW + gw) * 2] = c0; tc[((size_t)p * GRID * NW + gw) * 2 + 1] = c1; }
      continue;
    }
    if (busy) {
      if (ld && !(pre & 1)) load_w<SL1>(b, wb, row1, k0, g, t4);
      float acc[4][4][4];
      mmas<SL1>(acc, b, A4, lane);
      if (SL2) {
        if (ld) load_w<SL2 ? SL2 : 1>(b, wb, row2, k0, g, t4);  // hoisted: b is dead after task 1's last MMA
        if (st) store<MODE == 2 ? 2 : 0>(acc, dst1, g, t4);
        mmas<SL2 ? SL2 : 1>(acc, b, A4, lane);
        if (st) store<MODE>(acc, dst2, g, t4);
      } else if (st) {
        store<MODE>(acc, MODE ? dst2 : dst1, g, t4);
      }
      if (!st && acc[0][0][0] == -1.2345e30f) dst1[lane] = acc[3][3][3];
      __syncwarp();
      if (lane == 0) {
        unsigned long long t1 = gt(), c1 = clk();
        __threadfence();
        if (!(pre & 16)) { t1 = gt(); c1 = clk(); }
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt + 32 + (gw & 63) * 32) : "memory");
        tm[((size_t)p * GRID * NW + gw) * 2] = t0;
        tm[((size_t)p * GRID * NW + gw) * 2 + 1] = t1;
        tc[((size_t)p * GRID * NW + gw) * 2] = c0;
        tc[((size_t)p * GRID * NW + gw) * 2 + 1] = c1;
      }
    }
  }
}

static double pctl(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; }

struct Cfg { const char* name; void (*k)(const char*, size_t, int, int, int, int, int, unsigned*, float*, unsigned long long*, unsigned long long*);
             int nbusy, share, pre; };

int main(int argc, char** argv) {
  const int nph = argc > 1 ? atoi(argv[1]) : 100, reps = argc > 2 ? atoi(argv[2]) : 5;
  const size_t POOL = 4ull << 30;
  char* pool; unsigned* cnt; float* part; unsigned long long* tm; unsigned long long* tc;
  CK(cudaMalloc(&pool, POOL)); CK(cudaMalloc(&cnt, 64 * 1024)); CK(cudaMalloc(&part, (size_t)GRID * NW * T * 32 * 2 * 4));
  CK(cudaMalloc(&tm, (size_t)nph * GRID * NW * 16)); CK(cudaMalloc(&tc, (size_t)nph * GRID * NW * 16));
  CK(cudaMemset(pool, 0x3c, POOL)); CK(cudaMemset(part, 0, (size_t)GRID * NW * T * 32 * 2 * 4));
  const int smem = 117 * 1024, ALL = GRID * NW, A_BUSY = 80 * NW;
  std::vector<Cfg> cfgs = {
    // fp32 (today) vs bf16 partials; +16: time at store issue (before the fence) to split issue from drain
    {"k128_f32", k_task<4, 0, 0>, ALL, 1, 1}, {"k128_bf16", k_task<4, 0, 2>, ALL, 1, 1},
    {"k128_f32_issue", k_task<4, 0, 0>, ALL, 1, 17}, {"k128_bf16_issue", k_task<4, 0, 2>, ALL, 1, 17},
    {"k128_nostore", k_task<4, 0, 0>, ALL, 1, 9},
    {"A_packed_f32", k_task<4, 0, 0>, A_BUSY, 1, 1}, {"A_packed_bf16", k_task<4, 0, 2>, A_BUSY, 1, 1},
    {"A_spread_f32", k_task<4, 0, 0>, A_BUSY, 1, 3}, {"A_spread_bf16", k_task<4, 0, 2>, A_BUSY, 1, 3},
    {"A_spread_nostore", k_task<4, 0, 0>, A_BUSY, 1, 11},
    {"H_spread_f32", k_task<4, 0, 0>, 1024, 1, 3}, {"H_spread_bf16", k_task<4, 0, 2>, 1024, 1, 3},
    {"F_f32", k_task<4, 4, 0>, ALL, 1, 1}, {"F_bf16", k_task<4, 4, 2>, ALL, 1, 1},
    {"F_f32_again", k_task<4, 4, 0>, ALL, 1, 1},
    // added for U10: weights issued after the barrier (pre0), and the empty-task floor
    {"k128_f32_pre0", k_task<4, 0, 0>, ALL, 1, 0}, {"A_spread_f32_pre0", k_task<4, 0, 0>, A_BUSY, 1, 2}, {"H_spread_f32_pre0", k_task<4, 0, 0>, 1024, 1, 2},
    {"F_f32_pre0", k_task<4, 4, 0>, ALL, 1, 0}, {"empty", k_task<4, 0, 0>, ALL, 1, 32},
  };


  std::vector<unsigned long long> h((size_t)nph * GRID * NW * 2), hc(h.size());
  int phase0 = 0;
  for (auto& c : cfgs) {
    CK(cudaFuncSetAttribute(c.k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    std::vector<double> dur, crit, cdur, per, nspc;  // c*: clock64 (globaltimer t0 reads can be stale by ~1-2 us)
    for (int r = 0; r < reps; ++r) {
      CK(cudaMemset(cnt, 0, 64 * 1024)); CK(cudaMemset(tm, 0, (size_t)nph * GRID * NW * 16)); CK(cudaMemset(tc, 0, (size_t)nph * GRID * NW * 16));
      c.k<<<GRID, NT, smem>>>(pool, POOL, nph, phase0, c.nbusy, c.share, c.pre, cnt, part, tm, tc);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      phase0 += nph;
      CK(cudaMemcpy(h.data(), tm, h.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hc.data(), tc, hc.size() * 8, cudaMemcpyDeviceToHost));
      double k = 0;  // ns per cycle: first busy warp, globaltimer vs clock over phases 2..nph-1
      for (int w = 0; w < GRID * NW && !k; ++w) {
        const size_t a = ((size_t)2 * GRID * NW + w) * 2, b = ((size_t)(nph - 1) * GRID * NW + w) * 2;
        if (hc[a + 1] && hc[b + 1]) k = (double)(h[b] - h[a]) / (double)(hc[b] - hc[a]);
      }
      nspc.push_back(k);
      for (int p = 2; p < nph; ++p) {  // the first phases warm the instruction cache / TLB
        unsigned long long lo = ~0ull, hi = 0;
        for (int w = 0; w < GRID * NW; ++w) {
          const unsigned long long t0 = h[((size_t)p * GRID * NW + w) * 2], t1 = h[((size_t)p * GRID * NW + w) * 2 + 1];
          if (!t1) continue;  // idle warp
          dur.push_back((double)(t1 - t0)); lo = std::min(lo, t0); hi = std::max(hi, t1);
          const size_t i = ((size_t)p * GRID * NW + w) * 2;
          cdur.push_back((double)(hc[i + 1] - hc[i]) * k);
          if (p + 1 < nph && hc[i + 2 * GRID * NW]) per.push_back((double)(hc[i + 2 * GRID * NW] - hc[i]) * k);
        }
        crit.push_back((double)(hi - lo));
      }
    }
    printf("{\"u\":\"U9\",\"cfg\":\"%s\",\"busy_warps\":%d,\"share\":%d,\"pre\":%d,\"warp_ns_p50\":%.0f,\"warp_ns_p90\":%.0f,"
           "\"warp_ns_max_p50\":%.0f,\"phase_crit_ns_p50\":%.0f,\"phase_crit_ns_p90\":%.0f,\"ns_per_cycle\":%.4f,\"clk_warp_ns_p50\":%.0f,"
           "\"clk_warp_ns_p90\":%.0f,\"clk_warp_ns_max\":%.0f,\"clk_period_ns_p50\":%.0f}\n",
           c.name, c.nbusy, c.share, c.pre, pctl(dur, .5), pctl(dur, .9), pctl(dur, .999), pctl(crit, .5), pctl(crit, .9),
           pctl(nspc, .5), pctl(cdur, .5), pctl(cdur, .9), pctl(cdur, .999), pctl(per, .5));
    fflush(stdout);
  }
  return 0;
}
