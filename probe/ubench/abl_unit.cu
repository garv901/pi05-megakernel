// MB2: standalone microbenchmark of the A -> C-release window of the pi0.5 expert layer (split-K + A->B barrier + B vs the fused A+B
// unit of FUSE_ABL).  Spec: probe/specs/MB2_abl_unit.md.  148 CTAs x 256 thr (1 CTA/SM), cooperative launch, persistent over --iters iterations.
// Build: nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v probe/ubench/abl_unit.cu -o abl_unit
//
// One iteration = [prev phase: --fresh residual tasks + cp.async drain] -> G1 (production WPRE barrier: arrive, weight loads into registers,
// acquire wait) stamp s0 -> [--jitter_ns spin] s1 -> variant work s2..s6 -> G2 (plain counter barrier) s7.  Stamps: thread 0 after a CTA sync and a
// volatile ld.shared (the "settle" of cute/iket_marks.py), then %globaltimer.  stamps[iter][cta][8] u64.  A CTA that does not reach a stamp
// stores the previous one.
//
// V0   production split-K (fuse 18): CTA bid owns K chunk bid % 8 (every CTA stages it, as gemm_phase does), warp tile ng = (bid / 8) * 8 + warp,
//      tiles ng < 80 (CTAs 0..79); one-round register staging with adaRMS norm, fragment-interleaved tile (PI05_GEMM_FRAG == 2), MMA, fp32 partial
//      part[s][t][n]; then the A->B grid barrier, or with --abc 1 per-head counters (red.release per tile, block_wait target 64 x epoch);
//      L2 prefetch share, prefill (CTAs bid < --prefill_ctas, past the A tile), B: 70 tasks on CTAs 147 - task (reduce_part4 of 8 partials, rbf, RoPE).
// V1   fused unit (fuse 50, cute/abl.py): 80 units u = 8 hh + p on CTA place(u); columns 256 hh + 16 p + [0,16) and 256 hh + 128 + 16 p + [0,16).
//      Warp w = K chunk w.  All 8 chunks are staged with norm into 8 private tiles (--stage reg --la L: L chunks of raw loads in flight while
//      the current one is normalised, L = 0 fully serial, L = 1 = stage_a8; --stage cpa: all raw rows by cp.async into 8 row-major tiles
//      [64][160] bf16, wait, in-place normalise, PI05_GEMM_FRAG == 0 operand loads).  MMA, fp32 partials to smem [8][50][40] reusing the
//      tiles, 200 reduce tasks (sum s = 0..7 from +0.0, rbf, RoPE, stores), then (all CTAs) prefetch share and prefill.  --norm 0 drops the
//      normalisation arithmetic (the loads are kept).  --active N: only units u < N work.
// H    hybrid, --c in {1, 2, 4}: G = 8 / c K groups; unit u -> group gi = u / G, kg = u % G.  Warp w: pair tile gi G + w / c, chunk kg c + w % c.
//      The CTA stages its c chunks (register, la 1), MMA, fp32 partial to part[chunk][t][virtual column] (V0 layout), red.release on
//      counter[gi], block_wait(G x epoch); the G CTAs split the group's rows (kg: rows [50 kg / G, 50 (kg + 1) / G)) and reduce all 8
//      partials in order s = 0..7 from global.  (--active with a group only partly active waits for the active units only; the outputs of such
//      a run are not meaningful, the timing is.)
// ctl  idle control: G1, jitter, prefetch + prefill, G2.
//
// Barrier epochs: one counter word, targets ep * 148 with ep incremented per barrier (G1, [A->B], G2), identical on every CTA.  --abc / H
// counters use targets (iteration + 1) * arrivals.  The host zeroes counters and the barrier word before every launch.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef __nv_bfloat16 bf16;

constexpr int NT = 256, T = 50, W = 1024, KC = 128, NCH = 8, NSSP = 16, NV = 2560, HD = 256, NCTA = 148, NUNITS = 80;
constexpr int TILE_BYTES = 64 * KC * 2;            // fragment-interleaved A tile (16 KB)
constexpr int AS_STRIDE = KC + 32;                  // row-major tile stride (bf16)
constexpr int CPA_TILE_BYTES = 64 * AS_STRIDE * 2;  // 20 KB
constexpr int PB_STRIDE = 40;                       // fp32 partial buffer row stride (smem)
constexpr int CNT_STRIDE = 32;                      // one 128 B line per counter
constexpr int NRES = 800;                           // residual tasks (50 rows x 16 parts)
enum { V0 = 0, V1 = 1, VH = 2, VCTL = 3 };
enum { STG_REG = 0, STG_CPA = 1 };

// ---------------------------------------------------------------- helpers (copied from mk/pi05_mk.cu)
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* ptr) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory"); return v; }
__device__ __forceinline__ void red_release(unsigned* ptr) { asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(ptr) : "memory"); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* ptr) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(ptr) : "memory"); return v; }
__device__ __forceinline__ void spin_acquire(const unsigned* ptr, unsigned target) { while ((int)(ld_acquire(ptr) - target) < 0) {} }
// Whole block waits until *c >= target (thread 0 relaxed spin + acquire fence, CTA barrier).
__device__ __forceinline__ void block_wait(const unsigned* c, unsigned target) {
  if (threadIdx.x == 0) {
    while (ld_relaxed(c) < target) {}
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  }
  __syncthreads();
}
__device__ __forceinline__ void wpre_arrive(unsigned* bar) {
  __syncthreads();
  if (threadIdx.x == 0) { unsigned old; asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(bar) : "memory"); asm volatile("" ::"r"(old)); }
}
__device__ __forceinline__ void wpre_wait(unsigned* bar, unsigned target) {
  if (threadIdx.x == 0) spin_acquire(bar, target);
  __syncthreads();
}
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned epoch) { wpre_arrive(bar); wpre_wait(bar, epoch * gridDim.x); }
__device__ __forceinline__ void prefetch_l2(const void* base, size_t bytes) {
  constexpr size_t CH = 128;
  const size_t n = bytes / CH;
  const size_t g0 = (size_t)blockIdx.x * NT + threadIdx.x, nt = (size_t)gridDim.x * NT;
  const char* b = reinterpret_cast<const char*>(base);
  for (size_t i = g0; i < n; i += nt) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(b + i * CH));
}
__device__ __forceinline__ uint4 lds128(const void* smem_ptr) {
  uint4 r;
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}
// Staging loads / stores are volatile (PTX keeps volatile accesses in program order): without it ptxas hoists every chunk's raw loads
// to the top of the staging code (SASS check; ld.relaxed.gpu is hoisted too), which would turn the --la knob into a no-op. SASS: LDG.E.128.STRONG.SYS.
__device__ __forceinline__ uint4 ldv128(const void* p) {
  uint4 v; asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void sts128v(void* smem_ptr, uint4 v) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("st.volatile.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
template <int LDV>
__device__ __forceinline__ void ldv_nmod(float4& sc0, float4& sc1, float4& sh0, float4& sh1, const float* nmod, int k0pc) {
  const float4* sp = reinterpret_cast<const float4*>(nmod + k0pc);
  if constexpr (!LDV) { sc0 = __ldcg(sp); sc1 = __ldcg(sp + 1); sh0 = __ldcg(sp + W / 4); sh1 = __ldcg(sp + W / 4 + 1); return; }
  const uint4 a = ldv128(sp), b = ldv128(sp + 1), c = ldv128(sp + W / 4), d = ldv128(sp + W / 4 + 1);
  sc0 = *reinterpret_cast<const float4*>(&a); sc1 = *reinterpret_cast<const float4*>(&b); sh0 = *reinterpret_cast<const float4*>(&c); sh1 = *reinterpret_cast<const float4*>(&d);
}
__device__ __forceinline__ uint2 pack_bf16x4(float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  return make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}

// unit -> CTA placement (--place 0 = hi: 147 - u; 1 = spread: u * 148 / 80); returns the unit of CTA bid or -1 (units u < active only)
__host__ __device__ __forceinline__ int unit_of(int bid, int place, int active) {
  int u;
  if (place == 0) u = NCTA - 1 - bid;
  else { u = (bid * NUNITS + NCTA - 1) / NCTA; if (u * NCTA / NUNITS != bid) u = -1; }
  return (u >= 0 && u < NUNITS && u < active) ? u : -1;
}

struct Args {
  bf16* h; const bf16* h0; float* ssp; const float* nmod; const bf16* w; const float* rope; float* part; bf16* out;
  const bf16* pool_k; const bf16* pool_v; const char* pf_pool; unsigned* cnt; unsigned* bar; unsigned long long* stamps; unsigned* sink;
  size_t pf_bytes;
  int iters, fresh, jitter_ns, prefill_keys, prefill_ctas, place, active, pf_pool_n, pf_it0, abc;
};

// ---------------------------------------------------------------- staging (production gemm_phase, PI05_GEMM_FRAG == 2 layout)
__device__ __forceinline__ int row_of(int x, int q) { return (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7); }
struct Raw { uint4 av[4]; float4 sc0, sc1, sh0, sh1; };

// raw loads of chunk k0: this thread's 4 h pieces (16 B) and its 8 scale / 8 shift floats
template <int LDV>
__device__ __forceinline__ void load_raw(Raw& r, const bf16* h, const float* nmod, int k0) {
  const int q = threadIdx.x >> 4, pc = threadIdx.x & 15;
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    const int row = row_of(x, q);
    r.av[x] = row < T ? (LDV ? ldv128(h + (size_t)row * W + k0 + pc * 8) : __ldcg(reinterpret_cast<const uint4*>(h + (size_t)row * W + k0 + pc * 8))) : make_uint4(0, 0, 0, 0);
  }
  ldv_nmod<LDV>(r.sc0, r.sc1, r.sh0, r.sh1, nmod, k0 + pc * 8);
}
// rs per row piece x (production: 16 lane groups of ssp parts, fixed shuffle tree, broadcast from the half-warp's lane 0). RM = row-major row mapping.
template <bool RM>
__device__ __forceinline__ void compute_rs(float (&rs)[4], const float* ssp) {
  const int pc = threadIdx.x & 15, r0 = threadIdx.x >> 4;
  float ssv[4];
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    const int r = RM ? r0 + x * 16 : row_of(x, r0);
    ssv[x] = r < T ? __ldcg(ssp + r * NSSP + pc) : 0.f;
  }
  const int hbase = threadIdx.x & 16;
#pragma unroll
  for (int x = 0; x < 4; ++x) {
    float v = ssv[x];
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    v = __shfl_sync(0xffffffffu, v, hbase);
    rs[x] = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
  }
}
__device__ __forceinline__ void norm_scalars(const Raw& r, float (&sc)[8], float (&sh)[8]) {
  sc[0] = __fadd_rn(1.0f, r.sc0.x); sc[1] = __fadd_rn(1.0f, r.sc0.y); sc[2] = __fadd_rn(1.0f, r.sc0.z); sc[3] = __fadd_rn(1.0f, r.sc0.w);
  sc[4] = __fadd_rn(1.0f, r.sc1.x); sc[5] = __fadd_rn(1.0f, r.sc1.y); sc[6] = __fadd_rn(1.0f, r.sc1.z); sc[7] = __fadd_rn(1.0f, r.sc1.w);
  sh[0] = r.sh0.x; sh[1] = r.sh0.y; sh[2] = r.sh0.z; sh[3] = r.sh0.w; sh[4] = r.sh1.x; sh[5] = r.sh1.y; sh[6] = r.sh1.z; sh[7] = r.sh1.w;
}
__device__ __forceinline__ void norm_piece(uint4& v, float rs, const float (&sc)[8], const float (&sh)[8]) {
  uint32_t* wv = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    __nv_bfloat162 hv2 = *reinterpret_cast<__nv_bfloat162*>(&wv[e]);
    const float o0 = __fadd_rn(__fmul_rn(__fmul_rn(__low2float(hv2), rs), sc[2 * e]), sh[2 * e]);
    const float o1 = __fadd_rn(__fmul_rn(__fmul_rn(__high2float(hv2), rs), sc[2 * e + 1]), sh[2 * e + 1]);
    hv2 = __floats2bfloat162_rn(o0, o1);
    wv[e] = *reinterpret_cast<uint32_t*>(&hv2);
  }
}
// normalise (or, NORM == 0, just consume the loaded scale / shift) and store the chunk into a fragment-interleaved tile
template <int NORM, int LDV>
__device__ __forceinline__ void norm_store(Raw& r, const float (&rs)[4], char* tile, uint32_t& sink) {
  const int q = threadIdx.x >> 4, pc = threadIdx.x & 15;
  if constexpr (NORM) {
    float sc[8], sh[8];
    norm_scalars(r, sc, sh);
#pragma unroll
    for (int x = 0; x < 4; ++x)
      if (row_of(x, q) < T) norm_piece(r.av[x], rs[x], sc, sh);
  } else {
    sink ^= __float_as_uint(r.sc0.x) ^ __float_as_uint(r.sc0.w) ^ __float_as_uint(r.sc1.y) ^ __float_as_uint(r.sh0.z) ^ __float_as_uint(r.sh1.w);
  }
  uint4* A4 = reinterpret_cast<uint4*>(tile);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint4 u = r.av[2 * j], v = r.av[2 * j + 1];
    uint4* dst = A4 + ((j * 2 + (q >> 3)) * 4 + (pc >> 2)) * 2 * 32 + (q & 7) * 4 + (pc & 3);
    if constexpr (LDV) {
      sts128v(dst, make_uint4(u.x, v.x, u.y, v.y));
      sts128v(dst + 32, make_uint4(u.z, v.z, u.w, v.w));
    } else {
      dst[0] = make_uint4(u.x, v.x, u.y, v.y);
      dst[32] = make_uint4(u.z, v.z, u.w, v.w);
    }
  }
}
// Register staging of NC chunks chunk0 .. chunk0 + NC - 1 into tiles 0 .. NC - 1 (tile c = smem + c * TILE_BYTES).  LA chunks beyond the current
// one are in flight (ring of LA + 1 raw sets; LA = 0: load, wait, normalise, store, serial; LA = 1 = stage_a8: c + 1 in flight, c + 2 issued after c
// is stored).  rs is computed once (it does not depend on k0).
template <int LA, int NORM, int NC, int LDV>
__device__ __forceinline__ void stage_reg(char* smem, int chunk0, const bf16* h, const float* ssp, const float* nmod, uint32_t& sink) {
  constexpr int R = LA + 1;
  Raw ring[R];
#pragma unroll
  for (int c = 0; c < R && c < NC; ++c) load_raw<LDV>(ring[c], h, nmod, (chunk0 + c) * KC);
  float rs[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (NORM) compute_rs<false>(rs, ssp);
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    norm_store<NORM, LDV>(ring[c % R], rs, smem + c * TILE_BYTES, sink);
    if (c + R < NC) load_raw<LDV>(ring[c % R], h, nmod, (chunk0 + c + R) * KC);
  }
}
// cp.async staging: all 8 chunks' raw h rows (zero-filled rows >= T) into 8 row-major [64][AS_STRIDE] tiles, wait, in-place normalise.
template <int NORM>
__device__ __forceinline__ void stage_cpa(char* smem, const bf16* h, const float* ssp, const float* nmod, uint32_t& sink) {
  const int tid = threadIdx.x, pc = tid & 15, r0 = tid >> 4;
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int i = tid + x * NT, r = i >> 4, p = i & 15;
      cp_async16(smem + c * CPA_TILE_BYTES + (r * AS_STRIDE + p * 8) * 2, h + (size_t)(r < T ? r : 0) * W + c * KC + p * 8, r < T ? 16 : 0);
    }
  cp_commit();
  float rs[4] = {0.f, 0.f, 0.f, 0.f};
  Raw nm;
  if constexpr (NORM) compute_rs<true>(rs, ssp);
  ldv_nmod<1>(nm.sc0, nm.sc1, nm.sh0, nm.sh1, nmod, pc * 8);
  cp_wait0();
  __syncthreads();
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    Raw cur = nm;
    if (c + 1 < NCH) ldv_nmod<1>(nm.sc0, nm.sc1, nm.sh0, nm.sh1, nmod, (c + 1) * KC + pc * 8);
    float sc[8], sh[8];
    if constexpr (NORM) norm_scalars(cur, sc, sh);
    else sink ^= __float_as_uint(cur.sc0.x) ^ __float_as_uint(cur.sc0.w) ^ __float_as_uint(cur.sc1.y) ^ __float_as_uint(cur.sh0.z) ^ __float_as_uint(cur.sh1.w);
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int r = r0 + x * 16;
      if (r < T) {
        char* p = smem + c * CPA_TILE_BYTES + (r * AS_STRIDE + pc * 8) * 2;
        uint4 v = lds128(p);
        if constexpr (NORM) norm_piece(v, rs[x], sc, sh);
        sts128v(p, v);
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------- weights / MMA / epilogue
// B fragments of a 32-column tile: n-tile nt covers weight rows rowb[nt] .. + 7 (lane row g), k = k0 + 32 sl + 8 t4 ..: 16 x 128-bit loads (gemm_phase load_w)
__device__ __forceinline__ void load_b(uint4 (&b)[4][4], const bf16* w, const int (&rowb)[4], int k0, int g, int t4) {
#pragma unroll
  for (int sl = 0; sl < 4; ++sl)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      b[sl][nt] = __ldg(reinterpret_cast<const uint4*>(w + (size_t)(rowb[nt] + g) * W + k0 + sl * 32 + t4 * 8));
}
// PI05_GEMM_FRAG == 2 operand loads
__device__ __forceinline__ void mma_tile_fi(float (&acc)[4][4][4], const char* tile, int lane, const uint4 (&b)[4][4]) {
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
  for (int sl = 0; sl < 4; ++sl)
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const uint4* af = reinterpret_cast<const uint4*>(tile) + (mt * 4 + sl) * 2 * 32 + lane;
      const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[sl][nt].x, b[sl][nt].y);
        mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[sl][nt].z, b[sl][nt].w);
      }
    }
}
// PI05_GEMM_FRAG == 0 operand loads (row-major tile, stride AS_STRIDE)
__device__ __forceinline__ void mma_tile_rm(float (&acc)[4][4][4], const char* tile, int lane, const uint4 (&b)[4][4]) {
  const int g = lane >> 2, t4 = lane & 3;
  const bf16* As = reinterpret_cast<const bf16*>(tile);
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
  for (int sl = 0; sl < 4; ++sl)
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const uint4 a0 = lds128(As + (mt * 16 + g) * AS_STRIDE + sl * 32 + t4 * 8);
      const uint4 a1 = lds128(As + (mt * 16 + g + 8) * AS_STRIDE + sl * 32 + t4 * 8);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(acc[mt][nt], a0.x, a1.x, a0.y, a1.y, b[sl][nt].x, b[sl][nt].y);
        mma16816(acc[mt][nt], a0.z, a1.z, a0.w, a1.w, b[sl][nt].z, b[sl][nt].w);
      }
    }
}
// fp32 partial of chunk -> part[chunk][t][NV]; n-tile nt starts at virtual column colb[nt]
__device__ __forceinline__ void store_part_global(const float (&acc)[4][4][4], float* part, int chunk, const int (&colb)[4], int g, int t4) {
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = mt * 16 + g, c = colb[nt] + 2 * t4;
      if (r0 < T) *reinterpret_cast<float2*>(part + ((size_t)chunk * T + r0) * NV + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      if (r0 + 8 < T) *reinterpret_cast<float2*>(part + ((size_t)chunk * T + r0 + 8) * NV + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
}
// reduce_part4 of 8 partials (s = 0..7 from +0.0) -> rbf -> RoPE (heads < 9) -> two 64-bit bf16 stores; shared by V0 / V1 / H
__device__ __forceinline__ void reduce_rope_store(const float4 (&p1)[8], const float4 (&p2)[8], int hh, int t, int j, const float* rope, bf16* out) {
  float4 v1 = make_float4(0.f, 0.f, 0.f, 0.f), v2 = v1;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    v1.x += p1[s].x; v1.y += p1[s].y; v1.z += p1[s].z; v1.w += p1[s].w;
    v2.x += p2[s].x; v2.y += p2[s].y; v2.z += p2[s].z; v2.w += p2[s].w;
  }
  float x1[4] = {rbf(v1.x), rbf(v1.y), rbf(v1.z), rbf(v1.w)};
  float x2[4] = {rbf(v2.x), rbf(v2.y), rbf(v2.z), rbf(v2.w)};
  float o1[4], o2[4];
  if (hh < 9) {
    const float4* rp = reinterpret_cast<const float4*>(rope + (t * (HD / 2) + j) * 2);
    const float4 r01 = __ldcg(rp), r23 = __ldcg(rp + 1);
    const float cs[4] = {r01.x, r01.z, r23.x, r23.z}, sn[4] = {r01.y, r01.w, r23.y, r23.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      o1[e] = rbf(rbf(x1[e] * cs[e]) + rbf(-x2[e] * sn[e]));
      o2[e] = rbf(rbf(x2[e] * cs[e]) + rbf(x1[e] * sn[e]));
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) { o1[e] = x1[e]; o2[e] = x2[e]; }
  }
  bf16* dst = out + (size_t)t * NV + hh * HD + j;
  *reinterpret_cast<uint2*>(dst) = pack_bf16x4(o1[0], o1[1], o1[2], o1[3]);
  *reinterpret_cast<uint2*>(dst + HD / 2) = pack_bf16x4(o2[0], o2[1], o2[2], o2[3]);
}
__device__ __forceinline__ void load_part8(float4 (&p1)[8], float4 (&p2)[8], const float* part, int t, int c) {
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    p1[s] = __ldcg(reinterpret_cast<const float4*>(part + ((size_t)s * T + t) * NV + c));
    p2[s] = __ldcg(reinterpret_cast<const float4*>(part + ((size_t)s * T + t) * NV + c + HD / 2));
  }
}
// prefill: prefill_keys rows of K and of V (512 B each) from the prefix pool into smem at dst, no wait (drained at the start of the next iteration)
__device__ __forceinline__ void prefill(const Args& a, char* dst) {
  if (a.prefill_keys > 0 && (int)blockIdx.x < a.prefill_ctas) {
    __syncthreads();
    const int nk = a.prefill_keys;
    for (int i = threadIdx.x; i < nk * 32; i += NT) {
      const int kk = i >> 5, pc = i & 31, row = ((int)blockIdx.x * nk + kk) & 1023;
      cp_async16(dst + kk * 512 + pc * 16, reinterpret_cast<const char*>(a.pool_k) + (size_t)row * 512 + pc * 16, 16);
      cp_async16(dst + nk * 512 + kk * 512 + pc * 16, reinterpret_cast<const char*>(a.pool_v) + (size_t)row * 512 + pc * 16, 16);
    }
    cp_commit();
  }
}

// ---------------------------------------------------------------- kernel
// VAR: V0 / V1 / VH / VCTL;  STG, LA, NORM: V1 staging (NORM also V0 = 1);  CC: H's c;  ABC: V0 counters instead of the A->B barrier
template <int VAR, int STG, int LA, int NORM, int CC, int ABC, int LDV>
__global__ void __launch_bounds__(NT, 1) k_main(Args a) {
  extern __shared__ __align__(16) char sm[];
  const int tid = threadIdx.x, bid = blockIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  unsigned ep = 0;
  uint32_t sink = 0;
  unsigned long long last = 0;
  unsigned long long* stp = a.stamps;
  const int unit = (VAR == V1 || VAR == VH) ? unit_of(bid, a.place, a.active) : -1;
  constexpr int G = VAR == VH ? 8 / CC : 1;
  uint4 b[4][4];

  auto stamp = [&](int k, bool sync) {
    if (sync) __syncthreads();
    if (tid == 0) {
      unsigned settle;
      asm volatile("ld.volatile.shared.b32 %0, [%1];\n" : "=r"(settle) : "r"(0u));
      const unsigned long long t = gt();
      stp[k] = t; last = t;
    }
  };
  auto stamp_copy = [&](int k) { if (tid == 0) stp[k] = last; };

  for (int it = 0; it < a.iters; ++it) {
    stp = a.stamps + ((size_t)it * NCTA + bid) * 8;
    // ---- 1. prev phase: residual tasks (64 bf16 of h0 -> h, ssp part), drain of the previous prefill
    if (a.fresh) {
      for (int k = warp;; k += NT / 32) {
        const int task = bid + k * NCTA;
        if (task >= NRES) break;
        const int row = task >> 4, part = task & 15;
        const uint32_t x = reinterpret_cast<const uint32_t*>(a.h0 + (size_t)row * W + part * 64)[lane];
        reinterpret_cast<uint32_t*>(a.h + (size_t)row * W + part * 64)[lane] = x;
        const __nv_bfloat162 hv = *reinterpret_cast<const __nv_bfloat162*>(&x);
        const float f0 = __low2float(hv), f1 = __high2float(hv);
        float s = __fadd_rn(__fmul_rn(f0, f0), __fmul_rn(f1, f1));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) a.ssp[row * NSSP + part] = s;
      }
    }
    cp_wait0();  // the CTA sync of wpre_arrive follows: smem is free before any staging write
    // ---- 2. G1 = WPRE barrier: arrive, weight loads into registers, acquire wait
    wpre_arrive(a.bar);
    if constexpr (VAR == V0) {
      if (bid < 80) {
        const int ng = (bid >> 3) * 8 + warp;
        const int rb[4] = {ng * 32, ng * 32 + 8, ng * 32 + 16, ng * 32 + 24};
        load_b(b, a.w, rb, (bid & 7) * KC, g, t4);
      }
    } else if constexpr (VAR == V1) {
      if (unit >= 0) {
        const int vb = (unit >> 3) * HD + (unit & 7) * 16;
        const int rb[4] = {vb, vb + 8, vb + 128, vb + 136};
        load_b(b, a.w, rb, warp * KC, g, t4);
      }
    } else if constexpr (VAR == VH) {
      if (unit >= 0) {
        const int pt = (unit / G) * G + warp / CC, vb = (pt >> 3) * HD + (pt & 7) * 16;
        const int rb[4] = {vb, vb + 8, vb + 128, vb + 136};
        load_b(b, a.w, rb, ((unit % G) * CC + warp % CC) * KC, g, t4);
      }
    }
    ++ep;
    wpre_wait(a.bar, ep * NCTA);
    stamp(0, false);
    // ---- 3. jitter
    if (a.jitter_ns > 0) {
      unsigned hs = (unsigned)bid * 0x9E3779B1u ^ ((unsigned)it + 1u) * 0x85EBCA6Bu;
      hs ^= hs >> 15; hs *= 0x2C1B3C6Du; hs ^= hs >> 12; hs *= 0x297A2D39u; hs ^= hs >> 15;
      if (tid == 0) { const unsigned long long t0 = gt(), d = hs % (unsigned)(a.jitter_ns + 1); while (gt() - t0 < d) {} }
      __syncthreads();
    }
    stamp(1, true);
    const char* pfb = a.pf_bytes ? a.pf_pool + (size_t)((a.pf_it0 + it) % a.pf_pool_n) * a.pf_bytes : nullptr;

    // ---- 4. variant work
    if constexpr (VAR == V0) {
      float acc[4][4][4];
      stage_reg<0, 1, 1, LDV>(sm, bid & 7, a.h, a.ssp, a.nmod, sink);  // every CTA stages its chunk (gemm_phase does)
      __syncthreads();
      stamp(2, false);
      if (bid < 80) {
        const int ng = (bid >> 3) * 8 + warp;
        mma_tile_fi(acc, sm, lane, b);
        const int cb[4] = {ng * 32, ng * 32 + 8, ng * 32 + 16, ng * 32 + 24};
        store_part_global(acc, a.part, bid & 7, cb, g, t4);
        if (ABC) {  // publish this 32-column tile on its head's counter
          __syncwarp();
          if (lane == 0) red_release(a.cnt + (ng >> 3) * CNT_STRIDE);
        }
      }
      stamp(3, true);
      if (!ABC) { grid_barrier(a.bar, ++ep); stamp(4, false); }
      if (pfb) prefetch_l2(pfb, a.pf_bytes);
      prefill(a, sm + TILE_BYTES);
      if (bid >= NCTA - 70) {  // B task 147 - bid: head task / 7, 8-row slice task % 7
        const int task = NCTA - 1 - bid, hh = task / 7, t = (task % 7) * 8 + (tid >> 5), j = (tid & 31) * 4;
        if (ABC) { block_wait(a.cnt + hh * CNT_STRIDE, (unsigned)(it + 1) * 64u); stamp(4, false); }
        if (t < T) {
          float4 p1[8], p2[8];
          load_part8(p1, p2, a.part, t, hh * HD + j);
          reduce_rope_store(p1, p2, hh, t, j, a.rope, a.out);
        }
      } else if (ABC) stamp_copy(4);
      stamp(5, true);
      stamp_copy(6);
    } else if constexpr (VAR == V1) {
      if (unit >= 0) {
        float acc[4][4][4];
        if constexpr (STG == STG_REG) stage_reg<LA, NORM, NCH, LDV>(sm, 0, a.h, a.ssp, a.nmod, sink);
        else stage_cpa<NORM>(sm, a.h, a.ssp, a.nmod, sink);
        __syncthreads();
        stamp(2, false);
        if constexpr (STG == STG_REG) mma_tile_fi(acc, sm + warp * TILE_BYTES, lane, b);
        else mma_tile_rm(acc, sm + warp * CPA_TILE_BYTES, lane, b);
        __syncthreads();  // every warp is done reading the tiles (the partial buffer reuses them)
        {
          float* pb = reinterpret_cast<float*>(sm) + warp * T * PB_STRIDE;
#pragma unroll
          for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
              const int r0 = mt * 16 + g, c = nt * 8 + 2 * t4;
              if (r0 < T) *reinterpret_cast<float2*>(pb + r0 * PB_STRIDE + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
              if (r0 + 8 < T) *reinterpret_cast<float2*>(pb + (r0 + 8) * PB_STRIDE + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
            }
        }
        stamp(3, true);
        stamp_copy(4);
        if (tid < T * 4) {  // 200 reduce tasks: row t, quad qd of 4 first-half columns (tile cols 4 qd.., partner 16 + 4 qd..)
          const int t = tid >> 2, qd = tid & 3, hh = unit >> 3, j = (unit & 7) * 16 + qd * 4;
          const float* pbf = reinterpret_cast<const float*>(sm);
          float4 p1[8], p2[8];
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            p1[s] = *reinterpret_cast<const float4*>(pbf + (s * T + t) * PB_STRIDE + qd * 4);
            p2[s] = *reinterpret_cast<const float4*>(pbf + (s * T + t) * PB_STRIDE + 16 + qd * 4);
          }
          reduce_rope_store(p1, p2, hh, t, j, a.rope, a.out);
        }
        stamp(5, true);
      } else {
        stamp_copy(2); stamp_copy(3); stamp_copy(4); stamp_copy(5);
      }
      if (pfb) prefetch_l2(pfb, a.pf_bytes);
      prefill(a, sm);
      stamp(6, true);
    } else if constexpr (VAR == VH) {
      if (unit >= 0) {
        float acc[4][4][4];
        const int gi = unit / G, kg = unit % G;
        stage_reg<1, NORM, CC, LDV>(sm, kg * CC, a.h, a.ssp, a.nmod, sink);
        __syncthreads();
        stamp(2, false);
        mma_tile_fi(acc, sm + (warp % CC) * TILE_BYTES, lane, b);
        {
          const int pt = gi * G + warp / CC, vb = (pt >> 3) * HD + (pt & 7) * 16;
          const int cb[4] = {vb, vb + 8, vb + 128, vb + 136};
          store_part_global(acc, a.part, kg * CC + warp % CC, cb, g, t4);
        }
        stamp(3, true);  // CTA sync orders every warp's partial stores before thread 0's release
        int na = a.active - gi * G;
        na = na < 0 ? 0 : (na > G ? G : na);
        if (tid == 0) red_release(a.cnt + gi * CNT_STRIDE);
        block_wait(a.cnt + gi * CNT_STRIDE, (unsigned)(it + 1) * (unsigned)na);
        stamp(4, false);
        {
          const int r0 = T * kg / G, nrows = T * (kg + 1) / G - r0;
          const int item = tid >> 2;
          if (item < nrows * G) {
            const int qd = tid & 3, tl = item % G, t = r0 + item / G, pt = gi * G + tl, hh = pt >> 3, j = (pt & 7) * 16 + qd * 4;
            float4 p1[8], p2[8];
            load_part8(p1, p2, a.part, t, hh * HD + j);
            reduce_rope_store(p1, p2, hh, t, j, a.rope, a.out);
          }
        }
        stamp(5, true);
      } else {
        stamp_copy(2); stamp_copy(3); stamp_copy(4); stamp_copy(5);
      }
      if (pfb) prefetch_l2(pfb, a.pf_bytes);
      prefill(a, sm);
      stamp(6, true);
    } else {  // ctl
      stamp_copy(2); stamp_copy(3); stamp_copy(4); stamp_copy(5);
      if (pfb) prefetch_l2(pfb, a.pf_bytes);
      prefill(a, sm);
      stamp(6, true);
    }
    // ---- 5. G2
    grid_barrier(a.bar, ++ep);
    stamp(7, false);
  }
  cp_wait0();
  if (sink == 0x9e3779b9u) a.sink[0] = sink;
}

// ---------------------------------------------------------------- host
struct Opts {
  std::string variant = "V0", stage = "reg", place = "hi", dump;
  int la = 1, norm = 1, c = 1, abc = 0, fresh = 1, pf_kb = 4224, prefill_keys = 16, prefill_ctas = 84, jitter_ns = 0, active = 80, smem_kb = 132;
  int iters = 200, launches = 5, check = 0, pf_pool = 24, ldv = 1;
};
static uint64_t rng_state = 0x1234abcdULL;
static double urand() { rng_state += 0x9e3779b97f4a7c15ULL; uint64_t z = rng_state; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31; return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double nrand() { return sqrt(-2 * log(urand())) * cos(6.283185307179586 * urand()); }
static bf16 hb(float x) { return __float2bfloat16(x); }
static float bf(bf16 x) { return __bfloat162float(x); }
static float rbf_h(float x) { return bf(hb(x)); }
static double median(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double pct(std::vector<double> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }

struct Data {
  std::vector<bf16> h0, w; std::vector<float> ssp, nmod, rope;
  bf16 *dh, *dh0, *dw, *dout, *dpk, *dpv; float *dssp, *dnmod, *drope, *dpart; char* dpf; unsigned *dcnt, *dbar, *dsink; unsigned long long* dstamps;
  size_t pf_bytes = 0;
};
static Data D;
static void init_data(const Opts& o) {
  D.h0.resize((size_t)T * W); D.w.resize((size_t)NV * W); D.nmod.resize(2 * W); D.ssp.resize(T * NSSP); D.rope.resize((size_t)T * (HD / 2) * 2);
  for (auto& x : D.h0) x = hb((float)nrand());
  for (auto& x : D.w) x = hb((float)(nrand() / sqrt(32.0)));
  for (auto& x : D.nmod) x = (float)(0.1 * nrand());
  for (int r = 0; r < T; ++r)
    for (int p = 0; p < NSSP; ++p) {  // device residual task order: lane l holds elements 2l, 2l+1; xor butterfly 16, 8, 4, 2, 1
      float v[32];
      for (int l = 0; l < 32; ++l) { const float f0 = bf(D.h0[(size_t)r * W + p * 64 + 2 * l]), f1 = bf(D.h0[(size_t)r * W + p * 64 + 2 * l + 1]); v[l] = f0 * f0 + f1 * f1; }
      for (int off = 16; off > 0; off >>= 1) { float n[32]; for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ off]; memcpy(v, n, sizeof v); }
      D.ssp[r * NSSP + p] = v[0];
    }
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < HD / 2; ++j) {
      const float inv = 1.0f / powf(10000.0f, (float)(2 * j) / (float)HD);
      const float ph = (float)t * inv;
      D.rope[((size_t)t * (HD / 2) + j) * 2] = rbf_h(cosf(ph));
      D.rope[((size_t)t * (HD / 2) + j) * 2 + 1] = rbf_h(sinf(ph));
    }
  auto up = [](void** d, const void* h, size_t n) { CK(cudaMalloc(d, n)); CK(cudaMemcpy(*d, h, n, cudaMemcpyHostToDevice)); };
  up((void**)&D.dh, D.h0.data(), D.h0.size() * 2); up((void**)&D.dh0, D.h0.data(), D.h0.size() * 2); up((void**)&D.dw, D.w.data(), D.w.size() * 2);
  up((void**)&D.dssp, D.ssp.data(), D.ssp.size() * 4); up((void**)&D.dnmod, D.nmod.data(), D.nmod.size() * 4); up((void**)&D.drope, D.rope.data(), D.rope.size() * 4);
  CK(cudaMalloc(&D.dpart, (size_t)8 * T * NV * 4)); CK(cudaMemset(D.dpart, 0, (size_t)8 * T * NV * 4));
  CK(cudaMalloc(&D.dout, (size_t)T * NV * 2));
  std::vector<bf16> pk(1024 * 256), pv(1024 * 256);
  for (auto& x : pk) x = hb((float)nrand());
  for (auto& x : pv) x = hb((float)nrand());
  up((void**)&D.dpk, pk.data(), pk.size() * 2); up((void**)&D.dpv, pv.data(), pv.size() * 2);
  D.pf_bytes = (size_t)o.pf_kb * 1024;
  if (D.pf_bytes) { CK(cudaMalloc(&D.dpf, D.pf_bytes * o.pf_pool)); CK(cudaMemset(D.dpf, 1, D.pf_bytes * o.pf_pool)); }
  CK(cudaMalloc(&D.dcnt, 64 * CNT_STRIDE * 4)); CK(cudaMalloc(&D.dbar, 256)); CK(cudaMalloc(&D.dsink, 64));
  CK(cudaMalloc(&D.dstamps, (size_t)o.iters * NCTA * 8 * 8));
}
static Args mkargs(const Opts& o, int iters, int pf_it0) {
  Args a;
  a.h = D.dh; a.h0 = D.dh0; a.ssp = D.dssp; a.nmod = D.dnmod; a.w = D.dw; a.rope = D.drope; a.part = D.dpart; a.out = D.dout; a.pool_k = D.dpk; a.pool_v = D.dpv;
  a.pf_pool = D.dpf; a.cnt = D.dcnt; a.bar = D.dbar; a.stamps = D.dstamps; a.sink = D.dsink; a.pf_bytes = D.pf_bytes;
  a.iters = iters; a.fresh = o.fresh; a.jitter_ns = o.jitter_ns; a.prefill_keys = o.prefill_keys; a.prefill_ctas = o.prefill_ctas; a.place = o.place == "spread";
  a.active = o.active; a.pf_pool_n = o.pf_pool; a.pf_it0 = pf_it0; a.abc = o.abc;
  return a;
}
// host reference of V0's output (norm in float as the device formula with 1/sqrtf, bf16 rounded A, GEMM in double, rbf, RoPE in double)
static void reference(std::vector<double>& out) {
  std::vector<double> A((size_t)T * W);
  for (int t = 0; t < T; ++t) {
    float ss[NSSP]; for (int p = 0; p < NSSP; ++p) ss[p] = D.ssp[t * NSSP + p];
    float v[16]; memcpy(v, ss, sizeof v);
    for (int off = 8; off > 0; off >>= 1) { float n[16]; for (int l = 0; l < 16; ++l) n[l] = v[l] + v[l ^ off]; memcpy(v, n, sizeof v); }
    const float rs = 1.0f / sqrtf(v[0] * (1.0f / W) + 1e-6f);
    for (int k = 0; k < W; ++k) { const float o = (bf(D.h0[(size_t)t * W + k]) * rs) * (1.0f + D.nmod[k]) + D.nmod[W + k]; A[(size_t)t * W + k] = rbf_h(o); }
  }
  std::vector<double> acc((size_t)T * NV);
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < NV; ++n) { double d = 0; const bf16* w = &D.w[(size_t)n * W]; for (int k = 0; k < W; ++k) d += A[(size_t)t * W + k] * (double)bf(w[k]); acc[(size_t)t * NV + n] = (double)rbf_h((float)d); }
  out.assign((size_t)T * NV, 0);
  for (int t = 0; t < T; ++t)
    for (int hh = 0; hh < 10; ++hh)
      for (int j = 0; j < HD / 2; ++j) {
        const double x1 = acc[(size_t)t * NV + hh * HD + j], x2 = acc[(size_t)t * NV + hh * HD + j + HD / 2];
        if (hh < 9) {
          const double cs = D.rope[((size_t)t * (HD / 2) + j) * 2], sn = D.rope[((size_t)t * (HD / 2) + j) * 2 + 1];
          out[(size_t)t * NV + hh * HD + j] = x1 * cs - x2 * sn; out[(size_t)t * NV + hh * HD + j + HD / 2] = x2 * cs + x1 * sn;
        } else { out[(size_t)t * NV + hh * HD + j] = x1; out[(size_t)t * NV + hh * HD + j + HD / 2] = x2; }
      }
}

template <int VAR_, int STG_, int LA_, int NORM_, int CC_, int ABC_, int LDV_ = 1> struct KT { static constexpr int VAR = VAR_, STG = STG_, LA = LA_, NORM = NORM_, CC = CC_, ABC = ABC_, LDV = LDV_; };
// runtime knobs -> template instantiation (only the combinations mb2.sbatch uses); returns 2 for an unsupported combination
template <class Fn> static int dispatch(const Opts& o, Fn&& fn) {
  if (o.variant == "ctl") return fn(KT<VCTL, 0, 0, 1, 0, 0>{});
  if (!o.ldv) {
    if (o.variant == "V0" && !o.abc) return fn(KT<V0, 0, 0, 1, 0, 0, 0>{});
    if (o.variant == "V1" && o.stage == "reg" && o.la == 1 && o.norm) return fn(KT<V1, STG_REG, 1, 1, 0, 0, 0>{});
    fprintf(stderr, "--ldv 0 is built only for V0 (abc 0) and V1 reg la1 norm1\n");
    return 2;
  }
  if (o.variant == "V0") return o.abc ? fn(KT<V0, 0, 0, 1, 0, 1>{}) : fn(KT<V0, 0, 0, 1, 0, 0>{});
  if (o.variant == "V1") {
    if (o.stage == "cpa") return o.norm ? fn(KT<V1, STG_CPA, 0, 1, 0, 0>{}) : fn(KT<V1, STG_CPA, 0, 0, 0, 0>{});
    if (o.norm) {
      if (o.la == 0) return fn(KT<V1, STG_REG, 0, 1, 0, 0>{});
      if (o.la == 1) return fn(KT<V1, STG_REG, 1, 1, 0, 0>{});
      if (o.la == 2) return fn(KT<V1, STG_REG, 2, 1, 0, 0>{});
      if (o.la == 4) return fn(KT<V1, STG_REG, 4, 1, 0, 0>{});
    } else if (o.la == 1) return fn(KT<V1, STG_REG, 1, 0, 0, 0>{});
  }
  if (o.variant == "H" && o.norm) {
    if (o.c == 1) return fn(KT<VH, 0, 1, 1, 1, 0>{});
    if (o.c == 2) return fn(KT<VH, 0, 1, 1, 2, 0>{});
    if (o.c == 4) return fn(KT<VH, 0, 1, 1, 4, 0>{});
  }
  fprintf(stderr, "unsupported variant/knob combination (variant %s stage %s la %d norm %d c %d)\n", o.variant.c_str(), o.stage.c_str(), o.la, o.norm, o.c);
  return 2;
}
static size_t smem_needed(const Opts& o, bool cpa) {
  if (cpa) return (size_t)NCH * CPA_TILE_BYTES;
  size_t need = 0;
  if (o.variant == "V1") need = (size_t)NCH * TILE_BYTES;
  else if (o.variant == "H") need = (size_t)o.c * TILE_BYTES;
  else if (o.variant == "V0") need = TILE_BYTES;
  const size_t pf = (size_t)o.prefill_keys * 1024;
  return std::max(need, (o.variant == "V0" ? (size_t)TILE_BYTES : 0) + pf);
}

template <class KTag> static cudaError_t launch(const Opts& o, size_t smem, Args a, void (*kfn)(Args)) {
  void* kargs[] = {&a};
  return cudaLaunchCooperativeKernel((const void*)kfn, dim3(NCTA), dim3(NT), kargs, smem, 0);
}
static void reset_launch(bool clear_out) {
  CK(cudaMemset(D.dcnt, 0, 64 * CNT_STRIDE * 4)); CK(cudaMemset(D.dbar, 0, 256));
  if (clear_out) CK(cudaMemset(D.dout, 0xFF, (size_t)T * NV * 2));
}

static std::string role_of(const Opts& o, int bid) {
  std::vector<std::string> r;
  if (o.variant == "V0") { if (bid < 80) r.push_back("tile"); if (bid >= NCTA - 70) r.push_back("B"); }
  else if (o.variant == "V1" || o.variant == "H") { if (unit_of(bid, o.place == "spread", o.active) >= 0) r.push_back("unit"); }
  if (o.prefill_keys > 0 && bid < o.prefill_ctas) r.push_back("prefill");
  if (r.empty()) return "idle";
  std::string s; for (size_t i = 0; i < r.size(); ++i) s += (i ? "+" : "") + r[i];
  return s;
}
// CTAs whose (s_k - s_{k-1}) enters the stage median
static bool in_mask(const Opts& o, int k, int bid) {
  if (o.variant == "V0") {
    if (k == 3) return bid < 80;
    if (k == 5) return bid >= NCTA - 70;
    if (k == 4 && o.abc) return bid >= NCTA - 70;
    return true;
  }
  if (o.variant == "V1" || o.variant == "H") { if (k >= 2 && k <= 5) return unit_of(bid, o.place == "spread", o.active) >= 0; return true; }
  return true;
}

template <class KTag> static int run(const Opts& o, KTag) {
  auto kfn = &k_main<KTag::VAR, KTag::STG, KTag::LA, KTag::NORM, KTag::CC, KTag::ABC, KTag::LDV>;
  const bool cpa = KTag::VAR == V1 && KTag::STG == STG_CPA;
  Opts oe = o;
  if (cpa) oe.smem_kb = 160;
  const size_t smem = (size_t)oe.smem_kb * 1024;
  if (smem < smem_needed(oe, cpa)) { fprintf(stderr, "smem_kb %d too small (need %zu B)\n", oe.smem_kb, smem_needed(oe, cpa)); return 2; }
  CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kfn));
  int nb = 0, nsm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kfn, NT, smem)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  if (nb < 1 || nsm < NCTA) { fprintf(stderr, "cannot co-schedule %d CTAs (blocks/SM %d, SMs %d, smem %zu)\n", NCTA, nb, nsm, smem); return 1; }
  fprintf(stderr, "kernel ldv=%d variant=%s stage=%s la=%d norm=%d c=%d abc=%d: smem_kb=%d regs=%d local_bytes=%zu\n", o.ldv, o.variant.c_str(), o.stage.c_str(), o.la, o.norm, o.c, o.abc, oe.smem_kb,
          fa.numRegs, fa.localSizeBytes);
  const int nit = oe.check ? 3 : oe.iters, nl = oe.check ? 1 : oe.launches;
  std::vector<unsigned long long> st((size_t)nit * NCTA * 8);
  std::vector<double> windows, last_arr; std::vector<std::vector<double>> stage_med(6);
  std::map<int, int> crit;
  for (int r = 0; r < nl; ++r) {
    reset_launch(oe.check);
    Args a = mkargs(oe, nit, r * nit);
    CK(launch<KTag>(oe, smem, a, kfn)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if (oe.check) return 0;
    CK(cudaMemcpy(st.data(), D.dstamps, st.size() * 8, cudaMemcpyDeviceToHost));
    if (r == 0) continue;
    for (int it = 10; it < nit; ++it) {
      const unsigned long long* s = &st[(size_t)it * NCTA * 8];
      unsigned long long r1 = ~0ull, r2 = ~0ull, mx6 = 0; int cb = 0;
      for (int b = 0; b < NCTA; ++b) { r1 = std::min(r1, s[b * 8 + 0]); r2 = std::min(r2, s[b * 8 + 7]); if (s[b * 8 + 6] > mx6) { mx6 = s[b * 8 + 6]; cb = b; } }
      windows.push_back((double)(r2 - r1) / 1e3); last_arr.push_back((double)(mx6 - r1) / 1e3); crit[cb]++;
      for (int k = 1; k <= 6; ++k) {
        std::vector<double> d;
        for (int b = 0; b < NCTA; ++b) if (in_mask(oe, k, b)) d.push_back((double)((long long)s[b * 8 + k] - (long long)s[b * 8 + k - 1]) / 1e3);
        stage_med[k - 1].push_back(median(d));
      }
    }
  }
  if (!oe.dump.empty()) {
    FILE* f = fopen(oe.dump.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", oe.dump.c_str()); return 1; }
    fprintf(f, "MB2 stamps u64 [iters=%d][cta=%d][8] variant=%s stage=%s la=%d norm=%d c=%d abc=%d place=%s active=%d ns\n", nit, NCTA, oe.variant.c_str(), oe.stage.c_str(), oe.la, oe.norm, oe.c, oe.abc,
            oe.place.c_str(), oe.active);
    fwrite(st.data(), 8, st.size(), f); fclose(f);
  }
  std::vector<std::pair<int, int>> cv(crit.begin(), crit.end());
  std::sort(cv.begin(), cv.end(), [](auto& x, auto& y) { return x.second > y.second; });
  std::string cs = "[";
  for (size_t i = 0; i < cv.size() && i < 3; ++i) { char b[128]; snprintf(b, sizeof b, "%s[%d,%d,\"%s\"]", i ? "," : "", cv[i].first, cv[i].second, role_of(oe, cv[i].first).c_str()); cs += b; }
  cs += "]";
  printf("{\"u\":\"MB2\",\"variant\":\"%s\",\"stage\":\"%s\",\"la\":%d,\"norm\":%d,\"c\":%d,\"abc\":%d,\"ldv\":%d,\"fresh\":%d,\"pf_kb\":%d,\"prefill_keys\":%d,\"prefill_ctas\":%d,\"place\":\"%s\","
         "\"jitter_ns\":%d,\"active\":%d,\"iters\":%d,\"launches\":%d,\"pf_pool\":%d,\"smem_kb\":%d,\"regs\":%d,\"local_bytes\":%zu,\"window_us\":%.3f,\"window_p10\":%.3f,\"window_p90\":%.3f,"
         "\"st_us\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],\"last_arrival_us\":%.3f,\"crit\":%s}\n",
         oe.variant.c_str(), oe.variant == "V1" ? oe.stage.c_str() : "-", oe.la, oe.norm, oe.c, oe.abc, oe.ldv, oe.fresh, oe.pf_kb, oe.prefill_keys, oe.prefill_ctas, oe.place.c_str(), oe.jitter_ns, oe.active,
         oe.iters, oe.launches, oe.pf_pool, oe.smem_kb, fa.numRegs, fa.localSizeBytes, median(windows), pct(windows, 0.1), pct(windows, 0.9), median(stage_med[0]), median(stage_med[1]),
         median(stage_med[2]), median(stage_med[3]), median(stage_med[4]), median(stage_med[5]), median(last_arr), cs.c_str());
  fflush(stdout);
  return 0;
}

// --check: V0 (abc 0) vs the host reference, then the selected variant vs V0 byte for byte
static int check(const Opts& o) {
  Opts c = o; c.check = 1;
  Opts r0 = c; r0.variant = "V0"; r0.abc = 0; r0.norm = 1; r0.ldv = 1;
  std::vector<bf16> ref0((size_t)T * NV), sel((size_t)T * NV);
  if (int rc = run(r0, KT<V0, 0, 0, 1, 0, 0>{})) return rc;
  CK(cudaMemcpy(ref0.data(), D.dout, ref0.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<double> ref; reference(ref);
  double mx = 0, me = 0; int nf = 0;
  for (double v : ref) mx = std::max(mx, fabs(v));
  for (size_t i = 0; i < ref.size(); ++i) { const double v = bf(ref0[i]); if (!std::isfinite(v)) ++nf; me = std::max(me, fabs(v - ref[i])); }
  const bool pa = nf == 0 && me <= 2e-2 * mx;
  printf("check (a) V0 vs host reference: max_abs_err=%.4e max|ref|=%.4e (limit %.4e) nonfinite=%d %s\n", me, mx, 2e-2 * mx, nf, pa ? "PASS" : "FAIL");
  bool pb = true;
  if (!o.norm) printf("check (b) skipped (--norm 0 outputs differ by design)\n");
  else {
    if (int rc = dispatch(c, [&](auto kt) { return run(c, kt); })) return rc;
    CK(cudaMemcpy(sel.data(), D.dout, sel.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < sel.size(); ++i) if (memcmp(&sel[i], &ref0[i], 2) != 0) { if (!bad) first = i; ++bad; }
    pb = bad == 0;
    if (pb) printf("check (b) %s out == V0 out byte for byte: PASS\n", o.variant.c_str());
    else printf("check (b) %s out != V0 out: %zu of %zu elements differ, first t=%zu col=%zu (head %zu): got %04x want %04x FAIL\n", o.variant.c_str(), bad, sel.size(), first / NV, first % NV,
                (first % NV) / HD, (unsigned)*reinterpret_cast<uint16_t*>(&sel[first]), (unsigned)*reinterpret_cast<uint16_t*>(&ref0[first]));
  }
  printf("check %s\n", pa && pb ? "PASS" : "FAIL");
  return pa && pb ? 0 : 1;
}

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    auto is = [&](const char* s) { return !strcmp(argv[i], s) && i + 1 < argc; };
    if (is("--variant")) o.variant = argv[++i]; else if (is("--stage")) o.stage = argv[++i]; else if (is("--la")) o.la = atoi(argv[++i]);
    else if (is("--norm")) o.norm = atoi(argv[++i]); else if (is("--c")) o.c = atoi(argv[++i]); else if (is("--abc")) o.abc = atoi(argv[++i]);
    else if (is("--fresh")) o.fresh = atoi(argv[++i]); else if (is("--pf_kb")) o.pf_kb = atoi(argv[++i]); else if (is("--prefill_keys")) o.prefill_keys = atoi(argv[++i]);
    else if (is("--prefill_ctas")) o.prefill_ctas = atoi(argv[++i]); else if (is("--place")) o.place = argv[++i]; else if (is("--jitter_ns")) o.jitter_ns = atoi(argv[++i]);
    else if (is("--active")) o.active = atoi(argv[++i]); else if (is("--smem_kb")) o.smem_kb = atoi(argv[++i]); else if (is("--iters")) o.iters = atoi(argv[++i]);
    else if (is("--launches")) o.launches = atoi(argv[++i]); else if (is("--pf_pool")) o.pf_pool = atoi(argv[++i]); else if (is("--dump")) o.dump = argv[++i]; else if (is("--ldv")) o.ldv = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--check")) o.check = 1;
    else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
  }
  if (o.variant != "V0" && o.variant != "V1" && o.variant != "H" && o.variant != "ctl") { fprintf(stderr, "bad --variant\n"); return 2; }
  if (o.place != "hi" && o.place != "spread") { fprintf(stderr, "bad --place\n"); return 2; }
  if (o.stage != "reg" && o.stage != "cpa") { fprintf(stderr, "bad --stage\n"); return 2; }
  if (o.prefill_keys < 0 || o.prefill_keys > 64 || o.pf_pool < 1 || o.iters < 1 || o.launches < 1 || (o.iters <= 10 && !o.check)) { fprintf(stderr, "bad knob value\n"); return 2; }
  if (o.check) o.iters = 3;
  init_data(o);
  if (o.check) return check(o);
  return dispatch(o, [&](auto kt) { return run(o, kt); });
}
