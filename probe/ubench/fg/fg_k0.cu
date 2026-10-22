// fg_k0.cu: K0 = faithful HMMA F+G for the standalone harness (fg_common.cuh). gemm_phase (phase F) and the phase G body are
// VERBATIM copies from mk/pi05_mk.cu with the megakernel's default defines: GEMM_FRAG=2, LOAD_HOIST=1, GEMM_SPREAD=1,
// PART_BF16=1, WPRE=15 (bit 2: F's first-tile weights issued between the E->F arrive and wait), FUSE=18 (FUSE_FG off: no
// counters, real F->G barrier), __launch_bounds__(256,1) on the kernel. The probe-only FIT_* macros are compiled out (no-ops).
// Variants in one binary (runtime --variants null,k0,calib; ABAB-interleaved launches): null (every barrier kept, F+G work removed,
// G's next-layer prefetch kept), k0 (real F + G), calib (F slot spins --ncal cycles).
// Build (login node):  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a fg_k0.cu -o fg_k0
//   unstamped twin for gate 2:  ... -DFG_STAMPS=0 fg_k0.cu -o fg_k0_ns
// Run: fg_k0 [--reps 20] [--ncal 8700] [--ccyc 19000] [--cold 0|1] [--ctx 15] [--variants null,k0,calib] [--out DIR] [--name TAG]
#include "fg_common.cuh"

#define PI05_GEMM_FRAG 2
#define PI05_LOAD_HOIST 1
#define PI05_GEMM_SPREAD 1
#define PI05_TILE_ROT 0
#define FIT_OPEN() do {} while (0)
#define FIT_EV(ev) do {} while (0)
#define FIT_CLOSE() do {} while (0)
#define FIT_KEV(ev) do {} while (0)

// ---- VERBATIM from mk/pi05_mk.cu (phase F: gemm_phase; helpers) ----
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Volatile so the scheduler keeps fragment loads next to their mma (bounds registers).
__device__ __forceinline__ uint4 lds128(const void* smem_ptr) {
  uint4 r;
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
  const float kKappa = 0.044715f;
  float x3 = x * x * x;
  float inner = kBeta * (x + kKappa * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

constexpr int AS_STRIDE = KC + 32;  // bf16; 16B fragment loads are conflict-free

__device__ __noinline__ void gemm_phase(const bf16* __restrict__ A, int K, const bf16* w0, int n1, const bf16* w1, int n2,
                           const bf16* w2, int N, float* __restrict__ part, bf16* As,
                           unsigned* cnt = nullptr, int cdiv = 1, int cmod = 1,
                           const float* ssp = nullptr, const float* nmod = nullptr WPRE_PARAM) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int NG = N / 32, S = K / KC;
  const int vblk = PI05_TILE_ROT ? (blockIdx.x + PI05_TILE_ROT) % gridDim.x : blockIdx.x;
  const int s = vblk % S, slot = vblk / S;
  const int nblk = (gridDim.x - s + S - 1) / S;  // blocks sharing chunk s
  const int k0 = s * KC;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  int ng = PI05_GEMM_SPREAD ? warp * nblk + slot : slot * NWARPS + warp;
  const int ng_step = nblk * NWARPS;

  auto load_w = [&](int ngi, uint4 (&b)[4][4]) {
    const int n0 = ngi * 32;
    const bf16* Wb;
    int nl;
    if (n0 < n1) { Wb = w0; nl = n0; }
    else if (n0 < n2) { Wb = w1; nl = n0 - n1; }
    else { Wb = w2; nl = n0 - n2; }
#pragma unroll
    for (int sl = 0; sl < 4; ++sl)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        b[sl][nt] = __ldg(reinterpret_cast<const uint4*>(Wb + (size_t)(nl + nt * 8 + g) * K + k0 + sl * 32 + t4 * 8));
  };

  uint4 b[4][4];
  FIT_OPEN();
  FIT_EV(FIT_G0);
#if PI05_WPRE
  if (wp.bar != nullptr) {
    wpre_arrive(wp);
    if (ng < NG) load_w(ng, b);
    FIT_EV(FIT_WISS);
    wpre_wait(wp);
    FIT_EV(FIT_BAR);
  } else
#endif
  {
    if (ng < NG) load_w(ng, b);
    FIT_EV(FIT_WISS);
  }
  // Stage A[0:50, k0:k0+128] (rows 50..63 zero).
  // Norm mode (ssp != null): A is the residual h and the staged tile is its adaRMS
  // norm, rs[r] from the 8 per-chunk sums of squares ssp[r][0..7] (fixed order, as
  // block_sum) and scale/shift = nmod[k], nmod[W + k]; all loads issued in one round.
  {
    constexpr int PIECES = 64 * (KC / 8);  // 1024 16B pieces
    uint4 av[PIECES / NTHREADS];
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
#if PI05_GEMM_FRAG == 2
      const int q = threadIdx.x / (KC / 8), pc = threadIdx.x % (KC / 8);
      const int r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7);
#else
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
#endif
      av[x] = r < T ? __ldcg(reinterpret_cast<const uint4*>(A + (size_t)r * K + k0 + pc * 8)) : zero;
    }
    // Thread tid stages rows tid/16 + 16x (x < 4) at the same 8 columns (pc = tid % 16), so
    // its scale/shift are 4 float4 and each half-warp owns one row per x: lane j of the
    // half-warp loads ss part j and a fixed shuffle tree, broadcast from lane 0, gives rs.
    static_assert(KC / 8 == 16 && NSSP == 16, "norm staging layout");
    if (ssp != nullptr) {
      const int pc = threadIdx.x % (KC / 8), r0 = threadIdx.x / (KC / 8);
      float ssv[PIECES / NTHREADS];
#pragma unroll
      for (int x = 0; x < PIECES / NTHREADS; ++x) {
#if PI05_GEMM_FRAG == 2
        const int r = (x >> 1) * 32 + (r0 >> 3) * 16 + (x & 1) * 8 + (r0 & 7);
#else
        const int r = r0 + x * (NTHREADS / (KC / 8));
#endif
        ssv[x] = r < T ? __ldcg(ssp + r * NSSP + pc) : 0.f;
      }
      const float4* sp = reinterpret_cast<const float4*>(nmod + k0 + pc * 8);
      const float4 sc0 = __ldcg(sp), sc1 = __ldcg(sp + 1), sh0 = __ldcg(sp + W / 4), sh1 = __ldcg(sp + W / 4 + 1);
      const float sc[8] = {__fadd_rn(1.0f, sc0.x), __fadd_rn(1.0f, sc0.y), __fadd_rn(1.0f, sc0.z), __fadd_rn(1.0f, sc0.w),
                           __fadd_rn(1.0f, sc1.x), __fadd_rn(1.0f, sc1.y), __fadd_rn(1.0f, sc1.z), __fadd_rn(1.0f, sc1.w)};
      const float sh[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
      const int hbase = threadIdx.x & 16;  // lane 0 of this half-warp
#pragma unroll
      for (int x = 0; x < PIECES / NTHREADS; ++x) {
        float v = ssv[x];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        v = __shfl_sync(0xffffffffu, v, hbase);
        const float rs = rsqrtf(__fadd_rn(v * (1.0f / W), 1e-6f));
#if PI05_GEMM_FRAG == 2
        if ((x >> 1) * 32 + (r0 >> 3) * 16 + (x & 1) * 8 + (r0 & 7) < T) {
#else
        if (r0 + x * (NTHREADS / (KC / 8)) < T) {
#endif
          uint32_t* wv = reinterpret_cast<uint32_t*>(&av[x]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            __nv_bfloat162 hv2 = *reinterpret_cast<__nv_bfloat162*>(&wv[e]);
            const float o0 = __fadd_rn(__fmul_rn(__fmul_rn(__low2float(hv2), rs), sc[2 * e]), sh[2 * e]);
            const float o1 = __fadd_rn(__fmul_rn(__fmul_rn(__high2float(hv2), rs), sc[2 * e + 1]), sh[2 * e + 1]);
            hv2 = __floats2bfloat162_rn(o0, o1);
            wv[e] = *reinterpret_cast<uint32_t*>(&hv2);
          }
        }
      }
    }
#if PI05_GEMM_FRAG == 2
    // Fragment-interleaved tile (layout of gemm_phase_wsplit): uint4 [mt][sl][half][g * 4 + t4] =
    // {row g, row g + 8} x 2 words; this thread's pieces x = 2j, 2j + 1 are rows (r, r + 8), r % 16 < 8.
    {
      const int q = threadIdx.x / (KC / 8), pc = threadIdx.x % (KC / 8);
      uint4* A4 = reinterpret_cast<uint4*>(As);
#pragma unroll
      for (int j = 0; j < PIECES / NTHREADS / 2; ++j) {
        const uint4 u = av[2 * j], v = av[2 * j + 1];
        uint4* dst = A4 + ((j * 2 + (q >> 3)) * 4 + (pc >> 2)) * 2 * 32 + (q & 7) * 4 + (pc & 3);
        dst[0] = make_uint4(u.x, v.x, u.y, v.y);
        dst[32] = make_uint4(u.z, v.z, u.w, v.w);
      }
    }
#elif PI05_GEMM_FRAG
    // Fragment-interleaved tile (layout of gemm_phase_wsplit): uint4 [mt][sl][half][g * 4 + t4].
    static_assert(64 * KC * 2 <= 64 * AS_STRIDE * 2, "interleaved A tile");
    uint32_t* Ai = reinterpret_cast<uint32_t*>(As);
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      const int base = (((r >> 4) * 4 + (pc >> 2)) * 2 * 32 + (r & 7) * 4 + (pc & 3)) * 4 + ((r >> 3) & 1);
      Ai[base] = av[x].x;
      Ai[base + 2] = av[x].y;
      Ai[base + 32 * 4] = av[x].z;
      Ai[base + 32 * 4 + 2] = av[x].w;
    }
#else
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      *reinterpret_cast<uint4*>(As + r * AS_STRIDE + pc * 8) = av[x];
    }
#endif
  }
  FIT_EV(FIT_ADONE);
  __syncthreads();
  FIT_EV(FIT_SYNC);
  for (; ng < NG; ng += ng_step) {
    float acc[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int sl = 0; sl < 4; ++sl) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
#if PI05_GEMM_FRAG
        const uint4* af = reinterpret_cast<const uint4*>(As) + (mt * 4 + sl) * 2 * 32 + lane;
        const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[sl][nt].x, b[sl][nt].y);
          mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[sl][nt].z, b[sl][nt].w);
        }
        if (sl == 0 && mt == 0) FIT_EV(FIT_M0);  // after the first 8 MMAs: the sl = 0 weights have arrived
#else
        const uint4 a0 = lds128(As + (mt * 16 + g) * AS_STRIDE + sl * 32 + t4 * 8);
        const uint4 a1 = lds128(As + (mt * 16 + g + 8) * AS_STRIDE + sl * 32 + t4 * 8);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          mma16816(acc[mt][nt], a0.x, a1.x, a0.y, a1.y, b[sl][nt].x, b[sl][nt].y);
          mma16816(acc[mt][nt], a0.z, a1.z, a0.w, a1.w, b[sl][nt].z, b[sl][nt].w);
        }
        if (sl == 0 && mt == 0) FIT_EV(FIT_M0);  // after the first 8 MMAs: the sl = 0 weights have arrived
#endif
      }
    }
    FIT_EV(FIT_MMA);
    const int n0 = ng * 32;
#if PI05_LOAD_HOIST
    if (ng + ng_step < NG) {
      load_w(ng + ng_step, b);
      FIT_EV(FIT_WNEXT);
    }
#endif
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int r0 = mt * 16 + g, c = n0 + nt * 8 + 2 * t4;
#if PI05_PART_BF16
        if (r0 < T) part_st2(part, ((size_t)s * T + r0) * N + c, acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T) part_st2(part, ((size_t)s * T + r0 + 8) * N + c, acc[mt][nt][2], acc[mt][nt][3]);
#else
        if (r0 < T)
          *reinterpret_cast<float2*>(part + ((size_t)s * T + r0) * N + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
        if (r0 + 8 < T)
          *reinterpret_cast<float2*>(part + ((size_t)s * T + r0 + 8) * N + c) =
              make_float2(acc[mt][nt][2], acc[mt][nt][3]);
#endif
      }
    FIT_EV(FIT_ST);
#if !PI05_LOAD_HOIST
    if (ng + ng_step < NG) {
      load_w(ng + ng_step, b);
      FIT_EV(FIT_WNEXT);
    }
#endif
    if (cnt != nullptr) {  // publish this 32-column partial tile (fence overlaps the next loads)
      __syncwarp();
      if (lane == 0) red_release(cnt + ((n0 / cdiv) % cmod) * CNT_STRIDE);
    }
  }
  FIT_EV(FIT_GEND);
  FIT_CLOSE();
}

// Four consecutive columns of reduce_part with all S loads in flight.
template <int S>
__device__ __forceinline__ float4 reduce_part4(const float* part, int N, int t, int c) {
  float4 x[S];
#pragma unroll
#if PI05_PART_BF16
  for (int s = 0; s < S; ++s) x[s] = part_ld4(part, ((size_t)s * T + t) * N + c);
#else
  for (int s = 0; s < S; ++s) x[s] = __ldcg(reinterpret_cast<const float4*>(part + ((size_t)s * T + t) * N + c));
#endif
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int s = 0; s < S; ++s) { v.x += x[s].x; v.y += x[s].y; v.z += x[s].z; v.w += x[s].w; }
  return v;
}

__device__ __forceinline__ uint2 pack_bf16x4(float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  return make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}

struct K0 {
  static constexpr const char* kName = "fg_k0";
  static constexpr bool kFgBarrier = true;
  static constexpr size_t kSmem = FG_SMEM_DYN;
  // E->F: variant slot. w is the split barrier already created by the caller (epoch and stamp slot reserved).
  __device__ static void F(const Params& p, const FgArgs& a, int l, const WPre& w, bf16* smb) {
    if (a.variant == FGV_K0) {
      const bool defnorm = (p.fuse & FUSE_DEFNORM) != 0;
      const float* modstep = p.mod;
      gemm_phase(defnorm ? p.h : p.xn, W, a.wg + (size_t)l * FF * W, FF, a.wu + (size_t)l * FF * W, 2 * FF, nullptr,
                 2 * FF, p.part, smb, (p.fuse & FUSE_FG) ? p.cnt + CNT_GU * CNT_STRIDE : nullptr, 32, FF / 32,
                 defnorm ? p.ssp : nullptr, defnorm ? modstep + (size_t)(2 * l + 1) * 3 * W : nullptr, w);
    } else {
      fg_null_F(w);
      if (a.variant == FGV_CALIB) fg_spin(a.ncal);
    }
  }
  __device__ static void G(const Params& p, const FgArgs& a, int tid, unsigned it1, bool backoff) {
    if (a.variant != FGV_K0) return;
      // ---- VERBATIM phase G body (mk/pi05_mk.cu, default build); tid / it1 / backoff are defined by the caller ----
      {
        constexpr int GR = (T + 1) / 2;  // tasks: (32-col group, half of the rows), one quad per thread
        for (int task = gridDim.x - 1 - blockIdx.x; task < 2 * (FF / 32); task += gridDim.x) {
          const int jg = task >> 1, t = (task & 1) * GR + tid / 8, c = jg * 32 + (tid % 8) * 4;
          if (p.fuse & FUSE_FG) block_wait(p.cnt + (CNT_GU + jg) * CNT_STRIDE, it1 * ARR_GU, backoff);
          if (tid / 8 < GR && t < T) {
            const float4 gv = reduce_part4<W / KC>(p.part, 2 * FF, t, c);
            const float4 uv = reduce_part4<W / KC>(p.part, 2 * FF, t, FF + c);
            *reinterpret_cast<uint2*>(p.act + t * FF + c) =
                pack_bf16x4(rbf(gelu_tanh(rbf(gv.x))) * rbf(uv.x), rbf(gelu_tanh(rbf(gv.y))) * rbf(uv.y),
                            rbf(gelu_tanh(rbf(gv.z))) * rbf(uv.z), rbf(gelu_tanh(rbf(gv.w))) * rbf(uv.w));
          }
        }
      }

  }
};

int main(int argc, char** argv) { return fg_main<K0>(argc, argv); }
