// nvcc reference of the tensor-core tiles of mk/pi05_mk.cu (code copied from the kernel), for probes/mma_tile.py.
//   gemm : gemm_phase (PI05_GEMM_FRAG == 2), one block = 8 warps, 8 x 32 weight columns of one K chunk
//   wsplit : gemm_phase_wsplit<2>, 4 groups of 2 warps, one 32-column task each
//   fa : FA unit (PI05_ATT_FA), one warp pair on one 16-row m-tile; P is an input
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
constexpr int T = 50, HD = 256, KC = 128, NTHREADS = 256, NWARPS = 8;
constexpr int AS_STRIDE = KC + 32, WS_STRIDE = 40, KS_STRIDE = HD + 8, VS_STRIDE = HD + 8, ATT_KEYS = 64, ATT_NT = ATT_KEYS / 8;
constexpr int GEMM_SMEM = 64 * AS_STRIDE * 2;
constexpr int WSPLIT_SMEM = 64 * AS_STRIDE * 2 + NWARPS * T * WS_STRIDE * 4 + NWARPS * 2 * 4 * 32 * 16;
constexpr int FA_SMEM = (16 + 2 * ATT_KEYS) * KS_STRIDE * 2;

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ uint4 lds128(const void* smem_ptr) {
  uint4 r;
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

// A: [nb][64][K] (rows >= T zero), Wt: [nb][256][K], out: [nb][T][256]; chunk k0.
__global__ void __launch_bounds__(NTHREADS, 1) gemm_kernel(const bf16* __restrict__ Ain, const bf16* __restrict__ Win,
                                                           float* __restrict__ outp, int K, int k0) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* As = reinterpret_cast<bf16*>(smem);
  const bf16* A = Ain + (size_t)blockIdx.x * 64 * K;
  const bf16* Wb = Win + (size_t)blockIdx.x * 256 * K;
  float* part = outp + (size_t)blockIdx.x * T * 256;
  const int N = 256;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  const int ng = warp;
  uint4 b[4][4];
  {
    const int n0 = ng * 32, nl = n0;
#pragma unroll
    for (int sl = 0; sl < 4; ++sl)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        b[sl][nt] = __ldg(reinterpret_cast<const uint4*>(Wb + (size_t)(nl + nt * 8 + g) * K + k0 + sl * 32 + t4 * 8));
  }
  {
    constexpr int PIECES = 64 * (KC / 8);
    uint4 av[PIECES / NTHREADS];
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int q = threadIdx.x / (KC / 8), pc = threadIdx.x % (KC / 8);
      const int r = (x >> 1) * 32 + (q >> 3) * 16 + (x & 1) * 8 + (q & 7);
      av[x] = r < T ? __ldcg(reinterpret_cast<const uint4*>(A + (size_t)r * K + k0 + pc * 8)) : zero;
    }
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
  }
  __syncthreads();
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
      const uint4* af = reinterpret_cast<const uint4*>(As) + (mt * 4 + sl) * 2 * 32 + lane;
      const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[sl][nt].x, b[sl][nt].y);
        mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[sl][nt].z, b[sl][nt].w);
      }
    }
  }
  const int n0 = ng * 32;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = mt * 16 + g, c = n0 + nt * 8 + 2 * t4;
      if (r0 < T) *reinterpret_cast<float2*>(part + (size_t)r0 * N + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      if (r0 + 8 < T) *reinterpret_cast<float2*>(part + (size_t)(r0 + 8) * N + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
}

// WPT = 2. A: [nb][64][K], Wt: [nb][128][K], out: [nb][T][128]; chunk k0; group gi = warp / 2 takes columns 32 gi .. 32 gi + 31.
__global__ void __launch_bounds__(NTHREADS, 1) wsplit_kernel(const bf16* __restrict__ Ain, const bf16* __restrict__ Win,
                                                             float* __restrict__ outp, int K, int k0) {
  constexpr int WPT = 2, NSL = 4 / WPT;
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* As = reinterpret_cast<bf16*>(smem);
  const bf16* A = Ain + (size_t)blockIdx.x * 64 * K;
  const bf16* Wt = Win + (size_t)blockIdx.x * 128 * K;
  float* part = outp + (size_t)blockIdx.x * T * 128;
  const int N = 128;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int gi = warp / WPT, wi = warp % WPT;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  const int ng = gi;
  float* red = reinterpret_cast<float*>(As + 64 * AS_STRIDE);
  float* mine = red + warp * T * WS_STRIDE;
  const float* grp = red + gi * WPT * T * WS_STRIDE;
  uint4* wb = reinterpret_cast<uint4*>(red + NWARPS * T * WS_STRIDE) + warp * NSL * 4 * 32;
  {
    const int n0 = ng * 32;
#pragma unroll
    for (int j = 0; j < NSL; ++j)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        cp_async16(wb + (j * 4 + nt) * 32 + lane, Wt + (size_t)(n0 + nt * 8 + g) * K + k0 + (wi * NSL + j) * 32 + t4 * 8, 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  {
    constexpr int PIECES = 64 * (KC / 8);
    uint4 av[PIECES / NTHREADS];
#pragma unroll
    for (int x = 0; x < PIECES / NTHREADS; ++x) {
      const int i = threadIdx.x + x * NTHREADS, r = i / (KC / 8), pc = i % (KC / 8);
      av[x] = r < T ? __ldcg(reinterpret_cast<const uint4*>(A + (size_t)r * K + k0 + pc * 8)) : zero;
    }
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
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  float acc[4][4][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
  for (int j = 0; j < NSL; ++j) {
    const int sl = wi * NSL + j;
    uint4 b[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) b[nt] = lds128(wb + (j * 4 + nt) * 32 + lane);
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      const uint4* af = reinterpret_cast<const uint4*>(As) + (mt * 4 + sl) * 2 * 32 + lane;
      const uint4 x0 = lds128(af), x1 = lds128(af + 32);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        mma16816(acc[mt][nt], x0.x, x0.y, x0.z, x0.w, b[nt].x, b[nt].y);
        mma16816(acc[mt][nt], x1.x, x1.y, x1.z, x1.w, b[nt].z, b[nt].w);
      }
    }
  }
  const int n0 = ng * 32;
  __syncwarp();
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r0 = mt * 16 + g, c = nt * 8 + 2 * t4;
      if (r0 < T) *reinterpret_cast<float2*>(mine + r0 * WS_STRIDE + c) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
      if (r0 + 8 < T) *reinterpret_cast<float2*>(mine + (r0 + 8) * WS_STRIDE + c) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
    }
  asm volatile("bar.sync %0, %1;\n" ::"r"(1 + gi), "r"(32 * WPT) : "memory");
  const int rr = lane >> 3, c4 = (lane & 7) * 4;
#pragma unroll
  for (int q = wi; q < (T + 3) / 4; q += WPT) {
    const int r = q * 4 + rr;
    if (r < T) {
      float4 v = *reinterpret_cast<const float4*>(grp + r * WS_STRIDE + c4);
#pragma unroll
      for (int w = 1; w < WPT; ++w) {
        const float4 u = *reinterpret_cast<const float4*>(grp + (w * T + r) * WS_STRIDE + c4);
        v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
      }
      *reinterpret_cast<float4*>(part + (size_t)r * N + n0 + c4) = v;
    }
  }
}

// Q: [nb][16][HD], K, V: [nb][64][HD], P: [nb][16][64] (bf16 probabilities), S: [nb][16][64] f32, O: [nb][16][HD] f32.
__global__ void __launch_bounds__(64, 1) fa_kernel(const bf16* __restrict__ Qg, const bf16* __restrict__ Kg, const bf16* __restrict__ Vg,
                                                   const bf16* __restrict__ Pg, float* __restrict__ Sg, float* __restrict__ Og) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* Qs = reinterpret_cast<bf16*>(smem);
  bf16* Ks = Qs + 16 * KS_STRIDE;
  bf16* Vs = Ks + ATT_KEYS * KS_STRIDE;
  const int u = blockIdx.x;
  const bf16* q = Qg + (size_t)u * 16 * HD;
  const bf16* k = Kg + (size_t)u * ATT_KEYS * HD;
  const bf16* v = Vg + (size_t)u * ATT_KEYS * HD;
  const bf16* pg = Pg + (size_t)u * 16 * ATT_KEYS;
  float* S = Sg + (size_t)u * 16 * ATT_KEYS;
  float* O = Og + (size_t)u * 16 * HD;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  for (int i = tid; i < 16 * 32; i += 64) {
    const int rr = i >> 5, pc = i & 31;
    *reinterpret_cast<uint4*>(Qs + rr * KS_STRIDE + pc * 8) = *reinterpret_cast<const uint4*>(q + rr * HD + pc * 8);
  }
  for (int i = tid; i < ATT_KEYS * 32; i += 64) {
    const int rr = i >> 5, pc = i & 31;
    *reinterpret_cast<uint4*>(Ks + rr * KS_STRIDE + pc * 8) = *reinterpret_cast<const uint4*>(k + rr * HD + pc * 8);
    *reinterpret_cast<uint4*>(Vs + rr * VS_STRIDE + pc * 8) = *reinterpret_cast<const uint4*>(v + rr * HD + pc * 8);
  }
  __syncthreads();
  const int mt = 0, dh = warp;
  float sc[ATT_NT][4];
#pragma unroll
  for (int nt = 0; nt < ATT_NT; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) sc[nt][e] = 0.f;
  const bf16* qa = Qs + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KS_STRIDE + (lane >> 4) * 8;
  const bf16* kb = Ks + ((lane >> 4) * 8 + (lane & 7)) * KS_STRIDE + ((lane >> 3) & 1) * 8;
#pragma unroll 4
  for (int kk = 0; kk < HD / 16; ++kk) {
    uint32_t a0, a1, a2, a3;
    ldmatrix_x4(a0, a1, a2, a3, qa + kk * 16);
#pragma unroll
    for (int np = 0; np < ATT_NT / 2; ++np) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4(b0, b1, b2, b3, kb + np * 16 * KS_STRIDE + kk * 16);
      mma16816(sc[2 * np], a0, a1, a2, a3, b0, b1);
      mma16816(sc[2 * np + 1], a0, a1, a2, a3, b2, b3);
    }
  }
  if (dh == 0) {
#pragma unroll
    for (int nt = 0; nt < ATT_NT; ++nt) {
      *reinterpret_cast<float2*>(S + g * ATT_KEYS + nt * 8 + 2 * t4) = make_float2(sc[nt][0], sc[nt][1]);
      *reinterpret_cast<float2*>(S + (g + 8) * ATT_KEYS + nt * 8 + 2 * t4) = make_float2(sc[nt][2], sc[nt][3]);
    }
  }
  uint32_t pa[ATT_NT][2];
#pragma unroll
  for (int nt = 0; nt < ATT_NT; ++nt) {
    float pr[4];
    pr[0] = __bfloat162float(pg[g * ATT_KEYS + nt * 8 + 2 * t4]);
    pr[1] = __bfloat162float(pg[g * ATT_KEYS + nt * 8 + 2 * t4 + 1]);
    pr[2] = __bfloat162float(pg[(g + 8) * ATT_KEYS + nt * 8 + 2 * t4]);
    pr[3] = __bfloat162float(pg[(g + 8) * ATT_KEYS + nt * 8 + 2 * t4 + 1]);
    const __nv_bfloat162 lo = __floats2bfloat162_rn(pr[0], pr[1]);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(pr[2], pr[3]);
    pa[nt][0] = *reinterpret_cast<const uint32_t*>(&lo);
    pa[nt][1] = *reinterpret_cast<const uint32_t*>(&hi);
  }
  float acc[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < ATT_KEYS / 16; ++ks) {
    const int krow = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      uint32_t b0, b1, b2, b3;
      ldmatrix_x4_trans(b0, b1, b2, b3, Vs + krow * VS_STRIDE + dh * 128 + np * 16 + (lane >> 4) * 8);
      mma16816(acc[2 * np], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b0, b1);
      mma16816(acc[2 * np + 1], pa[2 * ks][0], pa[2 * ks][1], pa[2 * ks + 1][0], pa[2 * ks + 1][1], b2, b3);
    }
  }
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int c = dh * 128 + nt * 8 + 2 * t4;
    *reinterpret_cast<float2*>(O + g * HD + c) = make_float2(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<float2*>(O + (g + 8) * HD + c) = make_float2(acc[nt][2], acc[nt][3]);
  }
}

template <typename F>
static void launch(F kernel, int nb, int threads, int smem, std::function<void()> go) {
  TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "smem attr");
  go();
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "mma_ref launch failed");
}

#define BF(t) reinterpret_cast<const bf16*>(t.data_ptr())
void gemm(torch::Tensor A, torch::Tensor W, torch::Tensor out, int64_t k0) {
  const int nb = A.size(0), K = A.size(2);
  launch(gemm_kernel, nb, NTHREADS, GEMM_SMEM, [&] {
    gemm_kernel<<<nb, NTHREADS, GEMM_SMEM, at::cuda::getCurrentCUDAStream()>>>(BF(A), BF(W), out.data_ptr<float>(), K, (int)k0);
  });
}
void wsplit(torch::Tensor A, torch::Tensor W, torch::Tensor out, int64_t k0) {
  const int nb = A.size(0), K = A.size(2);
  launch(wsplit_kernel, nb, NTHREADS, WSPLIT_SMEM, [&] {
    wsplit_kernel<<<nb, NTHREADS, WSPLIT_SMEM, at::cuda::getCurrentCUDAStream()>>>(BF(A), BF(W), out.data_ptr<float>(), K, (int)k0);
  });
}
void fa(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor P, torch::Tensor S, torch::Tensor O) {
  const int nb = Q.size(0);
  launch(fa_kernel, nb, 64, FA_SMEM, [&] {
    fa_kernel<<<nb, 64, FA_SMEM, at::cuda::getCurrentCUDAStream()>>>(BF(Q), BF(K), BF(V), BF(P), S.data_ptr<float>(), O.data_ptr<float>());
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm);
  m.def("wsplit", &wsplit);
  m.def("fa", &fa);
}
