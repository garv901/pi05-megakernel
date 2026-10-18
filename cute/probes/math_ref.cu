// Elementwise nvcc reference of the scalar expressions of mk/pi05_mk.cu (C++ copied from the kernel), for
// probes/math_parity.py. in: [n_in][n] float32 (integer-valued inputs are stored as floats), out: [2][n] uint32 bits.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
constexpr int W = 1024, HD = 256, NSTEP = 10;

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x) {
  const float kBeta = 0.7978845608028654f;
  const float kKappa = 0.044715f;
  float x3 = x * x * x;
  float inner = kBeta * (x + kKappa * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

template <int E>
__global__ void ref_kernel(const float* __restrict__ in, uint32_t* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
#define IN(k) in[(size_t)(k) * n + i]
  float o0 = 0.f, o1 = 0.f;
  uint32_t u0 = 0;
  bool bits = false;
  if (E == 0) {
    o0 = rbf(IN(0));
  } else if (E == 1) {
    __nv_bfloat162 v = __floats2bfloat162_rn(IN(0), IN(1));
    u0 = *reinterpret_cast<uint32_t*>(&v);
    bits = true;
  } else if (E == 2) {
    o0 = (float)(double)IN(0);
  } else if (E == 3) {
    o0 = silu(IN(0));
  } else if (E == 4) {
    o0 = gelu_tanh(IN(0));
  } else if (E == 5) {
    o0 = rbf(gelu_tanh(rbf(IN(0)))) * rbf(IN(1));
    o1 = rbf(o0);
  } else if (E == 6) {
    o0 = rsqrtf(__fadd_rn(IN(0) * (1.0f / W), 1e-6f));
  } else if (E == 7) {
    const float h = IN(0), rs = IN(1), sc = IN(2), sh = IN(3);
    o0 = __fadd_rn(__fmul_rn(__fmul_rn(h, rs), __fadd_rn(1.0f, sc)), sh);
    o1 = rbf(o0);
  } else if (E == 8) {
    o0 = rbf(__fadd_rn(IN(0), rbf(__fmul_rn(rbf(IN(1)), rbf(IN(2))))));
  } else if (E == 9) {
    const bool Mk = IN(1) > 0.5f;
    o0 = Mk ? rbf(IN(0)) * 0.0625f : -INFINITY;
  } else if (E == 10) {
    const float m = IN(1);
    const float msafe = m == -INFINITY ? 0.f : m;
    o0 = rbf(expf(IN(0) - msafe));
  } else if (E == 11) {
    const float w = IN(0), M = IN(1);
    o0 = w == -INFINITY ? 0.f : expf(w - M);
  } else if (E == 12) {
    o0 = fmaf(IN(0), IN(1), IN(2));
    o1 = fmaf(IN(0), IN(3), IN(4));
  } else if (E == 13) {
    o0 = IN(0) / IN(1);
  } else if (E == 14) {
    const int j = (int)IN(0);
    o0 = 1.0f / powf(10000.0f, (float)(2 * j) / (float)HD);
  } else if (E == 15) {
    const int pos = (int)IN(0), j = (int)IN(1);
    const float inv = 1.0f / powf(10000.0f, (float)(2 * j) / (float)HD);
    const float ph = __fmul_rn((float)pos, inv);
    o0 = rbf(cosf(ph));
    o1 = rbf(sinf(ph));
  } else if (E == 16) {
    const float x1 = IN(0), x2 = IN(1), cs = IN(2), sn = IN(3);
    o0 = rbf(rbf(x1 * cs) + rbf(-x2 * sn));
    o1 = rbf(rbf(x2 * cs) + rbf(x1 * sn));
  } else if (E == 17) {
    const int s = (int)IN(0), j = (int)IN(1);
    const float tf = (float)(1.0 + (double)s * (-1.0 / NSTEP));
    const int steps = W / 2;
    const double step = 1.0 / (double)(steps - 1);
    const double frac = j < steps / 2 ? step * (double)j : 1.0 - step * (double)(steps - j - 1);
    const double period = 4e-3 * pow(4.0 / 4e-3, frac);
    const double freq = 1.0 / period * 2.0 * 3.141592653589793;
    const double ph = (double)tf * freq;
    o0 = (float)sin(ph);
    o1 = (float)cos(ph);
  } else if (E == 18) {
    float acc = IN(8);
    acc = fmaf(IN(4), IN(0), acc); acc = fmaf(IN(5), IN(1), acc);
    acc = fmaf(IN(6), IN(2), acc); acc = fmaf(IN(7), IN(3), acc);
    o0 = acc;
  } else if (E == 19) {
    float v0 = 0.f;
    v0 = fmaf(IN(0), IN(4), v0); v0 = fmaf(IN(1), IN(5), v0);
    v0 = fmaf(IN(2), IN(6), v0); v0 = fmaf(IN(3), IN(7), v0);
    o0 = rbf(v0 + IN(8));
    o1 = v0;
  } else if (E == 20) {
    const float h0 = IN(0), h1 = IN(1);
    o0 = fmaf(h1, h1, h0 * h0);
  } else if (E == 21) {
    float v = 0.f;
    v = fmaf(IN(0), IN(4), v); v = fmaf(IN(1), IN(5), v);
    v = fmaf(IN(2), IN(6), v); v = fmaf(IN(3), IN(7), v);
    const float vel = v + IN(8);
    o0 = __fadd_rn(IN(9), __fmul_rn(-0.1f, vel));
    o1 = vel;
  } else if (E == 22) {
    float ss = 0.f;
    ss = fmaf(IN(0), IN(0), ss); ss = fmaf(IN(1), IN(1), ss);
    ss = fmaf(IN(2), IN(2), ss); ss = fmaf(IN(3), IN(3), ss);
    o0 = ss;
  }
  out[i] = bits ? u0 : __float_as_uint(o0);
  out[n + i] = __float_as_uint(o1);
#undef IN
}

template <int E>
static void go(const torch::Tensor& in, torch::Tensor& out, int n) {
  ref_kernel<E><<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float*>(in.data_ptr()), reinterpret_cast<uint32_t*>(out.data_ptr()), n);
}

void run(int64_t e, torch::Tensor in, torch::Tensor out, int64_t n) {
  const int m = (int)n;
  switch (e) {
    case 0: go<0>(in, out, m); break;   case 1: go<1>(in, out, m); break;   case 2: go<2>(in, out, m); break;
    case 3: go<3>(in, out, m); break;   case 4: go<4>(in, out, m); break;   case 5: go<5>(in, out, m); break;
    case 6: go<6>(in, out, m); break;   case 7: go<7>(in, out, m); break;   case 8: go<8>(in, out, m); break;
    case 9: go<9>(in, out, m); break;   case 10: go<10>(in, out, m); break; case 11: go<11>(in, out, m); break;
    case 12: go<12>(in, out, m); break; case 13: go<13>(in, out, m); break; case 14: go<14>(in, out, m); break;
    case 15: go<15>(in, out, m); break; case 16: go<16>(in, out, m); break; case 17: go<17>(in, out, m); break;
    case 18: go<18>(in, out, m); break; case 19: go<19>(in, out, m); break; case 20: go<20>(in, out, m); break;
    case 21: go<21>(in, out, m); break; case 22: go<22>(in, out, m); break;
    default: TORCH_CHECK(false, "bad expression id");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "math_ref launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
