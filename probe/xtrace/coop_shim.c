// LD_PRELOAD shim for the Xtrace CUDA arm: interposes cudaLaunchCooperativeKernel (the pi05 extension's only launch).
// Off (default) it forwards untouched. On, it launches the Xtrace-spliced copy of the kernel (loaded with xt_load from
// an instrumented cubin, selected with xt_set) with the same grid / block / smem / stream / Params, then the slot's extra
// parameters set by xt_set_extra (addresses standing in for constant bank 4, see splice.retarget_bank4), then the trace
// buffer as the last parameter. The device code of the shipped extension is never rebuilt.
// Build: gcc -O2 -shared -fPIC coop_shim.c -I$CUDA/include -L$CUDA/lib64/stubs -lcuda -ldl -Wl,--version-script=coop_shim.map
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef cudaError_t (*launch_fn)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

static CUfunction g_fns[8];
static unsigned long long g_extra[8][8];
static int g_nextra[8];
static int g_nfn;
static CUdeviceptr g_buf;
static int g_slot = -1;  // instrumented function launched in place of the original; -1 = forward untouched
static unsigned long long g_count;

// Load an instrumented cubin; returns its slot (>= 0) or a negative / 1000+ / 2000+ error code.
int xt_load(const char* path, const char* name) {
  if (g_nfn == 8) return -3;
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* data = (char*)malloc(n);
  if (fread(data, 1, n, f) != (size_t)n) { fclose(f); free(data); return -2; }
  fclose(f);
  CUmodule mod;
  CUresult r = cuModuleLoadData(&mod, data);
  free(data);
  if (r != CUDA_SUCCESS) return 1000 + (int)r;
  r = cuModuleGetFunction(&g_fns[g_nfn], mod, name);
  return r == CUDA_SUCCESS ? g_nfn++ : 2000 + (int)r;
}

void xt_set(int slot, unsigned long long buf) { g_slot = slot; g_buf = (CUdeviceptr)buf; }

// The slot's parameters between the kernel's own and the trace buffer; returns 0, or -1 if out of range.
int xt_set_extra(int slot, int n, const unsigned long long* v) {
  if (slot < 0 || slot >= 8 || n < 0 || n > 8) return -1;
  for (int i = 0; i < n; i++) g_extra[slot][i] = v[i];
  g_nextra[slot] = n;
  return 0;
}
unsigned long long xt_count(void) { return g_count; }

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 grid, dim3 block, void** args, size_t smem,
                                        cudaStream_t stream) {
  static launch_fn real;
  if (!real) real = (launch_fn)dlvsym(RTLD_NEXT, "cudaLaunchCooperativeKernel", "libcudart.so.13");
  if (!real) real = (launch_fn)dlsym(RTLD_NEXT, "cudaLaunchCooperativeKernel");
  if (g_slot < 0) return real(func, grid, block, args, smem, stream);
  CUfunction g_fn = g_fns[g_slot];
  void* xargs[10] = {args[0]};
  int n = g_nextra[g_slot];
  for (int i = 0; i < n; i++) xargs[1 + i] = &g_extra[g_slot][i];
  xargs[1 + n] = &g_buf;
  if (cuFuncSetAttribute(g_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  CUresult r = cuLaunchCooperativeKernel(g_fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, (unsigned)smem,
                                         (CUstream)stream, xargs);
  g_count++;
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
}
