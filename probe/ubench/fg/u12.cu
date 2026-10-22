// U12: design-parameter microbenchmarks for the F+G redesign (plan Step 0). One binary, sections selected by argv[1] (a,b,c,d,coop or "all").
//   U12a  per-SM TMA ingress from L2 (148 CTAs, 1 CTA/SM): box size x ring depth sweep (flat 128 B-row map and the real weight layout
//         [rows][1024] bf16 with 64x64 boxes), warm (L2 resident) vs cold (HBM walk), single-shot (ring drains every pass) vs steady;
//         cluster multicast of a 100 KB (56x1024 bf16) activation tile at cluster 2/4 vs unicast; cudaOccupancyMaxActiveClusters;
//         cooperative + cluster launch attempts ("coop").
//   U12b  tcgen05.ld throughput by shape (32x32b, 16x256b, 16x32bx2) x .x1..x16, 1/4/8 warps on legal lane quadrants (warp w -> lanes
//         32*(w%4)), with/without a concurrent MMA stream; plus the U11 "320 ns per batch" anomaly decomposition (k_anom).
//   U12c  tcgen05.mma cta_group::1 kind::f16 M=64 N=56 K=16 x 64 chain fed from a TMA ring with per-stage tcgen05.commit -> empty
//         mbarriers: issue rate, completion latency, keeps-up-with-ingress, numeric check vs fp32 reference.
//   U12d  A-from-TMEM ([a_tmem]) MMA form, M=64 / M=128, kind::f16: does it assemble, does it run, numeric check.
// Timing rules: clock64 inside an SM, cudaEvent for totals, ns/cycle = globaltimer over >=100 us spans (never point reads after a
// barrier). Every timed config prints a verify field (bad:N) and a clk/event cross-check ("xchk", flagged when >2%).
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a u12.cu -o u12 -lcuda     Run: u12 [a|b|c|d|coop|all]
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef unsigned long long ull;
typedef __nv_bfloat16 bf16;
constexpr int NSM = 148;

__device__ unsigned g_to;  // mbarrier wait timeouts
__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ ull clk() { ull t; asm volatile("mov.u64 %0, %clock64;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
// bounded wait: returns false (and counts a timeout) instead of hanging
__device__ __forceinline__ bool mbar_wait(uint32_t b, uint32_t par) {
  uint32_t ok = 0;
  for (ull i = 0; i < (1ull << 22) && !ok; ++i)
    asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  if (!ok) atomicAdd(&g_to, 1u);
  return ok;
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void tma2d_mc(uint32_t dst, const CUtensorMap* m, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               ::"r"(dst), "l"((ull)m), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory"); }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void mma_tc(uint32_t d, ull da, ull db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc(uint32_t slot_smem, int ncols) {
  if (ncols == 256) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(slot_smem));
  else asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" ::"r"(slot_smem));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_free(uint32_t tb, int ncols) {
  if (ncols == 256) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tb));
  else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tb));
}
// UMMA smem descriptor: K-major, SWIZZLE_128B, SBO = 1024 B, sm100 version bit (as gemm_tc2.cu)
__device__ __forceinline__ ull sdesc(uint32_t addr) { return (ull)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
// instruction descriptor: f32 accum, bf16 x bf16, K-major A and B; N>>3 at bit 17, M>>4 at bit 24
__device__ __forceinline__ uint32_t idesc(int N, int M = 64) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
__device__ __forceinline__ bool elect_one() { uint32_t e; asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(e)); return e != 0; }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ float hashf(uint32_t i) { uint32_t h = i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; return ((float)(h & 0xffff) - 32768.f) * (1.f / 65536.f); }
__global__ void k_init(bf16* p, size_t n, uint32_t seed) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2bfloat16(hashf((uint32_t)i + seed)); }

// ---- generated tcgen05.ld wrappers: SH 0 = 32x32b (R=X regs, C=X cols), 1 = 16x256b (R=4X, C=8X), 2 = 16x32bx2 (R=X, C=2X, halfsplit=X)
template <int SH, int X> struct LD;
template <> struct LD<0,1> { static constexpr int R=1, C=1; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v[0]) : "r"(ta)); } };
template <> struct LD<0,2> { static constexpr int R=2, C=2; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "r"(ta)); } };
template <> struct LD<0,4> { static constexpr int R=4, C=4; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(ta)); } };
template <> struct LD<0,8> { static constexpr int R=8, C=8; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(ta)); } };
template <> struct LD<0,16> { static constexpr int R=16, C=16; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta)); } };
template <> struct LD<1,1> { static constexpr int R=4, C=8; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(ta)); } };
template <> struct LD<1,2> { static constexpr int R=8, C=16; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(ta)); } };
template <> struct LD<1,4> { static constexpr int R=16, C=32; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta)); } };
template <> struct LD<1,8> { static constexpr int R=32, C=64; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]) : "r"(ta)); } };
template <> struct LD<1,16> { static constexpr int R=64, C=128; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]), "=r"(v[32]), "=r"(v[33]), "=r"(v[34]), "=r"(v[35]), "=r"(v[36]), "=r"(v[37]), "=r"(v[38]), "=r"(v[39]), "=r"(v[40]), "=r"(v[41]), "=r"(v[42]), "=r"(v[43]), "=r"(v[44]), "=r"(v[45]), "=r"(v[46]), "=r"(v[47]), "=r"(v[48]), "=r"(v[49]), "=r"(v[50]), "=r"(v[51]), "=r"(v[52]), "=r"(v[53]), "=r"(v[54]), "=r"(v[55]), "=r"(v[56]), "=r"(v[57]), "=r"(v[58]), "=r"(v[59]), "=r"(v[60]), "=r"(v[61]), "=r"(v[62]), "=r"(v[63]) : "r"(ta)); } };
template <> struct LD<2,1> { static constexpr int R=1, C=2; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0}, [%1], 1;" : "=r"(v[0]) : "r"(ta)); } };
template <> struct LD<2,2> { static constexpr int R=2, C=4; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0,%1}, [%2], 2;" : "=r"(v[0]), "=r"(v[1]) : "r"(ta)); } };
template <> struct LD<2,4> { static constexpr int R=4, C=8; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x4.b32 {%0,%1,%2,%3}, [%4], 4;" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(ta)); } };
template <> struct LD<2,8> { static constexpr int R=8, C=16; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], 8;" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(ta)); } };
template <> struct LD<2,16> { static constexpr int R=16, C=32; __device__ __forceinline__ static void ld(uint32_t* v, uint32_t ta) {
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16], 16;" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta)); } };


// ---- host helpers
static PFN_cuTensorMapEncodeTiled_v12000 g_enc;
static void init_enc() { cudaDriverEntryPointQueryResult qr; CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&g_enc, 12000, cudaEnableDefault, &qr)); }
// bf16 2D map, dims {d0 (inner, elems), d1}, row pitch bytes, box {b0, b1}, SWIZZLE_128B (b0*2 == 128)
static CUtensorMap mkmap(void* base, ull d0, ull d1, ull pitch, unsigned b0, unsigned b1) {
  CUtensorMap m; cuuint64_t dims[2] = {d0, d1}, strides[1] = {pitch}; cuuint32_t box[2] = {b0, b1}, es[2] = {1, 1};
  CUresult r = g_enc(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "tensor map encode failed %d\n", (int)r); exit(1); }
  return m;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
static unsigned read_to() { unsigned t; CK(cudaMemcpyFromSymbol(&t, g_to, 4)); return t; }
static void clr_to() { unsigned z = 0; CK(cudaMemcpyToSymbol(g_to, &z, 4)); }

// ================================================================== U12a: TMA ingress ==================================================================
constexpr int A_ATOM = 7168, A_ATOMS = 16, A_BYTES = A_ATOM * A_ATOMS;  // 56 rows x 64 K x 2 B per atom; 112 KB = 50 tokens padded to N=56
constexpr int SLICE = 131072;                                           // weight bytes per CTA per pass (64 rows x 1024 K x 2 B)
constexpr int DYN_FORCE = 200 * 1024;                                   // dynamic smem requested so only 1 CTA fits per SM
struct AP { int D, S, NS, passes, drain, amode /*0 none,1 unicast,2 multicast*/, csz, real; ull flatStride, realStride; unsigned* bad; };

__global__ void __launch_bounds__(32, 1) k_tma(const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap ma, AP p, const bf16* pool, const bf16* Ag, ull* out) {
  extern __shared__ uint8_t smraw[];
  __shared__ __align__(8) uint64_t sbar[16];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u;
  const uint32_t bA = base, bR = base + (p.amode ? A_BYTES : 0), barF = su32(sbar), barA = barF + 8 * 8;
  const int lane = threadIdx.x, cta = blockIdx.x, D = p.D, S = p.S, NS = p.NS;
  const uint32_t rank = p.csz > 1 ? cluster_rank() : 0;
  if (lane == 0) { for (int i = 0; i < 8; ++i) mbar_init(barF + 8 * i, 1); mbar_init(barA, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  if (p.csz > 1) cluster_sync(); else __syncwarp();
  uint32_t ph[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int total = p.passes * NS;
  ull c0 = 0, g0 = 0, cm = 0, gm = 0;
  auto issue = [&](int pas, int j, int slot) {
    const uint32_t bar = barF + 8 * slot, dst = bR + slot * S;
    mbar_expect(bar, S);
    if (!p.real) tma2d(dst, &mw, bar, 0, cta * 1024 + j * (S / 128) + (int)(pas * p.flatStride));
    else { const int nb = S / 8192, rowc = cta * 64 + (int)(pas * p.realStride); for (int b = 0; b < nb; ++b) tma2d(dst + b * 8192, &mw, bar, (j * nb + b) * 64, rowc); }
  };
  auto issueA = [&]() {
    mbar_expect(barA, A_BYTES);
    if (p.amode == 1) { for (int a = 0; a < A_ATOMS; ++a) tma2d(bA + a * A_ATOM, &ma, barA, a * 64, 0); }
    else { const uint16_t mask = (uint16_t)((1u << p.csz) - 1); for (int a = rank; a < A_ATOMS; a += p.csz) tma2d_mc(bA + a * A_ATOM, &ma, barA, a * 64, 0, mask); }
  };
  if (lane == 0) { g0 = gt(); c0 = clk(); }
  if (p.amode == 0 && !p.drain) {
    if (lane == 0) {
      const int pre = min(D, total);
      for (int i = 0; i < pre; ++i) issue(i / NS, i % NS, i % D);
      for (int g = 0; g < total; ++g) {
        const int slot = g % D;
        if (g == total / 8) { cm = clk(); gm = gt(); }
        mbar_wait(barF + 8 * slot, ph[slot]); ph[slot] ^= 1;
        if (g + D < total) issue((g + D) / NS, (g + D) % NS, slot);
      }
    }
  } else {
    for (int pas = 0; pas < p.passes; ++pas) {
      if (lane == 0) {
        if (pas == p.passes / 8) { cm = clk(); gm = gt(); }
        if (p.amode) issueA();
        const int pre = min(D, NS);
        for (int j = 0; j < pre; ++j) issue(pas, j, j % D);
        for (int j = 0; j < NS; ++j) {
          const int slot = j % D;
          mbar_wait(barF + 8 * slot, ph[slot]); ph[slot] ^= 1;
          if (j + D < NS) issue(pas, j + D, slot);
        }
        if (p.amode) mbar_wait(barA, pas & 1);
      }
      if (p.csz > 1) cluster_sync(); else __syncwarp();
    }
  }
  __syncwarp();
  ull c1 = 0, g1 = 0;
  if (lane == 0) { c1 = clk(); g1 = gt(); out[cta * 4] = c1 - c0; out[cta * 4 + 1] = g1 - gm; out[cta * 4 + 2] = c1 - cm; }
  // verify: last min(D,NS) stages of the last pass (still resident, nothing issued after them) and the A tile
  unsigned bad = 0;
  const int pl = p.passes - 1, nv = min(D, NS);
  for (int j = NS - nv; j < NS; ++j) {
    const int slot = (p.drain || p.amode) ? j % D : (pl * NS + j) % D;
    const uint32_t* sm = (const uint32_t*)(smraw + (bR + slot * S - raw));
    if (!p.real) {
      const int R = S / 128; const size_t row0 = (size_t)cta * 1024 + j * R + (size_t)pl * p.flatStride;
      for (int r = lane; r < R; r += 32) {
        const uint32_t* g = (const uint32_t*)(pool + (row0 + r) * 64); uint32_t a = 0, b = 0;
        for (int w = 0; w < 32; ++w) { a += sm[r * 32 + w]; b += g[w]; }
        bad += (a != b);
      }
    } else {
      const int nb = S / 8192; const size_t rowc = (size_t)cta * 64 + (size_t)pl * p.realStride;
      for (int b0 = 0; b0 < nb; ++b0) for (int r = lane; r < 64; r += 32) {
        const uint32_t* g = (const uint32_t*)(pool + (rowc + r) * 1024 + (size_t)(j * nb + b0) * 64); uint32_t a = 0, b = 0;
        for (int w = 0; w < 32; ++w) { a += sm[b0 * 2048 + r * 32 + w]; b += g[w]; }
        bad += (a != b);
      }
    }
  }
  if (p.amode) {
    const uint32_t* sm = (const uint32_t*)(smraw + (bA - raw));
    for (int a = 0; a < A_ATOMS; ++a) for (int r = lane; r < 56; r += 32) {
      const uint32_t* g = (const uint32_t*)(Ag + (size_t)r * 1024 + a * 64); uint32_t x = 0, y = 0;
      for (int w = 0; w < 32; ++w) { x += sm[(a * A_ATOM) / 4 + r * 32 + w]; y += g[w]; }
      bad += (x != y);
    }
  }
  if (bad) atomicAdd(p.bad, bad);
}

__global__ void k_coop(unsigned* cnt, int* res, unsigned long long limit) {
  // grid-wide barrier with a bounded spin: res[0] = #CTAs that saw all arrive, res[1] = min cluster-agnostic arrivals seen
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(cnt, 1u);
    ull t0 = clk(); unsigned v;
    do { v = *((volatile unsigned*)cnt); } while (v < gridDim.x && clk() - t0 < limit);
    if (v >= gridDim.x) atomicAdd(&res[0], 1);
    atomicMin(&res[1], (int)v);
  }
}
__global__ void __cluster_dims__(2, 1, 1) k_coop_cd(unsigned* cnt, int* res, unsigned long long limit) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(cnt, 1u);
    ull t0 = clk(); unsigned v;
    do { v = *((volatile unsigned*)cnt); } while (v < gridDim.x && clk() - t0 < limit);
    if (v >= gridDim.x) atomicAdd(&res[0], 1);
    atomicMin(&res[1], (int)v);
  }
}

static cudaError_t launch_tma(int ncta, int csz, size_t dyn, const CUtensorMap& mw, const CUtensorMap& ma, AP p, const bf16* pool, const bf16* Ag, ull* out, bool coop = false) {
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(ncta); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = dyn; cfg.stream = 0;
  cudaLaunchAttribute at[2]; int na = 0;
  if (csz > 1) { at[na].id = cudaLaunchAttributeClusterDimension; at[na].val.clusterDim.x = csz; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1; ++na; }
  cfg.attrs = at; cfg.numAttrs = na;
  return cudaLaunchKernelEx(&cfg, k_tma, mw, ma, p, pool, Ag, out);
}

static void u12a() {
  printf("{\"u12a\":\"begin\"}\n"); fflush(stdout);
  const size_t POOL_COLD = (size_t)148 * SLICE * 100;  // 100 distinct passes of 18.9 MB
  bf16 *pool, *Ag; ull* out; unsigned* bad; char* flush;
  CK(cudaMalloc(&pool, POOL_COLD)); CK(cudaMalloc(&Ag, 64 * 1024 * 2)); CK(cudaMalloc(&out, 4 * 160 * 8)); CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&flush, 512u << 20));
  k_init<<<4096, 256>>>(pool, POOL_COLD / 2, 12345u); k_init<<<64, 256>>>(Ag, 64 * 1024, 777u); CK(cudaDeviceSynchronize());
  CK(cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_FORCE));
  CK(cudaFuncSetAttribute(k_tma, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  const ull rowsFlat = POOL_COLD / 128, rowsReal = POOL_COLD / 2048;
  CUtensorMap mflat[3] = {mkmap(pool, 64, rowsFlat, 128, 64, 64), mkmap(pool, 64, rowsFlat, 128, 64, 128), mkmap(pool, 64, rowsFlat, 128, 64, 256)};
  CUtensorMap mreal = mkmap(pool, 1024, rowsReal, 2048, 64, 64);
  CUtensorMap mA = mkmap(Ag, 1024, 56, 2048, 64, 56);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  // occupancy of clusters at the smem we need
  for (int csz : {2, 4, 8}) for (size_t smem : {(size_t)(A_BYTES + 32768 + 1024 + 128), (size_t)(A_BYTES + 65536 + 1024 + 128), (size_t)DYN_FORCE, (size_t)(227 * 1024 - 1024)}) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(NSM / csz * csz); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = csz; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)std::max<size_t>(smem, DYN_FORCE));
    int n = -1; cudaError_t e = cudaOccupancyMaxActiveClusters(&n, k_tma, &cfg);
    printf("{\"u12a\":\"occ\",\"cluster\":%d,\"dyn_smem\":%zu,\"max_active_clusters\":%d,\"max_ctas\":%d,\"err\":\"%s\"}\n", csz, smem, n, n * csz, cudaGetErrorString(e));
    cudaGetLastError();
  }
  CK(cudaFuncSetAttribute(k_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_FORCE));

  struct Cfg { const char* tag; int S, D, real, warm, drain, amode, csz; };
  std::vector<Cfg> cs;
  for (int warm = 1; warm >= 0; --warm) {
    for (int S : {8192, 16384, 32768}) for (int D : {2, 3, 4, 6}) cs.push_back({warm ? "flat_warm" : "flat_cold", S, D, 0, warm, 0, 0, 1});
    for (int S : {8192, 16384}) for (int D : {2, 3, 4, 6}) cs.push_back({warm ? "real_warm" : "real_cold", S, D, 1, warm, 0, 0, 1});
  }
  for (int S : {8192, 16384, 32768}) for (int D : {2, 4}) cs.push_back({"flat_warm_1shot", S, D, 0, 1, 1, 0, 1});
  for (int S : {8192, 16384}) for (int D : {2, 4}) cs.push_back({"real_warm_1shot", S, D, 1, 1, 1, 0, 1});
  // activation tile + weights (real layout, 8 KB x4 ring), one shot per pass
  cs.push_back({"A+W_unicast", 8192, 4, 1, 1, 1, 1, 1});
  for (int c : {2, 4}) { cs.push_back({"A+W_unicast_cl", 8192, 4, 1, 1, 1, 1, c}); cs.push_back({"A+W_mcast_cl", 8192, 4, 1, 1, 1, 2, c}); }
  cs.push_back({"A+W_unicast_16K", 16384, 4, 1, 1, 1, 1, 1});
  for (int c : {2, 4}) { cs.push_back({"A+W_unicast_cl_16K", 16384, 4, 1, 1, 1, 1, c}); cs.push_back({"A+W_mcast_cl_16K", 16384, 4, 1, 1, 1, 2, c}); }

  for (auto& c : cs) {
    AP p = {}; p.D = c.D; p.S = c.S; p.real = c.real; p.drain = c.drain; p.amode = c.amode; p.csz = c.csz;
    const CUtensorMap& mw = c.real ? mreal : mflat[c.S == 8192 ? 0 : c.S == 16384 ? 1 : 2];
    p.NS = SLICE / c.S; p.flatStride = c.warm ? 0 : (ull)NSM * 1024; p.realStride = c.warm ? 0 : (ull)NSM * 64;
    p.bad = bad;
    const size_t dyn = DYN_FORCE;
    if ((size_t)(c.amode ? A_BYTES : 0) + (size_t)c.D * c.S + 2048 > dyn) continue;
    int ncta = NSM;
    if (c.csz > 1) {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(NSM / c.csz * c.csz); cfg.blockDim = dim3(32); cfg.dynamicSmemBytes = dyn;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = c.csz; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
      cfg.attrs = at; cfg.numAttrs = 1; int n = 0; cudaOccupancyMaxActiveClusters(&n, k_tma, &cfg); cudaGetLastError();
      ncta = n * c.csz;
    }
    if (ncta == 0) { printf("{\"u12a\":\"%s\",\"S\":%d,\"D\":%d,\"cluster\":%d,\"err\":\"0 co-resident clusters\"}\n", c.tag, c.S, c.D, c.csz); continue; }
    // trial to pick pass count (>= ~400 us)
    int passes = c.warm ? 20 : 20;
    const int maxPasses = c.warm ? 4000 : 100;
    std::vector<double> evs, aggs, perSm, cyc_chk; double nspc = 0; unsigned badTot = 0; cudaError_t err = cudaSuccess; std::vector<double> pass_ns;
    for (int rep = -2; rep < 5; ++rep) {
      p.passes = rep == -2 ? 20 : passes;
      CK(cudaMemset(bad, 0, 4)); clr_to();
      if (!c.warm) CK(cudaMemsetAsync(flush, rep & 0xff, 512u << 20));
      CK(cudaEventRecord(e0));
      err = launch_tma(ncta, c.csz, dyn, mw, mA, p, pool, Ag, out);
      CK(cudaEventRecord(e1));
      if (err != cudaSuccess) break;
      CK(cudaEventSynchronize(e1)); CK(cudaDeviceSynchronize());
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      ull h[4 * NSM]; CK(cudaMemcpy(h, out, 4 * ncta * 8, cudaMemcpyDeviceToHost));
      unsigned hb; CK(cudaMemcpy(&hb, bad, 4, cudaMemcpyDeviceToHost)); badTot += hb + read_to();
      if (rep == -2) { passes = std::min(maxPasses, std::max(20, (int)(20 * 450.0 / std::max(20.0, (double)ms * 1000 - 8)))); continue; }
      if (rep == -1) continue;  // warm-up launch at final pass count
      std::vector<double> ns; double mx = 0, mxg = 0, ratio = 0;
      for (int i = 0; i < ncta; ++i) { ratio += (double)h[4 * i + 1] / h[4 * i + 2]; mx = std::max(mx, (double)h[4 * i]); mxg = std::max(mxg, (double)h[4 * i + 1]); }
      ratio /= ncta; nspc = ratio;
      const double bytesCta = (double)p.passes * (SLICE + (c.amode ? A_BYTES : 0));
      for (int i = 0; i < ncta; ++i) ns.push_back(bytesCta / (h[4 * i] * ratio));  // GB/s (bytes/ns)
      double sm = 0; for (double x : ns) sm += x;
      perSm.push_back(sm / ncta); aggs.push_back(bytesCta * ncta / (ms * 1e6)); evs.push_back(ms * 1000);
      cyc_chk.push_back((mx * ratio / 1000.0) / (ms * 1000));  // clock span / event time
      pass_ns.push_back(mx * ratio / p.passes);
      if (mxg < 100e3) badTot += 1000000;  // globaltimer span < 100 us: ns/cycle not trustworthy
    }
    // two-point delta cross-check: same config at P/2; (clk span delta) vs (event delta) cancels fixed launch/setup cost
    std::vector<double> dx;
    if (err == cudaSuccess) {
      std::vector<double> evh, mxh;
      for (int rep = -1; rep < 4; ++rep) {
        AP q = p; q.passes = std::max(2, p.passes / 2);
        CK(cudaMemset(bad, 0, 4)); clr_to();
        if (!c.warm) CK(cudaMemsetAsync(flush, rep & 0xff, 512u << 20));
        CK(cudaEventRecord(e0)); err = launch_tma(ncta, c.csz, dyn, mw, mA, q, pool, Ag, out); CK(cudaEventRecord(e1));
        if (err != cudaSuccess) break;
        CK(cudaEventSynchronize(e1)); CK(cudaDeviceSynchronize());
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        ull h[4 * NSM]; CK(cudaMemcpy(h, out, 4 * ncta * 8, cudaMemcpyDeviceToHost));
        if (rep < 0) continue;
        double mx = 0, ratio = 0; for (int i = 0; i < ncta; ++i) { mx = std::max(mx, (double)h[4 * i]); ratio += (double)h[4 * i + 1] / h[4 * i + 2]; }
        ratio /= ncta; evh.push_back(ms * 1000); mxh.push_back(mx * ratio / 1000.0);
      }
      if (err == cudaSuccess) {
        // spans: full-P spans in cyc_chk*evs
        std::vector<double> spP; for (size_t i = 0; i < cyc_chk.size(); ++i) spP.push_back(cyc_chk[i] * evs[i]);
        const double dclk = med(spP) - med(mxh), dev = med(evs) - med(evh);
        dx.push_back(dev > 0 ? dclk / dev : 0);
      }
    }
    if (err != cudaSuccess) { printf("{\"u12a\":\"%s\",\"S\":%d,\"D\":%d,\"cluster\":%d,\"err\":\"%s\"}\n", c.tag, c.S, c.D, c.csz, cudaGetErrorString(err)); cudaGetLastError(); continue; }
    const double xc = dx.empty() ? 0 : dx[0];
    printf("{\"u12a\":\"%s\",\"S_KB\":%d,\"D\":%d,\"drain\":%d,\"A\":%d,\"cluster\":%d,\"ctas\":%d,\"passes\":%d,\"per_sm_GBps\":%.1f,\"agg_TBps\":%.2f,\"pass_ns\":%.0f,\"ns_per_cyc\":%.4f,\"event_us\":%.0f,\"xchk_clk_over_event\":%.3f,\"xchk\":\"%s\",\"bad\":%u}\n",
           c.tag, c.S / 1024, c.D, c.drain, c.amode, c.csz, ncta, p.passes, med(perSm), med(aggs), med(pass_ns), nspc, med(evs), xc, (xc > 0.98 && xc < 1.02) ? "ok" : "FLAG", badTot);
    fflush(stdout);
  }
  cudaFree(pool); cudaFree(Ag); cudaFree(out); cudaFree(bad); cudaFree(flush);
}

// ---- cooperative + cluster launch attempts
static void u12coop() {
  unsigned* cnt; int* res; CK(cudaMalloc(&cnt, 4)); CK(cudaMalloc(&res, 8));
  int dev = 0; cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
  int cop = 0; CK(cudaDeviceGetAttribute(&cop, cudaDevAttrCooperativeLaunch, dev));
  printf("{\"coop\":\"info\",\"sms\":%d,\"cooperativeLaunch\":%d,\"max_dyn_smem_optin\":%zu}\n", pr.multiProcessorCount, cop, (size_t)pr.sharedMemPerBlockOptin);
  const unsigned long long lim = 2000000000ull;  // ~1 s of cycles
  struct T { const char* tag; int csz; bool coopAttr; int grid; size_t smem; int kind; };
  int occ2 = 0, occ4 = 0;
  CK(cudaFuncSetAttribute(k_coop, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024 - 2048));
  CK(cudaFuncSetAttribute(k_coop, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CK(cudaFuncSetAttribute(k_coop_cd, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024 - 2048));
  { cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(148); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 150 * 1024; cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim = {2, 1, 1}; cfg.attrs = at; cfg.numAttrs = 1; cudaOccupancyMaxActiveClusters(&occ2, k_coop, &cfg);
    at[0].val.clusterDim = {4, 1, 1}; cudaOccupancyMaxActiveClusters(&occ4, k_coop, &cfg); cudaGetLastError(); }
  int mb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_coop, 128, 150 * 1024));
  printf("{\"coop\":\"occ\",\"blocks_per_sm_at_150KB\":%d,\"max_clusters_c2\":%d,\"max_clusters_c4\":%d}\n", mb, occ2, occ4);
  std::vector<T> ts = {
    {"coopAttr+cluster2 grid148", 2, true, 148, 150 * 1024, 0}, {"coopAttr+cluster4 grid148", 4, true, 148, 150 * 1024, 0},
    {"coopAttr+cluster2 grid=maxcl*2", 2, true, occ2 * 2, 150 * 1024, 0}, {"coopAttr+cluster4 grid=maxcl*4", 4, true, occ4 * 4, 150 * 1024, 0},
    {"coopAttr no cluster grid148", 1, true, 148, 150 * 1024, 0},
    {"cluster2 only (no coop attr) grid148", 2, false, 148, 150 * 1024, 0}, {"cluster4 only (no coop attr) grid148", 4, false, 148, 150 * 1024, 0},
    {"cluster2 only grid=maxcl*2", 2, false, occ2 * 2, 150 * 1024, 0}, {"cluster4 only grid=maxcl*4", 4, false, occ4 * 4, 150 * 1024, 0},
    {"coopAttr+cluster2 grid148 smem227", 2, true, 148, 227 * 1024 - 2048, 0},
    {"cudaLaunchCooperativeKernel + __cluster_dims__(2) grid148", 2, true, 148, 150 * 1024, 1},
  };
  for (auto& t : ts) {
    CK(cudaMemset(cnt, 0, 4)); int init[2] = {0, 0x7fffffff}; CK(cudaMemcpy(res, init, 8, cudaMemcpyHostToDevice));
    cudaError_t e;
    if (t.kind == 1) {
      void* args[3] = {&cnt, &res, (void*)&lim}; ull l2 = lim; args[2] = &l2;
      e = cudaLaunchCooperativeKernel((void*)k_coop_cd, dim3(t.grid), dim3(128), args, t.smem, 0);
    } else {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(t.grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = t.smem;
      cudaLaunchAttribute at[2]; int na = 0;
      if (t.coopAttr) { at[na].id = cudaLaunchAttributeCooperative; at[na].val.cooperative = 1; ++na; }
      if (t.csz > 1) { at[na].id = cudaLaunchAttributeClusterDimension; at[na].val.clusterDim = {(unsigned)t.csz, 1, 1}; ++na; }
      cfg.attrs = at; cfg.numAttrs = na;
      e = cudaLaunchKernelEx(&cfg, k_coop, cnt, res, lim);
    }
    cudaError_t e2 = cudaSuccess; int h[2] = {-1, -1};
    if (e == cudaSuccess) { e2 = cudaDeviceSynchronize(); CK(cudaMemcpy(h, res, 8, cudaMemcpyDeviceToHost)); }
    printf("{\"coop\":\"%s\",\"grid\":%d,\"launch\":\"%s\",\"sync\":\"%s\",\"ctas_that_saw_full_barrier\":%d,\"min_arrivals_seen\":%d}\n", t.tag, t.grid, cudaGetErrorString(e), cudaGetErrorString(e2), h[0], h[1]);
    cudaGetLastError();
    fflush(stdout);
  }
  cudaFree(cnt); cudaFree(res);
}

// ================================================================== U12b: tcgen05.ld throughput ==================================================================
// smem: ones for the M=64 accumulate run (A 8 KB, B 16 KB), zeros 32 KB for the M=128 clear. TMEM 256 cols: [0,128) read by loaders, [128,192) concurrent-MMA target.
// The M=128 zero MMA first clears all 128 lanes of cols 0..127; then 16 MMAs M=64 N=128 (ones, K=16 each) give 256.0 in every valid element, so lanes the
// M=64 layout does NOT write read back 0 -> 32x32b lane 16..31 hits count the layout (expected 0 if rows q*16..+15 sit in lanes q*32..+15).
constexpr int LB_SMEM = 8192 + 16384 + 32768 + 1024;

__device__ __forceinline__ void ld_setup(uint32_t& tb, uint8_t*& sm, uint32_t& base, uint32_t* tslot, int tid, int warp, uint32_t nbar_addr, int nb) {
  extern __shared__ uint8_t smraw[];
  const uint32_t raw = su32(smraw); base = (raw + 1023) & ~1023u; sm = smraw + (base - raw);
  if (warp == 0) tmem_alloc(su32(tslot), 256);
  if (tid == 0) { for (int i = 0; i < nb; ++i) mbar_init(nbar_addr + 8 * i, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  for (int i = tid; i < (8192 + 16384) / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = 0x3F803F80u;
  for (int i = tid; i < 32768 / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(sm + 24576)[i] = 0u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); fence_after();
  tb = *tslot;
}

// issuer: clear with M=128, then 16 x M=64 N=128 ones; leaves 256.0 in the valid elements. Called by one thread; waits on bar.
__device__ __forceinline__ void init_acc(uint32_t tb, uint32_t base, uint32_t bar, uint32_t par) {
  mma_tc(tb, sdesc(base + 24576), sdesc(base + 24576 + 16384), idesc(128, 128), 0);
  for (int kk = 0; kk < 16; ++kk) mma_tc(tb, sdesc(base + (kk & 3) * 32), sdesc(base + 8192 + (kk & 3) * 32), idesc(128, 64), 1);
  commit(bar);
  mbar_wait(bar, par);
}

template <int SH, int X>
__global__ void __launch_bounds__(288, 1) k_ld(int nw, int mma, int mode, int iters, int lanemode, double* out, unsigned* bad) {
  typedef LD<SH, X> L; constexpr int C = L::C, R = L::R, NL = 128 / C;
  __shared__ uint32_t tslot; __shared__ __align__(8) uint64_t sb[4]; __shared__ volatile int stop; __shared__ ull wc[8], wg[2]; __shared__ unsigned mmacnt; __shared__ ull mmaspan;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t tb, base; uint8_t* sm;
  if (tid == 0) { stop = 0; mmacnt = 0; }
  ld_setup(tb, sm, base, &tslot, tid, warp, su32(sb), 2);
  if (tid == 256) init_acc(tb, base, su32(sb), 0);
  __syncthreads(); fence_after();
  if (warp == 8 && mma) {
    const bool el = elect_one();
    const ull dA0 = sdesc(base), dB0 = sdesc(base + 8192); const uint32_t id64 = idesc(64, 64);
    uint32_t par = 1; unsigned n = 0; const ull t0 = clk();
    while (!stop) {
      if (el) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) mma_tc(tb + 128, dA0 + (ull)((kk & 3) * 2), dB0 + (ull)((kk & 3) * 2), id64, 1);
        commit(su32(sb) + 8);
      }
      mbar_wait(su32(sb) + 8, par); par ^= 1; ++n;
    }
    if (el) { mmacnt = n; mmaspan = clk() - t0; }
  }
  uint32_t sink = 0;
  if (warp < nw) {
    const uint32_t ta0 = tb + ((uint32_t)(lanemode == 0 ? 32 * (warp & 3) : 0) << 16);
    uint32_t vb[NL * R];
    // verify (untimed)
    unsigned b = 0, off = 0;
#pragma unroll
    for (int i = 0; i < NL; ++i) L::ld(vb + i * R, ta0 + i * C);
    wait_ld();
#pragma unroll
    for (int i = 0; i < NL * R; ++i) {
      const bool hit = __uint_as_float(vb[i]) == 256.f;
      if (SH == 0) { if (lane < 16) b += !hit; else off += hit; } else b += !hit;
    }
    if (lanemode != 0) b = 0;  // illegal-lane variant: values not checked
    if (b) atomicAdd(&bad[0], b);
    if (off) atomicAdd(&bad[1], off);
    named_bar(1, nw * 32);
    const ull c0 = clk(); ull cm = c0, gm = 0;
    for (int it = 0; it < iters; ++it) {
      if (it == iters / 16) { cm = clk(); gm = gt(); }
      if (mode == 0) {
        uint32_t v[R];
#pragma unroll 4
        for (int i = 0; i < NL; ++i) { L::ld(v, ta0 + i * C); wait_ld(); sink ^= v[0]; }
      } else {
#pragma unroll
        for (int i = 0; i < NL; ++i) L::ld(vb + i * R, ta0 + i * C);
        wait_ld();
        sink ^= vb[0] ^ vb[NL * R - 1];
      }
    }
    const ull c1 = clk(), g1 = gt();
    if (lane == 0) { wc[warp] = c1 - c0; if (warp == 0) { wg[0] = g1 - gm; wg[1] = c1 - cm; } }
    if (sink == 0x9e3779b9u) bad[3] = 1;
    named_bar(1, nw * 32);
    if (tid == 0) stop = 1;
  }
  __syncthreads();
  if (tid == 0) {
    double s = 0, mx = 0; for (int w = 0; w < nw; ++w) { s += wc[w]; mx = (double)wc[w] > mx ? (double)wc[w] : mx; }
    out[0] = s / nw / iters; out[1] = mx / iters; out[2] = (double)wg[0] / (double)wg[1]; out[3] = mx;
  }
  __syncthreads();
  if (tid == 256 && mma) { out[4] = mmacnt; out[5] = (double)mmaspan; }  // (tid 256 = lane 0 of warp 8; the elected lane is lane 0)
  __syncthreads();
  if (warp == 0) tmem_free(tb, 256);
}

typedef void (*KLD)(int, int, int, int, int, double*, unsigned*);
struct LdK { const char* shape; int sh, X; KLD f; int C, R; };
#define LDK(SH, X, NAME) {NAME, SH, X, k_ld<SH, X>, LD<SH, X>::C, LD<SH, X>::R}
static std::vector<LdK> ldks() {
  return {LDK(0, 1, "32x32b"), LDK(0, 2, "32x32b"), LDK(0, 4, "32x32b"), LDK(0, 8, "32x32b"), LDK(0, 16, "32x32b"),
          LDK(1, 1, "16x256b"), LDK(1, 2, "16x256b"), LDK(1, 4, "16x256b"), LDK(1, 8, "16x256b"), LDK(1, 16, "16x256b"),
          LDK(2, 1, "16x32bx2"), LDK(2, 2, "16x32bx2"), LDK(2, 4, "16x32bx2"), LDK(2, 8, "16x32bx2"), LDK(2, 16, "16x32bx2")};
}

// ---- U11 anomaly decomposition: 4 loader warps, batch of 4 x 16x256b.x2 (cols 0..63) then one wait::ld (exactly U11's pattern)
// scn 0: idle, batches back to back. scn 1: after each 64-MMA chain (N=56, K=1024) completes (commit -> mbarrier), first/2nd/3rd/4th batch timed.
// scn 2: batches while a continuous MMA stream runs on other columns.
__global__ void __launch_bounds__(288, 1) k_anom(int nw, int scn, int lanemode, int fenceEach, int iters, double* out, unsigned* bad) {
  __shared__ uint32_t tslot; __shared__ __align__(8) uint64_t sb[4]; __shared__ volatile int stop; __shared__ double acc[4][8]; __shared__ ull tcommit, tseen[4]; __shared__ ull wspan[4], wg[2]; __shared__ double lat;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t tb, base; uint8_t* sm;
  if (tid == 0) { stop = 0; lat = 0; for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) acc[i][j] = 0; }
  ld_setup(tb, sm, base, &tslot, tid, warp, su32(sb), 3);
  if (tid == 256) init_acc(tb, base, su32(sb), 0);
  __syncthreads(); fence_after();
  const uint32_t b0 = su32(sb) + 8, b1 = su32(sb) + 16;
  if (warp == 8 && scn == 1) {  // whole warp 8 runs the loop (named barrier needs full warps); elected lane issues
    const bool el = elect_one();
    const ull dA0 = sdesc(base), dB0 = sdesc(base + 8192); const uint32_t id56 = idesc(56, 64);
    for (int t = 0; t < iters; ++t) {
      if (el) {
#pragma unroll 8
        for (int kk = 0; kk < 64; ++kk) mma_tc(tb, dA0 + (ull)((kk & 3) * 2), dB0 + (ull)((kk & 3) * 2), id56, kk > 0);
        tcommit = clk(); commit(b0);
      }
      __syncwarp();
      named_bar(2, (nw + 1) * 32);  // loaders finished with this trial
    }
  }
  if (warp == 8 && scn == 2) {
    const bool el = elect_one();
    const ull dA0 = sdesc(base), dB0 = sdesc(base + 8192); const uint32_t id56 = idesc(56, 64);
    uint32_t par = 0;
    while (!stop) {
      if (el) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) mma_tc(tb + 128, dA0 + (ull)((kk & 3) * 2), dB0 + (ull)((kk & 3) * 2), id56, 1);
        commit(b1);
      }
      mbar_wait(b1, par); par ^= 1;
    }
  }
  uint32_t sink = 0;
  if (warp < nw) {
    const uint32_t ta0 = tb + ((uint32_t)(lanemode == 0 ? 32 * (warp & 3) : 0) << 16);
    named_bar(1, nw * 32);
    double bt[4] = {0, 0, 0, 0}, seen = 0; int nb = 0; ull ws = 0;
    const ull cs = clk(); ull cm = cs, gm = 0;
    for (int t = 0; t < iters; ++t) {
      if (t == iters / 16) { cm = clk(); gm = gt(); }
      uint32_t v[4][8];
      if (scn == 1) { mbar_wait(b0, t & 1); fence_after(); if (lane == 0) tseen[warp] = clk(); }
      const int nbatch = scn == 1 ? 4 : 1;
      for (int bI = 0; bI < nbatch; ++bI) {
        const ull a = clk();
        if (fenceEach) fence_after();
#pragma unroll
        for (int j = 0; j < 4; ++j) LD<1, 2>::ld(v[j], ta0 + j * 16);
        wait_ld();
        sink ^= v[0][0] ^ v[3][7];
        const ull b = clk();
        bt[bI] += (double)(b - a);
        if (scn == 1 && bI == 0 && lane == 0 && t >= 0) seen += (double)(tseen[warp] - tcommit);
      }
      if (scn == 1) named_bar(2, (nw + 1) * 32);
      ++nb;
    }
    const ull ce = clk(), ge = gt();
    if (lane == 0) { for (int i = 0; i < 4; ++i) atomicAdd(&acc[i][0], bt[i] / nb); atomicAdd(&lat, seen / nb); wspan[warp] = ce - cs; if (warp == 0) { wg[0] = ge - gm; wg[1] = ce - cm; } }
    if (sink == 0x9e3779b9u) bad[3] = 1;
    // verify one batch (legal lanes only): values must be 256.0f in cols 0..55 (col 56..63 are unwritten in scn 1)
    if (lanemode == 0 && scn != 1) {
      uint32_t v[4][8];
      for (int j = 0; j < 4; ++j) LD<1, 2>::ld(v[j], ta0 + j * 16);
      wait_ld(); unsigned b = 0;
      for (int j = 0; j < 4; ++j) for (int r = 0; r < 8; ++r) b += (__uint_as_float(v[j][r]) != 256.f);
      if (b) atomicAdd(&bad[0], b);
    }
    named_bar(1, nw * 32);
    if (tid == 0) stop = 1;
  }
  __syncthreads();
  if (tid == 0) {
    double mx = 0; for (int w = 0; w < nw; ++w) mx = (double)wspan[w] > mx ? (double)wspan[w] : mx;
    for (int i = 0; i < 4; ++i) out[i] = acc[i][0] / nw;
    out[4] = lat / nw; out[5] = (double)wg[0] / (double)wg[1]; out[6] = mx; out[7] = iters;
  }
  __syncthreads();
  if (warp == 0) tmem_free(tb, 256);
}

static void u12b() {
  printf("{\"u12b\":\"begin\"}\n"); fflush(stdout);
  double* out; unsigned* bad; CK(cudaMalloc(&out, 64 * 8)); CK(cudaMalloc(&bad, 16));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto ks = ldks();
  for (auto& k : ks) CK(cudaFuncSetAttribute((const void*)k.f, cudaFuncAttributeMaxDynamicSharedMemorySize, LB_SMEM));
  CK(cudaFuncSetAttribute(k_anom, cudaFuncAttributeMaxDynamicSharedMemorySize, LB_SMEM));
  for (auto& k : ks) for (int nw : {1, 4, 8}) for (int mma : {0, 1}) for (int mode : {0, 1}) {
    int iters = 64; double res[8];
    auto run = [&](int it, float& ms) {
      CK(cudaMemset(bad, 0, 16)); clr_to();
      CK(cudaEventRecord(e0)); k.f<<<1, 288, LB_SMEM>>>(nw, mma, mode, it, 0, out, bad); CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); CK(cudaMemcpy(res, out, 64, cudaMemcpyDeviceToHost)); CK(cudaEventElapsedTime(&ms, e0, e1));
    };
    float ms; run(iters, ms);
    iters = std::min(1 << 20, std::max(64, (int)(64 * 500.0 / std::max(5.0, (double)ms * 1000 - 25))));
    std::vector<double> cyc, evs, nsc, spans, mc; unsigned hb[4] = {0, 0, 0, 0};
    for (int rep = 0; rep < 3; ++rep) {
      run(iters, ms); unsigned h[4]; CK(cudaMemcpy(h, bad, 16, cudaMemcpyDeviceToHost)); for (int i = 0; i < 4; ++i) hb[i] = std::max(hb[i], h[i]);
      cyc.push_back(res[0]); evs.push_back(ms * 1000); nsc.push_back(res[2]); spans.push_back(res[3] * res[2] / 1000.0); mc.push_back(res[4]);
    }
    double xc = 0;
    {  // two-point delta cross-check at iters/2
      std::vector<double> ev2, sp2; const int it2 = std::max(8, iters / 2);
      for (int rep = 0; rep < 3; ++rep) { run(it2, ms); ev2.push_back(ms * 1000); sp2.push_back(res[3] * res[2] / 1000.0); }
      const double dev = med(evs) - med(ev2); xc = dev > 0 ? (med(spans) - med(sp2)) / dev : 0;
    }
    const double cy = med(cyc), nspc = med(nsc), span = med(spans), ev = med(evs), ns = cy * nspc;
    const double useful = 16.0 * 128 * 4 * nw;  // bytes per iter over all active warps (16 valid lanes x 128 cols x 4 B)
    printf("{\"u12b\":\"ld\",\"shape\":\"%s\",\"x\":%d,\"mode\":\"%s\",\"warps\":%d,\"mma\":%d,\"loads_per_unit\":%d,\"regs_per_load\":%d,\"ns_per_unit128col\":%.1f,\"ns_per_ld\":%.1f,\"GBps_SM_useful\":%.1f,\"iters\":%d,\"span_us\":%.0f,\"ns_per_cyc\":%.4f,\"mma_chains\":%.0f,\"xchk_clk_over_event\":%.3f,\"xchk\":\"%s\",\"bad\":%u,\"bad_off_lanes\":%u,\"timeouts\":%u}\n",
           k.shape, k.X, mode ? "batch" : "wait_each", nw, mma, 128 / k.C, k.R, ns, ns / (128 / k.C), useful / ns, iters, span, nspc, med(mc), xc,
           (xc > 0.98 && xc < 1.02 && span >= 100) ? "ok" : "FLAG", hb[0], hb[1], read_to());
    fflush(stdout);
  }
  // anomaly decomposition
  struct An { const char* tag; int nw, scn, lanemode, fenceEach, iters; };
  std::vector<An> as = {
    {"idle_1warp_legal", 1, 0, 0, 0, 40000}, {"idle_4warp_legal", 4, 0, 0, 0, 40000}, {"idle_4warp_legal_fence_each", 4, 0, 0, 1, 40000},
    {"idle_4warp_ILLEGAL_all_lane0(U10 k_lat)", 4, 0, 1, 0, 40000}, {"idle_8warp_legal", 8, 0, 0, 0, 40000},
    {"concurrentMMA_4warp_legal", 4, 2, 0, 0, 40000}, {"concurrentMMA_1warp_legal", 1, 2, 0, 0, 40000},
    {"after_64MMA_chain_4warp_legal(batch#0..3)", 4, 1, 0, 0, 3000}, {"after_64MMA_chain_1warp_legal(batch#0..3)", 1, 1, 0, 0, 3000},
    {"after_64MMA_chain_4warp_fence_each", 4, 1, 0, 1, 3000},
  };
  for (auto& a : as) {
    double r[8]; float ms; CK(cudaMemset(bad, 0, 16)); clr_to();
    CK(cudaEventRecord(e0)); k_anom<<<1, 288, LB_SMEM>>>(a.nw, a.scn, a.lanemode, a.fenceEach, a.iters, out, bad); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); CK(cudaMemcpy(r, out, 64, cudaMemcpyDeviceToHost)); CK(cudaEventElapsedTime(&ms, e0, e1));
    unsigned h[4]; CK(cudaMemcpy(h, bad, 16, cudaMemcpyDeviceToHost));
    const double nspc = r[5], span = r[6] * nspc / 1000.0, xc = span / (ms * 1000);
    printf("{\"u12b\":\"anom\",\"cfg\":\"%s\",\"batch_ns\":[%.1f,%.1f,%.1f,%.1f],\"commit_to_seen_ns\":%.1f,\"ns_per_cyc\":%.4f,\"span_us\":%.0f,\"event_us\":%.0f,\"xchk_clk_over_event\":%.3f,\"xchk\":\"%s\",\"bad\":%u,\"timeouts\":%u}\n",
           a.tag, r[0] * nspc, r[1] * nspc, r[2] * nspc, r[3] * nspc, r[4] * nspc, nspc, span, ms * 1000, xc, (xc > 0.98 && xc < 1.02 && span >= 100) ? "ok" : "FLAG", h[0], read_to());
    fflush(stdout);
  }
  cudaFree(out); cudaFree(bad);
}

// ================================================================== U12c: M=64 N=56 MMA chain fed from a TMA ring ==================================================================
// Swap-AB: weights (64 rows x K) are the A/M operand, 50 tokens padded to N=56 are the B operand (A_BYTES tile resident in smem, loaded once).
// One pass = 64 MMAs (K=16 each, 4 per 64-K atom) = K 1024. Passes repeat back-to-back; the accumulator is double-buffered in TMEM (cols 0..55 / 64..119) so the
// epilogue of pass p overlaps MMAs of pass p+1. flags: 1 NOTMA (MMAs read pre-filled ring), 2 NOMMA (consumer just releases stages), 4 NOEPI, 8 ONECOMMIT.
constexpr int F_NOTMA = 1, F_NOMMA = 2, F_NOEPI = 4, F_ONECOMMIT = 8, F_ONES = 16, F_NOFENCE = 32, F_ONEBUF = 64, F_NOSTAMP = 128;
struct CP { int D, kpb, passes, flags; unsigned* bad; };
// stamp slots (smem, dumped per CTA)
enum { X_START = 0, X_ASEEN, X_FULL0, X_M0, X_ACC0, X_LD0, X_EPIEND, X_ISSEND, X_LASTCOMMIT, X_LASTM0, X_GT0, X_GT1, X_IGT0, X_IGT1, X_ICLK0, X_ICLK1, X_LF = 16, X_LM0 = 32, X_LM1 = 48, X_LE = 64, X_LACC = 80, X_LLD = 81, XN = 96 };

__global__ void __launch_bounds__(256, 1) k_mma(const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap ma, CP p, ull* st, float* Dout) {
  extern __shared__ uint8_t smraw[];
  __shared__ __align__(8) uint64_t sbar[24]; __shared__ uint32_t tslot; __shared__ ull sst[XN];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u;
  const uint32_t bA = base, bR = base + A_BYTES;
  const uint32_t bf = su32(sbar), be = bf + 64, bacc = bf + 128, bacce = bf + 144, bAb = bf + 160, bDone = bf + 168;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, D = p.D, kpb = p.kpb, SB = 8192 * kpb, NS = 16 / kpb, flags = p.flags, cta = blockIdx.x;
  const int passes = p.passes, last = passes - 1;
  if (warp == 0) tmem_alloc(su32(&tslot), 128);
  if (tid == 0) {
    for (int i = 0; i < 8; ++i) { mbar_init(bf + 8 * i, 1); mbar_init(be + 8 * i, 1); }
    mbar_init(bacc, 1); mbar_init(bacc + 8, 1); mbar_init(bacce, 4); mbar_init(bacce + 8, 4); mbar_init(bAb, 1); mbar_init(bDone, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  uint32_t* smw = reinterpret_cast<uint32_t*>(smraw + (base - raw));
  for (int i = tid; i < (A_BYTES + 8192 * 16) / 4 && i < (A_BYTES + D * SB) / 4; i += 256) { uint32_t h = (uint32_t)i * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; smw[i] = (flags & F_ONES) ? 0x3F803F80u : (h & 0x3F7F3F7Fu); }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); fence_after();
  const uint32_t tb = tslot;
  if (tid == 0) { sst[X_START] = clk(); sst[X_GT0] = gt(); }
  const int uw = __shfl_sync(0xffffffff, tid >> 5, 0);  // warp index known to be warp-uniform: keeps descriptors on the uniform datapath
  if (uw == 0) {  // producer warp: all lanes run the loop, the elected lane issues
    const bool el = elect_one();
    if (!(flags & F_NOTMA)) {
      if (el) { mbar_expect(bAb, A_BYTES); for (int a = 0; a < A_ATOMS; ++a) tma2d(bA + a * A_ATOM, &ma, bAb, a * 64, 0); }
      const int total = passes * NS;
      int slot = 0, lap = 0, j = 0, pasP = 0;  // incremental counters: runtime div/mod per stage costs ~150 cycles of issue time
      for (int g = 0; g < total; ++g) {
        if (lap > 0) { mbar_wait(be + 8 * slot, (lap - 1) & 1); if (el && pasP == last) sst[X_LE + j] = clk(); }
        if (el) {
          mbar_expect(bf + 8 * slot, SB);
          for (int b = 0; b < kpb; ++b) tma2d(bR + slot * SB + b * 8192, &mw, bf + 8 * slot, (j * kpb + b) * 64, cta * 64);
        }
        if (++slot == D) { slot = 0; ++lap; }
        if (++j == NS) { j = 0; ++pasP; }
      }
    }
  } else if (uw == 1) {  // MMA issuer warp
    const bool el = elect_one();
    const ull dA0 = sdesc(bR), dB0 = sdesc(bA);
    const uint32_t id56 = idesc(56, 64);
    if (!(flags & F_NOTMA)) { mbar_wait(bAb, 0); if (el) sst[X_ASEEN] = clk(); }
    int slot = 0, lap = 0;
    for (int pas = 0; pas < passes; ++pas) {
      const int buf = pas & 1;
      if (pas == passes / 8 && el) { sst[X_ICLK0] = clk(); sst[X_IGT0] = gt(); }  // (clock64, globaltimer) pair taken well after the CTA barrier
      if (!(flags & (F_NOMMA | F_NOEPI)) && pas >= 2) mbar_wait(bacce + 8 * buf, ((pas >> 1) - 1) & 1);
      for (int j = 0; j < NS; ++j) {
        if (!(flags & F_NOTMA)) mbar_wait(bf + 8 * slot, lap & 1);
        const ull tf = (flags & F_NOSTAMP) ? 0 : clk();
        if (el && pas == 0 && j == 0) sst[X_FULL0] = tf;
        if (el && pas == last) sst[X_LF + j] = tf;
        const int cslot = slot; if (++slot == D) { slot = 0; ++lap; }
        if (flags & F_NOMMA) { if (el) mbar_arrive(be + 8 * cslot); continue; }
        if (!(flags & F_NOFENCE)) fence_after();
        if (el) {  // descriptors are built once (dA0/dB0) and advanced with 64-bit adds
          const ull dA = dA0 + (ull)((cslot * SB) >> 4), dB = dB0 + (ull)(((j * kpb) * A_ATOM) >> 4);
          const uint32_t dacc = tb + ((flags & F_ONEBUF) ? 0 : buf * 64);
#pragma unroll
          for (int a = 0; a < 2; ++a) {
            if (a < kpb) {
#pragma unroll
              for (int kk = 0; kk < 4; ++kk)
                mma_tc(dacc, dA + (ull)((a * 8192 + kk * 32) >> 4), dB + (ull)((a * A_ATOM + kk * 32) >> 4), id56, (j | a | kk) != 0);
            }
          }
          if (pas == 0 && j == 0) sst[X_M0] = clk();
          if (pas == last) { sst[X_LM0 + j] = tf; sst[X_LM1 + j] = clk(); }
          if (!(flags & F_ONECOMMIT)) commit(be + 8 * cslot);
        }
      }
      if (el && !(flags & (F_NOMMA | F_NOEPI))) commit(bacc + 8 * buf);
      if (pas == last && !(flags & F_NOMMA) && el) { sst[X_LASTCOMMIT] = clk(); commit(bDone); }
    }
    if (!(flags & F_NOMMA)) mbar_wait(bDone, 0);
    if (el) { sst[X_ISSEND] = clk(); sst[X_ICLK1] = sst[X_ISSEND]; sst[X_IGT1] = gt(); }
  } else if (warp >= 4 && !(flags & (F_NOMMA | F_NOEPI))) {  // epilogue, quadrant q = warp & 3
    const int q = warp & 3;
    float sink = 0.f;
    for (int pas = 0; pas < passes; ++pas) {
      const int buf = pas & 1;
      mbar_wait(bacc + 8 * buf, (pas >> 1) & 1);
      const ull ts = clk();
      fence_after();
      const uint32_t ta = tb + buf * 64 + ((uint32_t)(q * 32) << 16);
      uint32_t v[28];
      asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                   : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]),
                     "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta));
      asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]) : "r"(ta + 32));
      asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]) : "r"(ta + 48));
      wait_ld();
      if (tid == 128) { if (pas == 0) { sst[X_ACC0] = ts; sst[X_LD0] = clk(); } if (pas == last) { sst[X_LACC] = ts; sst[X_LLD] = clk(); } }
      if (pas == last) {
        float* d = Dout + (size_t)cta * 64 * 56;
        const int r0 = q * 16 + (lane >> 2), c = (lane & 3) * 2;
#pragma unroll
        for (int g = 0; g < 7; ++g) {
          d[(size_t)r0 * 56 + g * 8 + c] = __uint_as_float(v[4 * g]); d[(size_t)r0 * 56 + g * 8 + c + 1] = __uint_as_float(v[4 * g + 1]);
          d[(size_t)(r0 + 8) * 56 + g * 8 + c] = __uint_as_float(v[4 * g + 2]); d[(size_t)(r0 + 8) * 56 + g * 8 + c + 1] = __uint_as_float(v[4 * g + 3]);
        }
      } else sink += __uint_as_float(v[0]) + __uint_as_float(v[27]);
      fence_before(); __syncwarp();
      if (lane == 0) mbar_arrive(bacce + 8 * buf);
    }
    if (tid == 128) sst[X_EPIEND] = clk();
    if (sink == 1.2345e30f) Dout[0] = sink;
  }
  __syncthreads();
  if (tid == 0) { sst[X_GT1] = gt(); }
  for (int i = tid; i < XN; i += 256) st[(size_t)cta * XN + i] = sst[i];
  fence_after();
  if (warp == 0) tmem_free(tb, 128);
}

__global__ void k_refc(const bf16* pool, const bf16* A, const float* Dout, unsigned* bad, float* maxerr) {
  const int cta = blockIdx.x;
  for (int i = threadIdx.x; i < 64 * 56; i += blockDim.x) {
    const int m = i / 56, n = i % 56; float acc = 0.f;
    const bf16* w = pool + (size_t)(cta * 64 + m) * 1024; const bf16* a = A + (size_t)n * 1024;
    for (int k = 0; k < 1024; ++k) acc += __bfloat162float(w[k]) * __bfloat162float(a[k]);
    const float err = fabsf(Dout[(size_t)cta * 3584 + i] - acc);
    atomicMax((unsigned*)maxerr, __float_as_uint(err));
    if (!(err <= 1e-3f + 1e-4f * fabsf(acc))) atomicAdd(bad, 1u);
  }
}

static void u12c() {
  printf("{\"u12c\":\"begin\"}\n"); fflush(stdout);
  const size_t POOLB = (size_t)148 * 64 * 2048;
  bf16 *pool, *Ag; ull* st; float* Dout; unsigned* bad; float* merr;
  CK(cudaMalloc(&pool, POOLB)); CK(cudaMalloc(&Ag, 64 * 1024 * 2)); CK(cudaMalloc(&st, (size_t)NSM * XN * 8)); CK(cudaMalloc(&Dout, (size_t)NSM * 3584 * 4));
  CK(cudaMalloc(&bad, 4)); CK(cudaMalloc(&merr, 4));
  k_init<<<2048, 256>>>(pool, POOLB / 2, 12345u); k_init<<<64, 256>>>(Ag, 64 * 1024, 777u); CK(cudaDeviceSynchronize());
  CUtensorMap mw = mkmap(pool, 1024, (ull)148 * 64, 2048, 64, 64), ma = mkmap(Ag, 1024, 56, 2048, 64, 56);
  const int dyn = DYN_FORCE;
  CK(cudaFuncSetAttribute(k_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  struct Cf { const char* tag; int D, kpb, flags, ncta; };
  std::vector<Cf> cs;
  cs.push_back({"full", 4, 1, 0, NSM});
  for (int D : {2, 3, 6}) cs.push_back({"full", D, 1, 0, NSM});
  for (int D : {2, 3, 4}) cs.push_back({"full_K128stage", D, 2, 0, NSM});
  cs.push_back({"no_epilogue", 4, 1, F_NOEPI, NSM});
  cs.push_back({"tma_only", 4, 1, F_NOMMA, NSM});
  cs.push_back({"mma_only_percommit", 4, 1, F_NOTMA | F_NOEPI, NSM});
  cs.push_back({"mma_only_onecommit", 4, 1, F_NOTMA | F_NOEPI | F_ONECOMMIT, NSM});
  cs.push_back({"mma_only_epi_percommit", 4, 1, F_NOTMA, NSM});
  cs.push_back({"mma_only_ones_data", 4, 1, F_NOTMA | F_NOEPI | F_ONES, NSM});
  cs.push_back({"mma_only_nofence", 4, 1, F_NOTMA | F_NOEPI | F_NOFENCE, NSM});
  cs.push_back({"mma_only_ones_nofence_onecommit", 4, 1, F_NOTMA | F_NOEPI | F_NOFENCE | F_ONES | F_ONECOMMIT, NSM});
  cs.push_back({"mma_only_onebuf", 4, 1, F_NOTMA | F_NOEPI | F_NOFENCE | F_ONES | F_ONECOMMIT | F_ONEBUF, NSM});
  cs.push_back({"mma_only_onebuf_nostamp", 4, 1, F_NOTMA | F_NOEPI | F_NOFENCE | F_ONES | F_ONECOMMIT | F_ONEBUF | F_NOSTAMP, NSM});
  cs.push_back({"full_1cta", 4, 1, 0, 1}); cs.push_back({"tma_only_1cta", 4, 1, F_NOMMA, 1});
  cs.push_back({"mma_only_percommit_1cta", 4, 1, F_NOTMA | F_NOEPI, 1}); cs.push_back({"mma_only_onecommit_1cta", 4, 1, F_NOTMA | F_NOEPI | F_ONECOMMIT, 1});
  for (auto& c : cs) {
    CP p = {c.D, c.kpb, 40, c.flags, bad};
    std::vector<double> spc, ev, xc, pas, st0, ldlat, mmaiv, chainlat, accl, fullw, aland, firstfull, p0end; double nspc = 1; unsigned badTot = 0; float me = 0;
    for (int rep = -2; rep < 5; ++rep) {
      CK(cudaMemset(bad, 0, 4)); CK(cudaMemset(merr, 0, 4)); clr_to();
      CK(cudaEventRecord(e0)); k_mma<<<c.ncta, 256, dyn>>>(mw, ma, p, st, Dout); CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      if (rep == -2) { p.passes = std::min(20000, std::max(40, (int)(40 * 500.0 / std::max(10.0, (double)ms * 1000 - 10)))); continue; }
      if (rep == -1) continue;
      std::vector<ull> h((size_t)c.ncta * XN); CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost));
      if (getenv("U12C_DBG") && rep == 0) { const ull* s0 = h.data(); printf("DBG %s:", c.tag); for (int i = 0; i < 12; ++i) printf(" %lld", (long long)(i < 2 || i > 9 ? (long long)s0[i] : (long long)(s0[i] - s0[0]))); printf(" igt %lld", (long long)(s0[X_IGT1]-s0[X_IGT0]));printf("\n LF:"); for (int i = 0; i < 16; ++i) printf(" %lld", (long long)(s0[X_LF + i] - s0[0])); printf("\n LM0:"); for (int i = 0; i < 16; ++i) printf(" %lld", (long long)(s0[X_LM0 + i] - s0[0])); printf("\n LM1:"); for (int i = 0; i < 16; ++i) printf(" %lld", (long long)(s0[X_LM1 + i] - s0[0])); printf("\n LE:"); for (int i = 0; i < 16; ++i) printf(" %lld", (long long)(s0[X_LE + i] - s0[0])); printf("\n"); }
      bool chk = !(c.flags & (F_NOTMA | F_NOMMA | F_NOEPI));
      if (chk) { k_refc<<<c.ncta, 256>>>(pool, Ag, Dout, bad, merr); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&me, merr, 4, cudaMemcpyDeviceToHost)); }
      unsigned hb; CK(cudaMemcpy(&hb, bad, 4, cudaMemcpyDeviceToHost)); badTot += hb + read_to();
      std::vector<double> sp, a_pas, a_ld, a_iv, a_cl, a_acc, a_p0, a_al, a_ff; double mxspan = 0, nsr = 0;
      for (int i = 0; i < c.ncta; ++i) {
        const ull* s = &h[(size_t)i * XN];
        const double ns = (double)(s[X_GT1] - s[X_GT0]) / (double)(0 + 1);  // placeholder; ratio below
        (void)ns;
      }
      // ns/cycle per CTA from globaltimer span over the whole kernel body
      std::vector<double> nsc; for (int i = 0; i < c.ncta; ++i) { const ull* s = &h[(size_t)i * XN]; nsc.push_back((double)(s[X_IGT1] - s[X_IGT0]) / (double)(s[X_ICLK1] - s[X_ICLK0])); }
      nspc = med(nsc);
      const bool epi = !(c.flags & (F_NOMMA | F_NOEPI));
      for (int i = 0; i < c.ncta; ++i) {
        const ull* s = &h[(size_t)i * XN];
        const ull tend = epi ? std::max(s[X_EPIEND], s[X_ISSEND]) : s[X_ISSEND];
        mxspan = std::max(mxspan, (double)(tend - s[X_START]));
        a_pas.push_back((double)(tend - s[X_START]) * nspc / p.passes);
        if (epi) a_ld.push_back((double)(s[X_EPIEND] - s[X_LD0]) * nspc / (p.passes - 1));
        const int NS = 16 / c.kpb;
        if (!(c.flags & F_NOMMA)) {
          a_iv.push_back((double)(s[X_LM1 + NS - 1] - s[X_LM0]) * nspc / (64.0 - 4.0 / 1) * 1.0);  // issue span / 60 inter-MMA gaps approx; refined below
          a_cl.push_back((double)(s[X_ISSEND] - s[X_LASTCOMMIT]) * nspc);
        }
        if (epi) { a_acc.push_back((double)(s[X_LACC] - s[X_LASTCOMMIT]) * nspc); }
        if (!(c.flags & F_NOTMA)) { a_al.push_back((double)(s[X_ASEEN] - s[X_START]) * nspc); a_ff.push_back((double)(s[X_FULL0] - s[X_START]) * nspc); }
        if (epi) a_p0.push_back((double)(s[X_LD0] - s[X_START]) * nspc);
      }
      ev.push_back(ms * 1000); spc.push_back(mxspan * nspc / 1000.0);
      pas.push_back(med(a_pas)); ldlat.push_back(a_ld.empty() ? 0 : med(a_ld)); mmaiv.push_back(a_iv.empty() ? 0 : med(a_iv)); chainlat.push_back(a_cl.empty() ? 0 : med(a_cl));
      accl.push_back(a_acc.empty() ? 0 : med(a_acc)); aland.push_back(a_al.empty() ? 0 : med(a_al)); firstfull.push_back(a_ff.empty() ? 0 : med(a_ff)); p0end.push_back(a_p0.empty() ? 0 : med(a_p0));
      if ((double)(h[X_ICLK1] - h[X_ICLK0]) * nspc < 100e3) badTot += 1000000;  // gt/clk pair span must be >= 100 us
    }
    double x = 0;
    {  // two-point delta cross-check at passes/2
      std::vector<double> ev2, sp2; CP q = p; q.passes = std::max(2, p.passes / 2);
      for (int rep = -1; rep < 4; ++rep) {
        CK(cudaMemset(bad, 0, 4)); clr_to();
        CK(cudaEventRecord(e0)); k_mma<<<c.ncta, 256, dyn>>>(mw, ma, q, st, Dout); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (rep < 0) continue;
        std::vector<ull> h((size_t)c.ncta * XN); CK(cudaMemcpy(h.data(), st, h.size() * 8, cudaMemcpyDeviceToHost));
        std::vector<double> nsc; for (int i = 0; i < c.ncta; ++i) { const ull* s = &h[(size_t)i * XN]; nsc.push_back((double)(s[X_IGT1] - s[X_IGT0]) / (double)(s[X_ICLK1] - s[X_ICLK0])); }
        const double nn = med(nsc); double mxs = 0; const bool epi = !(c.flags & (F_NOMMA | F_NOEPI));
        for (int i = 0; i < c.ncta; ++i) { const ull* s = &h[(size_t)i * XN]; const ull tend = epi ? std::max(s[X_EPIEND], s[X_ISSEND]) : s[X_ISSEND]; mxs = std::max(mxs, (double)(tend - s[X_START])); }
        ev2.push_back(ms * 1000); sp2.push_back(mxs * nn / 1000.0);
      }
      const double dev = med(ev) - med(ev2); x = dev > 0 ? (med(spc) - med(sp2)) / dev : 0;
    }
    const double ipass = med(pas);
    printf("{\"u12c\":\"%s\",\"D\":%d,\"stageKB\":%d,\"ctas\":%d,\"passes\":%d,\"pass_ns\":%.1f,\"steady_pass_ns\":%.1f,\"implied_ingress_GBps_per_SM\":%.1f,\"mma_issue_ns_per_mma_last_pass\":%.2f,\"last_commit_to_done_ns\":%.1f,\"last_commit_to_epilogue_seen_ns\":%.1f,\"pass0_A_landed_ns\":%.0f,\"pass0_first_stage_landed_ns\":%.0f,\"pass0_to_epilogue_ld_done_ns\":%.0f,\"ns_per_cyc\":%.4f,\"event_us\":%.0f,\"xchk_clk_over_event\":%.3f,\"xchk\":\"%s\",\"verify_maxerr\":%.2e,\"bad\":%u}\n",
           c.tag, c.D, 8 * c.kpb, c.ncta, p.passes, ipass, med(ldlat), SLICE / (med(ldlat) > 0 ? med(ldlat) : ipass), med(mmaiv), med(chainlat), med(accl), med(aland), med(firstfull), med(p0end), nspc, med(ev), x,
           (x > 0.98 && x < 1.02) ? "ok" : "FLAG", me, badTot);
    fflush(stdout);
  }
  cudaFree(pool); cudaFree(Ag); cudaFree(st); cudaFree(Dout); cudaFree(bad); cudaFree(merr);
}


// ---- MMA throughput sweep (no TMA): chain of L MMAs K=16 (ones operands) over nacc independent accumulators, commit + mbarrier wait per chain.
// time(L) = latency + L * per-MMA cost; loops for >= ~400 us of cycles; verifies D (=16 * MMAs per accumulator) on lane 0 col 0.
#define getenv_dbg 1
__global__ void __launch_bounds__(128, 1) k_msweep(int M, int N, int L, int nacc, int uni, ull budget, double* out, unsigned* bad) {
  extern __shared__ uint8_t smraw[];
  __shared__ uint32_t tslot; __shared__ __align__(8) uint64_t sb[2];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u; uint8_t* sm = smraw + (base - raw);
  const int tid = threadIdx.x, warp = tid >> 5;
  if (warp == 0) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(su32(&tslot))); asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  if (tid == 0) { mbar_init(su32(sb), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  for (int i = tid; i < 180224 / 4; i += 128) reinterpret_cast<uint32_t*>(sm)[i] = 0x3F803F80u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); fence_after();
  const uint32_t tb = tslot;
  double cyc = 0, nch = 0; ull cm = 0, gm = 0, ce = 0, ge = 0, sink_clk = 0;
  if (uni == 1 ? warp == 0 : tid == 0) {
    uint32_t par = 0; const uint32_t id = idesc(N, M);
    const bool el = uni == 1 ? elect_one() : true;  // uni: whole warp runs the loop with warp-uniform operands, one elected lane issues
    const ull s0 = clk(); ull t = s0; int n = 0;
    while (t - s0 < budget) {
      const ull a = clk();
      if (uni >= 3) {  // ring-like addressing: stage s = i/4 uses A slot (s%4)*8 KB and B atom (s%16)*7168 B; uni 3 = unrolled by 4, uni 4 = + a clock read per stage
        const ull da0 = sdesc(base), db0 = sdesc(base + 32768);
        for (int i = 0; i < L; i += 4) {
          const int s_ = i >> 2;
          const ull dA = da0 + (ull)(((s_ & (uni == 5 ? 0 : 3)) * 8192) >> 4), dB = db0 + (ull)(((s_ & (uni == 6 ? 0 : 15)) * 7168) >> 4);
#pragma unroll
          for (int u = 0; u < 4; ++u) mma_tc(tb, dA + (ull)(u * 2), dB + (ull)(u * 2), id, (i | u) != 0);
          if (uni == 4) sink_clk += clk();
        }
      } else
      if (uni == 2) {  // pre-computed descriptors, 8x unrolled, single accumulator
        const ull da0 = sdesc(base), db0 = sdesc(base + 16384);
        for (int i = 0; i < L; i += 8) {
#pragma unroll
          for (int u = 0; u < 8; ++u) mma_tc(tb, da0 + (ull)((u & 3) * 2), db0 + (ull)((u & 3) * 2), id, (i | u) != 0);
        }
      } else
      for (int i = 0; i < L; ++i) { if (el) mma_tc(tb + (i % nacc) * N, sdesc(base + (i & 3) * 32), sdesc(base + 16384 + (i & 3) * 32), id, i >= nacc); }
      if (el) commit(su32(sb));
      mbar_wait(su32(sb), par); par ^= 1;
      t = clk(); if (n >= 3) { cyc += (double)(t - a); nch += 1; } ++n;
      if (n == 8) { cm = t; gm = gt(); }
    }
    ce = t; ge = gt();
  }
  __syncthreads(); fence_after();
  // verify: accumulator 0, lane 0, col 0 must be 16 * ceil(L/nacc)
  uint32_t v = 0;
  if (warp == 0) { asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(tb)); wait_ld(); }
  if (tid == 0 && __uint_as_float(v) != 16.f * ((L + nacc - 1) / nacc)) { atomicAdd(bad, 1u); if (getenv_dbg) printf("msweep bad uni=%d M=%d N=%d L=%d got %f exp %f\n", uni, M, N, L, __uint_as_float(v), 16.f * ((L + nacc - 1) / nacc)); }
  if (sink_clk == 12345) out[7] = 1;
  if (tid == 0) { out[0] = cyc / nch; out[1] = (double)(ge - gm) / (double)(ce - cm); out[2] = nch; out[3] = (double)(ce - cm); }
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tb));
}
static void u12msweep() {
  printf("{\"u12c_mmasweep\":\"begin\"}\n"); fflush(stdout);
  double* out; unsigned* bad; CK(cudaMalloc(&out, 64)); CK(cudaMalloc(&bad, 4));
  CK(cudaFuncSetAttribute(k_msweep, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384 + 32768 + 150000));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int uni : {0, 1, 2, 3, 4, 5, 6}) for (int M : {64, 128}) for (int N : {8, 56, 64, 128, 256}) for (int nacc : {1, 4}) {
    if (uni >= 2 && nacc != 1) continue;
    if (uni >= 3 && (M != 64 || N != 56)) continue;
    if (nacc * N > 512) continue;
    double cyc[4] = {0, 0, 0, 0}; double nspc = 0, xc = 0; unsigned bd = 0; int Ls[4] = {4, 16, 64, 256};
    for (int li = 0; li < 4; ++li) {
      CK(cudaMemset(bad, 0, 4));
      const ull budget = 800000; if (uni >= 2) Ls[0] = 8;  // cycles (~440 us)
      CK(cudaEventRecord(e0)); k_msweep<<<1, 128, 16384 + 32768 + 150000>>>(M, N, Ls[li], nacc, uni, budget, out, bad); CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      double r[4]; CK(cudaMemcpy(r, out, 32, cudaMemcpyDeviceToHost)); unsigned hb; CK(cudaMemcpy(&hb, bad, 4, cudaMemcpyDeviceToHost)); bd += hb + read_to();
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      cyc[li] = r[0]; nspc = r[1]; if (li == 2) xc = (r[3] * r[1] / 1000.0 + 0) / (ms * 1000);  // clock span (after chain 64) / event (event also has ~64 chains + setup)
      if (r[3] * r[1] < 100e3) bd += 1000000;
    }
    // slope between L=64 and L=256 (cycles/MMA), latency from L=4 point
    const double slope = (cyc[3] - cyc[2]) / 192.0, lat = cyc[2] - 64 * slope;
    printf("{\"u12c_mmasweep\":1,\"uniform_elect\":%d,\"M\":%d,\"N\":%d,\"acc\":%d,\"chain_ns\":[%.0f,%.0f,%.0f,%.0f],\"ns_per_mma_slope\":%.1f,\"cyc_per_mma_slope\":%.1f,\"fixed_latency_ns\":%.0f,\"ns_per_cyc\":%.4f,\"TFLOPs_per_SM_slope\":%.2f,\"bad\":%u}\n",
           uni, M, N, nacc, cyc[0] * nspc, cyc[1] * nspc, cyc[2] * nspc, cyc[3] * nspc, slope * nspc, slope, lat * nspc, nspc, 2.0 * M * N * 16 / (slope * nspc * 1000.0), bd);
    (void)xc;
    fflush(stdout);
  }
  cudaFree(out); cudaFree(bad);
}

// ================================================================== U12d: A-from-TMEM MMA ==================================================================
// mode 0: SS reference (A from smem); 1: TS with A rows at TMEM lane = m; 2: TS with M=64 rows at lane (m/16)*32 + m%16 (same as M=128 for m<... n/a).
// A[m][k] = ((m*3+k) % 7) - 3, B[n][k] = ((n+2*k) % 5) - 2: small ints, exact in bf16/fp32.
__device__ __forceinline__ float aval(int m, int k) { return (float)(((m * 3 + k) % 7) - 3); }
__device__ __forceinline__ float bval(int n, int k) { return (float)(((n + 2 * k) % 5) - 2); }
__global__ void __launch_bounds__(128, 1) k_ts(int M, int mode, float* res, int* flags) {
  extern __shared__ uint8_t smraw[];
  __shared__ uint32_t tslot; __shared__ __align__(8) uint64_t sb[2];
  const uint32_t raw = su32(smraw), base = (raw + 1023) & ~1023u; uint8_t* sm = smraw + (base - raw);
  const uint32_t bB = base, bAs = base + 8192;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (warp == 0) tmem_alloc(su32(&tslot), 128);
  if (tid == 0) { mbar_init(su32(sb), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  for (int i = tid; i < 8192 + 16384; i += 128) sm[i] = 0;
  __syncthreads();
  auto put = [&](uint8_t* t, int row, int k, float v) { bf16 b = __float2bfloat16(v); *reinterpret_cast<bf16*>(t + row * 128 + ((((k * 2) >> 4) ^ (row & 7)) << 4) + ((k * 2) & 15)) = b; };
  for (int i = tid; i < 64 * 16; i += 128) put(sm, i >> 4, i & 15, bval(i >> 4, i & 15));
  if (mode == 0) for (int i = tid; i < M * 16; i += 128) put(sm + 8192, i >> 4, i & 15, aval(i >> 4, i & 15));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); fence_after();
  const uint32_t tb = tslot;
  if (mode != 0) {  // write A into TMEM columns 64..71 with tcgen05.st (warp w -> lane quadrant w)
    int row = -1;
    if (M == 128) row = warp * 32 + lane;
    else if (mode == 1) row = warp < 2 ? warp * 32 + lane : -1;
    else row = warp * 16 + (lane & 15);
    uint32_t r[8];
    for (int c = 0; c < 8; ++c) {
      const float a0 = row >= 0 ? aval(row, 2 * c) : 0.f, a1 = row >= 0 ? aval(row, 2 * c + 1) : 0.f;
      __nv_bfloat162 b2 = __floats2bfloat162_rn(a0, a1); r[c] = *reinterpret_cast<uint32_t*>(&b2);
    }
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(tb + 64 + ((uint32_t)(warp * 32) << 16)), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    fence_before();
  }
  __syncthreads(); fence_after();
  if (tid == 0) {
    if (mode == 0) mma_tc(tb, sdesc(bAs), sdesc(bB), idesc(64, M), 0);
    else asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 0, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(tb), "r"(tb + 64), "l"(sdesc(bB)), "r"(idesc(64, M)) : "memory");
    commit(su32(sb));
    if (!mbar_wait(su32(sb), 0)) flags[0] = 1;
  }
  __syncthreads(); fence_after();
  uint32_t v[16];
  for (int c0 = 0; c0 < 64; c0 += 16) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
                 : "r"(tb + c0 + ((uint32_t)(warp * 32) << 16)));
    wait_ld();
    for (int i = 0; i < 16; ++i) res[(warp * 32 + lane) * 64 + c0 + i] = __uint_as_float(v[i]);
  }
  __syncthreads();
  if (warp == 0) tmem_free(tb, 128);
}
// compile-only: the masked (disable_output_lane) TS form that CUTLASS emits
__global__ void k_ts_mask_asm_only(uint32_t d, uint32_t a, ull bd, uint32_t id, uint32_t m0) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%4, %4, %4, %4}, p;\n}" ::"r"(d), "r"(a), "l"(bd), "r"(id), "r"(m0));
}

static void u12d() {
  printf("{\"u12d\":\"begin\"}\n"); fflush(stdout);
  float* res; int* fl; CK(cudaMalloc(&res, 128 * 64 * 4)); CK(cudaMalloc(&fl, 4));
  CK(cudaFuncSetAttribute(k_ts, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192 + 16384 + 2048));
  struct T { int M, mode; const char* tag; };
  for (T t : {T{128, 0, "SS M=128 (reference)"}, T{64, 0, "SS M=64 (reference)"}, T{128, 1, "TS M=128 A rows at lane m"}, T{64, 1, "TS M=64 A rows at lane m (0..63)"}, T{64, 2, "TS M=64 A rows at lane (m/16)*32+m%16"}}) {
    CK(cudaMemset(fl, 0, 4)); CK(cudaMemset(res, 0xff, 128 * 64 * 4));
    k_ts<<<1, 128, 8192 + 16384 + 2048>>>(t.M, t.mode, res, fl);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> h(128 * 64); int hf = 0;
    if (e == cudaSuccess) { CK(cudaMemcpy(h.data(), res, h.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hf, fl, 4, cudaMemcpyDeviceToHost)); }
    else { printf("{\"u12d\":\"%s\",\"error\":\"%s\"}\n", t.tag, cudaGetErrorString(e)); cudaGetLastError(); exit(0); }
    auto ref = [](int m, int n) { float s = 0; for (int k = 0; k < 16; ++k) s += (float)(((m * 3 + k) % 7) - 3) * (float)(((n + 2 * k) % 5) - 2); return s; };
    auto cnt = [&](auto lane_of_row) { int bad = 0; for (int m = 0; m < t.M; ++m) for (int n = 0; n < 64; ++n) bad += (h[lane_of_row(m) * 64 + n] != ref(m, n)); return bad; };
    const int badId = cnt([](int m) { return m; });
    const int badQ = t.M == 64 ? cnt([](int m) { return (m / 16) * 32 + m % 16; }) : -1;
    printf("{\"u12d\":\"%s\",\"mismatches_rows_at_lane_m\":%d,\"mismatches_rows_at_lane_(m/16)*32+m%%16\":%d,\"mbar_timeout\":%d}\n", t.tag, badId, badQ, hf);
    fflush(stdout);
  }
  cudaFree(res); cudaFree(fl);
}

int main(int argc, char** argv) {
  const char* sel = argc > 1 ? argv[1] : "all";
  init_enc();
  cudaFree(0);
  auto want = [&](const char* s) { return !strcmp(sel, "all") || !strcmp(sel, s); };
  if (want("coop")) u12coop();
  if (want("a")) u12a();
  if (want("b")) u12b();
  if (want("c")) u12c();
  if (want("c") || want("msweep")) u12msweep();
  if (want("d")) u12d();
  return 0;
}
