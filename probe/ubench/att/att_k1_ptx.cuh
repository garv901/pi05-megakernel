// att_k1_ptx.cuh: PTX helpers for the tcgen05 attention variants (att_k1.cu, att_t1.cu). Patterns verified in probe/ubench/fg/fg_k1.cu / u12.cu / u13.cu
// (sdesc K-major SW128, idesc, cta_group::2 TMA / MMA / commit, tcgen05.ld/st shapes). Generated ld/st wrappers: tmem_ld32_xN / tmem_st32_xN.
#pragma once
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdint.h>
typedef unsigned long long ull_t;
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
// arrive on the barrier at the same smem offset in cluster CTA `rank` (release, cluster scope)
__device__ __forceinline__ void mbar_arrive_remote(uint32_t b, uint32_t rank) {
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n}" ::"r"(b), "r"(rank) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n.reg .pred P;\nmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void tma2d(uint32_t dst, const void* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull_t)m), "r"(bar), "r"(c0), "r"(c1) : "memory"); }
// cta_group::2 form: completion is signalled on the pair leader's (even rank) barrier at the same offset
__device__ __forceinline__ void tma2d2(uint32_t dst, const void* m, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               ::"r"(dst), "l"((ull_t)m), "r"(bar & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory"); }
__device__ __forceinline__ void mma_ss2(uint32_t d, ull_t da, ull_t db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d), "l"(da), "l"(db), "r"(id), "r"(acc) : "memory"); }
// A operand from TMEM (a = tmem address of the packed bf16 A columns)
__device__ __forceinline__ void mma_ts2(uint32_t d, uint32_t a, ull_t db, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}"
               ::"r"(d), "r"(a), "l"(db), "r"(id), "r"(acc) : "memory"); }
__device__ __forceinline__ void commit2(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"((unsigned short)3) : "memory"); }
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc2(uint32_t slot_smem, int ncols) {
  if (ncols == 512) asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(slot_smem));
  else if (ncols == 256) asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 256;" ::"r"(slot_smem));
  else asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 128;" ::"r"(slot_smem));
}
__device__ __forceinline__ void tmem_free2(uint32_t tb, int ncols) {
  if (ncols == 512) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tb));
  else if (ncols == 256) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 256;" ::"r"(tb));
  else asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 128;" ::"r"(tb));
}
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ bool elect_one() { uint32_t e; asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(e)); return e != 0; }
// K-major SW128 operand descriptor (SBO = 1024 B = 8 rows of 128 B), sm100 version bit
__device__ __forceinline__ ull_t sdesc(uint32_t addr) { return (ull_t)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 46) | (2ull << 61); }
// MN-major SW128 operand descriptor: LBO = byte distance between 64-element MN atoms, SBO = byte distance between 8-row (k) groups
__device__ __forceinline__ ull_t sdesc_mn(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (ull_t)((addr >> 4) & 0x3fff) | ((ull_t)((lbo >> 4) & 0x3fff) << 16) | ((ull_t)((sbo >> 4) & 0x3fff) << 32) | (1ull << 46) | (2ull << 61); }
// bf16 x bf16 -> f32 instruction descriptor; bmajor = 1: B is MN-major
__device__ __forceinline__ uint32_t idesc(int N, int M, int bmajor = 0) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)bmajor << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }

__device__ __forceinline__ void tmem_ld32_x4(uint32_t ta, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(ta)); }
__device__ __forceinline__ void tmem_st32_x4(uint32_t ta, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(ta), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory"); }

__device__ __forceinline__ void tmem_ld32_x8(uint32_t ta, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]) : "r"(ta)); }
__device__ __forceinline__ void tmem_st32_x8(uint32_t ta, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(ta), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]) : "memory"); }

__device__ __forceinline__ void tmem_ld32_x16(uint32_t ta, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(ta)); }
__device__ __forceinline__ void tmem_st32_x16(uint32_t ta, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(ta), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory"); }

__device__ __forceinline__ void tmem_ld32_x32(uint32_t ta, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]) : "r"(ta)); }
__device__ __forceinline__ void tmem_st32_x32(uint32_t ta, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(ta), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]) : "memory"); }

__device__ __forceinline__ void tmem_ld32_x64(uint32_t ta, uint32_t* v) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]), "=r"(v[32]), "=r"(v[33]), "=r"(v[34]), "=r"(v[35]), "=r"(v[36]), "=r"(v[37]), "=r"(v[38]), "=r"(v[39]), "=r"(v[40]), "=r"(v[41]), "=r"(v[42]), "=r"(v[43]), "=r"(v[44]), "=r"(v[45]), "=r"(v[46]), "=r"(v[47]), "=r"(v[48]), "=r"(v[49]), "=r"(v[50]), "=r"(v[51]), "=r"(v[52]), "=r"(v[53]), "=r"(v[54]), "=r"(v[55]), "=r"(v[56]), "=r"(v[57]), "=r"(v[58]), "=r"(v[59]), "=r"(v[60]), "=r"(v[61]), "=r"(v[62]), "=r"(v[63]) : "r"(ta)); }
__device__ __forceinline__ void tmem_st32_x64(uint32_t ta, const uint32_t* v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64};" ::"r"(ta), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]), "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]), "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]), "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(v[32]), "r"(v[33]), "r"(v[34]), "r"(v[35]), "r"(v[36]), "r"(v[37]), "r"(v[38]), "r"(v[39]), "r"(v[40]), "r"(v[41]), "r"(v[42]), "r"(v[43]), "r"(v[44]), "r"(v[45]), "r"(v[46]), "r"(v[47]), "r"(v[48]), "r"(v[49]), "r"(v[50]), "r"(v[51]), "r"(v[52]), "r"(v[53]), "r"(v[54]), "r"(v[55]), "r"(v[56]), "r"(v[57]), "r"(v[58]), "r"(v[59]), "r"(v[60]), "r"(v[61]), "r"(v[62]), "r"(v[63]) : "memory"); }

template <int N> __device__ __forceinline__ void tmem_ld32(uint32_t ta, uint32_t* v) {
  if constexpr (N == 4) tmem_ld32_x4(ta, v); else if constexpr (N == 8) tmem_ld32_x8(ta, v); else if constexpr (N == 16) tmem_ld32_x16(ta, v);
  else if constexpr (N == 32) tmem_ld32_x32(ta, v); else tmem_ld32_x64(ta, v);
}
template <int N> __device__ __forceinline__ void tmem_st32(uint32_t ta, const uint32_t* v) {
  if constexpr (N == 4) tmem_st32_x4(ta, v); else if constexpr (N == 8) tmem_st32_x8(ta, v); else if constexpr (N == 16) tmem_st32_x16(ta, v);
  else if constexpr (N == 32) tmem_st32_x32(ta, v); else tmem_st32_x64(ta, v);
}
