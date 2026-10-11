#pragma once
// Shared helpers. Target: sm_75 (Turing). No cp.async, no bf16 — fp16/__half2 only.
#include <cuda_fp16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

DEVI half2 u32_as_h2(uint32_t u) { return *reinterpret_cast<half2*>(&u); }
DEVI uint32_t h2_as_u32(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

// Streaming 128-bit load of weights: read-only path, don't pollute L1 (each weight byte is used once).
DEVI uint4 ld_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Dequantize the 8 nibbles of one packed word and FMA against 8 fp16 activations (as 4 half2).
// Word layout: elem 2i at bits [4i,4i+4), elem 2i+1 at bits [16+4i, 16+4i+4).
// (w>>4i) & 0x000F000F | 0x64006400 == half2(1024+q_2i, 1024+q_2i+1)  ("magic number" dequant).
// z1024 = zero + 1024 (exact), so (q+1024) - (z+1024) is exact and the only rounding is the * scale.
DEVI half2 dq_fma_word(uint32_t w, uint4 xv, half2 s2, half2 z1024, half2 acc) {
  const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    half2 q = u32_as_h2(((w >> (4 * i)) & 0x000F000Fu) | 0x64006400u);
    half2 wv = __hmul2(__hsub2(q, z1024), s2);
    acc = __hfma2(wv, u32_as_h2(xs[i]), acc);
  }
  return acc;
}
