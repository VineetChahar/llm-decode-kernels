// Fused SwiGLU on the output of the concatenated gate|up GEMV:  out[i] = silu(gu[i]) * gu[I + i]
// Rounds like HF: silu result to fp16, then fp16 multiply. Unfused (HF) = silu kernel + mul kernel
// + an extra write/read of the 11 KB intermediate.
#include "common.cuh"
#include <cuda_runtime.h>

__global__ void swiglu_kernel(const half* __restrict__ gu, half* __restrict__ out, int I) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= I) return;
  const uint4 gv = *reinterpret_cast<const uint4*>(gu + i);
  const uint4 uv = *reinterpret_cast<const uint4*>(gu + I + i);
  const uint32_t g[4] = {gv.x, gv.y, gv.z, gv.w}, u[4] = {uv.x, uv.y, uv.z, uv.w};
  uint32_t o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 gf = __half22float2(u32_as_h2(g[k]));
    const half2 s = __floats2half2_rn(gf.x / (1.f + expf(-gf.x)), gf.y / (1.f + expf(-gf.y)));
    o[k] = h2_as_u32(__hmul2(s, u32_as_h2(u[k])));
  }
  *reinterpret_cast<uint4*>(out + i) = make_uint4(o[0], o[1], o[2], o[3]);
}

void swiglu_launch(const void* gu, void* out, int I, cudaStream_t st) {
  const int threads = 128, n = I / 8;
  swiglu_kernel<<<(n + threads - 1) / threads, threads, 0, st>>>((const half*)gu, (half*)out, I);
}
