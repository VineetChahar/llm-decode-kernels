// Fused residual add + RMSNorm (one kernel instead of HF's add, to(float), pow, mean, rsqrt, mul, to(half), mul).
//   x   <- x + delta                 (fp16 add, as HF's residual add)       [delta may be null: plain RMSNorm]
//   out <- w * half(float(x) * rsqrt(mean(float(x)^2) + eps))               (HF LlamaRMSNorm rounding)
// One block: the vector is 4 KB, so this is pure latency — the win is 1 launch + 1 DRAM round trip vs ~8.
#include "common.cuh"
#include <cuda_runtime.h>

__global__ void __launch_bounds__(256)
add_rmsnorm_kernel(half* __restrict__ x, const half* __restrict__ delta, const half* __restrict__ w,
                   half* __restrict__ out, int H, float eps) {
  __shared__ float wsum[8];
  const int tid = threadIdx.x;
  float ss = 0.f;
  for (int i = tid * 8; i < H; i += blockDim.x * 8) {
    uint4 xv = *reinterpret_cast<const uint4*>(x + i);
    if (delta) {
      const uint4 dv = *reinterpret_cast<const uint4*>(delta + i);
      xv.x = h2_as_u32(__hadd2(u32_as_h2(xv.x), u32_as_h2(dv.x)));
      xv.y = h2_as_u32(__hadd2(u32_as_h2(xv.y), u32_as_h2(dv.y)));
      xv.z = h2_as_u32(__hadd2(u32_as_h2(xv.z), u32_as_h2(dv.z)));
      xv.w = h2_as_u32(__hadd2(u32_as_h2(xv.w), u32_as_h2(dv.w)));
      *reinterpret_cast<uint4*>(x + i) = xv;
    }
    const uint32_t u[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __half22float2(u32_as_h2(u[k]));
      ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
    }
  }
  ss = warp_sum(ss);
  if ((tid & 31) == 0) wsum[tid >> 5] = ss;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (blockDim.x >> 5) ? wsum[tid] : 0.f;
    v = warp_sum(v);
    if (tid == 0) wsum[0] = v;
  }
  __syncthreads();
  const float r = rsqrtf(wsum[0] / H + eps);
  for (int i = tid * 8; i < H; i += blockDim.x * 8) {
    const uint4 xv = *reinterpret_cast<const uint4*>(x + i);  // this thread's own write: visible
    const uint4 wv = *reinterpret_cast<const uint4*>(w + i);
    const uint32_t xu[4] = {xv.x, xv.y, xv.z, xv.w}, wu[4] = {wv.x, wv.y, wv.z, wv.w};
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __half22float2(u32_as_h2(xu[k]));
      const half2 hn = __floats2half2_rn(f.x * r, f.y * r);
      o[k] = h2_as_u32(__hmul2(u32_as_h2(wu[k]), hn));
    }
    *reinterpret_cast<uint4*>(out + i) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

void add_rmsnorm_launch(void* x, const void* delta, const void* w, void* out, int H, float eps, cudaStream_t st) {
  add_rmsnorm_kernel<<<1, 256, 0, st>>>((half*)x, (const half*)delta, (const half*)w, (half*)out, H, eps);
}
