// RoPE (rotate_half convention, pairs (i, i+D/2)) on q and k of the fused qkv vector, then append k, v
// into the paged KV cache at position *pos. q is rotated in place. One 32-thread block per head.
#include "common.cuh"
#include <cuda_runtime.h>

__global__ void rope_append_kernel(half* __restrict__ qkv, const float* __restrict__ cos_t,
                                   const float* __restrict__ sin_t, const int* __restrict__ pos_ptr,
                                   const int* __restrict__ bt, half* __restrict__ kc, half* __restrict__ vc, int n_q,
                                   int n_kv, int log2P) {
  constexpr int D = 64, HALF = 32;
  const int head = blockIdx.x, i = threadIdx.x;
  const int pos = *pos_ptr;
  half* x = qkv + head * D;
  const int page = bt[pos >> log2P], off = pos & ((1 << log2P) - 1);
  if (head < n_q + n_kv) {
    const float c = cos_t[pos * HALF + i], s = sin_t[pos * HALF + i];
    const float a = __half2float(x[i]), b = __half2float(x[i + HALF]);
    const half ra = __float2half(a * c - b * s), rb = __float2half(b * c + a * s);
    if (head < n_q) {
      x[i] = ra;
      x[i + HALF] = rb;
    } else {
      half* dst = kc + ((((size_t)page * n_kv + (head - n_q)) << log2P) + off) * D;
      dst[i] = ra;
      dst[i + HALF] = rb;
    }
  } else {
    half* dst = vc + ((((size_t)page * n_kv + (head - n_q - n_kv)) << log2P) + off) * D;
    dst[i] = x[i];
    dst[i + HALF] = x[i + HALF];
  }
}

void rope_append_launch(void* qkv, const float* cos_t, const float* sin_t, const int* pos, const int* bt, void* kc,
                        void* vc, int n_q, int n_kv, int log2P, cudaStream_t st) {
  rope_append_kernel<<<n_q + 2 * n_kv, 32, 0, st>>>((half*)qkv, cos_t, sin_t, pos, bt, (half*)kc, (half*)vc, n_q,
                                                    n_kv, log2P);
}
