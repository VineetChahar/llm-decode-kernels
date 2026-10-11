// W4A16 GEMV with fused group-wise dequantization:  y[N] = W[N,K] (int4, group G) . x[K] (fp16)
//
// Stages are kept side by side and all benchmarked:
//   stage 0  naive        one warp/row, 32-bit weight loads, scalar nibble extract, fp32 FMA, scalar x loads
//   stage 1  vectorized   one warp/row, 128-bit streaming weight loads + 128-bit x loads, magic-number
//                         half2 dequant, __half2 FMA, fp32 cross-uint4 accumulation, __shfl_xor_sync reduce
//   stage 2  reg-blocked  R rows/warp: each x chunk is loaded once and reused for R rows (x reloads are
//                         4x the weight bytes at R=1!), plus manual register double-buffering of the
//                         next weight chunk (Turing has no cp.async, so this is the only prefetch)
//   stage 3  autotuned    stage-2 template, (R, warps/block, prefetch) picked per shape by scripts/autotune.py
#include "common.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

// ---------------- stage 0: naive ----------------
__global__ void gemv_naive(const uint32_t* __restrict__ Wp, const half2* __restrict__ SZ,
                           const half* __restrict__ x, half* __restrict__ y, int N, int K, int log2G) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const int words = K >> 3, ng = K >> log2G;
  float acc = 0.f;
  for (int j = lane; j < words; j += 32) {
    const uint32_t w = Wp[(size_t)row * words + j];
    const half2 sz = SZ[(size_t)row * ng + ((j << 3) >> log2G)];
    const float s = __low2float(sz), z = __high2float(sz);
    for (int e = 0; e < 8; ++e) {
      const int sh = 4 * (e >> 1) + 16 * (e & 1);
      const float q = (float)((w >> sh) & 0xFu);
      const float wv = __half2float(__float2half((q - z) * s));
      acc += wv * __half2float(x[(j << 3) + e]);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row] = __float2half(acc);
}

// ---------------- stages 1-3: vectorized / register-blocked / autotuned ----------------
template <int R, bool PF>
__global__ void __launch_bounds__(256)
gemv_rb(const uint4* __restrict__ Wp, const half2* __restrict__ SZ, const uint4* __restrict__ X,
        half* __restrict__ y, int N, int K, int log2G) {
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
  if (row0 >= N) return;
  const int U = K >> 5;  // uint4 (= 32 int4 weights) per row
  const int ng = K >> log2G;
  const uint4* wrow[R];
  const half2* szrow[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int rr = min(row0 + r, N - 1);  // tail rows: clamp loads, skip the store
    wrow[r] = Wp + (size_t)rr * U;
    szrow[r] = SZ + (size_t)rr * ng;
  }
  const half2 k1024 = __float2half2_rn(1024.f);
  float acc[R];
  uint4 wc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
  if (PF && lane < U) {
#pragma unroll
    for (int r = 0; r < R; ++r) wc[r] = ld_stream(wrow[r] + lane);
  }
  for (int u = lane; u < U; u += 32) {
    uint4 wn[R];
    if (PF) {
      if (u + 32 < U) {  // issue next iteration's loads before consuming this one
#pragma unroll
        for (int r = 0; r < R; ++r) wn[r] = ld_stream(wrow[r] + u + 32);
      }
    } else {
#pragma unroll
      for (int r = 0; r < R; ++r) wc[r] = ld_stream(wrow[r] + u);
    }
    const int g = (u << 5) >> log2G;
    const uint4 x0 = __ldg(X + 4 * u), x1 = __ldg(X + 4 * u + 1), x2 = __ldg(X + 4 * u + 2), x3 = __ldg(X + 4 * u + 3);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const half2 sz = __ldg(szrow[r] + g);
      const half2 s2 = __half2half2(__low2half(sz));
      const half2 z2 = __hadd2(__half2half2(__high2half(sz)), k1024);
      half2 a = __float2half2_rn(0.f);
      a = dq_fma_word(wc[r].x, x0, s2, z2, a);
      a = dq_fma_word(wc[r].y, x1, s2, z2, a);
      a = dq_fma_word(wc[r].z, x2, s2, z2, a);
      a = dq_fma_word(wc[r].w, x3, s2, z2, a);
      acc[r] += __low2float(a) + __high2float(a);  // fp32 across uint4 chunks: only 16 products ever sum in half
    }
    if (PF) {
#pragma unroll
      for (int r = 0; r < R; ++r) wc[r] = wn[r];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float v = warp_sum(acc[r]);
    if (lane == r && row0 + r < N) y[row0 + r] = __float2half(v);
  }
}

template <int R, bool PF>
static void launch_rb(const void* Wp, const void* SZ, const void* x, void* y, int N, int K, int log2G, int warps,
                      cudaStream_t st) {
  const int rows_per_block = R * warps;
  const int blocks = (N + rows_per_block - 1) / rows_per_block;
  gemv_rb<R, PF><<<blocks, 32 * warps, 0, st>>>((const uint4*)Wp, (const half2*)SZ, (const uint4*)x, (half*)y, N,
                                                 K, log2G);
}

void gemv_w4a16_launch(int stage, int R, int warps, int pf, const void* Wp, const void* SZ, const void* x, void* y,
                       int N, int K, int log2G, cudaStream_t st) {
  if (stage == 0) {
    const int blocks = (N + warps - 1) / warps;
    gemv_naive<<<blocks, 32 * warps, 0, st>>>((const uint32_t*)Wp, (const half2*)SZ, (const half*)x, (half*)y, N, K,
                                              log2G);
    return;
  }
  if (stage == 1) { launch_rb<1, false>(Wp, SZ, x, y, N, K, log2G, warps, st); return; }
#define CASE(RR)                                                                  \
  case RR:                                                                        \
    if (pf) launch_rb<RR, true>(Wp, SZ, x, y, N, K, log2G, warps, st);            \
    else launch_rb<RR, false>(Wp, SZ, x, y, N, K, log2G, warps, st);              \
    return;
  switch (R) {
    CASE(1) CASE(2) CASE(4) CASE(8)
    default: throw std::runtime_error("gemv: R must be 1,2,4,8");
  }
#undef CASE
}

// ---------------- achievable-bandwidth microbenchmark ----------------
template <int UNROLL>
__global__ void bw_read_kernel(const uint4* __restrict__ p, size_t n, uint32_t* sink) {
  uint32_t a = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) v[k] = ld_stream(p + i + k * stride);
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) a ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
  }
  for (; i < n; i += stride) { uint4 v = ld_stream(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (a == 0x9e3779b9u) *sink = a;  // defeat dead-code elimination
}

void bw_read_launch(const void* p, size_t bytes, void* sink, int blocks, int threads, int unroll, cudaStream_t st) {
  const size_t n = bytes / 16;
  if (unroll == 4) bw_read_kernel<4><<<blocks, threads, 0, st>>>((const uint4*)p, n, (uint32_t*)sink);
  else bw_read_kernel<1><<<blocks, threads, 0, st>>>((const uint4*)p, n, (uint32_t*)sink);
}
