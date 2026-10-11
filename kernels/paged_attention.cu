// Paged KV-cache decode attention (single query token, GQA), vLLM-style block table.
//
// Cache layout per layer: k_cache/v_cache [num_pages, n_kv, P, D] fp16, D = 64, P = page size (power of 2).
// block_table[i] = physical page holding tokens [i*P, (i+1)*P). Context length = *pos + 1 (device int, so
// one CUDA graph serves every decode step).
//
// Split-K ("flash-decoding"): grid (n_kv, max_splits). Block (kvh, s) owns tokens [s*PS, min(ctx,(s+1)*PS))
// for all HPG query heads that share kv head kvh -> each K/V byte is read from DRAM exactly once.
// Inside a block, chunks of CH tokens are folded in with the online-softmax recurrence
//     m' = max(m, max_chunk s);  alpha = e^(m - m');  l' = l*alpha + sum e^(s - m');  o' = o*alpha + sum e^(s-m') v
// Each split writes UNNORMALIZED o, plus m and l; the combine kernel rescales:
//     M = max_s m_s;  L = sum_s l_s e^(m_s - M);  O = sum_s o_s e^(m_s - M) / L
// When only one split is live, the partial kernel normalizes and writes the output itself (combine exits).
//
// Thread map (128 threads = 4 warps): lane = 8*g + c. In the score and V phases, each warp covers 4 tokens
// per step (g = token slot, c = 8-wide dim chunk), so one warp-wide 16B/lane load reads 4 consecutive tokens'
// 128-byte rows = 512 contiguous bytes (tokens within a page are contiguous).
#include "common.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr int D = 64;
constexpr int CH = 64;
constexpr int NT = 128;

DEVI void unpack8(uint4 v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __half22float2(u32_as_h2(w[i]));
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
}  // namespace

// PRE=false (v1): each warp step loads one K row-chunk, computes, shuffles, then loads the next -> one exposed
//                 DRAM latency per 16 tokens, ~1 load in flight per lane.
// PRE=true  (v2): at chunk start every lane issues the K AND V loads for all its CH/16 token slots (8 x 16 B in
//                 flight per lane) before any math; V addresses do not depend on the scores, so they can fly
//                 alongside K. One exposed latency per chunk instead of 2*CH/16.
template <int HPG, bool PRE>
__global__ void __launch_bounds__(NT)
paged_attn_partial(const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc,
                   const int* __restrict__ bt, const int* __restrict__ pos_ptr, int n_kv, int log2P, int PS,
                   float scale, float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
                   half* __restrict__ out, int max_splits) {
  constexpr int SLOTS = CH / 16;
  __shared__ float q_s[HPG][D];
  __shared__ float s_s[HPG][CH];
  __shared__ float m_s[HPG], l_s[HPG], a_s[HPG];
  __shared__ float red[NT / 32][HPG][D];

  const int kvh = blockIdx.x, split = blockIdx.y;
  const int ctx = *pos_ptr + 1;
  const int nsplit = min((ctx + PS - 1) / PS, max_splits);  // clamp: never index past the split buffers
  if (split >= nsplit) return;
  const int t0 = split * PS, t1 = min(ctx, t0 + PS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 3, c = lane & 7;
  const int P = 1 << log2P;

  for (int i = tid; i < HPG * D; i += NT) q_s[i / D][i % D] = __half2float(q[kvh * HPG * D + i]) * scale;
  if (tid < HPG) { m_s[tid] = -CUDART_INF_F; l_s[tid] = 0.f; }
  __syncthreads();

  float acc[HPG][8];
#pragma unroll
  for (int h = 0; h < HPG; ++h)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[h][e] = 0.f;

  auto row = [&](const half* cache, int t) {  // this lane's 16 B of token t's 128 B row
    const size_t base = ((((size_t)__ldg(bt + (t >> log2P)) * n_kv + kvh) << log2P) + (t & (P - 1))) * D;
    return __ldg(reinterpret_cast<const uint4*>(cache + base) + c);
  };
  auto score = [&](uint4 kv, bool valid, int tt) {
    float sc[HPG];
    float kf[8];
    unpack8(kv, kf);
#pragma unroll
    for (int h = 0; h < HPG; ++h) {
      float a = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) a = fmaf(kf[e], q_s[h][c * 8 + e], a);
      sc[h] = valid ? a : 0.f;
    }
#pragma unroll
    for (int h = 0; h < HPG; ++h) {
      sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 1);
      sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 2);
      sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 4);
      if ((h & 7) == c) s_s[h][tt] = valid ? sc[h] : -CUDART_INF_F;
    }
  };
  auto accum = [&](uint4 vv, int tt) {
    float vf[8];
    unpack8(vv, vf);
#pragma unroll
    for (int h = 0; h < HPG; ++h) {
      const float p = s_s[h][tt];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[h][e] = fmaf(p, vf[e], acc[h][e]);
    }
  };

  for (int cs = t0; cs < t1; cs += CH) {
    uint4 kr[SLOTS], vr[SLOTS];
    if constexpr (PRE) {
#pragma unroll
      for (int j = 0; j < SLOTS; ++j) {
        const int t = cs + warp * 4 + g + 16 * j;
        if (t < t1) { kr[j] = row(kc, t); vr[j] = row(vc, t); }
        else { kr[j] = make_uint4(0, 0, 0, 0); vr[j] = kr[j]; }
      }
    }
    // ---- 1) scores for this chunk -> s_s
    if constexpr (PRE) {
#pragma unroll
      for (int j = 0; j < SLOTS; ++j) {
        const int tt = warp * 4 + g + 16 * j;
        score(kr[j], cs + tt < t1, tt);
      }
    } else {
      for (int tt = warp * 4 + g; tt < CH; tt += 16) {
        const int t = cs + tt;
        score(t < t1 ? row(kc, t) : make_uint4(0, 0, 0, 0), t < t1, tt);
      }
    }
    __syncthreads();

    // ---- 2) online-softmax update: 16 threads per head
    if (tid < HPG * 16) {
      constexpr unsigned mask = (HPG * 16 >= 32) ? 0xffffffffu : ((1u << (HPG * 16)) - 1u);
      const int h = tid >> 4, j = tid & 15;
      const float m_old = m_s[h];
      float mx = -CUDART_INF_F;
      for (int i = j; i < CH; i += 16) mx = fmaxf(mx, s_s[h][i]);
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(mask, mx, o));
      const float m_new = fmaxf(m_old, mx);  // finite: every chunk has >= 1 live token
      float sum = 0.f;
      for (int i = j; i < CH; i += 16) {
        const float p = __expf(s_s[h][i] - m_new);  // masked: exp(-inf) = 0
        s_s[h][i] = p;
        sum += p;
      }
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(mask, sum, o);
      __syncwarp(mask);
      if (j == 0) {
        const float alpha = __expf(m_old - m_new);  // m_old = -inf on first chunk -> 0
        a_s[h] = alpha;
        l_s[h] = l_s[h] * alpha + sum;
        m_s[h] = m_new;
      }
    }
    __syncthreads();

    // ---- 3) rescale running output, accumulate p * V
#pragma unroll
    for (int h = 0; h < HPG; ++h) {
      const float al = a_s[h];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[h][e] *= al;
    }
    if constexpr (PRE) {
#pragma unroll
      for (int j = 0; j < SLOTS; ++j) {
        const int tt = warp * 4 + g + 16 * j;
        if (cs + tt < t1) accum(vr[j], tt);
      }
    } else {
      for (int tt = warp * 4 + g; tt < CH; tt += 16)
        if (cs + tt < t1) accum(row(vc, cs + tt), tt);
    }
    __syncthreads();  // s_s / a_s are overwritten by the next chunk
  }

  // ---- reduce over the 4 token slots in each warp, then across the 4 warps
#pragma unroll
  for (int h = 0; h < HPG; ++h)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      acc[h][e] += __shfl_xor_sync(0xffffffffu, acc[h][e], 8);
      acc[h][e] += __shfl_xor_sync(0xffffffffu, acc[h][e], 16);
    }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < HPG; ++h)
#pragma unroll
      for (int e = 0; e < 8; ++e) red[warp][h][c * 8 + e] = acc[h][e];
  }
  __syncthreads();
  for (int i = tid; i < HPG * D; i += NT) {
    const int h = i / D, d = i % D, head = kvh * HPG + h;
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w) o += red[w][h][d];
    if (nsplit == 1) out[head * D + d] = __float2half(o / l_s[h]);
    else o_part[((size_t)head * max_splits + split) * D + d] = o;
  }
  if (nsplit > 1 && tid < HPG) {
    const int head = kvh * HPG + tid;
    m_part[head * max_splits + split] = m_s[tid];
    l_part[head * max_splits + split] = l_s[tid];
  }
}

__global__ void paged_attn_combine(const float* __restrict__ o_part, const float* __restrict__ m_part,
                                   const float* __restrict__ l_part, const int* __restrict__ pos_ptr, int PS,
                                   int max_splits, half* __restrict__ out) {
  const int head = blockIdx.x, d = threadIdx.x;
  const int ctx = *pos_ptr + 1;
  const int nsplit = min((ctx + PS - 1) / PS, max_splits);
  if (nsplit <= 1) return;
  const float* m = m_part + head * max_splits;
  const float* l = l_part + head * max_splits;
  float M = -CUDART_INF_F;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, m[s]);
  float L = 0.f, o = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float a = __expf(m[s] - M);
    L = fmaf(a, l[s], L);
    o = fmaf(a, o_part[((size_t)head * max_splits + s) * D + d], o);
  }
  out[head * D + d] = __float2half(o / L);
}

// v2 combine: 256 threads per head. Max and sum over splits are block-parallel reductions; the output
// accumulation runs 4 split-lanes x 64 dims (coalesced over d), so cost no longer grows serially with splits.
__global__ void __launch_bounds__(256)
paged_attn_combine_v2(const float* __restrict__ o_part, const float* __restrict__ m_part,
                      const float* __restrict__ l_part, const int* __restrict__ pos_ptr, int PS, int max_splits,
                      half* __restrict__ out) {
  __shared__ float wred[8];
  __shared__ float o_s[4][D];
  const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ctx = *pos_ptr + 1;
  const int nsplit = min((ctx + PS - 1) / PS, max_splits);
  if (nsplit <= 1) return;
  const float* m = m_part + head * max_splits;
  const float* l = l_part + head * max_splits;

  float mx = -CUDART_INF_F;
  for (int s = tid; s < nsplit; s += 256) mx = fmaxf(mx, m[s]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) wred[warp] = mx;
  __syncthreads();
  float M = wred[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) M = fmaxf(M, wred[w]);
  __syncthreads();

  float L = 0.f;
  for (int s = tid; s < nsplit; s += 256) L = fmaf(__expf(m[s] - M), l[s], L);
  L = warp_sum(L);
  if (lane == 0) wred[warp] = L;
  __syncthreads();
  L = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) L += wred[w];

  const int d = tid & (D - 1), sl = tid >> 6;
  float o = 0.f;
  const float* op = o_part + (size_t)head * max_splits * D + d;
#pragma unroll 4
  for (int s = sl; s < nsplit; s += 4) o = fmaf(__expf(m[s] - M), op[(size_t)s * D], o);
  o_s[sl][d] = o;
  __syncthreads();
  if (tid < D) out[head * D + tid] = __float2half((o_s[0][tid] + o_s[1][tid] + o_s[2][tid] + o_s[3][tid]) / L);
}

void paged_attention_launch(int version, const void* q, const void* kc, const void* vc, const int* bt, const int* pos,
                            int n_q, int n_kv, int log2P, int PS, int max_splits, float* o_part, float* m_part,
                            float* l_part, void* out, cudaStream_t st) {
  const int hpg = n_q / n_kv;
  const float scale = 1.0f / sqrtf((float)D);
  dim3 grid(n_kv, max_splits);
#define HCASE(H)                                                                                                  \
  case H:                                                                                                         \
    if (version >= 2)                                                                                             \
      paged_attn_partial<H, true><<<grid, NT, 0, st>>>((const half*)q, (const half*)kc, (const half*)vc, bt, pos,  \
                                                       n_kv, log2P, PS, scale, o_part, m_part, l_part, (half*)out, \
                                                       max_splits);                                               \
    else                                                                                                          \
      paged_attn_partial<H, false><<<grid, NT, 0, st>>>((const half*)q, (const half*)kc, (const half*)vc, bt, pos, \
                                                        n_kv, log2P, PS, scale, o_part, m_part, l_part,           \
                                                        (half*)out, max_splits);                                  \
    break;
  switch (hpg) {
    HCASE(1) HCASE(2) HCASE(4) HCASE(8)
    default: return;  // caller validates
  }
#undef HCASE
  if (max_splits > 1) {
    if (version >= 2)
      paged_attn_combine_v2<<<n_q, 256, 0, st>>>(o_part, m_part, l_part, pos, PS, max_splits, (half*)out);
    else
      paged_attn_combine<<<n_q, D, 0, st>>>(o_part, m_part, l_part, pos, PS, max_splits, (half*)out);
  }
}
