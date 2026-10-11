// Torch bindings. All launches go on the current stream so they are capturable into CUDA graphs.
// Alignment: every 128-bit load assumes 16-byte aligned base pointers and row strides; checked here once
// (cheap host-side asserts) rather than discovered as a misaligned-address fault inside a graph.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void gemv_w4a16_launch(int stage, int R, int warps, int pf, const void* Wp, const void* SZ, const void* x, void* y,
                       int N, int K, int log2G, cudaStream_t st);
void bw_read_launch(const void* p, size_t bytes, void* sink, int blocks, int threads, int unroll, cudaStream_t st);
void paged_attention_launch(const void* q, const void* kc, const void* vc, const int* bt, const int* pos, int n_q,
                            int n_kv, int log2P, int PS, int max_splits, float* o_part, float* m_part,
                            float* l_part, void* out, cudaStream_t st);
void add_rmsnorm_launch(void* x, const void* delta, const void* w, void* out, int H, float eps, cudaStream_t st);
void swiglu_launch(const void* gu, void* out, int I, cudaStream_t st);
void rope_append_launch(void* qkv, const float* cos_t, const float* sin_t, const int* pos, const int* bt, void* kc,
                        void* vc, int n_q, int n_kv, int log2P, cudaStream_t st);

#define CHECK_CUDA(t) TORCH_CHECK((t).is_cuda() && (t).is_contiguous(), #t " must be a contiguous CUDA tensor")
#define CHECK_ALIGN16(t) TORCH_CHECK(reinterpret_cast<uintptr_t>((t).data_ptr()) % 16 == 0, #t " must be 16B aligned")

static int ilog2_exact(int v, const char* what) {
  TORCH_CHECK(v > 0 && (v & (v - 1)) == 0, what, " must be a power of two, got ", v);
  return 31 - __builtin_clz(v);
}
static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }

void gemv_w4a16(int64_t stage, int64_t R, int64_t warps, int64_t pf, torch::Tensor packed, torch::Tensor sz,
                torch::Tensor x, torch::Tensor y, int64_t group) {
  CHECK_CUDA(packed); CHECK_CUDA(sz); CHECK_CUDA(x); CHECK_CUDA(y);
  const int N = packed.size(0), K = packed.size(1) * 8;
  TORCH_CHECK(K % 32 == 0 && K % group == 0, "K must be a multiple of 32 and of the group size");
  TORCH_CHECK(group >= 32, "group must be >= 32 (one uint4 never straddles a group)");
  TORCH_CHECK(x.numel() == K && y.numel() == N && sz.size(0) == N && sz.size(1) == K / group);
  TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf && sz.scalar_type() == at::kHalf);
  CHECK_ALIGN16(packed); CHECK_ALIGN16(x);  // row stride K/2 bytes is a multiple of 16 since K % 32 == 0
  gemv_w4a16_launch(stage, R, warps, pf, packed.data_ptr(), sz.data_ptr(), x.data_ptr(), y.data_ptr(), N, K,
                    ilog2_exact(group, "group"), cur());
}

void bw_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads, int64_t unroll) {
  CHECK_CUDA(buf); CHECK_ALIGN16(buf);
  bw_read_launch(buf.data_ptr(), buf.numel() * buf.element_size(), sink.data_ptr(), blocks, threads, unroll, cur());
}

void paged_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor bt, torch::Tensor pos,
                     int64_t part_size, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part,
                     torch::Tensor out) {
  CHECK_CUDA(q); CHECK_CUDA(kc); CHECK_CUDA(vc); CHECK_CUDA(bt); CHECK_CUDA(pos); CHECK_CUDA(out);
  TORCH_CHECK(kc.dim() == 4 && kc.size(3) == 64, "cache must be [pages, n_kv, P, 64]");
  CHECK_ALIGN16(kc); CHECK_ALIGN16(vc);
  TORCH_CHECK(bt.scalar_type() == at::kInt && pos.scalar_type() == at::kInt);
  const int n_q = q.numel() / 64, n_kv = kc.size(1), P = kc.size(2);
  const int hpg = n_q / n_kv;
  TORCH_CHECK(n_q % n_kv == 0 && (hpg == 1 || hpg == 2 || hpg == 4 || hpg == 8), "heads-per-group must be 1/2/4/8");
  const int max_splits = o_part.size(1);
  TORCH_CHECK(o_part.size(0) == n_q && m_part.size(1) == max_splits && l_part.size(1) == max_splits);
  paged_attention_launch(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), bt.data_ptr<int>(), pos.data_ptr<int>(), n_q,
                         n_kv, ilog2_exact(P, "page size"), part_size, max_splits, o_part.data_ptr<float>(),
                         m_part.data_ptr<float>(), l_part.data_ptr<float>(), out.data_ptr(), cur());
}

void add_rmsnorm(torch::Tensor x, c10::optional<torch::Tensor> delta, torch::Tensor w, torch::Tensor out,
                 double eps) {
  CHECK_CUDA(x); CHECK_CUDA(w); CHECK_CUDA(out);
  TORCH_CHECK(x.numel() % 8 == 0);
  CHECK_ALIGN16(x); CHECK_ALIGN16(w); CHECK_ALIGN16(out);
  const void* d = nullptr;
  if (delta.has_value()) { CHECK_CUDA(*delta); CHECK_ALIGN16(*delta); d = delta->data_ptr(); }
  add_rmsnorm_launch(x.data_ptr(), d, w.data_ptr(), out.data_ptr(), x.numel(), (float)eps, cur());
}

void swiglu(torch::Tensor gu, torch::Tensor out) {
  CHECK_CUDA(gu); CHECK_CUDA(out);
  TORCH_CHECK(out.numel() % 8 == 0 && gu.numel() == 2 * out.numel());
  CHECK_ALIGN16(gu); CHECK_ALIGN16(out);
  swiglu_launch(gu.data_ptr(), out.data_ptr(), out.numel(), cur());
}

void rope_append(torch::Tensor qkv, torch::Tensor cos_t, torch::Tensor sin_t, torch::Tensor pos, torch::Tensor bt,
                 torch::Tensor kc, torch::Tensor vc, int64_t n_q, int64_t n_kv) {
  CHECK_CUDA(qkv); CHECK_CUDA(cos_t); CHECK_CUDA(sin_t); CHECK_CUDA(kc); CHECK_CUDA(vc);
  TORCH_CHECK(qkv.numel() == (n_q + 2 * n_kv) * 64 && cos_t.size(1) == 32);
  rope_append_launch(qkv.data_ptr(), cos_t.data_ptr<float>(), sin_t.data_ptr<float>(), pos.data_ptr<int>(),
                     bt.data_ptr<int>(), kc.data_ptr(), vc.data_ptr(), n_q, n_kv,
                     ilog2_exact(kc.size(2), "page size"), cur());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_w4a16", &gemv_w4a16);
  m.def("bw_read", &bw_read);
  m.def("paged_attention", &paged_attention);
  m.def("add_rmsnorm", &add_rmsnorm, py::arg("x"), py::arg("delta"), py::arg("w"), py::arg("out"), py::arg("eps"));
  m.def("swiglu", &swiglu);
  m.def("rope_append", &rope_append);
}
