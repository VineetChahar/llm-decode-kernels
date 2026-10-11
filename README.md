# Low-latency quantized LLM decode kernels (T4 / sm_75)

From-scratch CUDA kernels for **batch-1 LLM decoding** — W4A16 GEMV with fused dequantization, paged
KV-cache split-K decode attention, fused residual+RMSNorm and SwiGLU — wired into a CUDA-graph decode
runtime for **TinyLlama-1.1B** and measured end-to-end against HF `transformers` fp16, a cuBLAS fp16
build of the same runtime, and `llama.cpp` Q4_K_M.

The question: *is time-per-token explained by weight bytes moved — and if not, by exactly what else?*
Answer and numbers: **[RESULTS.md](RESULTS.md)**. Design log, dead ends and measurement bugs: **[NOTEBOOK.md](NOTEBOOK.md)**.

## Reproduce (one command)

Needs a Kaggle account with API token in `~/.kaggle/` (free; T4 x2 GPU quota). Nothing else is pushed anywhere.

```bash
python3 -m venv .venv && .venv/bin/pip install kaggle matplotlib numpy torch transformers datasets
.venv/bin/python scripts/kaggle_run.py --tag full --cmd "bash scripts/run_all.sh"   # ~1.5 h on one T4
.venv/bin/python scripts/plot_all.py runs/<stamp>_full/results.json --llamacpp runs/<stamp>_llamacpp/llamacpp.json
```

Faster: split across Kaggle's two concurrent GPU sessions (each re-runs the accuracy gate first):
`--tag k --cmd "PART=kernels bash scripts/run_all.sh"` and `--slug llm-decode-e2e --tag e --cmd "PART=e2e bash scripts/run_all.sh"`,
then `plot_all.py runs/<k>/results.json runs/<e>/results.json --llamacpp …` (later files override earlier keys; the
merged file is written to `results.json`). The e2e part loads `configs/autotune.json` / `attn_policy.json`, so run the
kernel part first if you want freshly tuned configs.

`scripts/run_all.sh` runs the **accuracy gate first** (`tests/test_kernels.py`: exits non-zero on any
numerical failure, before any speed number exists), then `bench/kernel_bench.py` (bandwidth microbenchmark,
clock probe, autotuner, GEMV stage ladder, group sweep, attention sweep, ncu attempt), then `bench/e2e.py`
(perplexity, greedy agreement, tokens/s vs context, ablations, profiler breakdown). All results go to one
`results.json`; every plot is regenerated from it by `scripts/plot_all.py`.

llama.cpp reference (separate kernel, CUDA build ~10 min):
`scripts/kaggle_run.py --slug llm-decode-llamacpp --tag llamacpp --cmd "bash scripts/llamacpp_bench.sh"`.

Local, CPU-only checks (no GPU): `python tests/test_quant_local.py`, `tests/test_ref_local.py`,
`tests/test_runtime_local.py` (reference-backend runtime vs HF).

## Layout

```
kernels/     gemv_w4a16.cu (4 stages + bandwidth microbench), paged_attention.cu (v1, v2 + combines),
             fused_norm.cu, fused_swiglu.cu, rope_kv.cu, bindings.cpp
runtime/     quant.py (group-wise INT4 + nibble packing), model.py (decode runtime, CUDA graphs), ext.py (JIT build)
reference/   ref_ops.py — fp16 PyTorch reference for every kernel (also the CPU backend)
bench/       kernel_bench.py, e2e.py, timing.py (CUDA events, L2 flush, clock gating), analyze.py (models)
scripts/     kaggle_run.py (push -> poll -> pull), autotune.py, plot_all.py, run_all.sh, llamacpp_bench.sh
tests/       test_kernels.py (GPU gates), test_*_local.py (CPU)
configs/     autotune.json, attn_policy.json — written by the sweeps, loaded by the runtime at startup
runs/        pulled Kaggle outputs, one directory per run
plots/       regenerated figures
```
