# NOTEBOOK — Low-latency quantized LLM decode kernels (T4 / sm_75)

Running log of decisions, measurements, failures. Newest entries at the bottom of each section.

## Decisions (one line each)

- **Model: TinyLlama-1.1B-Chat-v1.0** — every dim (2048, 5632, 256) divides by 64/128/256 so the group-size sweep needs no padding; Qwen2.5-0.5B has 896 (not divisible by 256), QKV biases, and a 152K-vocab LM head that would swamp the body.
- **Single GPU** of the T4 x2 (`CUDA_VISIBLE_DEVICES=0`) — batch-1 decode is latency-bound; tensor-parallel over PCIe would only add sync cost.
- **Quantizer: asymmetric RTN min/max per group**, fp16 scale + integer-valued fp16 zero, nibbles in [0,15]. No GPTQ/AWQ: the project is about kernels, and RTN gives an honest lower bound on quality.
- **Packing: interleaved nibbles in uint32 words** (elem 8j+2i at bits 4i, elem 8j+2i+1 at bits 16+4i) so `(w>>4i)&0x000F000F | 0x64006400` yields a `__half2` of (1024+q_a, 1024+q_b) — the magic-number dequant with one and+or per 2 weights. Same format for every GEMV stage so stages are apples-to-apples.
- **Dequant = (q+1024 − (z+1024)) · s in half**: the subtraction is exact (integers ≤ 2048 are exact in fp16), so the kernel's per-element weight equals the reference `((q−z)·s).half()` bit-for-bit. The tempting FMA form `(1024+q)·s − (1024+z)·s` loses ~s/2 per weight — rejected.
- **Scale/zero stored as an interleaved half2 [N, K/G]** → one 4-byte load per group per row.
- **Prefill stays outside the custom kernels**: prompts are fed through the *decode* path token-by-token (graph-replayed). Simple, exercises exactly the kernels under test; prefill speed is not a goal.
- **CUDA graphs for the decode step**: kernels read `pos` from a device int, so one captured graph serves every token; grid sizes are fixed at the run's max context (surplus attention splits exit early).
- **L2 flush (16 MB memset) before every timed GEMV/attention iteration**: T4 L2 is 4 MB and a 2.6 MB QKV weight would otherwise be L2-resident across iterations, inflating bandwidth. In the real model consecutive layers never reuse weights.
- **Long-context speed runs fill the KV cache with random values** rather than prefilling 16K real tokens: timing is data-independent, and TinyLlama is trained to 2048 positions, so quality beyond 2K isn't meaningful anyway (speed-only past 2K, stated in RESULTS).
- **Block table is a random permutation of pages** in every test and benchmark, so paging is exercised, not a contiguous buffer in disguise.

## Environment (verified 2026-10-11, hello run)

Kaggle T4 x2: Tesla T4 sm_75, driver 580.178, CUDA 12.8, torch 2.11.0+cu128, transformers 5.16.1, Python 3.13.
`ncu` present at /usr/local/cuda/bin/ncu, `nsys` absent. 31 GB host RAM, 4 vCPU.

**Turing limitations that shape the design:** no `cp.async` (sm_80+) → double-buffering is manual register prefetch; no bf16 → everything fp16 / `__half2`; tensor cores are fp16 WMMA 16x16x16 only (no int4 MMA via WMMA on sm_75 in a usable form for W4A16).

## Log

### 2026-10-11 — harness, quantizer, kernels, first GPU test run
- Harness verified with a trivial kernel (63 s round trip). T4 x2 visible; pinned to GPU 0.
- Local CPU tests: packing hand-check (`[0..7]` -> `0x75316420`), pack/unpack roundtrip, RTN error <= scale/2, split-softmax combine model vs plain softmax (ctx 1..1000, part 1..4096) — all pass before any GPU time.
- Local CPU runtime (reference backend, fp16 weights) vs HF fp32 on a 13-token prompt: max |Δlogit| 0.125; the single top-1 flip is a near-tie (HF margin 0.003). INT4 reference path: 3/13 flips, largest Δlogit 15 at position 0 (BOS; extreme activations) — quantified properly by perplexity later.
- **First GPU run: 328/328 numerical tests pass.** Worst errors: GEMV max_rel 6.5e-4 (≈1 fp16 ulp of output scale, all stages, all G), paged attention max_rel 3.2e-4 (ctx 1..16K, flat + peaky, ragged partitions 16/32/48/64), norm/swiglu/rope bit-exact vs reference. Shape-1 (exact closed-form) and one-hot nibble sweep: zero error for all 26 configs.
- ptxas: attention partial 128 regs (HPG=8, acc[8][8] in registers), 12.4 KB smem; GEMV R=8+prefetch 120 regs, R=1 ~43–56 regs.

### 2026-10-11 — kbench1 (first kernel benchmarks) — and why most of it was thrown away
- Achievable read BW **278.5 GB/s** (own read kernel, 256 MB, 320 blocks x 256 thr) = 87% of the 320 GB/s spec. D2D copy 232 GB/s (read+write).
- `ncu`: **blocked, ERR_NVGPUCTRPERM** (recorded once, moving on). `nsys`: not installed in the image.
- **Measurement bug #1 — clock throttling.** NVML showed the SM clock at 1590 MHz during autotune, then 1050 -> 690 -> 585 MHz during the GEMV stage table and the start of the attention sweep. qkv GEMV: 17.7 µs at 1590 MHz (autotune grid) vs 24.6 µs at 1050 MHz (stage table). A "memory-bound" INT4 GEMV is not clock-independent: the magic-number dequant is ~5 instructions per 2 weights, and at 585 MHz issue rate becomes the limit. Fix: every measurement records clocks/temp/power/throttle reasons; if the post-run SM clock < 1400 MHz the GPU idles 15 s and the measurement is retried (max 3); still-throttled results are flagged, not hidden. Also added a 60 s sustained-load clock probe.
- **Measurement bug #2 — dirty-L2 flush.** The 16 MB *memset* used to evict L2 between iterations leaves L2 full of dirty lines; the next kernel pays their write-back while reading. That penalises small kernels most (qkv 2.8 MB) and does not happen in the real decode step (L2 holds clean weight lines). Replaced by a 16 MB read (`sum()`), which also draws less power.
- Despite both biases the shape of the result was already clear: W4A16 autotuned 2.0–4.1x faster than cuBLAS fp16 on every shape; gate_up/lm_head reach ~220 GB/s (80% of achievable); qkv/o (2–3 MB) only ~100–150 GB/s -> added a BW-vs-transfer-size sweep to quantify the small-kernel ceiling.
- Attention v1: split-K is 4–13x faster than no-split (no-split = only 4 blocks on 40 SMs). But even the best split reached only 117 GB/s at 16K. Cause: the score loop issues one 16 B load per lane, then shuffles + smem writes before the next load -> one exposed DRAM latency per 16 tokens. **v2**: at chunk start each lane issues the K and V loads for all 4 of its token slots (8 x 16 B in flight), V doesn't depend on scores so it flies alongside K; combine kernel rewritten as block-parallel reductions (v1 combine looped serially over up to 256 splits). v1 kept and benchmarked side by side.
- Harness: Kaggle caps batch GPU sessions at 2 — a relaunch while a run + the llama.cpp build were live was refused. Added `--attach` to poll/pull an already-running kernel. Also stopped piping the harness through `tail` (it masked a non-zero exit once).

### 2026-10-11 — llama.cpp Q4_K_M reference (commit 23b0202a)
- First build failed: `CUDA::cuda_driver` target not found (no libcuda stub in the image, needed by the VMM path) -> `-DGGML_CUDA_NO_VMM=ON`.
- `llama-bench -p 0 -n 128 -d <ctx> -r 5`, model file 669 MB (Q4_K_M; output layer is Q6_K):

| ctx | 512 | 1K | 2K | 4K | 8K | 16K |
|---|---|---|---|---|---|---|
| fa=0 tok/s | 204.5 | 195.5 | 184.8 | 162.3 | 127.1 | 84.2 |
| fa=1 tok/s | 219.3 | 216.4 | 210.9 | 199.9 | 183.3 | 159.6 |
