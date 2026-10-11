# Results — low-latency W4A16 decode kernels, TinyLlama-1.1B on a Tesla T4

Batch-1 greedy decode, CUDA-graph replay, everything below measured on Kaggle T4s (sm_75, 320 GB/s spec,
**277 GB/s achievable** by our own read microbenchmark). Every speed number has its accuracy number next to it;
every speed number was produced only after the accuracy gate (`tests/test_kernels.py`, **439/439 passed** in
each of the four GPU runs used here) exited 0.

Sources (all merged into `results.json` by `scripts/plot_all.py`):
`runs/20261011T170804Z_kbench2` (kernel benchmarks, autotuner), `runs/20261011T193218Z_e2e4` (perplexity,
agreement, end-to-end speed, profiles), `runs/20261011T102433Z_llamacpp2` (llama.cpp), and
`runs/20261011T191452Z_kbench3` (a node stuck at 585 MHz — used only as a clock-sensitivity control).

> **Clock caveat, up front.** Kaggle does not allow clock locking (`nvidia-smi -lgc`: permission denied), and the
> T4 throttles under sustained load. Kernel microbenchmarks (short, with idle gaps) ran at **1470–1590 MHz**.
> The end-to-end decode loops ran on a node in NVML `hw_slowdown` at **900–1065 MHz SM clock** (fp16 runtime:
> 480–885 MHz) even after the clock gate's cool-down retries; every such row is flagged `throttled` in
> `results.json`. The numbers are what this platform delivers, not best-case T4 numbers.

## 1. Headline

| ctx | **W4A16 + INT4 head** | W4A16 (fp16 head) | fp16 (our runtime, cuBLAS) | HF transformers fp16 | llama.cpp Q4_K_M (FA on) |
|---:|---:|---:|---:|---:|---:|
| 512 | **279.8** | 248.5 | 72.6 | 35.7 | 219.3 |
| 1K | **271.5** | 251.9 | 82.8 | 33.6 | 216.4 |
| 2K | **254.0** | 233.3 | 89.6 | 34.3 | 210.9 |
| 4K | **222.6** | 207.1 | 86.5 | 28.4 | 199.9 |
| 8K | 182.2 | 169.2 | 80.4 | OOM | **183.3** |
| 16K | 134.5 | 126.6 | 70.5 | OOM | **159.6** |

decode tokens/s, batch 1. HF = `model(next_token, past_key_values=…)` loop, wall-clock; it runs out of memory in the
8K/16K prefill on the 15 GB T4. llama.cpp = `llama-bench -n 128 -d <ctx>`, a different Kaggle node and its clocks are
not recorded by llama-bench.

**With accuracy next to it** (WikiText-2 test, 16 windows × 512 tokens = 8,176 scored tokens, BOS-prefixed;
greedy agreement vs HF fp16 on 16 fixed prompts × 64 new tokens):

| config | perplexity | Δ vs HF fp16 | greedy exact match | mean first divergence | teacher-forced top-1 agreement | weights read / token |
|---|---:|---:|---:|---:|---:|---:|
| HF transformers fp16 | 9.722 | — | — | — | — | 2.07 GB |
| our runtime fp16 | 9.721 | −0.004 % | 15/16 | 60.6 / 64 | 0.9990 | 2.07 GB |
| W4A16 G64 | 10.173 | +4.6 % | 2/16 | 13.0 | 0.9111 | 0.65 GB |
| **W4A16 G128** | 10.461 | +7.6 % | 1/16 | 14.3 | 0.9102 | 0.65 GB |
| W4A16 G256 | 10.607 | +9.1 % | 0/16 | 6.1 | 0.8896 | 0.63 GB |
| W4A16 G128 + INT4 LM head | 10.515 | +8.2 % | 1/16 | 11.8 | 0.9082 | 0.55 GB |

Reading it honestly:

* **Our fp16 path is numerically HF.** Same perplexity to 4 decimals; 15/16 greedy continuations identical for all 64
  tokens; the 16th diverges at token 9 (fp16 accumulation-order differences flip a close argmax).
* **The INT4 quality cost is real and is not hidden.** Plain round-to-nearest asymmetric INT4 (no GPTQ/AWQ) costs
  +7.6 % perplexity at G128; greedy text matches HF exactly only 1 time in 16 (it diverges after ~14 tokens on average
  and then stays fluent — the teacher-forced top-1 agreement is 91 %). This is the price of RTN on a 1.1 B model,
  not a kernel bug: the kernels reproduce the fp16 reference dequant-and-GEMV to ≤ 6.5e-4 relative error (§3).
* **vs llama.cpp:** we are 28 % faster at 512 tokens and ahead up to 4K; tied at 8K; **16 % slower at 16K**. The reason
  is attention (§5): llama.cpp's flash-attention decode kernel holds up better at long context. llama.cpp's Q4_K_M
  uses a different (better) quantizer, so its quality is not equal to our RTN INT4 — we did not measure its perplexity.
* **vs HF fp16:** 7.8× at 512 tokens. vs our own fp16 cuBLAS runtime (same kernels for everything except the
  weight GEMVs): 1.8–3.4× in the timed loops, which ran the fp16 config at even lower clocks (480–885 MHz); the
  profiled graph steps at 512 tokens are 10.56 ms vs 3.33 ms = **3.2×**, close to the 3.2× ratio of bytes per token.

TinyLlama is trained to 2,048 positions. Beyond 2K the KV cache is filled with random values and only **speed** is
meaningful there (quality claims stop at 2K).

![headline](plots/tokens_per_s_vs_context.png)

## 2. Predicted vs measured: does "bytes moved" explain time per token?

Two models, **neither fitted to the end-to-end numbers** (`bench/analyze.py`):

* **Model A** — `t = bytes(ctx) / 277 GB/s` (weights + KV cache + activations; pure roofline).
* **Model B** — `t = bytes(ctx) / BW_fit + n_kernels × (t0 + gap)`; `BW_fit` = 278.3 GB/s and `t0` = 5.6 µs from a
  linear fit of the read-bandwidth-vs-transfer-size sweep (fixed cost of one memory-bound kernel), `n_kernels` = 225
  and `gap` = 0.36 µs from the profiler's graph-replay trace.

| ctx | bytes / token | measured | Model A | Model B | A explains | **B explains** |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 659 MB | 4.03 ms | 2.38 ms | 3.71 ms | 59 % | **92 %** |
| 1K | 671 MB | 3.97 ms | 2.42 ms | 3.75 ms | 61 % | **94 %** |
| 2K | 694 MB | 4.29 ms | 2.50 ms | 3.83 ms | 58 % | **89 %** |
| 4K | 740 MB | 4.83 ms | 2.67 ms | 4.00 ms | 55 % | **83 %** |
| 8K | 832 MB | 5.91 ms | 3.00 ms | 4.33 ms | 51 % | **73 %** |
| 16K | 1017 MB | 7.90 ms | 3.67 ms | 4.99 ms | 46 % | **63 %** |

**Bytes alone explain ~60 % at short context.** Adding a fixed per-kernel cost explains **~90 % up to 2K**. The model
fails at long context, and the profiler says exactly where (graph replay, `torch.profiler`, same config):

| µs per token | ctx 512 measured | bytes / BW | ctx 16K measured | bytes / BW |
|---|---:|---:|---:|---:|
| W4A16 GEMVs (88 launches) | 2,189 | 1,850 | 2,280 | 1,850 |
| fp16 LM head (cuBLAS) | 512 | 471 | 515 | 471 |
| attention partial + combine (44) | 301 | 41 | **3,625** | **1,326** |
| norm, SwiGLU, RoPE+append, argmax, embed (93) | 245 | ≈0 | 292 | ≈0 |
| GPU idle between kernels | 81 | — | 111 | — |
| **step span (profiled)** | **3,327** | | **6,825** | |

* **What the 10 % at short context is:** GEMVs run at 85 % of achievable bandwidth (small GEMVs pay a launch-to-first-byte
  cost — a 2.8 MB read reaches only ~180 GB/s in the size sweep), plus ~90 latency-bound tiny kernels
  (attention at 512 tokens moves 2 MB in 301 µs). Model B's `t0` captures this only on average: against the
  profiled step it actually *over*-predicts (3.71 vs 3.33 ms); against the event-timed loop it under-predicts, because
  that loop ran at a lower, noisier clock (IQR 1.76 ms at ctx 512 — `hw_slowdown`).
* **What the 37 % at 16K is:** attention. It moves 369 MB per token at 16K and takes 3.6 ms → **102 GB/s in the graph,
  37 % of achievable bandwidth**. That excess (2.3 ms) is 79 % of Model B's 2.9 ms miss; the remaining ~1.1 ms is the
  profiled-vs-timed clock difference. A Model B that used *measured* attention bandwidth would be descriptive, not
  predictive, so we report the miss instead. Why attention is slow: §5.

## 3. Kernel accuracy (the gate)

`tests/test_kernels.py`, run first in every GPU job; any failure exits non-zero before a speed number exists.

| kernel group | tests | max abs err | max rel err | tolerance (rel) |
|---|---:|---:|---:|---:|
| GEMV shape-1 sanity (4-row hand-checked, one-hot over all 128 columns, every stage/config) | 52 | **0** | **0** | exact |
| GEMV real shapes × G ∈ {64,128,256} × all stages/configs | 170 | 3.9e-3 | 6.5e-4 | 5e-3 |
| paged attention shape-1: ctx 128 vs contiguous reference (head configs, page 16/32, split sizes, v1/v2) | 60 | 1.2e-4 | 2.2e-4 | 2e-3 |
| paged attention 63 – 16K, uniform and peaky scores, split sizes, v1/v2 | 138 | 4.9e-4 | 3.2e-4 | 2e-3 |
| fused add + RMSNorm, SwiGLU, RoPE + paged KV append | 19 | 0 | 0 | 2e-3 |

The references are the fp16 PyTorch ops in `reference/ref_ops.py` (HF rounding order). The GEMV reference dequantizes
with the same fp16 scale/zero, so the error is accumulation order only; the shape-1 cases are exact.

## 4. W4A16 GEMV

Format: asymmetric RTN INT4, group size G along K, fp16 scale + zero per group, 8 nibbles per `int32` in an interleaved
order so that `(w >> 4i) & 0x000F000F | 0x64006400` yields two halves `1024 + q` (the "magic number" dequant: one
`lop3`-style op + one `hsub2` per 2 weights, no int→float converts). One warp per row (or R rows), 128-bit
`ld.global.nc.L1::no_allocate` weight loads, `__half2` FMA within a 16-byte chunk then fp32 accumulation,
`__shfl_xor_sync` reduction.

Stage ladder at G128, L2 flushed between iterations (a 16 MB *read* — see NOTEBOOK for why a memset flush was a bug),
SM 1590 MHz (`kbench2`):

| shape (N×K) | 0 naive | 1 vectorized | 2 reg-blocked | **3 autotuned** | cuBLAS fp16 | speedup vs cuBLAS | % of 277 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv 2560×2048 | 28.8 µs | 17.0 | 18.6 | **15.1** | 46.8 | 3.1× | 67 % |
| o 2048×2048 | 24.5 | 15.8 | 16.3 | **13.1** | 38.8 | 3.0× | 62 % |
| gate_up 11264×2048 | 106.0 | 57.0 | 55.2 | **50.3** | 184.3 | 3.7× | 88 % |
| down 2048×5632 | 57.3 | 32.2 | 32.6 | **29.1** | 94.1 | 3.2× | 76 % |
| lm_head 32000×2048 | 773 † | 395 † | 168 † | **133.3** ‡ | 520 † | 3.9× | 95 % |

† measured throttled (585–960 MHz) in that run. ‡ from the autotuner grid at ~1500 MHz (same config).
End-to-end at ctx 512 the ladder is 114.5 → 198.3 → 241.6 → 248.5 tok/s (stages 0 → 3).

* **Vectorization is the big step** (1.7–1.9×): 32-bit → 128-bit loads, and it needs the alignment work — packed rows
  are K/8 int32 = 16-byte multiples for every shape here, `x` is 16-byte aligned (checked in the bindings: `CHECK_ALIGN16`).
* **Register blocking matters only where `x` traffic is a real fraction of the bytes**: for the body shapes the 4 KB
  activation stays cache-resident and R=1→2 is worth 10–14 %; for the LM head (32,000 rows) the best R=1 cell is
  170 µs vs 133 µs at R=2 (+28 %; the 1–4-warp R=1 cells, 216–406 µs, were partly measured throttled).
* **Occupancy vs ILP, resolved by sweep:** the autotuner (R ∈ {1,2,4,8} × warps/block ∈ {1,2,4,8} × prefetch on/off ×
  G ∈ {64,128,256}, 480 cells, `configs/autotune.json`, loaded by the runtime at startup) picks **R=2, 1 warp/block**
  for 14 of 15 shape×G cells (prefetch off in 12 of 15). Many small blocks (occupancy) beat deeper per-thread ILP.
* **Manual double-buffering** (register prefetch of the next 128-bit chunk — Turing has no `cp.async`) **won 0 of 12
  body cells** and the LM head only within noise (133.3 vs 133.9 µs). With enough warps resident, the memory system
  already has the requests it needs in flight.
* **Clock sensitivity (kbench3 control, same kernels at 585 MHz):** our INT4 GEMVs slow **1.69–1.79×**, cuBLAS fp16 only
  **1.10–1.36×**, while the pure read benchmark held 275 GB/s as the clock fell from 1550 to 1330 MHz in the clock probe.
  The INT4 GEMV is *not* purely bandwidth-bound: dequant is 4–5 ALU ops per 2 weights, and at low SM clock the kernel
  becomes issue-bound. On a throttled T4 this is the single biggest reason end-to-end lands below Model B.

![gemv](plots/gemv_stages.png) ![autotune](plots/autotune_heatmap.png)

## 5. Paged decode attention (split-K / flash-decoding)

KV cache `[layer, K/V, page, kv_head, 16 tokens, 64]` fp16, pages handed out by a **randomly permuted** block table (so
"paged" is actually exercised). One CTA per (kv head, split) handles all 8 query heads of the GQA group, online softmax
over 64-token chunks, unnormalized partials + (m, l) written per split, then a combine kernel. The context length is read
from a device-side `pos`, so one captured CUDA graph is valid at every step. Partition size per context comes from
`configs/attn_policy.json` (written by the sweep, loaded at startup).

| ctx | v1 best split | **v2 best split** | v2 no split | HF repeat_kv + SDPA (contiguous) | SDPA `enable_gqa` | v2 achieved BW |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 39.9 µs | **25.0** | 43.9 | 77.7 | 232.7 | 21 GB/s |
| 1K | 26.0 | **23.4** | 69.7 | 127.5 | 276.6 | 45 |
| 2K | 31.7 | **31.3** | 130.3 | 235.3 | 526.2 | 67 |
| 4K | 50.1 | **48.9** | 254.3 | 451.5 | 1037.8 | 86 |
| 8K | 81.9 | **78.4** | 503.5 | 1013.8 | 2336.8 | 107 |
| 16K | 147.0 | **141.4** | 1008.7 | 2782.1 | 4784.0 | 119 |

per layer (32 q heads, 4 kv heads, d=64), SM 1470–1590 MHz except ctx 512 (930 MHz, flagged).

* **Split-K crossover:** there is none in the "no split ever wins" sense — with only 4 KV heads, a no-split launch is
  4 CTAs on a 40-SM GPU, so splitting wins at every context ≥ 512 (1.8× at 512, 7.1× at 16K). The crossover that does
  exist is in *partition size*: the best partition grows 64 → 256 tokens from 512 → 4K (fewer, longer splits amortize
  the combine) and falls back to 128 at 8K–16K (where 64–128 splits are needed to fill the machine).
* **v2 vs v1** (K/V loads for all slots issued at the start of each chunk — manual prefetch in place of `cp.async`): 2×
  –2.7× faster without splitting, 1.6× at 512 with splitting, ≈ equal from 2K up where enough CTAs hide the latency.
* **vs PyTorch:** 3.1× (512) to 19.7× (16K) faster than HF's own path (repeat_kv + SDPA on a contiguous cache), while
  reading a paged cache.
* **Why it tops out at ~120 GB/s (43 % of achievable):** each chunk is load K → 8-lane dot products + shuffles →
  `__syncthreads` → online-softmax update → `__syncthreads` → V accumulate; three barriers per 64-token chunk serialize
  load and compute inside a CTA, and with 4 KV heads there are few CTAs per SM to overlap them. Turing has no `cp.async`
  to make a real multi-stage pipeline cheap, and `ncu` was blocked (below), so the stall reasons are inferred, not
  measured. This is the kernel to fix next: it is the entire long-context gap to llama.cpp.

![attention](plots/attention_sweep.png)

## 6. Fusion, CUDA graphs, group size, INT4 LM head

**Fusion ablation** (fused residual-add + RMSNorm, fused SiLU·mul vs the same runtime with HF-style torch ops):
225 vs 716 kernels per token; **−21 % time per token at 512, −11.5 % at 16K** (graph replay). The unfused path spends
1.2 ms in elementwise kernels and 0.2 ms in reductions per token (profile).

**CUDA graphs:** for the fused runtime (225 kernels) eager and graph are within noise (3.81 ms wall eager vs 4.02 ms
graph at 512 — the CPU issues 225 launches faster than the GPU drains them). For 716 kernels the CPU cannot keep up:
eager is **10.4 ms vs 5.1 ms** graph. Graphs matter in proportion to kernel count; fusion and graphs are complementary.

**Group size** — one configuration in every column: INT4 transformer body at group size G, **fp16 LM head** (the head
is 125 MiB and identical in every row, so it is left out of the byte and GEMV columns). GEMV speed is the per-token sum
from the autotuner grid at ~1500 MHz; e2e is the throttled decode loop (e2e4):

| G | body weight MiB / token | body GEMV µs / token | e2e ms / token @512 | perplexity |
|---:|---:|---:|---:|---:|
| 64 | 520 | 2,512 (+6 %) | 4.13 | **10.17** |
| 128 | 491 | 2,367 | 4.03 | 10.46 |
| 256 | 476 | 2,317 (−2 %) | 4.03 | 10.61 |

G64 closes 39 % of the INT4-vs-fp16 perplexity gap for +6 % GEMV time (≈ +3 % end-to-end); G256 saves 2 % and gives
back 0.15 perplexity. **If quality matters, G64 is the better operating point**; G128 stays the default only because it
is the brief's spec.

**INT4 LM head** — the data-driven call from the profile: the fp16 LM head is 131 MB, **20 % of all bytes per token
and 512 µs (15 %) of the step** at 512 context. Quantizing it (G128, our GEMV) gives **+12.6 % tok/s at 512, +6 % at
16K**, for +0.05 perplexity (10.46 → 10.51) and top-1 agreement 0.910 → 0.908. Worth it. (Embedding stays fp16: it is
one 4 KB row gather per token, 4 µs in the profile — not worth a kernel, per the brief's "only if profiling says so".)

![fusion](plots/fusion_ablation.png) ![groups](plots/group_size_tradeoff.png) ![profile](plots/profile_breakdown.png)

## 7. Roofline

Arithmetic intensity: W4A16 GEMV ≈ 3.8 FLOP/byte, fp16 GEMV ≈ 1, decode attention ≈ 8 (8 q heads share each KV byte).
The T4 ridge is **59 FLOP/byte against the `__half2` CUDA-core peak** (2,560 lanes × 2 × 2 FLOP × 1.59 GHz ≈ 16.3 TFLOP/s
over 277 GB/s) and ≈ 235 against the 65 TFLOP/s tensor-core peak — none of these kernels uses tensor cores, so 59 is
the relevant one. **Every decode kernel is 7–59× left of it** (attention 7×, W4A16 GEMV 15×, fp16 GEMV 59×): decode is
a bandwidth problem, and the only compute concern is the dequant issue rate at
throttled clocks (§4). The achieved points sit at 62–95 % of measured bandwidth for GEMV and 21–43 % for attention.

![roofline](plots/roofline.png) ![bandwidth](plots/bandwidth_and_clocks.png)

## 8. Platform notes and limitations (stated, not hidden)

* **Turing (sm_75):** no `cp.async` → manual register double-buffering (GEMV prefetch, attention v2 slot prefetch);
  no bf16 → fp16 everywhere with `__half2` math and fp32 accumulation across chunks; no `ldmatrix`-based W4 tensor-core
  path was attempted.
* **Prefill** is not a separate kernel: prompt tokens are teacher-forced through the decode step (one graph replay per
  token). Correct and simple; slow for long prompts. The WMMA prefill GEMM stretch goal was **cut** for time, as the
  brief allows.
* **`ncu` blocked** (`ERR_NVGPUCTRPERM`: Kaggle does not expose performance counters); **`nsys` not available**.
  `torch.profiler` (CUPTI activity tracing, no counters) was used for per-kernel timing, idle gaps and kernel counts.
* **Clock control:** not permitted; see the caveat at the top. The timing harness samples NVML clocks *while the GPU is
  busy*, retries after cool-downs, and flags every throttled measurement (the first version sampled after
  `synchronize()` and read the 585 MHz idle clock — NOTEBOOK).
* **Quantization** is plain RTN; GPTQ/AWQ would cut most of the perplexity gap without changing a kernel.
* **llama.cpp** was measured on a different Kaggle node (`llama-bench` doesn't record clocks), with its own quantizer.

## 9. Reproduce

`README.md`: one command (`scripts/kaggle_run.py --cmd "bash scripts/run_all.sh"`), then `scripts/plot_all.py` over the
run's `results.json`. All numbers above are in `results.json`; all figures regenerate from it.
