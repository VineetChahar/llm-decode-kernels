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
- **L2 flush (16 MB read; originally a memset — see measurement bug #2) before every timed GEMV/attention iteration**: T4 L2 is 4 MB and a 2.6 MB QKV weight would otherwise be L2-resident across iterations, inflating bandwidth. In the real model consecutive layers never reuse weights.
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

## full1 post-mortem — the clock gate measured the idle clock (2026-10-11)
- Accuracy gate: **439/439 passed**, 0 FAIL (tests.log in runs/20261011T102747Z_full1).
- The run then hit Kaggle's 6 h limit inside the autotuner (~24 min per shape x G instead of ~1 min).
  Cause: `time_fn` read NVML clocks *after* `torch.cuda.synchronize()`; by then the T4 has already dropped to its
  585 MHz idle clock, so every measurement looked throttled and paid 3 x 15 s of cool-down. The clock probe under
  continuous load showed 1590 -> 1440 MHz over 60 s, i.e. the GPU was never actually at 585 MHz while busy.
- Fix: queue a 20 ms `torch.cuda._sleep` spin behind the timed loop and sample NVML while it runs (the loaded clock);
  same fix in e2e `time_decode`. Retries 3 -> 2, cool-down 15 -> 10 s, plus a 900 s per-process cool-down budget so
  a bad gate can never again eat the run.
- Decision: split the rerun across two concurrent sessions (`PART=kernels` / `PART=e2e`, each re-runs the accuracy
  gate). The e2e session can't wait for the new autotune, so `configs/autotune.json` is seeded from full1's printed
  winners (R=4, pf=0, warps 1-2 for every body shape; lm_head from kbench1). kbench2 re-tunes; any difference is
  reported in RESULTS.md.
- Observation worth keeping: prefetch (pf=1) never won a single autotune cell, and R=4 won every one in full1.

## kbench2 / kbench3 — which numbers are trusted (2026-10-11)
- kbench2 (runs/20261011T170804Z_kbench2): gate 439/439. Fixed clock gate works; autotune grid 434/480 cells >= 1400 MHz.
  The 900 s cool-down budget was spent inside the 720-point grid, so the stage ladder's lm_head rows (5/25) and the
  group sweep ran throttled; 12/78 attention points throttled. All flagged.
- kbench3 (runs/20261011T191452Z_kbench3), rerun of gemv/groups/attn: gate 439/439, but this instance sat at **585 MHz for
  the entire run at 61 C with no NVML throttle reason** (a capped node, not thermal). Every timing ~1.7-2x slower; 100% flagged.
  Not used for headline numbers. Kept as a natural experiment: same kernels, SM clock 585 vs ~1500 MHz.
- Decision: no more kernel reruns (each costs a contended GPU slot, and the node lottery can repeat). Trusted sources:
  GEMV ladder = kbench2 (lm_head tuned time from the autotune grid at ~1500 MHz); group-size GEMV speed = per-token sums
  from the autotune grid; attention = kbench2 unthrottled points; bandwidth/clock probe = kbench2.
- Clock-sensitivity finding: at 585 MHz the W4A16 GEMVs slow 1.6-1.9x while the 277 GB/s read microbenchmark held its
  bandwidth down to ~1330 MHz in the probe — the dequant ALU work makes the INT4 GEMV issue-bound at low SM clock.

## e2e4 + write-up (2026-10-11/12)
- e2e3 crashed at `load_dataset("wikitext")` — new huggingface_hub rejects un-namespaced ids -> `Salesforce/wikitext`.
  Made e2e sections independent (one failing section no longer costs the rest).
- e2e4 (runs/20261011T193218Z_e2e4): gate 439/439; all four sections ran. Quality is clock-independent and clean.
  Speed rows all ran at 900-1065 MHz with NVML `hw_slowdown` (fp16 config 480-885 MHz), even after gate retries.
- Decision: report e2e speed as measured with the clock next to it, no more reruns. Reason: kernel microbenchmarks with
  idle gaps reach ~1500 MHz but sustained decode loops on Kaggle T4s hit hw_slowdown; another rerun is a node lottery
  that also competes with two other projects for the account's two GPU slots.
- Model B (not fitted) explains 92 % at 512 -> 63 % at 16K. Profile attributes 79 % of the 16K miss to attention running
  at 102 GB/s in-graph; the rest is profiled-vs-timed clock difference. At 512 Model B over-predicts the profiled step
  (t0 from isolated kernels includes overhead graph replay hides) and under-predicts the throttled timed loop.
- HF fp16 OOMs at 8K/16K (eager prefill). Recorded as OOM, not dropped.
- INT4 LM head: +12.6 % tok/s at 512 for +0.05 ppl -> recommended. G64: closes 39 % of the ppl gap for +6 % GEMV time.
- WMMA prefill: cut (budget), as the brief allows. llama.cpp kept (already measured).

## Fix pass: attention v3, review fixes, llama.cpp perplexity (2026-10-12)
- llama.cpp perplexity (runs/20261011T205851Z_llamappl, `scripts/llamacpp_ppl.sh`): Q8_0 19.975, Q4_K_M 20.292 ->
  **+1.6 %** from 8-bit to Q4_K_M, vs our RTN G128 **+7.6 %** from fp16. Absolute values are not comparable to ours
  (llama-perplexity -c 512 scores the second half of each chunk, no BOS); the relative cost is. Their quantizer is better.
- Attention v3 (`paged_attn_partial_v3<HPG>`): v2 had three `__syncthreads` per 64-token chunk shared by the whole CTA.
  v3 gives each warp its own 16-token chunks with its own online-softmax state (no barrier in the main loop), prefetches
  the next chunk's K/V into registers while computing the current one, and merges the warps' (m, l, acc) once at the end.
  Decision: keep v1/v2 in the binary and let the sweep pick by geometric-mean best-split time (version stored in
  attn_policy.json). Reason: the choice is then measured, not asserted, and the old kernels stay as baselines.
- final run (runs/20261011T212701Z_final): gate 538/538 (v3 in every attention case). The node again sat at 585-720 MHz
  for most of the attention sweep (88/117 points flagged), so the comparison is same-node, same-clock relative:
  v3 vs v2 best split = 1.14x (512), 1.31x, 1.40x, 1.58x, 2.28x (8K), 1.61x (16K, both at 1590 MHz: 84 vs 135 us).
  Sweep selected v3; configs/attn_policy.json updated (partition 64 -> 1024 tokens from 512 -> 16K: v3 has 4 warps per
  CTA working independently, so it wants longer splits than v2). GEMV body ladder reproduced kbench2 within 1 %.
- Review findings fixed:
  * Policy lookup bug: the runtime picked the first policy entry with max_ctx >= ctx_cap; ctx_cap has headroom
    (e.g. 1104 for ctx 1024), so every e2e point used the next bucket's split size. Now: entry tuned at the largest
    context <= cap. e2e4's speed numbers used the wrong split sizes (one bucket up) — superseded by e2efinal.
  * Runtime defaults now match the recommendation: lm_head="auto" (INT4 when weights are INT4), attention version from
    the policy file. e2e configs pin lm_head explicitly so each row means one thing.
  * Context-cap guard: run_step raises if the host-tracked position exceeds max_splits * part_size (silent wrong
    attention before). Bindings reject part_size <= 0 and unknown versions.
  * Tests added for the runtime's actual launch conditions: spare split buffers (more splits allocated than used),
    PS=128 at real contexts, HPG=2.
  * e2e speed points start with cool() (idle until < 60 C, bounded) — e2e4 ran in hw_slowdown at 900-1065 MHz.
  * LM-head rows of the GEMV ladder were throttled in both kbench2 and final because they run last, after the
    cool-down budget is spent: added `kernel_bench.py gemv_lm` (LM head only, cool-down before each row), run as its
    own job; plot_all substitutes its rows.
  * RESULTS corrections: roofline ridge (59 is the half2 ridge, not tensor-core), Model B residual arithmetic,
    attention bytes at 512 (11.5 MB), vectorization range, test context range (1-16K), group-size table mixed configs,
    graph-vs-eager compared wall to wall, removed an unverified "CPU test shows the same flips" claim.
- lmhead job (runs/20261011T225838Z_lmhead): gate 667/667 (full new suite incl. runtime-condition cases). The
  LM-head-only ladder with cool() before each row still sampled 585-825 MHz at 40-44 C (no NVML reason). Third node in
  a row where only the long LM-head loops throttle. Decision: stop re-running it; report as a platform finding, keep the
  autotune-grid 133 us for stage 3 and mark the LM-head speedup vs cuBLAS as not comparable.
