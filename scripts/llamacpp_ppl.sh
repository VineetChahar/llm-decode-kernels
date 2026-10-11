#!/usr/bin/env bash
# llama.cpp quality reference: WikiText-2 test perplexity (llama-perplexity, ctx 512) of TinyLlama Q4_K_M vs Q8_0.
# No fp16 GGUF is published for this model, so Q8_0 is the near-lossless baseline; compare *relative* deltas
# (Q4_K_M vs Q8_0 here, our INT4 vs fp16 in RESULTS.md), since the chunking protocol differs from bench/e2e.py.
# Run on Kaggle via: scripts/kaggle_run.py --slug llm-decode-llamacpp --tag llamappl --cmd "bash scripts/llamacpp_ppl.sh"
set -euo pipefail
OUT="${RESULTS_DIR:-.}"
cd /tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git rev-parse HEAD | tee "$OUT/llamacpp_commit.txt"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_VMM=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs > /dev/null
cmake --build build -j4 --target llama-perplexity 2>&1 | tail -3
python - <<'EOF'
from huggingface_hub import hf_hub_download
from datasets import load_dataset
for q in ("Q4_K_M", "Q8_0"):
    print(hf_hub_download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", f"tinyllama-1.1b-chat-v1.0.{q}.gguf", local_dir="/tmp"))
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open("/tmp/wiki.test.raw", "w").write("".join(ds["text"]))
EOF
export CUDA_VISIBLE_DEVICES=0
for q in Q8_0 Q4_K_M; do
  ./build/bin/llama-perplexity -m /tmp/tinyllama-1.1b-chat-v1.0.$q.gguf -f /tmp/wiki.test.raw -c 512 -ngl 99 \
    2>&1 | tee "$OUT/ppl_$q.log" | grep -E "Final estimate|error" || true
done
python - "$OUT" <<'EOF'
import json, re, sys, os
out = {}
for q in ("Q8_0", "Q4_K_M"):
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", open(os.path.join(sys.argv[1], f"ppl_{q}.log")).read())
    out[q] = {"ppl": float(m.group(1)), "stderr": float(m.group(2))} if m else None
out["rel_delta_pct_Q4KM_vs_Q8"] = 100 * (out["Q4_K_M"]["ppl"] / out["Q8_0"]["ppl"] - 1)
out["protocol"] = "llama-perplexity -c 512, full WikiText-2 test (\"\".join of HF raw lines)"
json.dump(out, open(os.path.join(sys.argv[1], "llamacpp_ppl.json"), "w"), indent=1)
print(out)
EOF
