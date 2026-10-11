#!/usr/bin/env bash
# External sanity reference: llama.cpp (CUDA, sm_75) on TinyLlama-1.1B-Chat Q4_K_M, decode tok/s vs context depth.
# Run on Kaggle via: scripts/kaggle_run.py --slug llm-decode-llamacpp --cmd "bash scripts/llamacpp_bench.sh"
set -euo pipefail
OUT="${RESULTS_DIR:-.}"
cd /tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git rev-parse HEAD | tee "$OUT/llamacpp_commit.txt"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_VMM=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs > /dev/null
cmake --build build -j4 --target llama-bench 2>&1 | tail -3
python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", local_dir="/tmp")
print(p)
EOF
export CUDA_VISIBLE_DEVICES=0
./build/bin/llama-bench -m /tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -ngl 99 -p 0 -n 128 \
  -d 512,1024,2048,4096,8192,16384 -fa 0,1 -r 5 -o json | tee "$OUT/llamacpp.json"
ls -la /tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | tee -a "$OUT/llamacpp_commit.txt"
