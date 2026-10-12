#!/usr/bin/env bash
# Same-node head-to-head vs llama.cpp (bench/h2h.py). Accuracy gate first, then llama.cpp at the commit used for the
# original reference (llamacpp_bench.sh), then both decoders alternating on this GPU with clocks recorded.
# Run on Kaggle via: scripts/kaggle_run.py --slug llm-decode-h2h --tag h2h --cmd "bash scripts/h2h.sh"
set -uo pipefail
OUT="${RESULTS_DIR:-.}"
python tests/test_kernels.py > "$OUT/tests.log" 2>&1
rc=$?
tail -1 "$OUT/tests.log"
if [ $rc -ne 0 ]; then echo "ACCURACY GATE FAILED"; grep FAIL "$OUT/tests.log" | head -40; exit 1; fi
set -e
REPO=$PWD
COMMIT=23b0202a189c44a54625aadcb37a946dd1d6278d
mkdir -p /tmp/llama.cpp && cd /tmp/llama.cpp
git init -q && git remote add origin https://github.com/ggml-org/llama.cpp && git fetch -q --depth 1 origin $COMMIT
git checkout -q FETCH_HEAD && git rev-parse HEAD | tee "$OUT/llamacpp_commit.txt"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_VMM=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs > /dev/null
cmake --build build -j4 --target llama-bench 2>&1 | tail -3
python - <<'EOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", local_dir="/tmp"))
EOF
cd "$REPO"
python bench/h2h.py --llama-bench /tmp/llama.cpp/build/bin/llama-bench --gguf /tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
