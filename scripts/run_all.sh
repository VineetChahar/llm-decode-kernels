#!/usr/bin/env bash
# One-command pipeline (what the Kaggle job runs). Accuracy gate first: any failing numerical test aborts
# before a single speed number is produced.
set -uo pipefail
OUT="${RESULTS_DIR:-.}"
python tests/test_kernels.py > "$OUT/tests.log" 2>&1
rc=$?
tail -1 "$OUT/tests.log"
if [ $rc -ne 0 ]; then echo "ACCURACY GATE FAILED"; grep FAIL "$OUT/tests.log" | head -40; exit 1; fi
set -e
python bench/kernel_bench.py ${KERNEL_SECTIONS:-}
python bench/e2e.py
