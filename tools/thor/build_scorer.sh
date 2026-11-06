#!/usr/bin/env bash
# Build the native perplexity scorer in the pinned compiler image (same flags as liquid-lfm3).
# Usage (in container): build_scorer.sh OUTDIR
set -euo pipefail
out=$1; N=/workspace/liquid-CUDA/native
cd "$out"
sha256sum $N/tests/lfm3_score.cu $N/src/lfm3_main.cu $N/src/safetensors.cpp > sources.sha256
nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $N/include -o lfm3-score $N/tests/lfm3_score.cu $N/src/safetensors.cpp -lcublas > build.log 2>&1 || { grep -iE "error" build.log | head -10; exit 1; }
echo "lfm3-score built sha256 $(sha256sum lfm3-score | cut -c1-16)"
