#!/usr/bin/env bash
# Build audio-encoder-bench (validation/audio-speed-v1 rule 0) in the pinned compiler image, with the engine's flags.
# Usage (Thor host): build_audio_bench.sh OUT_TAG [SRC]   SRC: the source tree, relative to the workspace's
#   liquid-CUDA (default: liquid-CUDA itself; or a worktree copy such as build/src-audio).
#   -> build/OUT_TAG/{audio-encoder-bench, sources.sha256, build-audio-bench.log}
set -euo pipefail
tag=$1; src=${2:-.}; W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
C=/workspace/liquid-CUDA/build/$tag; S=/workspace/liquid-CUDA/$src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=$S/native
  (cd $S && sha256sum native/tests/audio_encoder_bench.cu native/src/lfm3_main.cu native/src/safetensors.cpp native/include/lfm/audio_*.hpp native/include/lfm/audio_encoder.cuh native/include/lfm/fp32_ops.cuh native/include/lfm/wav.hpp) > sources.sha256
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I \$N/include -DLFM3_NATIVE_GEMM -o audio-encoder-bench \$N/tests/audio_encoder_bench.cu \$N/src/safetensors.cpp > build-audio-bench.log 2>&1
  ! grep -iE ' error|warning' build-audio-bench.log; chown -R 1000:1000 ."
readelf -d $W/liquid-CUDA/build/$tag/audio-encoder-bench | grep -q cublas && { echo "audio-encoder-bench links cuBLAS"; exit 1; }
(cd $W/liquid-CUDA/build/$tag && sha256sum audio-encoder-bench)
