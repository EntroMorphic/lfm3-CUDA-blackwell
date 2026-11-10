#!/usr/bin/env bash
# Build lfm3-speech (#2 step 4) in the pinned compiler image, with the engine's flags (native GEMM, no cuBLAS).
# Usage (Thor host): build_speech.sh OUT_TAG   -> build/OUT_TAG/{lfm3-speech, sources.sha256, build-lfm3-speech.log}
set -euo pipefail
tag=$1; W=/home/ztflynn/Projects/000-research/lfm3-omni-2B; C=/workspace/liquid-CUDA/build/$tag
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=/workspace/liquid-CUDA/native
  F='-O3 --fmad=false -std=c++20 -arch=sm_110 -I '\$N/include
  (cd /workspace/liquid-CUDA && sha256sum native/src/lfm3_speech.cuh native/src/lfm3_batch.cuh native/src/lfm3_main.cu native/tests/lfm3_speech.cu native/include/lfm/audio_*.hpp native/include/lfm/speech_text.hpp native/include/lfm/speech_run.hpp native/include/lfm/lfm3_text.hpp native/include/lfm/codec.cuh native/include/lfm/depth.cuh) > sources.sha256
  nvcc \$F -DLFM3_NATIVE_GEMM -o lfm3-speech \$N/tests/lfm3_speech.cu \$N/src/safetensors.cpp > build-lfm3-speech.log 2>&1
  ! grep -iE ' error|warning' build-*.log; chown -R 1000:1000 ."
readelf -d $W/liquid-CUDA/build/$tag/lfm3-speech | grep -q cublas && { echo "lfm3-speech links cuBLAS"; exit 1; }
(cd $W/liquid-CUDA/build/$tag && sha256sum lfm3-speech)
