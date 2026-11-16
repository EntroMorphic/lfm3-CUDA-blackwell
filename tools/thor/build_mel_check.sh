#!/usr/bin/env bash
# Build audio-mel-check (validation/audio-frontend-v1 rule 1) in the pinned compiler image, compiled as the engine
# compiles log_mel: nvcc's host pass of a .cu at -O3 (lfm3-speech, the worker and the server include the header from
# .cu translation units).
# Usage (Thor host): LFM3_WORKSPACE=... build_mel_check.sh OUT_TAG [SRC]   SRC: the source tree relative to the
#   workspace's liquid-CUDA (default: liquid-CUDA itself).  -> build/OUT_TAG/{audio-mel-check, sources.sha256, build.log}
set -euo pipefail
tag=$1; src=${2:-.}; W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
C=/workspace/liquid-CUDA/build/$tag; S=/workspace/liquid-CUDA/$src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=$S/native
  (cd $S && sha256sum native/tests/audio_mel_check.cpp native/include/lfm/audio_features.hpp native/include/lfm/wav.hpp) > sources.sha256
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I \$N/include -x cu -o audio-mel-check \$N/tests/audio_mel_check.cpp > build.log 2>&1
  ! grep -iE ' error|warning' build.log; chown -R 1000:1000 ."
(cd $W/liquid-CUDA/build/$tag && sha256sum audio-mel-check)
