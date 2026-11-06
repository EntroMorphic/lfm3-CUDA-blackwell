#!/usr/bin/env bash
# Build the native targets exactly as production was configured (CMake from native/ mounted at
# /src/native, LIQUID_LFM3_OPS_TESTS=ON, no build type). Run as root in the pinned compiler image:
#   docker run --rm -v $W/liquid-CUDA/native:/src/native:ro -v $W:/workspace IMAGE bash build_native.sh BUILD_DIR
set -euo pipefail
out=$1
cmake -S /src/native -B "$out" -DLIQUID_LFM3_OPS_TESTS=ON > "$out.cmake.log" 2>&1 || { tail -20 "$out.cmake.log"; exit 1; }
cmake --build "$out" --target liquid-lfm3 liquid-lfm3-worker text-sampling-oracle lfm3-decode-batch-probe -j4 > "$out.build.log" 2>&1 \
  || { grep -iE " error|error:" "$out.build.log" | head -10; exit 1; }
cd "$out" && sha256sum liquid-lfm3 liquid-lfm3-worker text-sampling-oracle lfm3-decode-batch-probe
