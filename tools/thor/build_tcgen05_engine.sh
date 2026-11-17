#!/usr/bin/env bash
# validation/tcgen05-gemm-v1: build the engine with lfm::gemm's tcgen05 path ON (or OFF, the same sources without it),
# in the pinned compiler image, from a source tree under the workspace (default: build/src-tcgen05):
#   lfm3-speech and lfm3-vision-check as build_speech.sh builds them (nvcc, -O3 --fmad=false, sm_110), the server,
#   worker and liquid-lfm3 by CMake (LIQUID_LFM3_NATIVE_GEMM=ON, LIQUID_LFM3_TCGEN05=ON|OFF), and with ON the rule-1
#   check binary. The tcgen05 unit (src/gemm_tcgen05.cu) alone is compiled for sm_110a.
# Usage (Thor host): build_tcgen05_engine.sh OUT_TAG on|off [SRC_REL]   -> build/OUT_TAG/
set -euo pipefail
tag=$1; mode=$2; src=${3:-build/src-tcgen05}
[[ $mode == on || $mode == off ]] || { echo "mode: on|off"; exit 2; }
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
C=/workspace/liquid-CUDA/build/$tag; S=/workspace/liquid-CUDA/$src
test -f $W/liquid-CUDA/build/images-v1/decode-aarch64/ljt/lib/libjpeg.a || { echo "no libjpeg-turbo build"; exit 1; }
mkdir -p $W/liquid-CUDA/build/$tag
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C; N=$S/native
  L=/workspace/liquid-CUDA/build/images-v1/decode-aarch64/ljt
  F='-O3 --fmad=false -std=c++20 -arch=sm_110 -I '\$N/include
  (cd $S && find native tools -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.sh' -o -name 'CMakeLists.txt' \) | sort | xargs sha256sum) > sources.sha256
  nvcc --version | tail -2 > nvcc.txt
  T=''; O=''
  if [ $mode = on ]; then
    nvcc -O3 -std=c++20 -gencode arch=compute_110a,code=sm_110a -DLFM_GEMM_TCGEN05 -I \$N/include -Xptxas -v \
      -c \$N/src/gemm_tcgen05.cu -o gemm_tcgen05.o > build-gemm-tcgen05.log 2>&1
    T='-DLFM_GEMM_TCGEN05'; O=gemm_tcgen05.o
  fi
  g++ -std=c++20 -O2 -Wall -Wextra -I \$N/include -I \$L/include -c \$N/src/image_decode.cpp -o image_decode.o > build-image.log 2>&1
  g++ -std=c++20 -O2 -Wall -Wextra -I \$N/include -c \$N/src/image_preprocess.cpp -o image_preprocess.o >> build-image.log 2>&1
  nvcc \$F \$T -DLFM3_NATIVE_GEMM -o lfm3-speech \$N/apps/lfm3_speech.cu \$N/src/safetensors.cpp image_decode.o image_preprocess.o \$L/lib/libjpeg.a \$O > build-lfm3-speech.log 2>&1
  nvcc \$F \$T -o lfm3-vision-check \$N/tests/lfm3_vision_check.cu \$N/src/safetensors.cpp \$O > build-lfm3-vision-check.log 2>&1
  nvcc \$F \$T -DLFM3_NATIVE_GEMM -o lfm3-batch \$N/apps/lfm3_batch.cu \$N/src/safetensors.cpp \$O > build-lfm3-batch.log 2>&1
  if [ $mode = on ]; then
    nvcc \$F \$T -o lfm3-tcgen05-gemm-check \$N/tests/lfm3_tcgen05_gemm_check.cu \$O > build-check.log 2>&1
  fi
  TC=OFF; [ $mode = on ] && TC=ON
  cmake -S \$N -B cmake -DLIQUID_LFM3_OPS_TESTS=ON -DLIQUID_LFM3_NATIVE_GEMM=ON -DLIQUID_LFM3_TCGEN05=\$TC > cmake.log 2>&1
  cmake --build cmake --target liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server -j4 > cmake-build.log 2>&1
  cp cmake/liquid-lfm3 cmake/liquid-lfm3-worker cmake/liquid-lfm3-server .
  ! grep -iE ' error|warning' build-*.log
  chown -R 1000:1000 ."
cd $W/liquid-CUDA/build/$tag
for b in lfm3-speech lfm3-batch liquid-lfm3-server liquid-lfm3-worker liquid-lfm3; do readelf -d $b | grep -q cublas && { echo "$b links cuBLAS"; exit 1; }; done
sha256sum lfm3-speech lfm3-vision-check lfm3-batch liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server $( [ $mode = on ] && echo lfm3-tcgen05-gemm-check ) | tee binaries.sha256
