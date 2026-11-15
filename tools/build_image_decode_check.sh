#!/usr/bin/env bash
# Build the vendored libjpeg-turbo 3.2.0 (static, Release, its default SIMD where the platform has it: NEON on
# aarch64, NASM on x86 if installed) and lfm3-image-decode-check against it. Host only, no network.
# Usage: build_image_decode_check.sh OUT_DIR [extra compiler flags, e.g. -fsanitize=address,undefined]
set -euo pipefail
OUT=$(realpath -m "${1:?OUT_DIR}"); shift
N=$(cd "$(dirname "$0")/../native" && pwd)
mkdir -p "$OUT"
if [ ! -f "$OUT/ljt/lib/libjpeg.a" ]; then
  rm -rf "$OUT/ljt-src" "$OUT/ljt-build"
  mkdir -p "$OUT/ljt-src"
  echo "6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e  $N/third_party/libjpeg-turbo/libjpeg-turbo-3.2.0.tar.gz" | sha256sum -c --quiet
  tar xzf "$N/third_party/libjpeg-turbo/libjpeg-turbo-3.2.0.tar.gz" -C "$OUT/ljt-src" --strip-components=1
  cmake -S "$OUT/ljt-src" -B "$OUT/ljt-build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OUT/ljt" \
    -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON > "$OUT/ljt-cmake.log" 2>&1
  cmake --build "$OUT/ljt-build" -j"$(nproc)" > "$OUT/ljt-build.log" 2>&1
  cmake --install "$OUT/ljt-build" > /dev/null
  grep -i "simd" "$OUT/ljt-cmake.log" | head -3 || true
fi
g++ -std=c++20 -O2 -g -Wall -Wextra "$@" -I "$N/include" -I "$OUT/ljt/include" \
  "$N/tests/lfm3_image_decode_check.cpp" "$N/src/image_decode.cpp" "$N/src/image_preprocess.cpp" \
  "$OUT/ljt/lib/libjpeg.a" \
  -o "$OUT/lfm3-image-decode-check"
"$OUT/lfm3-image-decode-check" --selftest
