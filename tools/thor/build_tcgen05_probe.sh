#!/usr/bin/env bash
# Build the tcgen05 GEMM probe (validation/tcgen05-probe-v1) inside the pinned compiler image. Usage (in container):
#   build_tcgen05_probe.sh SRC_ROOT OUTDIR
# The probe's translation unit is sm_110a (tcgen05 needs the arch-specific target); lfm::gemm is compiled apart with
# the engine's own -arch=sm_110 so the comparison runs the engine's kernels as the engine builds them.
set -euo pipefail
src=$1; out=$2
mkdir -p "$out"; cd "$out"
P=$src/native/probes
sha256sum "$P/tcgen05_gemm.cuh" "$P/tcgen05_gemm_probe.cu" "$P/tcgen05_probe_engine_gemm.cu" \
  "$src/native/include/lfm/gemm.cuh" > sources.sha256
nvcc --version | tail -2 > nvcc.txt
nvcc -O3 -std=c++20 -gencode arch=compute_110a,code=sm_110a -I"$src/native/include" -I"$P" -Xptxas -v \
  -c "$P/tcgen05_gemm_probe.cu" -o probe.o > probe.build.log 2>&1 || { grep -iE "error|warning" probe.build.log | head -20; exit 1; }
nvcc -O3 -std=c++20 -arch=sm_110 -I"$src/native/include" -Xptxas -v \
  -c "$P/tcgen05_probe_engine_gemm.cu" -o engine_gemm.o > engine_gemm.build.log 2>&1 || { grep -iE "error|warning" engine_gemm.build.log | head -20; exit 1; }
nvcc -gencode arch=compute_110a,code=sm_110a probe.o engine_gemm.o -o tcgen05-gemm-probe \
  -L/usr/local/cuda/lib64/stubs -lcuda -lcublas > link.log 2>&1 || { cat link.log; exit 1; }
grep -iE "warning" probe.build.log engine_gemm.build.log || echo "warnings: 0"
grep -E "spill" probe.build.log engine_gemm.build.log | grep -v " 0 bytes spill stores, 0 bytes spill loads" || echo "spills: 0"
cuobjdump -sass tcgen05-gemm-probe > probe.sass
echo "SASS: $(grep -c UTCHMMA probe.sass) UTCHMMA, $(grep -c UTMALDG probe.sass) UTMALDG, $(grep -c LDTM probe.sass) LDTM"
sha256sum tcgen05-gemm-probe | tee binary.sha256
