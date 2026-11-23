#!/usr/bin/env bash
# Rules C1-C5 of validation/tcgen05-probe-v1 for one variant, on the Thor host (runs the probe in lfm3-profile-v1).
#   tcgen05_probe_correct.sh BUILD_TAG VARIANT [probe options...]
# Writes build/tcgen05-BUILD_TAG/correct-vVARIANT/{correct.jsonl,progress.log,rc}.
set -uo pipefail
tag=$1; variant=$2; shift 2
W=~/Projects/000-research/lfm3-omni-2B
out=$W/liquid-CUDA/build/tcgen05-$tag/correct-v$variant
mkdir -p "$out"
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 \
  /workspace/liquid-CUDA/build/tcgen05-$tag/tcgen05-gemm-probe correct "$variant" "$@" \
  > "$out/correct.jsonl" 2> "$out/progress.log"
echo $? > "$out/rc"
