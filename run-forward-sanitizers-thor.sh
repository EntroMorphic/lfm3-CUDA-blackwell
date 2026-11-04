#!/usr/bin/env bash
# Integrated native path; run in the Thor copy after check_lfm3_forward.py passes.
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(dirname "$PWD")"
GATES="${GATES:-build/forward-gates-v2}"
test -f "$GATES/summary.json"
for tool in memcheck racecheck synccheck initcheck; do
  extra=()
  if [[ "$tool" == synccheck ]]; then
    # Known NVJET instrumentation failure, not a custom-kernel exclusion.
    extra+=(--kernel-name-exclude kns=nvjet)
  fi
  docker run --rm --runtime=nvidia --gpus all \
    --mount "type=bind,source=$PROJECT,target=/workspace" \
    --mount "type=bind,source=$PWD/build/cuda-13.0.88-sanitizer,target=/opt/sanitizer,readonly" \
    -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib \
    lfm3-omni-minimal:dev /opt/sanitizer/compute-sanitizer \
    --tool "$tool" --error-exitcode 86 "${extra[@]}" \
    --log-file "/workspace/liquid-CUDA/build/forward-$tool.log" \
    /workspace/liquid-CUDA/build/cmake/liquid-lfm3 --experimental \
    /workspace/models/LFM3-Omni-2B \
    "/workspace/liquid-CUDA/$GATES/window257/generation-request.json" \
    "/workspace/liquid-CUDA/build/forward-$tool.json" \
    > "build/forward-$tool-stdout.log" 2>&1
  echo "$tool: PASS"
done
