#!/usr/bin/env bash
# validation/audio-ttft-v1, a diagnostic: alloc-effect-probe (native/tests/alloc_effect_probe.cu) on a quiet Thor, in
# the container as the engine runs, and on the host. The always-on service stopped for the window.
# Usage (Thor host, copied to build/alloc-probe/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=alloc-probe; S=ttft-src-a6
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]lloc-effect"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include \
  -DLFM3_NATIVE_GEMM -o alloc-effect-probe $C/$S/native/tests/alloc_effect_probe.cu $C/$S/native/src/safetensors.cpp > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build exit $?"; sha256sum alloc-effect-probe
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
echo "== in the container $(date +%T)"; $X $C/$O/alloc-effect-probe 5
echo "== on the host $(date +%T)"; ./alloc-effect-probe 5
echo "ALLOC-PROBE-FINISHED $(date +%T)"
