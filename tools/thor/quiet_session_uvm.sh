#!/usr/bin/env bash
# prefill-stall-v1's UVM test (2026-11-17; rules in the README, set before the data). Creates the throwaway container
# lfm3-uvm-test-v1 as a copy of lfm3-profile-v1's configuration (lfm3-profile-v1 is not touched), checks that
# /dev/nvidia-uvm opens in it, then window 8's probe: one control run in lfm3-profile-v1, two in the test container,
# each with the engine's maps checked for /dev/nvidia-uvm 60 s in. Timing: quiet Thor. The test container is left
# stopped, not removed, for the record (docker rm it once the README has the result).
# Usage (Thor host): quiet_session_uvm.sh   (results: build/quiet-uvm/)
set -uo pipefail
export LFM3_WORKSPACE=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}
W=$LFM3_WORKSPACE; R=$W/liquid-CUDA; B=$R/build; O=quiet-uvm; TC=lfm3-uvm-test-v1
IMG=sha256:2a9b0054deac9e70b38e31bfdc0b789a4918af50a07e9832263c47bcbc5b08da
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
if docker inspect $TC > /dev/null 2>&1; then echo "FAIL: $TC exists"; echo "QUIET-UVM-FINISHED $(date +%T)"; exit 1; fi
echo "== create $TC $(date +%T)"
docker run -d --name $TC --runtime nvidia --gpus all --network none --memory 103079215104 --workdir /workspace \
  -v $B/racecheck-pinned-libs-v1:/pinned-cuda:ro -v $W:/workspace -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nsys:ro \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia \
  -e PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -e DEBIAN_FRONTEND=noninteractive \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e HF_HOME=/workspace/cache/huggingface -e PYTHONUNBUFFERED=1 \
  $IMG sleep infinity > create.log 2>&1; echo "create exit $?"
for k in lfm3-profile-v1 $TC; do
  docker inspect $k --format "$k: image {{.Image}} runtime {{.HostConfig.Runtime}} memory {{.HostConfig.Memory}} network {{.HostConfig.NetworkMode}} mounts {{len .Mounts}}"
  docker exec $k python3 -c "
import os
for d in ('/dev/nvidia-uvm','/dev/nvidia0','/dev/nvidiactl'):
    try: os.close(os.open(d, os.O_RDWR)); print('  $k', d, 'opens')
    except OSError as e: print('  $k', d, 'FAIL', e)" 2>&1
done | tee devices.txt
if ! grep -q "$TC /dev/nvidia-uvm opens" devices.txt; then
  echo "VOID: /dev/nvidia-uvm does not open in $TC"; docker stop -t 10 $TC > /dev/null; echo "QUIET-UVM-FINISHED $(date +%T)"; exit 1
fi
probe() {  # container, tag
  ( sleep 60; p=$(pgrep -f "^/[^ ]*/lfm3-speech .*/$2/list.txt" | head -1)
    if [ -n "$p" ]; then echo "uvm mappings: $(grep -c nvidia-uvm /proc/$p/maps)"; else echo "uvm mappings: no engine at 60 s"; fi ) > maps-$2.txt &
  STALL_USER=1000 STALL_CONTAINER=$1 bash $R/tools/thor/stall_probe.sh vimg-a5 $2 > probe-$2.log 2>&1
  wait
  echo "$2 ($1): $(grep -E "requests" probe-$2.log | head -1); stalls $(grep -c "^STALL" probe-$2.log); $(cat maps-$2.txt)"
}
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
echo "== control: lfm3-profile-v1 $(date +%T)"; probe lfm3-profile-v1 $O-control
echo "== test 1: $TC $(date +%T)"; probe $TC $O-test1
echo "== test 2: $TC $(date +%T)"; probe $TC $O-test2
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
docker stop -t 10 $TC > /dev/null; echo "stopped $TC (not removed)"
echo "QUIET-UVM-FINISHED $(date +%T)"
