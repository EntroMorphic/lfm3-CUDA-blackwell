#!/usr/bin/env bash
# One FP32 scoring window at length L with attention CFG; records peak system memory use and what is returned after.
# Intervenes only with a graceful SIGTERM if available memory falls below FLOOR GB (SIGKILL leaked driver memory).
set -u
L=$1; CFG=$2; floor_gb=${3:-40}; B=/home/ztflynn/Projects/000-research/lfm3-omni-2B/liquid-CUDA/build; out=fp32-probe-$CFG-$L-${4:-v1}
rm -rf $B/$out
avail(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
before=$(avail); min=$before; stopped=0
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 \
  /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt \
  --output /workspace/liquid-CUDA/build/$out --configs $CFG --dtype float32 --lengths $L --max-windows 1 > $B/$out.log 2>&1 &
job=$!
while kill -0 $job 2>/dev/null; do
  a=$(avail); [ $a -lt $min ] && min=$a
  if [ $a -lt $floor_gb ] && [ $stopped = 0 ]; then docker exec lfm3-profile-v1 pkill -TERM -f ppl_reference.py; stopped=1; echo "SIGTERM at ${a} GB available"; fi
  sleep 0.5
done
wait $job; rc=$?; sleep 5
echo "$CFG $L: rc=$rc stopped=$stopped available before ${before} GB, minimum ${min} GB (peak ~$((before-min)) GB), after $(avail) GB"
