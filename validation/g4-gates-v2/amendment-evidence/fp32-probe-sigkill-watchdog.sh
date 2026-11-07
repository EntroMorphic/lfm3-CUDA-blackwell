#!/usr/bin/env bash
# Measure peak system memory of FP32 SDPA scoring at one length on one window, under a watchdog that kills the job
# if available memory falls below a floor (Thor's GPU shares system memory; an exhausted system hangs).
set -u
L=$1; floor_gb=${2:-16}; B=/home/ztflynn/Projects/000-research/lfm3-omni-2B/liquid-CUDA/build; out=fp32-probe-$L
rm -rf $B/$out
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 \
  /opt/venv/bin/python3 /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt \
  --output /workspace/liquid-CUDA/build/$out --configs sdpa --dtype float32 --lengths $L --max-windows 1 > $B/$out.log 2>&1 &
job=$!
start_avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); min_avail=$start_avail; killed=0
while kill -0 $job 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ $a -lt $min_avail ] && min_avail=$a
  if [ $a -lt $floor_gb ]; then
    docker exec lfm3-profile-v1 pkill -9 -f ppl_reference.py; killed=1; echo "WATCHDOG: available ${a} GB < ${floor_gb} GB, killed"; break
  fi
  sleep 0.5
done
wait $job; rc=$?
echo "length $L: rc=$rc killed=$killed available start ${start_avail} GB, minimum ${min_avail} GB, peak use ~$((start_avail-min_avail)) GB"
grep '^{' $B/$out.log | cut -c1-140
