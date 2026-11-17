#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 3, the one cell left: racecheck on the 4,096-token text prefill, run on the Thor
# host (125 GB, no memory cgroup; in lfm3-profile-v1 it was OOM-killed at 100 GB against the 96 GiB limit), with the
# pinned runtime libraries (build/racecheck-pinned-libs-v1, as build/stall-v4/run.sh). A plain host run of the same
# request first; the racecheck run's outputs must be byte-identical to it (the reply tokens for the .json).
# Waits (never overlapping anything memory-heavy): until the audio round's racecheck has finished
# (build/audio-fe-a2-san-race/{run.log,wait.log} contains "RACECHECK RUN DONE") and no compute-sanitizer runs.
# During the run, free memory is logged every minute; if MemAvailable drops under 8 GB the run is stopped with
# SIGTERM and recorded as NOT COMPLETED.
# Usage (Thor host, no arguments): nohup bash tcgen05_host_racecheck.sh > ... &   Results: build/tgemm-r3-host/
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; O=$B/tgemm-r3-host
SAN=$B/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
BIN=$B/tgemm-on1/lfm3-speech; M=$W/models/LFM3-Omni-2B; REQ=$O/text4096.req.json
export LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia
mkdir -p $O && cd $O || exit 1
log() { echo "$(date +%T) $*" | tee -a run.log; }
log "waiting for the audio racecheck (RACECHECK RUN DONE)"
until grep -qs "RACECHECK RUN DONE" $B/audio-fe-a2-san-race/run.log $B/audio-fe-a2-san-race/wait.log; do sleep 60; done
log "audio racecheck done; waiting for no compute-sanitizer"
while pgrep -f "[c]ompute-sanitizer" > /dev/null; do sleep 60; done
cp $B/tgemm-r3b/text/text4096.req.json $REQ || { log "FAIL: no request"; exit 1; }
sha256sum $BIN $REQ $SAN $B/racecheck-pinned-libs-v1/* > inputs.sha256
log "plain run (host)"
$BIN $M $REQ $O/plain > plain.log 2>&1; p=$?
log "plain exit $p"
[ $p = 0 ] || { log "FAIL: plain run failed; racecheck not run"; exit 1; }
mem() { awk '/MemTotal/{t=$2}/MemFree/{f=$2}/MemAvailable/{a=$2}/^Buffers/{b=$2}/^Cached/{c=$2}/SReclaimable/{s=$2}
  END{printf "%s MemAvailable %.1f GiB free %.1f GiB buff/cache %.1f GiB\n", strftime("%T"), a/1048576, f/1048576, (b+c+s)/1048576}' /proc/meminfo; }
log "racecheck starting"
t0=$(date +%s)
$SAN --tool racecheck --error-exitcode 99 --print-limit 50 $BIN $M $REQ $O/racecheck > racecheck.log 2>&1 &
SPID=$!
stopped=0
while kill -0 $SPID 2>/dev/null; do
  mem >> memory.log
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt $((8 * 1024 * 1024)) ]; then
    log "MemAvailable under 8 GB: SIGTERM to the racecheck run"
    stopped=1
    kill -TERM $SPID
    for t in $(pgrep -f "[l]fm3-speech .*tgemm-r3-host/text4096"); do kill -TERM $t; done
    break
  fi
  sleep 60
done
wait $SPID; s=$?
mem >> memory.log
secs=$(( $(date +%s) - t0 ))
if [ $stopped = 1 ]; then
  log "racecheck NOT COMPLETED (stopped at MemAvailable < 8 GB after $secs s)"; exit 2
fi
same=identical; n=0
for f in plain.*; do
  case $f in *.log) continue;; esac
  n=$((n+1)); suf=${f#plain}
  if [[ $suf == .json ]]; then
    python3 -c "import json,sys; a,b=(json.load(open(x)) for x in sys.argv[1:]); sys.exit(a.get('text_tokens')!=b.get('text_tokens'))" $f racecheck$suf 2>/dev/null || same=DIFFER
  else cmp -s $f racecheck$suf || same=DIFFER; fi
done
[ $n = 0 ] && same="NOT COMPARED"
log "racecheck (host, 4,096 tokens): exit $s, $secs s; $(grep -E 'RACECHECK SUMMARY' racecheck.log | tr '\n' ' '); $n outputs $same; peak use in memory.log"
log "HOST RACECHECK DONE"
[ $s = 0 ] && [ "$same" = identical ] && exit 0 || exit 2
