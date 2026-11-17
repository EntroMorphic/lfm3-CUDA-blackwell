#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 3 under amendment 3: racecheck on a 2,048-token text prefill (the largest that fits
# on Thor; 4,096 tokens exceeded 110 GiB), on the Thor host with the pinned runtime libraries
# (build/racecheck-pinned-libs-v1, as build/stall-v4/run.sh). A plain host run of the same request first; the
# racecheck run's outputs must be byte-identical to it (the reply tokens for the .json).
# Waits until quiet window 7 has finished (build/quiet-w7.log contains QUIET-V7-FINISHED) and no compute-sanitizer
# runs. Free memory is logged every minute; if MemAvailable drops under 8 GB the run is stopped with SIGTERM and
# recorded as NOT COMPLETED. (Named without "tcgen05_" so quiet-window runners' busy checks do not see the waiter.)
# Usage (Thor host, no arguments): nohup bash tgemm_host_race2048.sh > ... &   Results: build/tgemm-r3-host2048/
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; O=$B/tgemm-r3-host2048; N=2048
SAN=$B/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
BIN=$B/tgemm-on1/lfm3-speech; M=$W/models/LFM3-Omni-2B; REQ=$O/text$N.req.json
export LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia
mkdir -p $O && cd $O || exit 1
log() { echo "$(date +%T) $*" | tee -a run.log; }
log "waiting for quiet window 7 (QUIET-V7-FINISHED in build/quiet-w7.log)"
until grep -qs "QUIET-V7-FINISHED" $B/quiet-w7.log; do sleep 60; done
log "window 7 finished; waiting for no compute-sanitizer"
while pgrep -f "[c]ompute-sanitizer" > /dev/null; do sleep 60; done
python3 - $B/speech-e2e-v1/requests/il-fridge-s1.req.json $REQ $N <<'PY' || { log "FAIL: request"; exit 1; }
import json, sys
src, out, n = json.load(open(sys.argv[1]))["prompt_ids"], sys.argv[2], int(sys.argv[3])
ids = (src * (n // len(src) + 1))[:n]
json.dump({"prompt_ids": ids, "mode": "text", "max_new_tokens": 4, "seed": 1, "capacity": n + 256}, open(out, "w"))
PY
sha256sum $BIN $REQ $SAN $B/racecheck-pinned-libs-v1/* > inputs.sha256
log "plain run (host, $N tokens)"
$BIN $M $REQ $O/plain > plain.log 2>&1; p=$?
log "plain exit $p"
[ $p = 0 ] || { log "FAIL: plain run failed; racecheck not run"; exit 1; }
mem() { awk '/MemFree/{f=$2}/MemAvailable/{a=$2}/^Buffers/{b=$2}/^Cached/{c=$2}/SReclaimable/{s=$2}
  END{printf "%s MemAvailable %.1f GiB free %.1f GiB buff/cache %.1f GiB\n", strftime("%T"), a/1048576, f/1048576, (b+c+s)/1048576}' /proc/meminfo; }
log "racecheck starting ($N tokens)"
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
    for t in $(pgrep -f "[l]fm3-speech .*tgemm-r3-host2048/text"); do kill -TERM $t; done
    break
  fi
  sleep 60
done
wait $SPID; s=$?
mem >> memory.log
secs=$(( $(date +%s) - t0 ))
if [ $stopped = 1 ]; then log "racecheck NOT COMPLETED (stopped at MemAvailable < 8 GB after $secs s)"; exit 2; fi
same=identical; n=0
for f in plain.*; do
  case $f in *.log) continue;; esac
  n=$((n+1)); suf=${f#plain}
  if [[ $suf == .json ]]; then
    python3 -c "import json,sys; a,b=(json.load(open(x)) for x in sys.argv[1:]); sys.exit(a.get('text_tokens')!=b.get('text_tokens'))" $f racecheck$suf 2>/dev/null || same=DIFFER
  else cmp -s $f racecheck$suf || same=DIFFER; fi
done
[ $n = 0 ] && same="NOT COMPARED"
log "racecheck (host, $N tokens): exit $s, $secs s; $(grep -E 'RACECHECK SUMMARY' racecheck.log | tr '\n' ' '); $n outputs $same"
log "HOST RACECHECK 2048 DONE"
[ $s = 0 ] && [ "$same" = identical ] && exit 0 || exit 2
