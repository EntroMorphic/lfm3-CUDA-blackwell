#!/usr/bin/env bash
# Quiet window 8 (2026-11-17), after window 7 and the tcgen05 2,048-token racecheck:
# 1. prefill-stall-v1, the container user: the probe inside lfm3-profile-v1 as uid 1000, as root, as uid 1000.
#    Pre-registered reading: running as root is the trigger if the root run stalls (>= 5 of 630) and both uid 1000
#    runs have none; uid 1000 stalling as root does (>= 5 each) means the user is not it; anything else, not decisive.
# 2. audio-frontend-v1's remaining server rules: build/audio-src-b/tools/thor/quiet_audio_rest.sh, as delivered.
# Usage (Thor host): quiet_session_v8.sh OUT_TAG
set -uo pipefail
O=${1:?OUT_TAG}
export LFM3_WORKSPACE=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build
mkdir -p $B/$O && cd $B/$O
until grep -q "^QUIET-V7-FINISHED\|^NOT QUIET" $B/quiet-w7.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
for run in user1000-1 root user1000-2; do
  echo "== stall probe $run $(date +%T)"
  if [ $run = root ]; then U=; else U=1000; fi
  STALL_USER=$U bash $W/liquid-CUDA/tools/thor/stall_probe.sh vimg-a5 $O-$run > probe-$run.log 2>&1
  cat $B/$O-$run/limits.txt; grep -E "requests" probe-$run.log; grep -c "^STALL" probe-$run.log; grep "^STALL" probe-$run.log | head -3 | cut -c1-250
done
s=$B/audio-src-b/tools/thor/quiet_audio_rest.sh
echo "== $(basename $s) $(date +%T)"
if [ -f $s ]; then bash $s > $(basename $s .sh).log 2>&1; echo "exit $?"; tail -12 $(basename $s .sh).log; else echo "MISSING $s"; fi
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V8-FINISHED $(date +%T)"
