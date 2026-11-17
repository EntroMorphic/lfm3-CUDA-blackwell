#!/usr/bin/env bash
# Quiet window 12 (2026-11-17), after window 11: audio-frontend-v1's remaining server rules (rules 0b, 3, 6 and the
# memory bound), build/audio-src-b/tools/thor/quiet_audio_rest.sh as delivered at commit 067054c of the audio branch
# (window 8's attempt was not runnable: its stall tool did not retry 429). Exit 0: every gated rule passed; 1: could
# not run; 2: a gated rule failed (build/quiet-audio-rest/summary.json lists which). ~2.5-3 h.
# Usage (Thor host): quiet_session_v12.sh
set -uo pipefail
export LFM3_WORKSPACE=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build
until grep -q "^QUIET-V11-FINISHED\|^NOT QUIET" $B/quiet-w11.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/quiet-w12 && cd $B/quiet-w12 || exit 1
s=$B/audio-src-b/tools/thor/quiet_audio_rest.sh
sha256sum $s $B/audio-src-b/tools/measure_lfm3_audio_stall.py > sources.sha256
echo "== $(basename $s) $(date +%T)"
bash $s > quiet_audio_rest.log 2>&1; echo "exit $?"; tail -15 quiet_audio_rest.log | cut -c1-300
echo "QUIET-V12-FINISHED $(date +%T)"
