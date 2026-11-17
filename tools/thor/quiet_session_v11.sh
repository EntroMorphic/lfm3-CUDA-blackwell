#!/usr/bin/env bash
# Quiet window 11 (2026-11-17), after window 10: prefill-stall-v1, where the faulting instruction is. Window 8's four
# catches (inside the container, uid 1000) stopped the main thread at one PC, 0xfffe9f4f3344, every time, but host gdb
# has no symbols across the PID namespace. tools/stall_catch.py now resolves each frame-0 PC against /proc/PID/maps
# (mapping, file, file offset) and saves the maps. Same catcher as window 8, otherwise unchanged. Diagnostic: no verdict;
# the README records which file and offset (and the symbol, resolved offline) the stalls sit in.
# Usage (Thor host): quiet_session_v11.sh   (results: build/quiet-w11-catch/)
set -uo pipefail
export LFM3_WORKSPACE=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
O=quiet-w11-catch
until grep -q "^QUIET-V10-FINISHED\|^NOT QUIET" $B/quiet-w10.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
echo "== stall catcher in the container as uid 1000 $(date +%T)"
mkdir -p catch/out && cp $B/stall-v4/libptracer_any.so catch/
sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$C/$O/catch/out/\1#" $B/speech-normalize-v1/dev/tts/list.txt > catch/list.txt
sha256sum $W/liquid-CUDA/tools/stall_catch.py $B/vimg-a5/lfm3-speech > sources.sha256
STALL_PREFIX="docker exec -u 1000 -e LD_PRELOAD=$C/$O/catch/libptracer_any.so -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1" \
  python3 $W/liquid-CUDA/tools/stall_catch.py $C/vimg-a5/lfm3-speech $M $C/$O/catch/list.txt $B/$O/catch $C/$O/catch/libptracer_any.so 4 > catch.log 2>&1
tail -2 catch.log; grep CATCH catch/monitor.txt | head -8
for f in catch/catch-*.txt; do echo "-- $f"; sed -n '/^--- frame-0 PCs/,/^--- \/proc/p' $f | head -8; done
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V11-FINISHED $(date +%T)"
