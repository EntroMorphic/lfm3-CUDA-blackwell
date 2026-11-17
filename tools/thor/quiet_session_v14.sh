#!/usr/bin/env bash
# Quiet window 14 (2026-11-17), after window 13: prefill-stall-v1, which mapping the faulting store writes to. Window
# 11 put all four catches at libcuda.so.1.1 file offset 0x3d3344, `str w2, [x0]` (a single 32-bit store to a base
# from a driver structure plus an offset, then a return: the shape of a doorbell write). The catcher now also records
# every thread's registers (gdb reads them without symbols) and resolves those that point into a non-library mapping,
# and takes a maps snapshot 120 s in. Runs: (1) in the container as uid 1000, as windows 8 and 11; (2) on the host,
# the same engine and requests (0 stalls expected, so no catch: its maps snapshot is the comparison). Diagnostic: no
# verdict; the README records x0's mapping and how the two runs' device mappings differ.
# Usage (Thor host): quiet_session_v14.sh   (results: build/quiet-w14-catch/)
set -uo pipefail
export LFM3_WORKSPACE=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}
W=$LFM3_WORKSPACE; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
O=quiet-w14-catch
grep -q "STALL_MAPS_AT" $W/liquid-CUDA/tools/stall_catch.py || { echo "NOT STARTED: catcher not synced"; echo "QUIET-V14-FINISHED $(date +%T)"; exit 1; }
until grep -q "^QUIET-V13-FINISHED\|^NOT QUIET" $B/quiet-w13.log 2>/dev/null; do sleep 60; done
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
sha256sum $W/liquid-CUDA/tools/stall_catch.py $B/vimg-a5/lfm3-speech > sources.sha256
echo "== (1) catcher in the container as uid 1000 $(date +%T)"
mkdir -p container/out && cp $B/stall-v4/libptracer_any.so container/
sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$C/$O/container/out/\1#" $B/speech-normalize-v1/dev/tts/list.txt > container/list.txt
STALL_MAPS_AT=120 STALL_PREFIX="docker exec -u 1000 -e LD_PRELOAD=$C/$O/container/libptracer_any.so -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1" \
  python3 $W/liquid-CUDA/tools/stall_catch.py $C/vimg-a5/lfm3-speech $M $C/$O/container/list.txt $B/$O/container $C/$O/container/libptracer_any.so 4 > container.log 2>&1
tail -1 container.log
for f in container/catch-*.txt; do echo "-- $f"; sed -n '/^--- registers/,/^--- \/proc/p' $f | grep -v "^--- /proc" | head -14; done
echo "== (2) the same engine on the host $(date +%T)"
mkdir -p host/out && cp $B/stall-v4/libptracer_any.so host/
sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$B/$O/host/out/\1#; s#$C/#$B/#g" $B/speech-normalize-v1/dev/tts/list.txt > host/list.txt
STALL_MAPS_AT=120 LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia \
  python3 $W/liquid-CUDA/tools/stall_catch.py $B/vimg-a5/lfm3-speech $W/models/LFM3-Omni-2B $B/$O/host/list.txt $B/$O/host $B/$O/host/libptracer_any.so 4 > host.log 2>&1
tail -1 host.log
echo "== device mappings at 120 s (count, path, perms, size)"
for r in container host; do
  echo "-- $r"
  [ -f $r/maps-snapshot.txt ] && awk '$6 ~ /^\/dev\// {split($1,a,"-"); printf "%s %s %d KiB\n", $6, $2, (strtonum("0x" a[2])-strtonum("0x" a[1]))/1024}' $r/maps-snapshot.txt | sort | uniq -c | sort -rn
done
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V14-FINISHED $(date +%T)"
