#!/usr/bin/env bash
# Quiet window 13 (2026-11-17), after window 12: image-cache-v1 C3-C5, the steps window 10 could not measure, with the
# amendment set before this run (README: "Amendment to the C3-C5 procedure"): C3's location and C5 with the thread-CPU
# reading as the process total minus the engine thread; the server under memcheck with every thread's SigBlk recorded
# before the SIGTERM. Window 10's builds and images, unchanged (sha256 checked against window 10's record). ~15 min.
# Usage (Thor host): quiet_session_v13.sh   (results: build/quiet-w13-c3c5/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w13-c3c5; P=quiet-w10-c3c5; M=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
SRV=vimg-srv-c3; FLT=vimg-srv-c3f; E2E=vimg-e2e-a4
grep -q "def process_cpu" $R/tools/check_lfm3_image_cache_fixes.py && grep -q "def signal_masks" $R/tools/check_lfm3_image_cache_fixes.py \
  || { echo "NOT STARTED: the amended tool is not synced"; echo "QUIET-V13-FINISHED $(date +%T)"; exit 1; }
until grep -q "^QUIET-V12-FINISHED\|^NOT QUIET" $B/quiet-w12.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
(cd $B && sha256sum -c $P/binaries.sha256) > binaries.check 2>&1 || { echo "FAIL: window 10's binaries changed"; cat binaries.check; exit 1; }
(cd $B/$P/images && sha256sum -c ../images.sha256) > images.check 2>&1 || { echo "FAIL: window 10's images changed"; exit 1; }
sha256sum $R/tools/check_lfm3_image_cache_fixes.py > tool.sha256
echo "== C3: where the evicted image is decoded $(date +%T)"
$X python3 $T/check_lfm3_image_cache_fixes.py c3 $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$P/images/prime \
  $C/$P/images/fresh/c3-31.png $C/$O/c3.json > c3.log 2>&1; tail -1 c3.log
echo "== C5: the fault build $(date +%T)"
$X python3 $T/check_lfm3_image_cache_fixes.py c5 $C/$SRV/liquid-lfm3-server $C/$FLT/liquid-lfm3-server $M $C/$E2E $C/$O/c5.json > c5.log 2>&1; tail -8 c5.log
echo "== C4: memcheck, the server at --image-cache-mb 1 $(date +%T)"
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck - $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server-plain.log 2>&1; tail -1 memcheck-server-plain.log
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck $SAN $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server.log 2>&1; tail -1 memcheck-server.log
for f in plain memcheck; do
  python3 -c "
import json,sys
d=json.load(open('memcheck-server/$f.json')); m=d.get('signal_masks',{})
un=[(t,v['name'],v['sigblk']) for t,v in m.items() if isinstance(v,dict) and not v['sigterm_blocked']]
print('$f: exit', d.get('exit'), len(m), 'threads;', len(un), 'not blocking SIGTERM:', un[:8])"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V13-FINISHED $(date +%T)"
