#!/usr/bin/env bash
# Quiet window 17 (2026-11-17), after window 16: image-cache-v1's server memcheck gate (exit 0), rerun on a server with
# the stop-signal forwarding (lfm/stop_signals.hpp, main 064c1bb). Window 13: 0 errors, but exit 15, with one thread
# under memcheck not blocking SIGTERM. Same check as windows 10 and 13 (check_lfm3_image_cache_fixes.py memcheck, at
# --image-cache-mb 1), plain and under memcheck, with every thread's SigBlk recorded. Gate as written: 0 errors, exit 0,
# texts equal to plain, the expected cache counts. Exactness only (no timing), but queued so nothing overlaps.
# Usage (Thor host): quiet_session_v17.sh   (results: build/quiet-w17-memcheck/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w17-memcheck; M=/workspace/models/LFM3-Omni-2B; S=sig-src; SRV=sig-srv; E2E=vimg-e2e-a4
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
T=$C/$S/tools

missing=0
for f in $B/$S/native/include/lfm/stop_signals.hpp $B/$S/tools/check_lfm3_image_cache_fixes.py ${SAN/#$C/$B} $B/$E2E/out; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V17-FINISHED $(date +%T)"; exit 1; }

until grep -q "^QUIET-V16-FINISHED\|^NOT QUIET" $B/quiet-w16.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
wait_quiet() {  # 3 quiet checks in a row, 30 s apart, within $1 checks
  local quiet=0 i
  for i in $(seq 1 $1); do
    if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && return 0; else quiet=0; fi
    [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
  done
  [ -z "$(busy)" ]
}
wait_quiet 1440 || { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
(cd $B/$S && cat SOURCE_COMMIT 2>/dev/null; sha256sum native/include/lfm/stop_signals.hpp native/src/lfm3_server.cu \
  tools/check_lfm3_image_cache_fixes.py tools/thor/quiet_session_v17.sh) > sources.txt

echo "== build $(date +%T)"
if [ -e $B/$SRV ] && [ -z "${REUSE_BUILDS:-}" ]; then echo "FAIL: build/$SRV exists (REUSE_BUILDS=1 to keep it)"; echo "QUIET-V17-FINISHED $(date +%T)"; exit 1; fi
if ! [ -x $B/$SRV/liquid-lfm3-server ]; then
  docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
    "bash /workspace/liquid-CUDA/build/$S/tools/thor/build_native.sh $C/$SRV; rc=\$?; chown -R 1000:1000 $C/$SRV $C/$SRV.cmake.log $C/$SRV.build.log 2>/dev/null; exit \$rc" \
    > build.log 2>&1; echo "build exit $?"
fi
[ -x $B/$SRV/liquid-lfm3-server ] || { echo "FAIL build"; tail -5 build.log; echo "QUIET-V17-FINISHED $(date +%T)"; exit 1; }
(cd $B && sha256sum $SRV/liquid-lfm3-server) | tee binaries.sha256
wait_quiet 120 || { echo "NOT QUIET after the build"; echo "QUIET-V17-FINISHED $(date +%T)"; exit 1; }

echo "== the server at --image-cache-mb 1, plain and under memcheck $(date +%T)"
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck - $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server-plain.log 2>&1; tail -1 memcheck-server-plain.log
$X python3 $T/check_lfm3_image_cache_fixes.py memcheck $SAN $C/$SRV/liquid-lfm3-server $M $C/$E2E $C/$O/memcheck-server > memcheck-server.log 2>&1; tail -1 memcheck-server.log
for f in plain memcheck; do
  python3 -c "
import json,sys
d=json.load(open('memcheck-server/$f.json')); m=d.get('signal_masks',{})
un=[(t,v['name'],v['sigblk']) for t,v in m.items() if isinstance(v,dict) and not v['sigterm_blocked']]
print('$f: exit', d.get('exit'), len(m), 'threads;', len(un), 'not blocking SIGTERM:', un[:8])"
done
echo "QUIET-V17-FINISHED $(date +%T)"
