#!/usr/bin/env bash
# Quiet window 15 (2026-11-17), after window 14: always-on-v1 rules 0, 1, 2, 3 (RSS), 4 (startup) and 6, on a server
# built from main at the always-on merge (7ce8546), copied to build/always-on-src so the windows before it keep their
# tree. In the container (lfm3-profile-v1), as every server check so far; rule 5 (the unit, on the host, after linger)
# is a later window.
#   A. tools/measure_lfm3_first_request.py, parts a-d: rule 0 (the cold 1st/2nd/3rd requests), rule 1 (3 fresh
#      --warmup servers), rule 3's idle RSS, rule 4's startup. Timing: quiet Thor.
#   B. Rule 2 (with --warmup, through a wrapper that adds it) and rule 6 (the same build without it): the 11 stage-0
#      image requests (check_lfm3_image_server.py), check_lfm3_image_cache.py (32 of 32), the speech stream check's
#      cases, the audio-in server check. Not in this window: rule 2's three soak text bodies.
# Verdicts are read from the logs against the README's rules; this script only runs and records.
# Usage (Thor host): quiet_session_v15.sh   (results: build/quiet-w15-always-on/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w15-always-on; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
S=always-on-src; SRV=always-on-srv; SPE=vimg-c3; E2E=vimg-e2e-a4
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
T=$C/$S/tools
step() { echo "== $1 $(date +%T)"; }

missing=0
for f in $B/$S/native/include/lfm/lfm3_warmup.hpp $B/$S/tools/measure_lfm3_first_request.py $B/$SPE/lfm3-speech \
         $B/$E2E/out $B/vimg-e2e-a1/req/cherry.json $B/audio-in-listen/q0-16k.wav; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
LISTEN=${LISTEN_DIR:-}
if [ -z "$LISTEN" ]; then
  for d in $B/*/req $B/*/*/req; do [ -f $d/01-china.json ] && [ -f $d/08-mm05.json ] && { LISTEN=$d; break; }; done
fi
[ -n "$LISTEN" ] || { echo "MISSING the listening set's req/"; missing=1; }
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V15-FINISHED $(date +%T)"; exit 1; }
LISTEN_C=${LISTEN/#$B/$C}

until grep -q "^QUIET-V14-FINISHED\|^NOT QUIET" $B/quiet-w14.log 2>/dev/null; do sleep 60; done
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
(cd $B/$S && cat SOURCE_COMMIT 2>/dev/null; sha256sum native/src/lfm3_server.cu native/include/lfm/lfm3_http.hpp \
  native/include/lfm/lfm3_warmup.hpp tools/measure_lfm3_first_request.py tools/thor/quiet_session_v15.sh) > sources.txt

step "0. build"
if [ -e $B/$SRV ] && [ -z "${REUSE_BUILDS:-}" ]; then echo "FAIL: build/$SRV exists (REUSE_BUILDS=1 to keep it)"; exit 1; fi
if ! [ -x $B/$SRV/liquid-lfm3-server ]; then
  docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
    "bash /workspace/liquid-CUDA/build/$S/tools/thor/build_native.sh $C/$SRV; rc=\$?; chown -R 1000:1000 $C/$SRV $C/$SRV.cmake.log $C/$SRV.build.log 2>/dev/null; exit \$rc" \
    > build.log 2>&1; echo "build exit $?"
fi
[ -x $B/$SRV/liquid-lfm3-server ] || { echo "FAIL build"; tail -5 build.log; echo "QUIET-V15-FINISHED $(date +%T)"; exit 1; }
grep -a -q "warming_up" $B/$SRV/liquid-lfm3-server || { echo "FAIL: the build has no --warmup"; echo "QUIET-V15-FINISHED $(date +%T)"; exit 1; }
readelf -d $B/$SRV/liquid-lfm3-server | grep -q cublas && { echo "FAIL links cuBLAS"; exit 1; }
(cd $B && sha256sum $SRV/liquid-lfm3-server $SRV/liquid-lfm3-worker $SPE/lfm3-speech) | tee binaries.sha256
grep -E "LIQUID_LFM3_NATIVE_GEMM:" $B/$SRV/CMakeCache.txt
# the --warmup wrapper (rule 2): every argument as given, then --warmup
printf '#!/bin/sh\nexec %s "$@" --warmup\n' $C/$SRV/liquid-lfm3-server > srv-warmup.sh && chmod 755 srv-warmup.sh
WRAP=$C/$O/srv-warmup.sh
wait_quiet 120 || { echo "NOT QUIET after the build"; echo "QUIET-V15-FINISHED $(date +%T)"; exit 1; }

bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
step "A. rules 0, 1, 3 (RSS), 4: measure_lfm3_first_request.py, parts a-d"
$X python3 $T/measure_lfm3_first_request.py $C/$SRV/liquid-lfm3-server $M $C $C/$O/first-request.json > first-request.log 2>&1
echo "exit $?"; tail -25 first-request.log
bash $R/tools/thor/power_state.sh power-mid.json power-start.json > /dev/null || echo "VOID power changed during A"

step "B. rule 2 (--warmup) and rule 6 (without), each check on a fresh server"
for v in warm cold; do
  [ $v = warm ] && BIN=$WRAP || BIN=$C/$SRV/liquid-lfm3-server
  mkdir -p $v/audio-work
  echo "-- $v: the 11 stage-0 image requests"
  $X python3 $T/check_lfm3_image_server.py $BIN $M $C/$E2E $C/$O/$v/image-server.json > $v/image-server.log 2>&1; echo "exit $?"; tail -1 $v/image-server.log
  echo "-- $v: check_lfm3_image_cache.py"
  $X python3 $T/check_lfm3_image_cache.py $BIN $M $C/$E2E $C/$O/$v/image-cache.json $LISTEN_C > $v/image-cache.log 2>&1; echo "exit $?"; tail -1 $v/image-cache.log
  echo "-- $v: the speech stream check"
  $X python3 $T/check_lfm3_speech_stream.py $BIN $M $C $L $C/$O/$v/stream-check.json > $v/stream-check.log 2>&1; echo "exit $?"; tail -1 $v/stream-check.log
  echo "-- $v: the audio-in server check"
  $X python3 $T/check_lfm3_audio_in_server.py $BIN $C/$SPE/lfm3-speech $M $L $C/$O/$v/audio-work $C/$O/$v/audio-in-server.json > $v/audio-in-server.log 2>&1; echo "exit $?"; tail -1 $v/audio-in-server.log
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
step "summary"
for f in first-request.log warm/*.log cold/*.log; do echo "$f: $(tail -1 $f | cut -c1-220)"; done
echo "QUIET-V15-FINISHED $(date +%T)"
