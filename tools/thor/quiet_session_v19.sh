#!/usr/bin/env bash
# Quiet window 19 (launched by hand after Thor's reboot for always-on-v1 rule 5): validation/audio-handles-v1's rules on Thor. This script runs and records; the
# verdicts are read from its logs against the round's README. Sources: the audio-handles-v1 branch synced to
# build/audio-src-h; the base server's sources (main at a350659) in build/audio-src-base.
#   1. tools/thor/quiet_audio_rest.sh from build/audio-src-h, with its own names (QAR_OUT=quiet-w19-handles-rest,
#      QAR_TAG=audio-h-w): the builds (lfm3-speech, the server and worker, the bench, from audio-src-h); rule 3 as
#      written (the bar, then the stall at 8 and 16 rows, all five clips; the base server's again for the record); the
#      memory bound; rule 6's list (rule 5 here). Its exit: 0 all gated rules pass, 1 not runnable, 2 a gated rule failed.
#   2. Rule 1, exactness, on those builds: the server's audio-in check with each front end (LFM3_AUDIO_FEATURES dft,
#      exact-f2); the same with a clip left out (LFM3_AUDIO_AHEAD_SKIP=1: the two-clip case goes as base64, mixed);
#      8 and 32 requests at once (check_lfm3_audio_frontend_server.py).
#   3. Rule 4: compute-sanitizer memcheck (leak check) on the server path: the audio-in check with the server run under
#      memcheck (its own log per process); the check's wall-time bound may fail under the sanitizer, the sanitizer's
#      ERROR SUMMARY is what is read.
#   4. Rule 3 (reported): an nsys trace (host) of one 90-second stall run at 8 rows on the new server;
#      tools/nsys_audio_admission.py reads the engine thread's CPU-only time before the encoder (window 16: 35.3 ms).
# Usage (Thor host): quiet_session_v19.sh   (results: build/quiet-w19-handles/ and build/quiet-w19-handles-rest/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w19-handles; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
S=audio-src-h; TAG=audio-h-w; SRV=$TAG-srv; SPE=$TAG-speech
J90=server-stream-v3-check/work/joined-90s.wav
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
T=$C/$S/tools
step() { echo "== $1 $(date +%T)"; }

missing=0
for f in $B/$S/tools/thor/quiet_audio_rest.sh $B/$S/tools/nsys_audio_admission.py $B/$S/tools/check_lfm3_audio_frontend_server.py \
         $B/$S/tools/measure_lfm3_audio_stall.py $B/audio-src-base/native/src/lfm3_server.cu $B/$J90 $B/audioin-v2/cases.json \
         $B/sanitizer-13.5.25; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
grep -q "audio_ref" $B/$S/native/src/lfm3_worker.cu 2>/dev/null || { echo "MISSING audio_ref in build/$S (not the audio-handles-v1 tree)"; missing=1; }
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V19-FINISHED $(date +%T)"; exit 1; }

# the always-on service (always-on-v1), if running, is stopped for this window and started again when it ends, however
# it ends (a running server is "busy" to every timing check)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
SVC=0
if systemctl --user is-active --quiet liquid-lfm3 2>/dev/null; then
  systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 for this window $(date +%T)"
fi
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
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
(cd $B/$S && git -C . log --oneline -1 2>/dev/null; sha256sum native/include/lfm/audio_features.hpp native/include/lfm/lfm3_http.hpp \
  native/src/lfm3_worker.cu native/src/lfm3_speech.cuh tools/thor/quiet_session_v19.sh tools/thor/quiet_audio_rest.sh) > sources.txt

step "1. quiet_audio_rest.sh from $S (builds, rule 3, the memory bound, rule 6's list)"
LFM3_WORKSPACE=$W QAR_OUT=$O-rest QAR_TAG=$TAG bash $B/$S/tools/thor/quiet_audio_rest.sh > rest.log 2>&1
echo "quiet_audio_rest.sh exit $?"; grep -E "^PASS|^FAIL|NOT RUNNABLE|DONE|PASS$|FAIL$" rest.log | tail -40
[ -x $B/$SRV/liquid-lfm3-server ] && [ -x $B/$SPE/lfm3-speech ] || { echo "NO BUILDS: stopping"; echo "QUIET-V19-FINISHED $(date +%T)"; exit 1; }
grep -a -q "audio_ref" $B/$SRV/liquid-lfm3-server || { echo "FAIL: the server build has no audio_ref"; echo "QUIET-V19-FINISHED $(date +%T)"; exit 1; }
(cd $B && sha256sum $SRV/liquid-lfm3-server $SRV/liquid-lfm3-worker $SPE/lfm3-speech) | tee binaries.sha256
wait_quiet 120 || { echo "NOT QUIET after step 1"; echo "QUIET-V19-FINISHED $(date +%T)"; exit 1; }

bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
step "2. rule 1: exactness"
for mode in dft exact-f2 skip1; do
  env=(-e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia)
  case $mode in
    dft) env+=(-e LFM3_AUDIO_FEATURES=dft) ;;
    exact-f2) env+=(-e LFM3_AUDIO_FEATURES=exact-f2) ;;
    skip1) env+=(-e LFM3_AUDIO_AHEAD_SKIP=1) ;;  # mixed: a request's second clip left out (sent as base64)
  esac
  mkdir -p $mode
  docker exec "${env[@]}" lfm3-profile-v1 python3 $T/check_lfm3_audio_in_server.py $C/$SRV/liquid-lfm3-server $C/$SPE/lfm3-speech \
    $M $L $C/$O/$mode/work $C/$O/$mode/audio-in-server.json > $mode/audio-in-server.log 2>&1
  echo "audio-in server ($mode): exit $?; $(tail -1 $mode/audio-in-server.log)"
done
for n in 8 32; do
  docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 $T/check_lfm3_audio_frontend_server.py \
    $C/$SRV/liquid-lfm3-server $C/$SPE/lfm3-speech $M $C/audioin-v2/cases.json $C/$O/at-once-$n-work $n $C/$O/at-once-$n.json \
    --queue-size 40 --handlers 48 > at-once-$n.log 2>&1
  echo "at once $n: exit $?; $(tail -1 at-once-$n.log)"
done
bash $R/tools/thor/power_state.sh power-mid.json power-start.json > /dev/null || echo "VOID power changed during 2"

step "3. rule 4: memcheck on the server path (the audio-in check)"
mkdir -p memcheck
printf '#!/bin/sh\nexec %s --tool memcheck --leak-check full --log-file %s/memcheck/memcheck-%%p.log %s "$@"\n' \
  $SAN $C/$O $C/$SRV/liquid-lfm3-server > srv-memcheck.sh && chmod 755 srv-memcheck.sh
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 $T/check_lfm3_audio_in_server.py \
  $C/$O/srv-memcheck.sh $C/$SPE/lfm3-speech $M $L $C/$O/memcheck/work $C/$O/memcheck/audio-in-server.json > memcheck/audio-in-server.log 2>&1
echo "audio-in under memcheck: exit $? (the check's own verdict; its wall bound may not hold under the sanitizer)"
grep -h -E "ERROR SUMMARY|LEAK SUMMARY" memcheck/memcheck-*.log 2>/dev/null || echo "no memcheck summary found"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O/memcheck 2>/dev/null

step "4. rule 3 (reported): nsys of one 90-second stall run at 8 rows (host)"
if command -v nsys > /dev/null; then
  printf '[{"name": "joined90", "wav": "%s"}]\n' $B/$J90 > clips-90-host.json
  run_nsys() {
    nsys profile --trace=cuda,nvtx,osrt "$@" --output nsys-stall-90 --force-overwrite true \
      python3 $B/$S/tools/measure_lfm3_audio_stall.py $B/$SRV/liquid-lfm3-server $W/models/LFM3-Omni-2B $B/$O/clips-90-host.json 8 \
      $B/$O/nsys-stall-90.json 1 > nsys.log 2>&1
  }
  run_nsys --sample=process-tree --cpuctxsw=process-tree; rc=$?
  if [ $rc != 0 ]; then echo "nsys with CPU sampling: exit $rc; again without it"; run_nsys --sample=none --cpuctxsw=none; rc=$?; fi
  echo "nsys exit $rc; $(tail -1 nsys.log)"
  if [ -f nsys-stall-90.nsys-rep ]; then
    nsys export --type sqlite --force-overwrite true --output nsys-stall-90.sqlite nsys-stall-90.nsys-rep >> nsys.log 2>&1
    python3 $B/$S/tools/nsys_audio_admission.py nsys-stall-90.sqlite admission.json
  fi
else
  echo "no nsys on the host: skipped"
fi
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"

step "summary"
grep -E "^PASS|^FAIL|NOT RUNNABLE|QUIET AUDIO REST DONE" rest.log | cut -c1-200
for f in */audio-in-server.log at-once-*.log; do [ -f $f ] && echo "$f: $(tail -1 $f | cut -c1-200)"; done
[ -f admission.json ] && python3 -c "import json; [print('admission:', a) for a in json.load(open('admission.json'))['admissions']]"
echo "QUIET-V19-FINISHED $(date +%T)"
