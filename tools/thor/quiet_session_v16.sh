#!/usr/bin/env bash
# Quiet window 16 (2026-11-17), after window 15: validation/audio-frontend-v1 diagnostics. Diagnostic only: this
# script runs and records; nothing here is a rule's verdict. Sources: this branch's tree as synced to
# build/audio-src-b (it carries LFM3_MEL_THREADS and audio_parse_bench.cpp).
#   0. builds: the server and worker with LFM3_MEL_THREADS (audio-fe-v16-srv) and audio-parse-bench (audio-fe-v16-parse),
#      both from build/audio-src-b, as the engine is compiled.
#   1. exactness: images through the server (check_lfm3_image_server.py) on audio-fe-w-srv against vimg-e2e-a4, the
#      current reference (window 12 checked against the stale a1).
#   2. the stall of the 90-second clip at 8 rows (measure_lfm3_audio_stall.py, 3 repeats) on audio-fe-v16-srv, with
#      LFM3_MEL_THREADS=4 (the default), then 1, then 4 again (hypothesis 1: the handler's log_mel threads contending
#      with the engine thread).
#   3. the engine thread's per-clip parse and decode on one CPU (audio-parse-bench: Json::parse of the request line,
#      clip_from_base64, the cache lookup) for the 90-second and the 35-second clips (hypothesis 2).
#   4. an nsys trace (host nsys; the server and the stall tool on the host, outside the container) of one stall run of
#      the 90-second clip at 8 rows, with CUDA, NVTX and OS-runtime traces and CPU sampling where permitted.
# Usage (Thor host): quiet_session_v16.sh   (results: build/quiet-w16-audio/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w16-audio; M=/workspace/models/LFM3-Omni-2B
S=audio-src-b; OLD=audio-fe-w-srv; SRV=audio-fe-v16-srv; PB=audio-fe-v16-parse; E2E=vimg-e2e-a4
J90=server-stream-v3-check/work/joined-90s.wav; J35=/workspace/data/librispeech/wav/4507-16021-0047.wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
T=$C/$S/tools
step() { echo "== $1 $(date +%T)"; }

missing=0
for f in $B/$S/native/tests/audio_parse_bench.cpp $B/$S/tools/measure_lfm3_audio_stall.py $B/$S/tools/check_lfm3_image_server.py $B/racecheck-pinned-libs-v1 \
         $B/$OLD/liquid-lfm3-server $B/$E2E/out $B/$J90 $W/data/librispeech/wav/4507-16021-0047.wav; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
grep -q "LFM3_MEL_THREADS" $B/$S/native/include/lfm/audio_features.hpp 2>/dev/null || { echo "MISSING LFM3_MEL_THREADS in build/$S"; missing=1; }
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V16-FINISHED $(date +%T)"; exit 1; }

until grep -q "^QUIET-V15-FINISHED\|^NOT QUIET" $B/quiet-w15.log 2>/dev/null; do sleep 60; done
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
(cd $B/$S && sha256sum native/include/lfm/audio_features.hpp native/include/lfm/lfm3_http.hpp native/tests/audio_parse_bench.cpp \
  tools/measure_lfm3_audio_stall.py tools/thor/quiet_session_v16.sh) > sources.txt

step "0. builds"
for d in $SRV $PB; do
  if [ -e $B/$d ] && [ -z "${REUSE_BUILDS:-}" ]; then echo "FAIL: build/$d exists (REUSE_BUILDS=1 to keep it)"; echo "QUIET-V16-FINISHED $(date +%T)"; exit 1; fi
done
if ! [ -x $B/$SRV/liquid-lfm3-server ]; then
  docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
    "bash /workspace/liquid-CUDA/build/$S/tools/thor/build_native.sh $C/$SRV; rc=\$?; chown -R 1000:1000 $C/$SRV $C/$SRV.cmake.log $C/$SRV.build.log 2>/dev/null; exit \$rc" \
    > build-server.log 2>&1; echo "server build exit $?"
fi
if ! [ -x $B/$PB/audio-parse-bench ]; then
  mkdir -p $B/$PB
  docker run --rm --network none -v $W:/workspace $IMG bash -c \
    "cd $C/$PB && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -x cu -o audio-parse-bench $C/$S/native/tests/audio_parse_bench.cpp; rc=\$?; chown -R 1000:1000 $C/$PB; exit \$rc" \
    > build-parse.log 2>&1; echo "parse-bench build exit $?"
fi
[ -x $B/$SRV/liquid-lfm3-server ] && [ -x $B/$PB/audio-parse-bench ] || { echo "FAIL build"; tail -5 build-*.log; echo "QUIET-V16-FINISHED $(date +%T)"; exit 1; }
grep -a -q "LFM3_MEL_THREADS" $B/$SRV/liquid-lfm3-server || { echo "FAIL: the server build has no LFM3_MEL_THREADS"; echo "QUIET-V16-FINISHED $(date +%T)"; exit 1; }
readelf -d $B/$SRV/liquid-lfm3-server | grep -q cublas && { echo "FAIL links cuBLAS"; exit 1; }
(cd $B && sha256sum $OLD/liquid-lfm3-server $SRV/liquid-lfm3-server $SRV/liquid-lfm3-worker $PB/audio-parse-bench) | tee binaries.sha256
wait_quiet 120 || { echo "NOT QUIET after the builds"; echo "QUIET-V16-FINISHED $(date +%T)"; exit 1; }

bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
step "1. exactness: images through the server on $OLD against $E2E"
$X python3 $T/check_lfm3_image_server.py $C/$OLD/liquid-lfm3-server $M $C/$E2E $C/$O/image-server-a4.json > image-server-a4.log 2>&1
echo "exit $?"; tail -1 image-server-a4.log

step "2. the 90-second stall at 8 rows, LFM3_MEL_THREADS 4, 1, 4"
printf '[{"name": "joined90", "wav": "%s"}]\n' $C/$J90 > clips-90.json
for run in 1 2 3; do
  n=$([ $run = 2 ] && echo 1 || echo 4)
  docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_MEL_THREADS=$n lfm3-profile-v1 \
    python3 $T/measure_lfm3_audio_stall.py $C/$SRV/liquid-lfm3-server $M $C/$O/clips-90.json 8 $C/$O/stall-90-mel$n-run$run.json 3 \
    > stall-90-mel$n-run$run.log 2>&1
  echo "mel threads $n (run $run): exit $?; $(tail -1 stall-90-mel$n-run$run.log)"
done
bash $R/tools/thor/power_state.sh power-mid.json power-start.json > /dev/null || echo "VOID power changed during 2"

step "3. the engine thread's parse and decode on one CPU"
for clip in $C/$J90 $J35; do
  nm=$(basename "$clip" .wav)
  $X $C/$PB/audio-parse-bench $clip 20 > parse-$nm.json 2> parse-$nm.log; rc=$?
  echo "$nm: exit $rc; $(cat parse-$nm.json)"
done

step "4. nsys trace of one 90-second stall run at 8 rows (host)"
if command -v nsys > /dev/null; then
  printf '[{"name": "joined90", "wav": "%s"}]\n' $B/$J90 > clips-90-host.json
  run_nsys() {
    # the host has no CUDA 13.5 runtime on its default path: the pinned libraries, as window 14's host run
    LD_LIBRARY_PATH=$B/racecheck-pinned-libs-v1:/usr/lib/aarch64-linux-gnu/nvidia nsys profile --trace=cuda,nvtx,osrt "$@" --output nsys-stall-90 --force-overwrite true \
      python3 $B/$S/tools/measure_lfm3_audio_stall.py $B/$SRV/liquid-lfm3-server $W/models/LFM3-Omni-2B $B/$O/clips-90-host.json 8 \
      $B/$O/nsys-stall-90.json 1 > nsys.log 2>&1
  }
  run_nsys --sample=process-tree --cpuctxsw=process-tree; rc=$?
  if [ $rc != 0 ]; then echo "nsys with CPU sampling: exit $rc; again without it"; run_nsys --sample=none --cpuctxsw=none; rc=$?; fi
  echo "nsys exit $rc; $(tail -1 nsys.log)"
  [ -f nsys-stall-90.nsys-rep ] && nsys stats --report cuda_gpu_kern_sum,osrt_sum,nvtx_sum --format csv --output nsys-stall-90 \
    nsys-stall-90.nsys-rep >> nsys.log 2>&1
  ls nsys-stall-90* 2>/dev/null
else
  echo "no nsys on the host: skipped"
fi
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"

step "summary"
for f in image-server-a4.log stall-90-*.log parse-*.json nsys.log; do [ -f $f ] && echo "$f: $(tail -1 $f | cut -c1-220)"; done
echo "QUIET-V16-FINISHED $(date +%T)"
