#!/usr/bin/env bash
# validation/audio-ttft-v1 stage 1, the parts of rules 4 and 5 that are not timing (may overlap other work; nothing here
# is timed). Sources: build/ttft-src (this round), build/merge-src (main as deployed). lfm3-speech: build/ttft-speech-a2.
#   0. builds: the server and worker (ttft-srv), lfm3-vision-check from both trees (vcheck-ttft, vcheck-main);
#   1. rule 5, the vision encoder: both checks on images-v1's 31 cases (x2/x2), every stage file compared byte for byte;
#   2. rule 5, lfm3-speech: the 36-case regression; the audio-in files (40 dumped clips, 93 edge files) and e2e/split with
#      LFM3_AUDIO_ENCODER=fp32 against the validated runs (F2's 5 known differences only), and again by default (x2: the
#      audio-in files differ by design, e2e and split identical); the concurrent engine check;
#   3. rule 5, the server (ttft-srv, default x2): batch, audio in against ttft-speech-a2, 8 and 32 at once, TTS, the text
#      gate, images through the server;
#   4. rule 4: compute-sanitizer memcheck (leaks), racecheck, synccheck, initcheck on lfm3-speech with the shortest
#      dumped clip and the 90 s clip; outputs compared with the plain run's.
# Usage (Thor host, copied to build/ttft-exact/run.sh): bash run.sh > run.log 2>&1   (verdicts read from the log)
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=ttft-exact
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=ttft-src; SP=ttft-speech-a2; SV=ttft-srv
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
T=$C/$S/tools; TH=$B/$S/tools/thor
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
cd $B/$O || exit 1
step() { echo "== $1 $(date +%T)"; }

step "0. builds"
if [ ! -x $B/$SV/liquid-lfm3-server ]; then
  docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
    "bash $C/$S/tools/thor/build_native.sh $C/$SV; rc=\$?; chown -R 1000:1000 $C/$SV $C/$SV.cmake.log $C/$SV.build.log; exit \$rc" \
    > build-srv.log 2>&1; echo "server build exit $?"
fi
for v in ttft:$S main:merge-src; do tag=vcheck-${v%%:*}; src=${v#*:}
  [ -x $B/$tag/lfm3-vision-check ] || docker run --rm --network none -v $W:/workspace $IMG bash -c "mkdir -p $C/$tag && cd $C/$tag && \
    nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$src/native/include -DLFM3_NATIVE_GEMM -o lfm3-vision-check \
    $C/$src/native/tests/lfm3_vision_check.cu $C/$src/native/src/safetensors.cpp > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
  echo "$tag build exit $?"
done
(cd $B && sha256sum $SV/liquid-lfm3-server $SV/liquid-lfm3-worker $SP/lfm3-speech vcheck-ttft/lfm3-vision-check vcheck-main/lfm3-vision-check) | tee binaries.sha256

step "1. rule 5: the vision encoder, 31 cases"
n=0; same=0
for d in $B/vision-ref/hf-5.20.2/float32/*/ $B/vision-ref/heldout/float32/*/; do
  case=$(basename $d); set_=$(basename $(dirname $(dirname $d)))
  for v in ttft main; do
    $X $C/vcheck-$v/lfm3-vision-check $M $C/vision-ref/$set_/float32/$case $C/$O/vision-$v/$case --linear x2 --attention x2 > /dev/null 2>vision-$v-$case.err \
      || echo "RUN FAILED $v $case: $(tail -1 vision-$v-$case.err)"
  done
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  for f in vision-main/$case/*.f32; do n=$((n+1)); cmp -s $f vision-ttft/$case/$(basename $f) && same=$((same+1)) || echo "  DIFFER $case $(basename $f)"; done
done
echo "vision: $same of $n stage files identical"

step "2. rule 5: lfm3-speech"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh $SP > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
for mode in fp32 x2; do
  sed -e "s|^EXEC=\"docker exec |EXEC=\"docker exec -e LFM3_AUDIO_ENCODER=$mode |" -e "s|O=\$B/\$T-rule6; C=\$E/\$T-rule6|O=\$B/\$T-rule6-$mode; C=\$E/\$T-rule6-$mode|" \
    $TH/vision_rule6_speech.sh > rule6-$mode.sh
  grep -q "LFM3_AUDIO_ENCODER=$mode" rule6-$mode.sh && grep -q "rule6-$mode" rule6-$mode.sh || { echo "RULE6 SCRIPT EDIT FAILED $mode"; continue; }
  LFM3_WORKSPACE=$W bash rule6-$mode.sh $SP > rule6-$mode.log 2>&1
  python3 -c "
import json; d=json.load(open('$B/$SP-rule6-$mode/dump.json')); e=json.load(open('$B/$SP-rule6-$mode/edge.json'))
print('$mode: dump differ', len(d['differ']), sorted(d['differ'])[:8], '| missing', len(d['missing']), '| edge', e['identical'], 'of', e['files'])"
  grep -E "^(e2e|split): " rule6-$mode.log
done
$X python3 $T/check_lfm3_speech_concurrent.py $C/$SP/lfm3-speech $M $C $L $C/$O/conc $C/$O/engine-exact.json $C/speed-final2/conc/alone \
  > engine-exact.log 2>&1; echo "concurrent engine exit $?"

step "3. rule 5: the server"
$X python3 $T/check_lfm3_speech_batch_server.py $C/$SV/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L \
  $C/$O/batch-server.json > batch-server.log 2>&1; echo "batch server exit $?"
$X python3 $T/check_lfm3_audio_in_server.py $C/$SV/liquid-lfm3-server $C/$SP/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in-server.log 2>&1; echo "audio-in server exit $?: $(tail -1 audio-in-server.log | cut -c1-200)"
for k in 8 32; do
  $X python3 $T/check_lfm3_audio_frontend_server.py $C/$SV/liquid-lfm3-server $C/$SP/lfm3-speech $M $C/audioin-v2/cases.json \
    $C/$O/at-once-$k-work $k $C/$O/at-once-$k.json --queue-size 40 --handlers 48 > at-once-$k.log 2>&1
  echo "at once $k exit $?: $(tail -1 at-once-$k.log | cut -c1-200)"
done
$X python3 $T/check_lfm3_speech_server.py $C/$SV/liquid-lfm3-server $M $C $C/$O/speech-check.json > speech-check.log 2>&1; echo "TTS server exit $?"
$X python3 $T/check_lfm3_native_server.py --checkpoint $M --server $C/$SV/liquid-lfm3-server --worker $C/$SV/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1; echo "text gate exit $?"
$X python3 $T/check_lfm3_image_server.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-server.json > image-server.log 2>&1
echo "images through the server exit $?: $(tail -1 image-server.log | cut -c1-150)"

step "4. rule 4: sanitizers on lfm3-speech (shortest dumped clip, 90 s)"
short=$(python3 -c "
import json, wave
from pathlib import Path
best = None
for line in open('$B/audioin-v2/list-dump.txt'):
    req = line.split()[0].replace('/workspace/', '$W/')
    r = json.load(open(req)); w = r['audio'][0].replace('/workspace/', '$W/')
    with wave.open(w) as f: n = f.getnframes()
    if best is None or n < best[0]: best = (n, line.split()[0])
print(best[1])")
mkdir -p san
printf '%s %s\n%s %s\n' $short $C/$O/san/plain-short $C/quiet-w3-audio/req/joined90.req.json $C/$O/san/plain-90 > san/list-plain.txt
$X $C/$SP/lfm3-speech $M --batch $C/$O/san/list-plain.txt > san/plain.log 2>&1; echo "plain exit $?"
for tool in memcheck racecheck synccheck initcheck; do
  extra=""; [ $tool = memcheck ] && extra="--leak-check full"
  printf '%s %s\n%s %s\n' $short $C/$O/san/$tool-short $C/quiet-w3-audio/req/joined90.req.json $C/$O/san/$tool-90 > san/list-$tool.txt
  timeout 21600 $X $SAN --tool $tool $extra --log-file $C/$O/san/$tool.log $C/$SP/lfm3-speech $M --batch $C/$O/san/list-$tool.txt \
    > san/$tool.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tool exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY|RACECHECK SUMMARY' san/$tool.log | tr '\n' ' ')"
  for k in short 90; do python3 $B/$S/tools/compare_speech_outputs.py san/plain-$k san/$tool-$k san/cmp-$tool-$k.json > /dev/null 2>&1; \
    echo "  $tool $k against plain: $(python3 -c "import json; d=json.load(open('san/cmp-$tool-$k.json')); print(d.get('identical'), 'of', d.get('files'), 'differ', d.get('differ'))")"; done
done
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "TTFT-EXACT-FINISHED $(date +%T)"
