#!/usr/bin/env bash
# validation/audio-ttft-v1 amendment 5, the rules that are not timing (may overlap other work): rules 1-3 of the
# amendment. Source: build/ttft-src-a5 (the amendment's commit). Earlier binaries: ttft-speech-a2 (x2 with [W | W]).
#   0. builds: lfm3-speech (a5-speech), server and worker (a5-srv), audio-encoder-bench (a5-bench), lfm3-vision-check
#      (vcheck-a5), gemm-wrap-check (a5-gemm);
#   1. gemm-wrap-check (amendment rule 1);
#   2. amendment rule 2: lfm3-speech by default against ttft-speech-a2-rule6-x2 (dump 360, edge 93, byte for byte), e2e
#      and split against the validated runs; the bench's encoded hashes against rule 4's;
#   3. amendment rule 3: vision 31 cases against main's; TTS regression; rule6 in fp32 mode; the concurrent engine in
#      fp32 mode; the server (audio in, 8 and 32 at once, TTS, text gate, images); rule 4's sanitizers on the bench.
#      The batch server in fp32 mode has timing sub-checks and runs in the quiet window instead.
# Usage (Thor host, copied to build/a5-exact/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=a5-exact
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=ttft-src-a5; SP=a5-speech; SV=a5-srv; BENCH=a5-bench
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
XF="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
T=$C/$S/tools; TH=$B/$S/tools/thor
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
cd $B/$O || exit 1
step() { echo "== $1 $(date +%T)"; }

step "0. builds"
LFM3_WORKSPACE=$W bash $TH/build_speech.sh $SP build/$S > build-speech.log 2>&1; echo "lfm3-speech build exit $?"
docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "bash $C/$S/tools/thor/build_native.sh $C/$SV; rc=\$?; chown -R 1000:1000 $C/$SV $C/$SV.cmake.log $C/$SV.build.log; exit \$rc" \
  > build-srv.log 2>&1; echo "server build exit $?"
LFM3_WORKSPACE=$W bash $TH/build_audio_bench.sh $BENCH build/$S > build-bench.log 2>&1; echo "bench build exit $?"
for t in vcheck-a5:lfm3_vision_check.cu:lfm3-vision-check a5-gemm:gemm_wrap_check.cu:gemm-wrap-check; do
  tag=${t%%:*}; rest=${t#*:}; srcf=${rest%%:*}; bin=${rest#*:}
  docker run --rm --network none -v $W:/workspace $IMG bash -c "mkdir -p $C/$tag && cd $C/$tag && \
    nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -DLFM3_NATIVE_GEMM -o $bin \
    $C/$S/native/tests/$srcf $C/$S/native/src/safetensors.cpp > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
  echo "$tag build exit $?"
done
(cd $B && sha256sum $SP/lfm3-speech $SV/liquid-lfm3-server $SV/liquid-lfm3-worker $BENCH/audio-encoder-bench vcheck-a5/lfm3-vision-check a5-gemm/gemm-wrap-check) | tee binaries.sha256

step "1. gemm-wrap-check"
$X $C/a5-gemm/gemm-wrap-check > gemm-wrap.log 2>&1; echo "gemm-wrap-check exit $?: $(tail -2 gemm-wrap.log | tr '\n' ' ')"

step "2. byte-identical to ttft-speech-a2's x2"
LFM3_WORKSPACE=$W bash $TH/vision_rule6_speech.sh $SP > rule6-x2.log 2>&1
for k in dump edge; do
  python3 $B/$S/tools/compare_speech_outputs.py $B/ttft-speech-a2-rule6-x2/$k $B/$SP-rule6/$k a2-vs-a5-$k.json > /dev/null 2>&1
  python3 -c "import json; d=json.load(open('a2-vs-a5-$k.json')); print('$k against ttft-speech-a2 x2:', d.get('identical'), 'of', d.get('files'), '| differ', d.get('differ'), '| missing', d.get('missing'), '| extra', d.get('extra'))"
done
grep -E "^(e2e|split): " rule6-x2.log
mkdir -p hashes
$XF $C/$BENCH/audio-encoder-bench $M $C/$O/hashes/plain.json --repeat 1 $(cat $B/ttft-post/san-args.txt) > hashes/plain.log 2>&1; echo "bench exit $?"
python3 -c "
import json
a = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('$B/ttft-post/san/plain.json'))['clips']}
b = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('hashes/plain.json'))['clips']}
for k in a: print('bench', k, 'hashes', 'IDENTICAL to rule 4' if a[k] == b.get(k) else f'DIFFER {b.get(k)} vs {a[k]}')"

step "3. nothing else changes"
n=0; same=0
for d in $B/vision-ref/hf-5.20.2/float32/*/ $B/vision-ref/heldout/float32/*/; do
  case=$(basename $d); set_=$(basename $(dirname $(dirname $d)))
  $X $C/vcheck-a5/lfm3-vision-check $M $C/vision-ref/$set_/float32/$case $C/$O/vision-a5/$case --linear x2 --attention x2 > /dev/null 2>vision-$case.err \
    || echo "RUN FAILED $case: $(tail -1 vision-$case.err)"
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  for f in $B/ttft-exact/vision-main/$case/*.f32; do n=$((n+1)); cmp -s $f vision-a5/$case/$(basename $f) && same=$((same+1)) || echo "  DIFFER $case $(basename $f)"; done
done
echo "vision: $same of $n stage files identical to main's"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh $SP > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
sed -e "s|^EXEC=\"docker exec |EXEC=\"docker exec -e LFM3_AUDIO_ENCODER=fp32 |" -e "s|O=\$B/\$T-rule6; C=\$E/\$T-rule6|O=\$B/\$T-rule6-fp32; C=\$E/\$T-rule6-fp32|" \
  $TH/vision_rule6_speech.sh > rule6-fp32.sh
grep -q "LFM3_AUDIO_ENCODER=fp32" rule6-fp32.sh && grep -q "rule6-fp32" rule6-fp32.sh || echo "RULE6 SCRIPT EDIT FAILED"
LFM3_WORKSPACE=$W bash rule6-fp32.sh $SP > rule6-fp32.log 2>&1
python3 -c "
import json; d=json.load(open('$B/$SP-rule6-fp32/dump.json')); e=json.load(open('$B/$SP-rule6-fp32/edge.json'))
print('fp32: dump differ', len(d['differ']), sorted(d['differ']), '| missing', len(d['missing']), '| edge', e['identical'], 'of', e['files'])"
grep -E "^(e2e|split): " rule6-fp32.log
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_ENCODER=fp32 lfm3-profile-v1 \
  python3 $T/check_lfm3_speech_concurrent.py $C/$SP/lfm3-speech $M $C $L $C/$O/conc-fp32 $C/$O/engine-fp32.json $C/speed-final2/conc/alone \
  > engine-fp32.log 2>&1; echo "concurrent engine (fp32 mode) exit $?: $(tail -1 engine-fp32.log)"
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
mkdir -p san
for tool in memcheck racecheck synccheck initcheck; do
  extra=""; [ $tool = memcheck ] && extra="--leak-check full"
  $XF timeout -s TERM 21600 $SAN --tool $tool $extra --log-file $C/$O/san/$tool.log $C/$BENCH/audio-encoder-bench $M $C/$O/san/$tool.json \
    --repeat 1 $(cat $B/ttft-post/san-args.txt) > san/$tool.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tool exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY|RACECHECK SUMMARY' san/$tool.log | tr '\n' ' ')"
  python3 -c "
import json, sys
a = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('hashes/plain.json'))['clips']}
try: b = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('san/$tool.json'))['clips']}
except Exception as e: print('  $tool: NO OUTPUT', e); sys.exit()
for k in a: print('  $tool', k, 'hashes', 'IDENTICAL to plain' if b.get(k) == a[k] else f'DIFFER {b.get(k)} vs {a[k]}')"
done
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "A5-EXACT-FINISHED $(date +%T)"
