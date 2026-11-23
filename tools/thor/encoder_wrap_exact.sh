#!/usr/bin/env bash
# validation/encoder-wrap-v1 rules 1-4 (not timing; may overlap other work). Sources: build/wrap-src (the candidate),
# build/wrap-main-src (main at the branch point, for the machine-code comparison).
#   0. builds: the candidate and main, tcgen05 ON as deployed (wrap-on, wrap-main), by build_tcgen05_engine.sh;
#      gemm-wrap-check (with wrap-on's gemm_tcgen05.o);
#   1. rule 1: gemm-wrap-check;
#   2. rule 2: compare_sass.py, main against the candidate: liquid-lfm3-server, lfm3-speech, lfm3-vision-check;
#   3. rule 3: byte identity, tcgen05-v2's rule-2 suite on wrap-on against main's outputs;
#   4. rule 4: memcheck (leak check), synccheck, initcheck on wrap-on's lfm3-speech with the 90 s clip, cherry and two;
#      outputs against the plain run; memcheck on gemm-wrap-check.
# Usage (Thor host, copied to build/wrap-exact/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=wrap-exact; S=wrap-src
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$B/$S/tools/thor; ON=wrap-on
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
cd $B/$O || exit 1
# after ttft-rest-v2 stage 0's quiet window, which this run's builds and checks would disturb
until grep -q "^REST2-S0-FINISHED" $B/rest2-s0/run.log 2>/dev/null; do sleep 60; done
step() { echo "== $1 $(date +%T)"; }
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
step "0. builds"
LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh wrap-on on build/$S > build-on.log 2>&1; echo "build on exit $?"
LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh wrap-main on build/wrap-main-src > build-main.log 2>&1; echo "build main exit $?"
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 -std=c++20 -arch=sm_110 -DLFM_GEMM_TCGEN05 \
  -I $C/$S/native/include -o gemm-wrap-check $C/$S/native/tests/gemm_wrap_check.cu $C/$ON/gemm_tcgen05.o > build-check.log 2>&1; \
  rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build check exit $?"
(cd $B && sha256sum $ON/lfm3-speech $ON/liquid-lfm3-server $ON/liquid-lfm3-worker $ON/lfm3-vision-check wrap-main/liquid-lfm3-server \
  wrap-main/lfm3-speech $O/gemm-wrap-check) | tee binaries.sha256
step "1. rule 1: gemm-wrap-check"
$X $C/$O/gemm-wrap-check > wrap-check.log 2>&1; echo "gemm-wrap-check exit $?: $(tail -2 wrap-check.log | tr '\n' ' ')"
step "2. rule 2: machine code"
for b in liquid-lfm3-server lfm3-speech lfm3-vision-check; do
  python3 $B/$S/tools/compare_sass.py $B/wrap-main/$b $B/$ON/$b sass-$b.json /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/cu++filt \
    > sass-$b.log 2>&1; echo "sass $b exit $?: $(head -1 sass-$b.log)"
  grep -E "^  (DIFFER|missing)" sass-$b.log | head -20
done
step "3. rule 3: byte identity"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh $ON > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
LFM3_WORKSPACE=$W bash $TH/vision_rule6_speech.sh $ON > rule6.log 2>&1
for k in dump edge; do
  python3 $B/$S/tools/compare_speech_outputs.py $B/mel3-speech-rule6/$k $B/$ON-rule6/$k main-vs-on-$k.json > /dev/null 2>&1
  python3 -c "import json; d=json.load(open('main-vs-on-$k.json')); print('$k ON against main:', d.get('identical'), 'of', d.get('files'), '| differ', d.get('differ'), '| missing', d.get('missing'), '| extra', d.get('extra'))"
done
grep -E "^(e2e|split): " rule6.log
n=0; same=0
for d in $B/vision-ref/hf-5.20.2/float32/*/ $B/vision-ref/heldout/float32/*/; do
  case=$(basename $d); set_=$(basename $(dirname $(dirname $d)))
  $X $C/$ON/lfm3-vision-check $M $C/vision-ref/$set_/float32/$case $C/$O/vision-on/$case --linear x2 --attention x2 > /dev/null 2>vision-$case.err \
    || echo "RUN FAILED $case: $(tail -1 vision-$case.err)"
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  for f in $B/ttft-exact/vision-main/$case/*.f32; do n=$((n+1)); cmp -s $f vision-on/$case/$(basename $f) && same=$((same+1)) || echo "  DIFFER $case $(basename $f)"; done
done
echo "vision: $same of $n stage files identical to main's"
$X python3 $T/check_lfm3_audio_in_server.py $C/$ON/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in-server.log 2>&1; echo "audio-in server exit $?: $(tail -1 audio-in-server.log | cut -c1-200)"
for k in 8 32; do
  $X python3 $T/check_lfm3_audio_frontend_server.py $C/$ON/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $C/audioin-v2/cases.json \
    $C/$O/at-once-$k-work $k $C/$O/at-once-$k.json --queue-size 40 --handlers 48 > at-once-$k.log 2>&1
  echo "at once $k exit $?: $(tail -1 at-once-$k.log | cut -c1-200)"
done
$X python3 $T/check_lfm3_image_server.py $C/$ON/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-server.json > image-server.log 2>&1; echo "images exit $?: $(tail -1 image-server.log | cut -c1-150)"
$X python3 $T/check_lfm3_image_cache.py $C/$ON/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-cache.json > image-cache.log 2>&1; echo "image cache exit $?: $(tail -1 image-cache.log | cut -c1-200)"
$X python3 $T/check_lfm3_speech_server.py $C/$ON/liquid-lfm3-server $M $C $C/$O/speech-check.json > speech-check.log 2>&1; echo "TTS exit $?"
$X python3 $T/check_lfm3_native_server.py --checkpoint $M --server $C/$ON/liquid-lfm3-server --worker $C/$ON/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1; echo "text gate exit $?"
step "4. rule 4: sanitizers"
mkdir -p san
for tag in plain memcheck synccheck initcheck; do
  printf '%s %s\n%s %s\n%s %s\n' $C/quiet-w3-audio/req/joined90.req.json $C/$O/san/$tag-90 $C/vimg-e2e-a4/req/cherry.json $C/$O/san/$tag-cherry $C/vimg-e2e-a4/req/two.json $C/$O/san/$tag-two > san/list-$tag.txt
  if [ $tag = plain ]; then $X $C/$ON/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/plain.log 2>&1; echo "plain exit $?"; continue; fi
  extra=""; [ $tag = memcheck ] && extra="--leak-check full"
  $X timeout -s TERM 21600 $SAN --tool $tag $extra --log-file $C/$O/san/$tag.log $C/$ON/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/$tag.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tag exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY' san/$tag.log | tr '\n' ' ')"
  for k in 90 cherry two; do
    python3 - san/plain-$k san/$tag-$k <<'PY'
import json, sys, glob, os, filecmp
a, b = sys.argv[1], sys.argv[2]
ja, jb = json.load(open(a + '.json')), json.load(open(b + '.json'))
drop = lambda d: {k: v for k, v in d.items() if 'second' not in k and 'real_time' not in k}
files = [f for f in glob.glob(a + '.*') if not f.endswith('.json')]
same = all(filecmp.cmp(f, b + f[len(a):], shallow=False) for f in files)
print(f"  {os.path.basename(b)} against plain: json {'same' if drop(ja) == drop(jb) else 'DIFFER'}, {len(files)} other files {'same' if same else 'DIFFER'}")
PY
  done
done
$X timeout -s TERM 21600 $SAN --tool memcheck --leak-check full --log-file $C/$O/san/wrap-check-memcheck.log $C/$O/gemm-wrap-check > san/wrap-check-memcheck.out 2>&1
echo "gemm-wrap-check memcheck exit $?: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY' san/wrap-check-memcheck.log | tr '\n' ' ') $(tail -1 san/wrap-check-memcheck.out)"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "WRAP-EXACT-FINISHED $(date +%T)"
