#!/usr/bin/env bash
# validation/tcgen05-v2 rules 1-3 (not timing; may overlap other work). Source: build/tc3-src (amendment 2).
#   0. builds: ON (tc3-on) and OFF (tc3-off) by build_tcgen05_engine.sh;
#   1. rule 1: lfm3-tcgen05-gemm-check, every shape (the audio encoder's added), the default M list;
#   2. rule 2: lfm3-speech ON: TTS regression; rule-6 audio-in files against mel3-speech-rule6 (main's), e2e, split;
#      the vision encoder ON against main's stage files; the server ON: audio in and 8/32 at once against mel3-speech,
#      images, the image cache, TTS, the text gate;
#   3. rule 3: memcheck (leak check), synccheck, initcheck on lfm3-speech ON with a 4,096-token text prefill and the
#      90 s clip; outputs against the plain run.
# Usage (Thor host, copied to build/tc3-exact/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=tc3-exact; S=tc3-src
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$B/$S/tools/thor; ON=tc3-on
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
cd $B/$O || exit 1
step() { echo "== $1 $(date +%T)"; }
step "0. builds"
for m in on off; do LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh tc3-$m $m build/$S > build-$m.log 2>&1; echo "build $m exit $?"; done
(cd $B && sha256sum tc3-on/lfm3-speech tc3-on/liquid-lfm3-server tc3-on/liquid-lfm3-worker tc3-on/lfm3-vision-check tc3-on/lfm3-tcgen05-gemm-check \
  tc3-off/lfm3-speech tc3-off/liquid-lfm3-server tc3-off/liquid-lfm3-worker) | tee binaries.sha256
step "1. rule 1: the entry point"
$X $C/$ON/lfm3-tcgen05-gemm-check > gemm-check.jsonl 2> gemm-check.err; echo "gemm check exit $?: $(tail -1 gemm-check.jsonl)"
python3 -c "
import json
rows = [json.loads(l) for l in open('gemm-check.jsonl') if l.startswith('{') and 'summary' not in l]
bad = [r for r in rows if not r.get('pass')]
print('cases', len(rows), 'failed', len(bad), 'audio-encoder cases', sum(1 for r in rows if str(r.get('shape', '')).startswith('a.')), bad[:3])"
step "2. rule 2: byte identity"
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
step "3. rule 3: sanitizers"
mkdir -p san
for tag in plain memcheck synccheck initcheck; do
  printf '%s %s\n%s %s\n' $C/alloc-text/req/L4096.req.json $C/$O/san/$tag-L4096 $C/quiet-w3-audio/req/joined90.req.json $C/$O/san/$tag-90 > san/list-$tag.txt
  if [ $tag = plain ]; then $X $C/$ON/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/plain.log 2>&1; echo "plain exit $?"; continue; fi
  extra=""; [ $tag = memcheck ] && extra="--leak-check full"
  $X timeout -s TERM 21600 $SAN --tool $tag $extra --log-file $C/$O/san/$tag.log $C/$ON/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/$tag.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tag exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY' san/$tag.log | tr '\n' ' ')"
  for k in L4096 90; do
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
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "TC3-EXACT-FINISHED $(date +%T)"
