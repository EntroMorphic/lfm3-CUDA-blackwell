#!/usr/bin/env bash
# validation/scratch-keep-v1 rules 1, 2 and 7 (not timing; may overlap other work). A = build/merge6-srv (main 8120c19,
# deployed as 0012be944a78); B = keep-on, built here from build/keep-src the same way (build_tcgen05_engine.sh, ON).
#   0. build B;
#   1a. byte identity on B: the speech regression; the rule-6 suite (dump, edge, e2e, split) against main's; audio in; 8 and
#       32 at once; images; the image cache; TTS; the text gate;
#   1b. order: one lfm3-speech --batch list (90 s, short, typical, 90 s, cherry, short, 35 s) on A and on B, every output
#       file compared;
#   2. memcheck (leak check), synccheck, initcheck on B's lfm3-speech over 1b's list, outputs against B's plain run;
#   7. compare_sass.py A against B: liquid-lfm3-server, lfm3-speech, lfm3-vision-check.
# Usage (Thor host, copied to build/keep-exact/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=keep-exact; S=keep-src
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$B/$S/tools/thor; ON=keep-on; A=merge6-srv
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
cd $B/$O || exit 1
step() { echo "== $1 $(date +%T)"; }
# compares two lfm3-speech output prefixes: the json without its timing fields, and every other file byte for byte
cmp_out() {
  python3 - "$1" "$2" <<'PY'
import json, sys, glob, os, filecmp
a, b = sys.argv[1], sys.argv[2]
ja, jb = json.load(open(a + '.json')), json.load(open(b + '.json'))
drop = lambda d: {k: v for k, v in d.items() if 'second' not in k and 'real_time' not in k}
files = [f for f in glob.glob(a + '.*') if not f.endswith('.json')]
same = all(os.path.exists(b + f[len(a):]) and filecmp.cmp(f, b + f[len(a):], shallow=False) for f in files)
extra = len([f for f in glob.glob(b + '.*') if not f.endswith('.json')]) - len(files)
print(f"  {os.path.basename(b)} against {os.path.basename(a)}: json {'same' if drop(ja) == drop(jb) else 'DIFFER'}, "
      f"{len(files)} other files {'same' if same and not extra else 'DIFFER'}")
PY
}
step "0. build"
LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh $ON on build/$S > build-on.log 2>&1; echo "build on exit $?"
(cd $B && sha256sum $ON/liquid-lfm3-server $ON/liquid-lfm3-worker $ON/lfm3-speech $ON/lfm3-vision-check $A/liquid-lfm3-server \
  $A/lfm3-speech $A/lfm3-vision-check) | tee binaries.sha256

step "7. rule 7: machine code"
for b in liquid-lfm3-server lfm3-speech lfm3-vision-check; do
  python3 $B/$S/tools/compare_sass.py $B/$A/$b $B/$ON/$b sass-$b.json /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/cu++filt \
    > sass-$b.log 2>&1; echo "sass $b exit $?: $(head -1 sass-$b.log)"
  grep -E "^  (DIFFER|missing)" sass-$b.log | head -20
done

step "1a. rule 1a: byte identity"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh $ON > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
LFM3_WORKSPACE=$W bash $TH/vision_rule6_speech.sh $ON > rule6.log 2>&1
for k in dump edge; do
  python3 $B/$S/tools/compare_speech_outputs.py $B/mel3-speech-rule6/$k $B/$ON-rule6/$k main-vs-on-$k.json > /dev/null 2>&1
  python3 -c "import json; d=json.load(open('main-vs-on-$k.json')); print('$k ON against main:', d.get('identical'), 'of', d.get('files'), '| differ', d.get('differ'), '| missing', d.get('missing'), '| extra', d.get('extra'))"
done
grep -E "^(e2e|split): " rule6.log
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

step "1b. rule 1b: order"
Q=$C/quiet-w3-audio/req; I=$C/vimg-e2e-a4/req
ITEMS="90:$Q/joined90.req.json short:$Q/short.req.json typical:$Q/typical.req.json 90b:$Q/joined90.req.json cherry:$I/cherry.json short2:$Q/short.req.json long:$Q/long.req.json"
mkdir -p order san
list() { for it in $ITEMS; do echo "${it#*:} $C/$O/$1-${it%%:*}"; done > $2; }
list order/A order/list-A.txt; list order/B order/list-B.txt
$X $C/$A/lfm3-speech $M --batch $C/$O/order/list-A.txt > order/A.log 2>&1; echo "order A exit $?"
$X $C/$ON/lfm3-speech $M --batch $C/$O/order/list-B.txt > order/B.log 2>&1; echo "order B exit $?"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
for it in $ITEMS; do cmp_out order/A-${it%%:*} order/B-${it%%:*}; done

step "2. rule 2: sanitizers"
list san/plain san/list-plain.txt
$X $C/$ON/lfm3-speech $M --batch $C/$O/san/list-plain.txt > san/plain.log 2>&1; echo "plain exit $?"
for tag in memcheck synccheck initcheck; do
  list san/$tag san/list-$tag.txt
  extra=""; [ $tag = memcheck ] && extra="--leak-check full"
  $X timeout -s TERM 21600 $SAN --tool $tag $extra --log-file $C/$O/san/$tag.log $C/$ON/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/$tag.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tag exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY' san/$tag.log | tr '\n' ' ')"
  for it in $ITEMS; do cmp_out san/plain-${it%%:*} san/$tag-${it%%:*}; done
done
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "KEEP-EXACT-FINISHED $(date +%T)"
