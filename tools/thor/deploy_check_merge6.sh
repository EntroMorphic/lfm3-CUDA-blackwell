#!/usr/bin/env bash
# The merged server's checks before it replaces the always-on service's (validation/always-on-v1). This deploy: main
# 8120c19, the merges of encoder-wrap-v1, ttft-rest-v2 and pinned-uploads-v1 on e41e36c. Each round passed its own
# exactness rules against main separately; this checks the three together. build/merge6-srv by build_tcgen05_engine.sh
# (tcgen05 ON, as deployed; the server, worker and lfm3-speech/lfm3-vision-check from build/merge6-src).
#   exact (the service running; not timing): byte identity as in encoder-wrap-v1 rule 3 (the speech regression, the
#     rule-6 dump/edge/e2e/split suite, 713 vision stage files against main's); audio in; 8 and 32 at once; images; the
#     image cache; TTS; the text gate; memcheck (leak check) on lfm3-speech with the 90 s clip, cherry and two, outputs
#     against the plain run;
#   quiet (the service stopped): the speech stream check; time to first token on the five clips (1 untimed + 5 timed),
#     texts against tc3-r5's ON runs, times beside merge5-check's.
# Usage (Thor host, copied to build/merge6-check/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=merge6-check
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=merge6-src; T=$C/$S/tools; TH=$B/$S/tools/thor; SV=merge6-srv
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
step() { echo "== $1 $(date +%T)"; }
cd $B/$O || exit 1
step "build"
LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh $SV on build/$S > build.log 2>&1; echo "build exit $?"
(cd $B && sha256sum $SV/liquid-lfm3-server $SV/liquid-lfm3-worker $SV/lfm3-speech $SV/lfm3-vision-check) | tee binaries.sha256
grep -q "LIQUID_LFM3_TCGEN05:BOOL=ON" $B/$SV/cmake/CMakeCache.txt && echo "cache: LIQUID_LFM3_TCGEN05 ON" || echo "CACHE: TCGEN05 NOT ON"

step "exact: byte identity"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh $SV > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
LFM3_WORKSPACE=$W bash $TH/vision_rule6_speech.sh $SV > rule6.log 2>&1
for k in dump edge; do
  python3 $B/$S/tools/compare_speech_outputs.py $B/mel3-speech-rule6/$k $B/$SV-rule6/$k main-vs-merge6-$k.json > /dev/null 2>&1
  python3 -c "import json; d=json.load(open('main-vs-merge6-$k.json')); print('$k merge6 against main:', d.get('identical'), 'of', d.get('files'), '| differ', d.get('differ'), '| missing', d.get('missing'), '| extra', d.get('extra'))"
done
grep -E "^(e2e|split): " rule6.log
n=0; same=0
for d in $B/vision-ref/hf-5.20.2/float32/*/ $B/vision-ref/heldout/float32/*/; do
  case=$(basename $d); set_=$(basename $(dirname $(dirname $d)))
  $X $C/$SV/lfm3-vision-check $M $C/vision-ref/$set_/float32/$case $C/$O/vision/$case --linear x2 --attention x2 > /dev/null 2>vision-$case.err \
    || echo "RUN FAILED $case: $(tail -1 vision-$case.err)"
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  for f in $B/ttft-exact/vision-main/$case/*.f32; do n=$((n+1)); cmp -s $f vision/$case/$(basename $f) && same=$((same+1)) || echo "  DIFFER $case $(basename $f)"; done
done
echo "vision: $same of $n stage files identical to main's"

step "exact: the server"
$X python3 $T/check_lfm3_audio_in_server.py $C/$SV/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in.log 2>&1; echo "audio in exit $?: $(tail -1 audio-in.log | cut -c1-200)"
for k in 8 32; do
  $X python3 $T/check_lfm3_audio_frontend_server.py $C/$SV/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $C/audioin-v2/cases.json \
    $C/$O/at-once-$k-work $k $C/$O/at-once-$k.json --queue-size 40 --handlers 48 > at-once-$k.log 2>&1
  echo "at once $k exit $?: $(tail -1 at-once-$k.log | cut -c1-200)"
done
$X python3 $T/check_lfm3_image_server.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-server.json > image-server.log 2>&1; echo "images exit $?: $(tail -1 image-server.log | cut -c1-150)"
$X python3 $T/check_lfm3_image_cache.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-cache.json > image-cache.log 2>&1; echo "image cache exit $?: $(tail -1 image-cache.log | cut -c1-200)"
$X python3 $T/check_lfm3_speech_server.py $C/$SV/liquid-lfm3-server $M $C $C/$O/speech-check.json > speech-check.log 2>&1; echo "TTS exit $?"
$X python3 $T/check_lfm3_native_server.py --checkpoint $M --server $C/$SV/liquid-lfm3-server --worker $C/$SV/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1; echo "text gate exit $?"

step "exact: memcheck"
mkdir -p san
for tag in plain memcheck; do
  printf '%s %s\n%s %s\n%s %s\n' $C/quiet-w3-audio/req/joined90.req.json $C/$O/san/$tag-90 $C/vimg-e2e-a4/req/cherry.json $C/$O/san/$tag-cherry $C/vimg-e2e-a4/req/two.json $C/$O/san/$tag-two > san/list-$tag.txt
  if [ $tag = plain ]; then $X $C/$SV/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/plain.log 2>&1; echo "plain exit $?"; continue; fi
  $X timeout -s TERM 21600 $SAN --tool memcheck --leak-check full --log-file $C/$O/san/$tag.log $C/$SV/lfm3-speech $M --batch $C/$O/san/list-$tag.txt > san/$tag.out 2>&1; rc=$?
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

step "quiet"
cp $B/tc2-quiet/clips.json clips.json
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
echo "== the stream check $(date +%T)"
$X python3 $T/check_lfm3_speech_stream.py $C/$SV/liquid-lfm3-server $M $C $L $C/$O/stream-check.json > stream-check.log 2>&1; echo "stream check exit $?: $(tail -1 stream-check.log | cut -c1-200)"
echo "== time to first token $(date +%T)"
$X python3 $T/measure_lfm3_audio_ttft.py $C/$SV/liquid-lfm3-server $M $C/$O/clips.json $C/$O/ttft.json 5 > ttft.log 2>&1; echo "ttft exit $?"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
python3 - <<'PY'
import collections, json, os
rows = collections.defaultdict(list)
for line in open('samples.log'):
    p = line.rstrip('\n').split(' ', 2)
    if len(p) >= 2: rows[p[0]].append(p[1:])
n = foreign = 0
for t, r in rows.items():
    if not any(x[0] == 'T' for x in r): continue
    n += 1
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]; dock = [x[1] for x in r if x[0] == 'D' and len(x) > 1]
    foreign += any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or 'lfm3-speech' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
B = os.path.expanduser('~/Projects/000-research/lfm3-omni-2B/liquid-CUDA/build/')
m = json.load(open('ttft.json'))['medians']; m5 = json.load(open(B + 'merge5-check/ttft.json'))['medians']
b = [json.load(open(B + f'tc3-r5/{t}.json'))['medians'] for t in ('B1', 'B2')]
for c in m:
    print(f"{c:9} merge6 {m[c]['from_file_seconds']:.4f} s | merge5 (deployed) {m5[c]['from_file_seconds']:.4f} | text same as tc3 ON: {m[c]['text'] == b[0][c]['text'] == b[1][c]['text']}")
PY
)
echo "MERGE6-CHECK-FINISHED $(date +%T)"
