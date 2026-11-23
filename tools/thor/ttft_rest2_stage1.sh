#!/usr/bin/env bash
# validation/ttft-rest-v2 stage 1 on Thor. Source: build/rest2-src1.
#   builds: the server as deployed (tcgen05 ON) by build_tcgen05_engine.sh (rest2-on); the same tree with
#     -DLFM3_TTFT_TRACE (rest2-trace1);
#   rule 2 (not timing; may overlap other exactness work): the server's audio-in, 8 and 32 at once, images, the image
#     cache, TTS, the text gate;
#   rule 3 (quiet; after encoder-wrap-v1's quiet window): time to first token on the five clips, A = deployed
#     (merge5-srv) and B = rest2-on, A1 B1 A2 B2; then one trace pass of the new tree (reported).
# Usage (Thor host, copied to build/rest2-s1/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=rest2-s1; S=rest2-src1
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$B/$S/tools/thor; ON=rest2-on
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[t]race_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[l]fm3-vision-check|[a]lloc-|[g]emm-wrap-check"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
echo "== builds $(date +%T)"
LFM3_WORKSPACE=$W bash $TH/build_tcgen05_engine.sh $ON on build/$S > build-on.log 2>&1; echo "build on exit $?"
mkdir -p $B/rest2-trace1
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C/rest2-trace1
  cmake -S $C/$S/native -B cmake -DLIQUID_LFM3_NATIVE_GEMM=ON -DLIQUID_LFM3_TCGEN05=ON \
    -DCMAKE_CUDA_FLAGS=-DLFM3_TTFT_TRACE -DCMAKE_CXX_FLAGS=-DLFM3_TTFT_TRACE > cmake.log 2>&1
  cmake --build cmake --target liquid-lfm3-server -j4 > cmake-build.log 2>&1
  cp cmake/liquid-lfm3-server .
  ! grep -iE ' error|warning' cmake-build.log
  chown -R 1000:1000 $C/rest2-trace1"; echo "trace build exit $?"
(cd $B && sha256sum $ON/liquid-lfm3-server $ON/liquid-lfm3-worker rest2-trace1/liquid-lfm3-server merge5-srv/liquid-lfm3-server) | tee binaries.sha256
echo "== rule 2 $(date +%T)"
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
echo "== rule 2 done $(date +%T)"
until grep -q "^WRAP-QUIET-FINISHED" $B/wrap-quiet/run.log 2>/dev/null; do sleep 60; done
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
srv() { [ ${1:0:1} = A ] && echo $C/merge5-srv/liquid-lfm3-server || echo $C/$ON/liquid-lfm3-server; }
for tag in A1 B1 A2 B2; do
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== $tag $(date +%T)"
  $X python3 $T/trace_lfm3_audio_ttft.py $(srv $tag) $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
bash $R/tools/thor/power_state.sh power-T.json power-start.json > /dev/null || echo "POWER DIFFERS before T"
echo "== T $(date +%T)"
$X env LFM3_TTFT_TRACE=$C/$O/trace-T.txt python3 $T/trace_lfm3_audio_ttft.py $C/rest2-trace1/liquid-lfm3-server $M $C/$O/clips.json $C/$O/T.json 5 > T.log 2>&1; echo "T exit $?"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
python3 - <<'PY'
import collections, json, statistics
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
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
ok = bool(n) and not foreign
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2', 'T')}
print(f"{'clip':10} {'A1':>8} {'B1':>8} {'A2':>8} {'B2':>8} {'saved':>8} {'B/A':>6} texts   (from the first byte, s)")
for c in m['A1']:
    a = [m[t][c]['total'] for t in ('A1', 'A2')]; b = [m[t][c]['total'] for t in ('B1', 'B2')]
    texts = len({m[t][c]['text'] for t in m}) == 1 and all(m[t][c]['texts_identical'] for t in m)
    saved = statistics.mean(a) - statistics.mean(b)
    good = statistics.mean(b) <= statistics.mean(a) and texts and (c != 'joined90' or saved >= 0.006); ok &= good
    print(f"{c:10} {a[0]:8.4f} {b[0]:8.4f} {a[1]:8.4f} {b[1]:8.4f} {saved:8.4f} {statistics.mean(b) / statistics.mean(a):6.3f} {texts}{'' if good else '  FAIL'}")
print("RULE 3:", "PASS" if ok else "FAIL")
print("trace of the new tree (T, medians, ms):")
for k in ('send_to_head', 'body', 'strict_json', 'validate', 'encode_speech', 'log_mel', 'encoder', 'lm_forward_call', 'total'):
    print(f"  {k:16} " + " ".join(f"{1000 * m['T'][c].get(k, float('nan')):8.2f}" for c in m['T']))
PY
echo "REST2-S1-FINISHED $(date +%T)"
