#!/usr/bin/env bash
# validation/ttft-rest-v1 stage 1 rules 2 and 3 on Thor. Source: build/rest-src (this stage's commit).
#   builds: the server and worker (rest-srv), audio-handler-bench (rest-bench);
#   quiet: rule 3, audio-handler-bench (20 repeats) on the five clips; time to first token, the deployed server
#     (merge3-srv) A1, rest-srv B1, A2, B2;
#   then rule 2 (not timed): audio in against mel3-speech, 8 and 32 at once, images, the image cache, TTS, the text gate.
# Usage (Thor host, copied to build/rest-s1/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=rest-s1; S=rest-src1
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; SV=rest-srv
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-input-tool|[a]udio-handler-bench"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
echo "== builds $(date +%T)"
docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "bash $C/$S/tools/thor/build_native.sh $C/$SV; rc=\$?; chown -R 1000:1000 $C/$SV $C/$SV.cmake.log $C/$SV.build.log; exit \$rc" > build-srv.log 2>&1; echo "server exit $?"
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include \
  -x cu -o audio-handler-bench $C/$S/native/tests/audio_handler_bench.cpp > build-bench.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"; echo "bench exit $?"
(cd $B && sha256sum $SV/liquid-lfm3-server $SV/liquid-lfm3-worker $O/audio-handler-bench merge3-srv/liquid-lfm3-server) | tee binaries.sha256
cp $B/mel3-run/clips.json clips.json
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
python3 -c "import json; [print(c['name'], c['wav']) for c in json.load(open('clips.json'))]" | while read name wav; do
  $X $C/$O/audio-handler-bench $wav 20 > bench-$name.json 2> bench-$name.err; echo "bench $name exit $?"
done
for run in merge3-srv:A1 $SV:B1 merge3-srv:A2 $SV:B2; do srv=${run%%:*}; tag=${run#*:}
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== $tag $srv $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $C/$srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
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
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or 'lfm3-speech' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
print(f"{'clip':9} {'utf8':>7} {'strict':>7} {'decode':>7} {'b64':>7} {'put':>7} {'handler w/o log_mel (new: one decode)':>38}")
seq90 = None
for c in ('short', 'median', 'typical', 'long', 'joined90'):
    b = json.load(open(f'bench-{c}.json'))
    seq = b['strict_json'] + b['parse_messages'] + b['clip_from_base64'] + b['cache_put']
    if c == 'joined90': seq90 = seq
    print(f"{c:9} {b['valid_utf8']:7.4f} {b['strict_json']:7.4f} {b['clip_from_base64']:7.4f} {b['base64_decode']:7.4f} {b['cache_put']:7.4f} {seq:38.4f}")
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
ok = seq90 <= 0.015
print(f"{'clip':9} {'deployed A1':>11} {'new B1':>9} {'deployed A2':>11} {'new B2':>9} {'saved':>8} texts")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; bb = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    texts = len({m[t][c]['text'] for t in m}) == 1; saved = statistics.mean(a) - statistics.mean(bb)
    good = max(bb) <= statistics.mean(a) and texts and (c != 'joined90' or saved >= 0.025); ok &= good
    print(f"{c:9} {a[0]:11.4f} {bb[0]:9.4f} {a[1]:11.4f} {bb[1]:9.4f} {saved:8.4f} {texts}{'' if good else '  FAIL'}")
print("RULE 3 (handler without log_mel at 90 s <= 0.015 s; at most the deployed at every clip; 90 s saves >= 0.025 s):", "PASS" if ok else "FAIL")
PY
echo "== rule 2 $(date +%T)"
$X python3 $T/check_lfm3_audio_in_server.py $C/$SV/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in-server.log 2>&1; echo "audio-in server exit $?: $(tail -1 audio-in-server.log | cut -c1-200)"
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
echo "REST-S1-FINISHED $(date +%T)"
