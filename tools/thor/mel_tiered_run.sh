#!/usr/bin/env bash
# validation/mel-tiered-v1 rules 2-4 (rule 1: mel_tiered_rule1.sh). Source: build/mel3-rule-src (this round).
#   builds: audio-encoder-bench (mel3-bench), server and worker (mel3-srv), lfm3-speech (mel3-speech);
#   the edge clip with the most second-tier frames (audio-mel-check per clip, not timed);
#   quiet window: rule 2, audio-encoder-bench log_mel alone (--repeat 5) with LFM3_AUDIO_FEATURES=exact-f2 and exact-f3,
#     4 threads, the five clips and that edge clip; exact-f3 at 1 thread (reported); rule 3, time to first token
#     (measure_lfm3_audio_ttft.py, 1 untimed + 5 timed) of the deployed server (merge2-srv, F2 by default) and this
#     round's (mel3-srv, F3 by default), ABAB, no LFM3_AUDIO_FEATURES set;
#   then rule 4 (not timed): the TTS regression; rule-6 audio-in files against a6-speech-rule6, e2e and split; the
#     server's audio in, 8 and 32 at once, the text gate.
# Usage (Thor host, copied to build/mel3-run/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=mel3-run; S=mel3-rule-src
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$B/$S/tools/thor
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-input-tool"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
echo "== builds $(date +%T)"
LFM3_WORKSPACE=$W bash $TH/build_audio_bench.sh mel3-bench build/$S > build-bench.log 2>&1; echo "bench exit $?"
docker run --rm --network none -v $B/$S/native:/src/native:ro -v $W:/workspace $IMG bash -c \
  "bash $C/$S/tools/thor/build_native.sh $C/mel3-srv; rc=\$?; chown -R 1000:1000 $C/mel3-srv $C/mel3-srv.cmake.log $C/mel3-srv.build.log; exit \$rc" > build-srv.log 2>&1; echo "server exit $?"
LFM3_WORKSPACE=$W bash $TH/build_speech.sh mel3-speech build/$S > build-speech.log 2>&1; echo "lfm3-speech exit $?"
(cd $B && sha256sum mel3-bench/audio-encoder-bench mel3-srv/liquid-lfm3-server mel3-srv/liquid-lfm3-worker mel3-speech/lfm3-speech merge2-srv/liquid-lfm3-server) | tee binaries.sha256
cp $B/a6-run/clips.json clips.json; cp $B/a6-run/clip-args.txt clip-args.txt
echo "== the edge clip with the most second-tier frames $(date +%T)"
best=""; bestn=-1
while read name wav rest; do
  echo "$name $wav" > one.txt; mkdir -p edge-out
  n=$($X $C/mel3-round/audio-mel-check --threads 1 --transform f3 --write $C/$O/edge-out $C/$O/one.txt $C/$O/edge-one.json 2>&1 | tail -1 | sed -n 's/.*recomputed by the second tier \([0-9]*\).*/\1/p')
  echo "  $name: $n"; [ "${n:-0}" -gt $bestn ] && bestn=$n && best="$name=$wav"
done < $B/mel3-rule1/edge.txt
echo "chosen: $best ($bestn frames)"; echo "$best" > edge-arg.txt
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
pw() { bash $R/tools/thor/power_state.sh power-$1.json power-start.json > /dev/null || echo "POWER DIFFERS before $1"; }
for v in exact-f2:4 exact-f3:4 exact-f3:1; do mode=${v%%:*}; th=${v#*:}
  pw bench-$mode-$th; echo "== bench $mode, $th threads $(date +%T)"
  docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=$mode lfm3-profile-v1 \
    $C/mel3-bench/audio-encoder-bench $M $C/$O/bench-$mode-$th.json --repeat 5 --mel-threads $th $(cat clip-args.txt) $(cat edge-arg.txt) > bench-$mode-$th.log 2>&1
  echo "bench $mode $th exit $?"
done
for run in merge2-srv:A1 mel3-srv:B1 merge2-srv:A2 mel3-srv:B2; do srv=${run%%:*}; tag=${run#*:}
  pw $tag; echo "== $tag $srv $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $C/$srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
pw end
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
b = {k: {c['name']: c for c in json.load(open(f'bench-{k}.json'))['clips']} for k in ('exact-f2-4', 'exact-f3-4', 'exact-f3-1')}
print(f"{'clip':28} {'F2 4t':>8} {'F3 4t':>8} {'F3 1t':>8}  mel hash F2 == F3")
ok2 = True
for c in b['exact-f2-4']:
    f2, f3, f31 = (b[k][c]['log_mel_seconds'] for k in ('exact-f2-4', 'exact-f3-4', 'exact-f3-1'))
    same = b['exact-f2-4'][c]['mel_fnv1a'] == b['exact-f3-4'][c]['mel_fnv1a'] == b['exact-f3-1'][c]['mel_fnv1a']
    ok2 &= f3 <= f2 and same
    print(f"{c:28} {f2:8.4f} {f3:8.4f} {f31:8.4f}  {same}")
ok2 &= b['exact-f3-4']['joined90']['log_mel_seconds'] <= 0.060
print("RULE 2 (F3 at 90 s at most 0.060 s; F3 at most F2 at every clip; hashes equal):", "PASS" if ok2 else "FAIL")
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
ok3 = True
print(f"{'clip':9} {'deployed A1':>11} {'F3 B1':>9} {'deployed A2':>11} {'F3 B2':>9} {'saved':>8} texts")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; bb = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    texts = len({m[t][c]['text'] for t in m}) == 1
    ok = max(bb) <= statistics.mean(a) and texts; ok3 &= ok
    print(f"{c:9} {a[0]:11.4f} {bb[0]:9.4f} {a[1]:11.4f} {bb[1]:9.4f} {statistics.mean(a) - statistics.mean(bb):8.4f} {texts}{'' if ok else '  OVER'}")
print("RULE 3 (the new server at most the deployed at every clip; texts identical):", "PASS" if ok3 else "FAIL")
PY
echo "MEL3-QUIET-FINISHED $(date +%T)"
echo "== rule 4 $(date +%T)"
LFM3_WORKSPACE=$W bash $TH/speech_regression_check.sh mel3-speech > speech-regression.log 2>&1
echo "speech regression: $(grep -c '^SAME' speech-regression.log) SAME, $(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log) differ or failed"
LFM3_WORKSPACE=$W bash $TH/vision_rule6_speech.sh mel3-speech > rule6.log 2>&1
for k in dump edge; do
  python3 $B/$S/tools/compare_speech_outputs.py $B/a6-speech-rule6/$k $B/mel3-speech-rule6/$k a6-vs-mel3-$k.json > /dev/null 2>&1
  python3 -c "import json; d=json.load(open('a6-vs-mel3-$k.json')); print('$k against a6-speech-rule6:', d.get('identical'), 'of', d.get('files'), '| differ', d.get('differ'), '| missing', d.get('missing'), '| extra', d.get('extra'))"
done
grep -E "^(e2e|split): " rule6.log
$X python3 $T/check_lfm3_audio_in_server.py $C/mel3-srv/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in-server.log 2>&1; echo "audio-in server exit $?: $(tail -1 audio-in-server.log | cut -c1-200)"
for k in 8 32; do
  $X python3 $T/check_lfm3_audio_frontend_server.py $C/mel3-srv/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $C/audioin-v2/cases.json \
    $C/$O/at-once-$k-work $k $C/$O/at-once-$k.json --queue-size 40 --handlers 48 > at-once-$k.log 2>&1
  echo "at once $k exit $?: $(tail -1 at-once-$k.log | cut -c1-200)"
done
$X python3 $T/check_lfm3_native_server.py --checkpoint $M --server $C/mel3-srv/liquid-lfm3-server --worker $C/mel3-srv/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1; echo "text gate exit $?"
echo "MEL3-ALL-FINISHED $(date +%T)"
