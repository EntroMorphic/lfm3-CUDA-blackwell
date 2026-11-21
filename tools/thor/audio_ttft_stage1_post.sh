#!/usr/bin/env bash
# validation/audio-ttft-v1 stage 1, after the quiet window (build/ttft-s1q):
#   Q. quiet (the always-on service stopped, the 1 s sampler, the power state): the batch-server check with
#      LFM3_AUDIO_ENCODER=fp32 (its spoken-q cases differ from the old references under x2 by design; in fp32 mode they
#      must not; it has timing sub-checks, hence quiet);
#   4. rule 4 as amendment 3 states it: memcheck (leaks), racecheck, synccheck, initcheck on audio-encoder-bench built
#      from build/ttft-src-a3 (e611281), the short and 90 s clips, --repeat 1; 0 errors and mel_fnv1a, encoded_fnv1a
#      equal to the plain run's. Not timed; the service runs again by then.
# Usage (Thor host, copied to build/ttft-post/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=ttft-post
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=ttft-src-a3; SV=ttft-srv; BENCH=ttft-bench-a3
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
until grep -q "^TTFT-S1Q-FINISHED" $B/ttft-s1q/run.log 2>/dev/null; do sleep 60; done
echo "== the quiet window finished; build the bench from $S $(date +%T)"
LFM3_WORKSPACE=$W bash $B/$S/tools/thor/build_audio_bench.sh $BENCH build/$S > build-bench.log 2>&1 || { echo "BENCH BUILD FAILED: $(tail -2 build-bench.log)"; }
(cd $B && sha256sum $SV/liquid-lfm3-server $BENCH/audio-encoder-bench) | tee binaries.sha256
cp $B/ttft-s1q/clip-args.txt . ; grep -oE "(short|joined90)=[^ ]+" clip-args.txt | tr '\n' ' ' > san-args.txt; cat san-args.txt; echo

(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
echo "== Q. the batch server, LFM3_AUDIO_ENCODER=fp32 $(date +%T)"
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_ENCODER=fp32 lfm3-profile-v1 \
  python3 $C/ttft-src/tools/check_lfm3_speech_batch_server.py $C/$SV/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L \
  $C/$O/batch-server-fp32.json > batch-server-fp32.log 2>&1; echo "batch server fp32 exit $?"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
python3 - <<'PY'
import collections
foreign = more = defunct = n = 0
rows = collections.defaultdict(list)
for line in open('samples.log'):
    p = line.split(' ', 2)
    if len(p) >= 2: rows[p[0]].append(p[1:])
for t, r in rows.items():
    if not any(x[0].strip() == 'T' for x in r): continue
    n += 1
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]
    dock = [x[1].strip() for x in r if x[0] == 'D' and len(x) > 1]
    if any('<defunct>' in q for q in procs): defunct += 1
    if any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(('kin_' in q or 'venvs/hf-' in q or 'nsys' in q or 'compute-sanitizer' in q) for q in procs): foreign += 1
    if sum('liquid-lfm3-server' in q and '/.local/lib/' in q for q in procs): more += 1
print(f"sampler: {n} samples; foreign in {foreign}; the deployed service in {more}; defunct (allowed) in {defunct}")
print("MONITOR:", "CLEAN" if n and not foreign and not more else "NOT CLEAN")
PY
)

echo "== 4. rule 4 on audio-encoder-bench $(date +%T)"
mkdir -p san
$X $C/$BENCH/audio-encoder-bench $M $C/$O/san/plain.json --repeat 1 $(cat san-args.txt) > san/plain.log 2>&1; echo "plain exit $?"
for tool in memcheck racecheck synccheck initcheck; do
  extra=""; [ $tool = memcheck ] && extra="--leak-check full"
  $X timeout -s TERM 21600 $SAN --tool $tool $extra --log-file $C/$O/san/$tool.log $C/$BENCH/audio-encoder-bench $M $C/$O/san/$tool.json \
    --repeat 1 $(cat san-args.txt) > san/$tool.out 2>&1; rc=$?
  docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
  echo "$tool exit $rc: $(grep -h -E 'ERROR SUMMARY|LEAK SUMMARY|RACECHECK SUMMARY' san/$tool.log | tr '\n' ' ')"
  python3 -c "
import json, sys
a = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('san/plain.json'))['clips']}
try: b = {c['name']: (c['mel_fnv1a'], c['encoded_fnv1a']) for c in json.load(open('san/$tool.json'))['clips']}
except Exception as e: print('  $tool: NO OUTPUT', e); sys.exit()
for k in a: print('  $tool', k, 'hashes', 'IDENTICAL to plain' if b.get(k) == a[k] else f'DIFFER {b.get(k)} vs {a[k]}')"
done
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
echo "TTFT-POST-FINISHED $(date +%T)"
