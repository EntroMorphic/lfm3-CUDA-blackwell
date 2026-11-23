#!/usr/bin/env bash
# The merged server's checks before it replaces the always-on service's (validation/always-on-v1; this round:
# validation/tcgen05-v2). build/merge5-srv, built by CMake from main's merge of tcgen05-v2, tcgen05 ON by default
# (native/ as tc3-on's, which passed the round's exactness rules). On a quiet Thor, the service stopped, the server on
# its own defaults:
#   timed: the speech stream check; time to first token on the five clips (1 untimed + 5 timed) against tc3-r5's ON runs;
#   exact: images through the server; the image cache; audio in against mel3-speech; 8 and 32 at once; TTS; the text gate.
# Usage (Thor host, copied to build/merge5-check/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=merge5-check
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=merge5-src; T=$C/$S/tools; SV=merge5-srv
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
XP="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
(cd $B && sha256sum $SV/liquid-lfm3-server $SV/liquid-lfm3-worker) | tee binaries.sha256
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
import collections, json, os, statistics
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
m = json.load(open('ttft.json'))['medians']; b = [json.load(open(os.path.expanduser('~/Projects/000-research/lfm3-omni-2B/liquid-CUDA/build/') + f'tc3-r5/{t}.json'))['medians'] for t in ('B1', 'B2')]
for c in m:
    print(f"{c:9} merge5 {m[c]['from_file_seconds']:.4f} s | tc3 ON B1 {b[0][c]['from_file_seconds']:.4f} B2 {b[1][c]['from_file_seconds']:.4f} | text same as tc3 ON: {m[c]['text'] == b[0][c]['text']}")
PY
echo "== exact checks $(date +%T)"
$XP python3 $T/check_lfm3_image_server.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-server.json > image-server.log 2>&1; echo "images exit $?: $(tail -1 image-server.log | cut -c1-150)"
$XP python3 $T/check_lfm3_image_cache.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-cache.json > image-cache.log 2>&1; echo "image cache exit $?: $(tail -1 image-cache.log | cut -c1-200)"
$XP python3 $T/check_lfm3_audio_in_server.py $C/$SV/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $L $C/$O/audio-in-work $C/$O/audio-in-server.json > audio-in.log 2>&1; echo "audio in exit $?: $(tail -1 audio-in.log | cut -c1-200)"
for k in 8 32; do
  $XP python3 $T/check_lfm3_audio_frontend_server.py $C/$SV/liquid-lfm3-server $C/mel3-speech/lfm3-speech $M $C/audioin-v2/cases.json $C/$O/at-once-$k-work $k $C/$O/at-once-$k.json --queue-size 40 --handlers 48 > at-once-$k.log 2>&1
  echo "at once $k exit $?: $(tail -1 at-once-$k.log | cut -c1-200)"
done
$XP python3 $T/check_lfm3_speech_server.py $C/$SV/liquid-lfm3-server $M $C $C/$O/speech-check.json > speech-check.log 2>&1; echo "TTS exit $?"
$XP python3 $T/check_lfm3_native_server.py --checkpoint $M --server $C/$SV/liquid-lfm3-server --worker $C/$SV/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1; echo "text gate exit $?"
)
echo "MERGE5-CHECK-FINISHED $(date +%T)"
