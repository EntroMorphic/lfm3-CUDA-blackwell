#!/usr/bin/env bash
# validation/pinned-uploads-v1 rules 2 and 5 (timing: a quiet Thor, the service stopped, the samplers), after
# build/pu-exact. A = main as deployed (build/merge5-srv, f19f4ee40708), B = the candidate (build/pu-on).
#   rule 2: time to first token on tcgen05-v2's five clips (measure_lfm3_audio_ttft.py, 1 untimed + 5 timed), A1 B1 A2 B2;
#           images: measure_lfm3_image_turns.py on cherry and two (4 conversations, the first discarded), A1 B1 A2 B2;
#   rule 5: measure_lfm3_audio_device_memory.py (the short clip, then the 90 s clip twice), A1 B1 A2 B2.
# Usage (Thor host, copied to build/pu-quiet/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=pu-quiet; S=pu-src
M=/workspace/models/LFM3-Omni-2B; T=$C/$S/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[l]fm3-vision-check|[a]lloc-"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
read SHORT LONG < <(python3 -c "import json; c = {x['name']: x['wav'] for x in json.load(open('clips.json'))}; print(c['short'], c['joined90'])")
until grep -q "^PU-EXACT-FINISHED" $B/pu-exact/run.log 2>/dev/null; do sleep 60; done
(cd $B && sha256sum merge5-srv/liquid-lfm3-server pu-on/liquid-lfm3-server) | tee binaries.sha256
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
srv() { [ ${1:0:1} = A ] && echo $C/merge5-srv/liquid-lfm3-server || echo $C/pu-on/liquid-lfm3-server; }
for tag in A1 B1 A2 B2; do
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== audio $tag $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $(srv $tag) $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "audio $tag exit $?"
done
for n in cherry two; do for tag in A1 B1 A2 B2; do
  echo "== image $n $tag $(date +%T)"
  $X python3 $T/measure_lfm3_image_turns.py $(srv $tag) $M $C/vimg-e2e-a1/req/$n.json $C/$O/img-$n-$tag.json 4 > img-$n-$tag.log 2>&1; echo "image $n $tag exit $?"
done; done
echo "== device memory, ABAB $(date +%T)"
$X python3 $T/measure_lfm3_audio_device_memory.py $SHORT $LONG $M $C/$O/device-memory.json A1=$(srv A1) B1=$(srv B1) A2=$(srv A2) B2=$(srv B2) \
  > device-memory.log 2>&1; echo "device memory exit $?"
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
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
print(f"{'item':12} {'A1':>8} {'B1':>8} {'A2':>8} {'B2':>8} {'bound':>8} {'B/A':>7} texts   (time to first text, s)")
def row(name, a, b, texts):
    global ok
    bound = statistics.mean(a) * 1.01; good = max(b) <= bound and texts; ok &= good
    print(f"{name:12} {a[0]:8.4f} {b[0]:8.4f} {a[1]:8.4f} {b[1]:8.4f} {bound:8.4f} {statistics.mean(b) / statistics.mean(a):7.3f} {texts}{'' if good else '  OVER'}")
    return statistics.mean(b) / statistics.mean(a)
ratio = {}
for c in m['A1']:
    ratio[c] = row(c, [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')], [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')],
                   len({m[t][c]['text'] for t in m}) == 1)
for n_ in ('cherry', 'two'):
    d = {t: json.load(open(f'img-{n_}-{t}.json')) for t in ('A1', 'B1', 'A2', 'B2')}
    texts = len({d[t]['runs'][1][0]['text'] for t in d}) == 1 and all(d[t]['texts_repeatable'] for t in d)
    ratio[n_] = row(f'img {n_}', [d[t]['medians'][0]['first_text'] for t in ('A1', 'A2')], [d[t]['medians'][0]['first_text'] for t in ('B1', 'B2')], texts)
pred = ratio.get('joined90', 9) <= 0.97; ok &= pred
print(f"joined90 prediction (mean B <= 0.97 x mean A): {'MET' if pred else 'NOT MET'} ({ratio.get('joined90', float('nan')):.3f})")
print("RULE 2:", "PASS" if ok else "FAIL")
PY
echo "== device memory"; tail -12 device-memory.log
echo "PU-QUIET-FINISHED $(date +%T)"
