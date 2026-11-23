#!/usr/bin/env bash
# validation/tcgen05-v2 amendment 1: rule 5's audio part again, fresh. Quiet: time to first token, OFF A1, ON B1, OFF
# A2, ON B2 (measure_lfm3_audio_ttft.py, 1 untimed + 5 timed). Then, not timed: nsys on the ON server, 2 requests per
# clip, counting gemm_1cta / gemm_2cta (tcgen05) kernels. The amended verdict per clip.
# Usage (Thor host, copied to build/tc2-r5/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=tc2-r5; S=tc2-src
M=/workspace/models/LFM3-Omni-2B; T=$C/$S/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
(cd $B && sha256sum tc2-on/liquid-lfm3-server tc2-off/liquid-lfm3-server) | tee binaries.sha256
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
for run in off:A1 on:B1 off:A2 on:B2; do m=${run%%:*}; tag=${run#*:}
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== $tag ($m) $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $C/tc2-$m/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
echo "== the launch check (not timed) $(date +%T)"
python3 -c "import json; [print(c['name'], c['wav']) for c in json.load(open('clips.json'))]" | while read name wav; do
  w=${wav/\/workspace\//$W/}
  python3 $B/$S/tools/profile_lfm3_audio_server.py /usr/local/bin/nsys $B/tc2-on/liquid-lfm3-server $W/models/LFM3-Omni-2B $w 2 40 prof-$name > prof-$name.log 2>&1
  nsys export --type sqlite --force-overwrite true -o prof-$name.sqlite prof-$name.nsys-rep > /dev/null 2>&1
  python3 -c "
import sqlite3; c = sqlite3.connect('prof-$name.sqlite'); n = dict(c.execute('select id, value from StringIds'))
k = [n[i] for (i,) in c.execute('select shortName from CUPTI_ACTIVITY_KIND_KERNEL')]
print('$name', sum(1 for x in k if x in ('gemm_1cta', 'gemm_2cta')), 'tcgen05 kernels of', len(k))" | tee -a launches.txt
done
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
launches = {l.split()[0]: int(l.split()[1]) for l in open('launches.txt')}
void = launches.get('long', 0) == 0 or launches.get('joined90', 0) == 0
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
ok = not void and n and not foreign
print(f"{'clip':9} {'OFF A1':>8} {'ON B1':>8} {'OFF A2':>8} {'ON B2':>8} {'tcgen05':>8} {'bound':>8} {'saved':>8} texts")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; b = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    bound = statistics.mean(a) * (1.01 if launches.get(c, 1) == 0 else 1.0)
    texts = len({m[t][c]['text'] for t in m}) == 1; good = max(b) <= bound and texts; ok &= good
    print(f"{c:9} {a[0]:8.4f} {b[0]:8.4f} {a[1]:8.4f} {b[1]:8.4f} {launches.get(c, -1):8d} {bound:8.4f} {statistics.mean(a) - statistics.mean(b):8.4f} {texts}{'' if good else '  OVER'}")
print("launch check:", "VOID (no tcgen05 launches on the 35 or 90 s clip)" if void else "valid")
print("RULE 5 (amendment 1), audio:", "PASS" if ok else "FAIL")
PY
echo "TC2-R5-FINISHED $(date +%T)"
