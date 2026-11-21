#!/usr/bin/env bash
# validation/audio-ttft-v1, a diagnostic (not a rule): the deployed server (build/merge-srv) and this round's
# (build/ttft-srv) timed in one quiet window, ABAB, tools/measure_lfm3_audio_ttft.py on the five clips (1 untimed + 5
# timed passes each), so the change in time to first token is read without crossing windows. Runs after build/ttft-mem.
# Usage (Thor host, copied to build/ttft-ab/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=ttft-ab
M=/workspace/models/LFM3-Omni-2B; S=ttft-src-a4
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
fin() { echo "TTFT-AB-FINISHED $(date +%T)"; }
cd $B/$O || exit 1
until grep -q "^TTFT-MEM-FINISHED" $B/ttft-mem/run.log 2>/dev/null; do sleep 60; done
cp $B/ttft-s1q/clips.json clips.json
(cd $B && sha256sum merge-srv/liquid-lfm3-server ttft-srv/liquid-lfm3-server) | tee binaries.sha256
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; fin; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; fin; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; fin; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
for run in merge-srv:A1 ttft-srv:B1 merge-srv:A2 ttft-srv:B2; do srv=${run%%:*}; tag=${run#*:}
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== $tag $srv $(date +%T)"
  $X python3 $C/$S/tools/measure_lfm3_audio_ttft.py $C/$srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1
  echo "$tag exit $?"
done
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
python3 - <<'PY'
import collections, json
rows = collections.defaultdict(list)
for line in open('samples.log'):
    p = line.rstrip('\n').split(' ', 2)
    if len(p) >= 2: rows[p[0]].append(p[1:])
n = foreign = 0
for t, r in rows.items():
    if not any(x[0] == 'T' for x in r): continue
    n += 1
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]
    dock = [x[1] for x in r if x[0] == 'D' and len(x) > 1]
    foreign += any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(
        ('kin_' in q or 'venvs/hf-' in q or 'nsys' in q or 'compute-sanitizer' in q or 'lfm3-speech' in q
         or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
print(f"{'clip':10} {'merge A1':>9} {'ttft B1':>9} {'merge A2':>9} {'ttft B2':>9} {'saved':>8} | prefill: {'merge':>7} {'ttft':>7}")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; b = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    pa = [m[t][c]['prefill_seconds'] for t in ('A1', 'A2')]; pb = [m[t][c]['prefill_seconds'] for t in ('B1', 'B2')]
    print(f"{c:10} {a[0]:9.4f} {b[0]:9.4f} {a[1]:9.4f} {b[1]:9.4f} {sum(a)/2 - sum(b)/2:8.4f} | {sum(pa)/2:7.4f} {sum(pb)/2:7.4f}"
          f"  texts: merge {m['A1'][c]['text'] == m['A2'][c]['text']}, ttft {m['B1'][c]['text'] == m['B2'][c]['text']}")
PY
fin
