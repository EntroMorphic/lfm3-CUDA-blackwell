#!/usr/bin/env bash
# validation/audio-ttft-v1, a diagnostic (not a rule): after ttft-srv's first audio request, the language model's large
# prefill GEMMs (grids 60x9 and 48x9) ran about 75% slower (build/ttft-prof); merge-srv's never did. Which of the
# round's kept state does it: the scratch, or the doubled weights? On a quiet Thor, nsys on four 90 s requests each:
#   merge-srv (control); ttft-srv (x2: scratch kept, doubled weights); ttft-srv with LFM3_AUDIO_ENCODER=fp32 (scratch
#   kept, no doubled weights, no split); a5-srv (amendment 5: scratch kept, no doubled weights, x2 by the wrap).
# Per request: the encode span, and from the encode's end to the first token (the first greedy_rows) with its GEMM time.
# Then amendment 5's exactness script (build/a5-exact/run.sh), its builds done here first.
# Usage (Thor host, copied to build/a5-diag/run.sh): bash run.sh WAIT_PID > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; O=a5-diag; S=ttft-src-a5
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
while kill -0 ${1:-0} 2>/dev/null; do sleep 10; done
echo "== builds $(date +%T)"
(cd $B/a5-exact && BUILD_ONLY=1 bash run.sh > builds.log 2>&1); tail -8 $B/a5-exact/builds.log
LONG=$B/server-stream-v3-check/work/joined-90s.wav
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
for e in merge:merge-srv:default x2:ttft-srv:default fp32:ttft-srv:fp32 a5:a5-srv:default; do
  name=${e%%:*}; rest=${e#*:}; srv=${rest%%:*}; mode=${rest#*:}
  echo "== $name ($srv, encoder $mode) $(date +%T)"
  if [ $mode = default ]; then python3 $B/$S/tools/profile_lfm3_audio_server.py /usr/local/bin/nsys $B/$srv/liquid-lfm3-server $W/models/LFM3-Omni-2B $LONG 4 60 $name > $name.log 2>&1
  else LFM3_AUDIO_ENCODER=$mode python3 $B/$S/tools/profile_lfm3_audio_server.py /usr/local/bin/nsys $B/$srv/liquid-lfm3-server $W/models/LFM3-Omni-2B $LONG 4 60 $name > $name.log 2>&1; fi
  echo "$name exit $?"; grep '"request"' $name.log | python3 -c "
import json, sys
for l in sys.stdin: d = json.loads(l); print('  request', d['request'], 'TTFT %.4f prefill %.4f' % (d['from_file_seconds'], d['prefill_seconds']), d['text'][:30])"
  nsys export --type sqlite --force-overwrite true -o $name.sqlite $name.nsys-rep > /dev/null 2>&1
done
)
python3 - <<'PY'
import sqlite3, collections
for name in ("merge", "x2", "fp32", "a5"):
    try: c = sqlite3.connect(f"{name}.sqlite"); names = dict(c.execute("select id, value from StringIds"))
    except Exception as e: print(name, "NO REPORT", e); continue
    k = [(s, e, names[n], gx, gy) for s, e, n, gx, gy in c.execute("select start, end, shortName, gridX, gridY from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    starts = [i for i, y in enumerate(k) if y[2] == "conv_in"]
    for j, i in enumerate(starts):
        m = next(x for x in range(i, len(k)) if k[x][2] == "to_bf16")
        f = next(x for x in range(m, len(k)) if k[x][2] == "greedy_rows")
        seg = k[m + 1:f + 1]
        g = collections.defaultdict(float)
        for s, e, n, gx, gy in seg:
            if n == "wide_kernel": g[(gx, gy)] += (e - s) / 1e6
        print(f"{name} request {j}: encode {(k[m][1] - k[i][0]) / 1e6:.1f} ms; encode end -> first token {(k[f][1] - k[m][1]) / 1e6:.1f} ms; "
              f"GEMM grids 60x9 {g.get((60, 9), 0):.1f} ms, 48x9 {g.get((48, 9), 0):.1f} ms, all wide {sum(g.values()):.1f} ms")
PY
echo "A5-DIAG-FINISHED $(date +%T)"
echo "== amendment 5's exactness script"
cd $B/a5-exact && bash run.sh > run.log 2>&1
echo "A5-EXACT-DONE $(date +%T)"
