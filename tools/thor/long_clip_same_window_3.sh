#!/usr/bin/env bash
# validation/audio-handles-v1, "The long clip, third run of part A" (2026-11-20): the second run's part A, with defunct
# entries counted but allowed, and one throwaway run (M, 8 rows) before round 1. No part B.
# The second run's header follows. Part A: the first run's comparison
# (F audio-fe-w-srv, H audio-h-w-srv, M merge-srv; long clip; 8 and 16 rows; 3 rotated rounds) with a 1 s sampler of
# every GPU-work process and docker ps for the whole of part A, and the power state checked before every run.
# Part B (reported): nsys of F and H on the long clip at 8 rows (host, 3 repeats), tools/nsys_audio_admission.py.
# The always-on service is stopped for the window and started again however it ends. Waits for 15 quiet minutes.
# Usage (Thor host, copied to build/long-clip-w3/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; O=long-clip-w3
M=/workspace/models/LFM3-Omni-2B; T=$B/merge-src/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
declare -A SRV=([F]=audio-fe-w-srv [H]=audio-h-w-srv [M]=merge-srv)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
fin() { echo "LONG-CLIP-3-FINISHED $(date +%T)"; }
cd $B/$O || exit 1
(cd $B && sha256sum ${SRV[F]}/liquid-lfm3-server ${SRV[H]}/liquid-lfm3-server ${SRV[M]}/liquid-lfm3-server) | tee binaries.sha256
python3 -c "
import json; c=[x for x in json.load(open('$B/quiet-w19-handles-rest/clips.json')) if x['name']=='long']
assert len(c)==1 and c[0]['case']=='4507-16021-0047'
json.dump(c, open('clips-long.json','w'), indent=1)
h=[dict(c[0], wav=c[0]['wav'].replace('/workspace/', '$W/'))]; json.dump(h, open('clips-long-host.json','w'), indent=1)
import os; assert os.path.exists(h[0]['wav'])" || { echo "NO CLIP"; fin; exit 1; }
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; fin; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; fin; exit 1; }
echo "== quiet $(date +%T)"
bash $B/../tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; fin; exit 1; }

echo "== part A $(date +%T)"
# the sampler: every process matching the patterns (this window's included) and docker ps, once a second
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
power_bad=0
bash $B/../tools/thor/power_state.sh power-throwaway.json power-start.json > /dev/null || { power_bad=1; echo "POWER DIFFERS before the throwaway"; }
$X python3 $C/merge-src/tools/measure_lfm3_audio_stall.py $C/${SRV[M]}/liquid-lfm3-server $M $C/$O/clips-long.json 8 \
  $C/$O/throwaway-M-8.json 3 > throwaway-M-8.log 2>&1
echo "throwaway M 8 rows (left out): exit $? $(date +%T): $(tail -1 throwaway-M-8.log | cut -c1-160)"
for round in 1 2 3; do
  case $round in 1) order="F H M";; 2) order="H M F";; 3) order="M F H";; esac
  for s in $order; do for rows in 8 16; do
    bash $B/../tools/thor/power_state.sh power-r$round-$s-$rows.json power-start.json > /dev/null || { power_bad=1; echo "POWER DIFFERS before round $round $s $rows"; }
    $X python3 $C/merge-src/tools/measure_lfm3_audio_stall.py $C/${SRV[$s]}/liquid-lfm3-server $M $C/$O/clips-long.json $rows \
      $C/$O/stall-$s-$rows-r$round.json 3 > stall-$s-$rows-r$round.log 2>&1
    echo "round $round $s ${SRV[$s]} $rows rows: exit $? $(date +%T): $(tail -1 stall-$s-$rows-r$round.log | cut -c1-160)"
  done; done
done
kill -TERM $SAMPLER; wait $SAMPLER 2>/dev/null; SAMPLER=""
bash $B/../tools/thor/power_state.sh power-end-a.json power-start.json > /dev/null || { power_bad=1; echo "POWER DIFFERS at the end of part A"; }
echo "power: $([ $power_bad = 0 ] && echo "unchanged before every run and at the end" || echo "CHANGED: part A void")"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null

# the sampler's verdict: anything but this window's own processes and the two containers
python3 - "$C" <<'PY'
import collections, re, sys
C = sys.argv[1]
ours = [f"{C}/{d}/" for d in ("audio-fe-w-srv", "audio-h-w-srv", "merge-srv")]
ticks, foreign, servers, defunct = collections.OrderedDict(), [], collections.Counter(), []
for line in open("samples.log"):
    parts = line.rstrip("\n").split(" ", 2)
    if len(parts) < 2: continue
    t, kind = parts[0], parts[1]
    ticks.setdefault(t, [])
    if kind == "T": continue
    rest = parts[2] if len(parts) > 2 else ""
    if kind == "D":
        if rest not in ("lfm3-profile-v1", "lfm3-omni-dev"): foreign.append((t, "container " + rest))
        continue
    cmd = rest.split(" ", 1)[1] if " " in rest else rest
    if cmd.endswith("<defunct>"):
        defunct.append((t, cmd[:80])); continue
    if "long-clip-w3" in cmd: continue  # the stall tool and its docker exec (their arguments name this window)
    exe = cmd.split(" ", 1)[0]
    if any(exe.startswith(p) for p in ours) and re.search(r"liquid-lfm3-(server|worker)$", exe):
        if exe.endswith("server"): servers[t] += 1
        continue
    foreign.append((t, cmd[:160]))
too_many = [t for t, n in servers.items() if n > 1]
print(f"sampler: defunct entries (allowed, counted) in {len({t for t, _ in defunct})} samples: {sorted({c for _, c in defunct})}")
n = sum(1 for t in ticks)
print(f"sampler: {n} samples; foreign process or container in {len({t for t, _ in foreign})} of them; "
      f"more than one server in {len(too_many)}")
for t, what in foreign[:10]: print("  foreign", t, what)
gaps = [float(b) - float(a) for a, b in zip(list(ticks), list(ticks)[1:])]
print(f"sampler: longest interval between samples {max(gaps, default=0):.2f} s")
print("PART A MONITOR:", "CLEAN" if not foreign and not too_many and n > 0 else "VOID")
PY

python3 - <<'PY'
import json, statistics as st
r = {}
for s in "FHM":
    for rows in (8, 16):
        v = [json.load(open(f"stall-{s}-{rows}-r{k}.json"))["clips"]["long"]["stall_seconds"] for k in (1, 2, 3)]
        r[f"{s}-{rows}"] = dict(stalls=v, median=st.median(v), spread=max(v) - min(v))
def verdict(a, b):  # b against a
    out = {}
    for rows in (8, 16):
        A, Bv = r[f"{a}-{rows}"], r[f"{b}-{rows}"]
        d, sp = Bv["median"] - A["median"], max(A["spread"], Bv["spread"])
        out[rows] = dict(diff=d, spread=sp, reg=d > 0.003 and d > sp, none=d <= max(0.003, sp))
    v = "regression" if all(x["reg"] for x in out.values()) else "no regression shown" if all(x["none"] for x in out.values()) else "mixed"
    return dict(cells=out, verdict=v)
res = dict(figures=r, H_vs_F=verdict("F", "H"), M_vs_H=verdict("H", "M"))
json.dump(res, open("summary-a.json", "w"), indent=1)
for k, x in r.items(): print(k, [round(y, 4) for y in x["stalls"]], "median", round(x["median"], 4), "spread", round(x["spread"], 4))
for k in ("H_vs_F", "M_vs_H"):
    print(k, res[k]["verdict"], {rows: (round(c["diff"] * 1000, 1), round(c["spread"] * 1000, 1)) for rows, c in res[k]["cells"].items()}, "(diff ms, spread ms)")
PY

bash $B/../tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power unchanged at the end" || echo "POWER DIFFERS at the end"
fin
