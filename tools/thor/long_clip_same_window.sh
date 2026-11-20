#!/usr/bin/env bash
# validation/audio-handles-v1, "The long clip, same window" (2026-11-20): the long clip's stall on three servers in one
# quiet window. F audio-fe-w-srv (audio-frontend-v1), H audio-h-w-srv (handles), M merge-srv (the merge, deployed).
# 3 rounds, order rotated (F H M, H M F, M F H), 8 and 16 rows, measure_lfm3_audio_stall.py (merged tree) with 3 repeats.
# The always-on service is stopped for the window and started again however it ends. Waits for 15 quiet minutes.
# Usage (Thor host, copied to build/long-clip-w/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; O=long-clip-w
M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
declare -A SRV=([F]=audio-fe-w-srv [H]=audio-h-w-srv [M]=merge-srv)
busy() { { pgrep -af "[l]fm3-speech|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench" | grep -v "$O"; pgrep -af "[l]iquid-lfm3-server" | grep -v "/.local/lib/liquid-lfm3/" | grep -v "measure_lfm3_audio_stall"; docker ps --format "{{.Names}}" | grep -i "kin-pilot"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
(cd $B && sha256sum ${SRV[F]}/liquid-lfm3-server ${SRV[H]}/liquid-lfm3-server ${SRV[M]}/liquid-lfm3-server) | tee binaries.sha256
python3 -c "import json; c=[x for x in json.load(open('$B/quiet-w19-handles-rest/clips.json')) if x['name']=='long']; assert len(c)==1 and c[0]['case']=='4507-16021-0047'; json.dump(c, open('clips-long.json','w'), indent=1)" || { echo "NO CLIP"; echo LONG-CLIP-FINISHED; exit 1; }
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; echo LONG-CLIP-FINISHED; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; echo LONG-CLIP-FINISHED; exit 1; }
echo "== quiet $(date +%T)"
bash $B/../tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; echo LONG-CLIP-FINISHED; exit 1; }
seen_busy=""
for round in 1 2 3; do
  case $round in 1) order="F H M";; 2) order="H M F";; 3) order="M F H";; esac
  for s in $order; do for rows in 8 16; do
    b=$(busy); [ -n "$b" ] && seen_busy="$seen_busy | $(date +%T) $b"
    $X python3 $C/merge-src/tools/measure_lfm3_audio_stall.py $C/${SRV[$s]}/liquid-lfm3-server $M $C/$O/clips-long.json $rows \
      $C/$O/stall-$s-$rows-r$round.json 3 > stall-$s-$rows-r$round.log 2>&1
    echo "round $round $s ${SRV[$s]} $rows rows: exit $? $(date +%T): $(tail -1 stall-$s-$rows-r$round.log | cut -c1-160)"
  done; done
done
bash $B/../tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power unchanged" || echo "VOID power changed"
b=$(busy); [ -n "$b" ] && seen_busy="$seen_busy | end $b"
echo "busy during the window: ${seen_busy:-none}"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null
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
json.dump(res, open("summary.json", "w"), indent=1)
for k, x in r.items(): print(k, [round(y, 4) for y in x["stalls"]], "median", round(x["median"], 4), "spread", round(x["spread"], 4))
for k in ("H_vs_F", "M_vs_H"):
    print(k, res[k]["verdict"], {rows: (round(c["diff"] * 1000, 1), round(c["spread"] * 1000, 1)) for rows, c in res[k]["cells"].items()}, "(diff ms, spread ms)")
PY
echo "LONG-CLIP-FINISHED $(date +%T)"
