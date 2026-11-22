#!/usr/bin/env bash
# validation/mel-tiered-v1 rule 2's reported edge item (missed by mel_tiered_run.sh: it read rule 1's edge list before
# rule 1 had written it): the edge clip with the most second-tier frames, F2 against F3, 4 threads, --repeat 5, quiet.
# Usage (Thor host, copied to build/mel3-edge/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=mel3-edge
M=/workspace/models/LFM3-Omni-2B
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-input-tool"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
best=""; bestn=-1
while read name wav rest; do
  echo "$name $wav" > one.txt; mkdir -p edge-out
  n=$(docker exec lfm3-profile-v1 $C/mel3-round/audio-mel-check --threads 1 --transform f3 --write $C/$O/edge-out $C/$O/one.txt $C/$O/edge-one.json 2>&1 | tail -1 | sed -n 's/.*recomputed by the second tier \([0-9]*\).*/\1/p')
  echo "  $name: $n second-tier frames"; [ "${n:-0}" -gt $bestn ] && bestn=$n && best="$name=$wav"
done < $B/mel3-rule1/edge.txt
[ -n "$best" ] || { echo "NO EDGE CLIP"; exit 1; }
echo "chosen: $best ($bestn frames)"
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
for mode in exact-f2 exact-f3; do
  docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=$mode lfm3-profile-v1 \
    $C/mel3-bench/audio-encoder-bench $M $C/$O/bench-$mode.json --repeat 5 --mel-threads 4 "$best" > bench-$mode.log 2>&1; echo "bench $mode exit $?"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
)
python3 -c "
import json
a = json.load(open('bench-exact-f2.json'))['clips'][0]; b = json.load(open('bench-exact-f3.json'))['clips'][0]
print(a['name'], a['audio_seconds'], 's: log_mel F2', a['log_mel_seconds'], 'F3', b['log_mel_seconds'], '| hashes equal', a['mel_fnv1a'] == b['mel_fnv1a'])"
echo "MEL3-EDGE-FINISHED $(date +%T)"
