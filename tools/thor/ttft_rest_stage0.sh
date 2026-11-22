#!/usr/bin/env bash
# validation/ttft-rest-v1 stage 0 (a measurement): audio-handler-bench on the five clips (20 repeats, one thread), and
# one pass of time to first token of the deployed server (merge3-srv), in one quiet window.
# Usage (Thor host, copied to build/rest-s0/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=rest-s0; S=rest-src
M=/workspace/models/LFM3-Omni-2B
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-input-tool|[a]udio-handler-bench"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include \
  -x cu -o audio-handler-bench $C/$S/native/tests/audio_handler_bench.cpp > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"; echo "build exit $?"
sha256sum audio-handler-bench $B/merge3-srv/liquid-lfm3-server | tee binaries.sha256
cp $B/mel3-run/clips.json clips.json
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
python3 -c "import json; [print(c['name'], c['wav']) for c in json.load(open('clips.json'))]" | while read name wav; do
  $X $C/$O/audio-handler-bench $wav 20 > bench-$name.json 2> bench-$name.err; echo "bench $name exit $?"
done
$X python3 $C/$S/tools/measure_lfm3_audio_ttft.py $C/merge3-srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/ttft.json 5 > ttft.log 2>&1; echo "ttft exit $?"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
)
python3 - <<'PY'
import json
m = json.load(open('ttft.json'))['medians']
keys = ("valid_utf8", "strict_json", "parse_messages", "clip_from_base64", "base64_decode", "wav_parse", "clip_copy", "log_mel", "cache_put")
print(f"{'clip':9} " + " ".join(f"{k[:12]:>12}" for k in keys) + f" {'handler':>9} {'from send':>9} {'prefill':>8} {'remainder':>9}")
for c in m:
    b = json.load(open(f'bench-{c}.json'))
    rem = m[c]['from_send_seconds'] - m[c]['prefill_seconds'] - b['handler_sequence']
    print(f"{c:9} " + " ".join(f"{b[k]:12.4f}" for k in keys) + f" {b['handler_sequence']:9.4f} {m[c]['from_send_seconds']:9.4f} {m[c]['prefill_seconds']:8.4f} {rem:9.4f}")
PY
echo "REST-S0-FINISHED $(date +%T)"
