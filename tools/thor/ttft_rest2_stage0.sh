#!/usr/bin/env bash
# validation/ttft-rest-v2 stage 0 (a measurement): the trace build, then one quiet window.
#   builds (from build/rest2-src): the server with -DLFM3_TTFT_TRACE, tcgen05 ON as deployed (rest2-trace);
#     audio-handler-bench with the strict_json split (rest2-bench);
#   quiet: the bench (20 repeats) on the five clips; trace_lfm3_audio_ttft.py (1 untimed + 5 timed) on the trace
#     server and the deployed server (build/merge5-srv, f19f4ee40708), T1 A1 T2 A2.
# Usage (Thor host, copied to build/rest2-s0/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=rest2-s0; S=rest2-src
M=/workspace/models/LFM3-Omni-2B; T=$C/$S/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[t]race_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[l]fm3-vision-check|[a]lloc-"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
echo "== builds $(date +%T)"
mkdir -p $B/rest2-trace
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; N=$C/$S/native
  cd $C/rest2-trace
  cmake -S \$N -B cmake -DLIQUID_LFM3_NATIVE_GEMM=ON -DLIQUID_LFM3_TCGEN05=ON \
    -DCMAKE_CUDA_FLAGS=-DLFM3_TTFT_TRACE -DCMAKE_CXX_FLAGS=-DLFM3_TTFT_TRACE > cmake.log 2>&1
  cmake --build cmake --target liquid-lfm3-server -j4 > cmake-build.log 2>&1
  cp cmake/liquid-lfm3-server .
  ! grep -iE ' error|warning' cmake-build.log
  cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I \$N/include -x cu -o audio-handler-bench \
    $C/$S/native/tests/audio_handler_bench.cpp > build-bench.log 2>&1
  chown -R 1000:1000 $C/rest2-trace $C/$O"; echo "builds exit $?"
(cd $B && sha256sum rest2-trace/liquid-lfm3-server $O/audio-handler-bench merge5-srv/liquid-lfm3-server) | tee binaries.sha256
# grep -c reads to the end (grep -q would stop early, strings would get SIGPIPE, and pipefail would fail the check)
[ "$(strings -a $B/rest2-trace/liquid-lfm3-server | grep -cxE 's_enqueued|c_t0|o_text|h_sent')" -eq 4 ] && echo "trace build: marks present" || { echo "trace build: NO MARKS"; exit 1; }
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
for name in $(python3 -c "import json; print(' '.join(c['name'] for c in json.load(open('clips.json'))))"); do
  wav=$(python3 -c "import json,sys; print({c['name']: c['wav'] for c in json.load(open('clips.json'))}['$name'])")
  bash $R/tools/thor/power_state.sh power-bench-$name.json power-start.json > /dev/null || echo "POWER DIFFERS before bench $name"
  $X $C/$O/audio-handler-bench $wav 20 > bench-$name.json 2> bench-$name.err; echo "bench $name exit $?"
done
for tag in T1 A1 T2 A2; do
  bash $R/tools/thor/power_state.sh power-$tag.json power-start.json > /dev/null || echo "POWER DIFFERS before $tag"
  echo "== $tag $(date +%T)"
  if [ ${tag:0:1} = T ]; then
    $X env LFM3_TTFT_TRACE=$C/$O/trace-$tag.txt python3 $T/trace_lfm3_audio_ttft.py $C/rest2-trace/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1
  else
    $X python3 $T/trace_lfm3_audio_ttft.py $C/merge5-srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1
  fi
  echo "$tag exit $?"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
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
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]; dock = [x[1] for x in r if x[0] == 'D' and len(x) > 1]
    foreign += any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}"); print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('T1', 'A1', 'T2', 'A2')}
texts = {c: len({m[t][c]['text'] for t in m}) == 1 and all(m[t][c]['texts_identical'] for t in m) for c in m['T1']}
print("texts identical across T1 A1 T2 A2:", texts)
print(f"{'clip':10} {'T1':>8} {'A1':>8} {'T2':>8} {'A2':>8}   (from the first byte, s)")
for c in m['T1']:
    print(f"{c:10} " + " ".join(f"{m[t][c]['total']:8.4f}" for t in ('T1', 'A1', 'T2', 'A2')))
keys = [k for k in m['T1']['joined90'] if k not in ('text', 'texts_identical')]
print(f"{'interval':22} " + " ".join(f"{c:>9}" for c in m['T1']) + "   (T1 / T2 medians, s)")
for k in keys:
    print(f"{k:22} " + " ".join(f"{m['T1'][c].get(k, float('nan')):9.5f}" for c in m['T1']))
    print(f"{'':22} " + " ".join(f"{m['T2'][c].get(k, float('nan')):9.5f}" for c in m['T2']))
b = {c: json.load(open(f'bench-{c}.json')) for c in m['T1']}
print("bench (median of 20, s):")
for k in ('valid_utf8', 'depth_scan', 'body_copy', 'json_parse', 'tree_copy', 'strict_json'):
    print(f"  {k:12} " + " ".join(f"{b[c][k]:9.5f}" for c in b))
PY
echo "REST2-S0-FINISHED $(date +%T)"
