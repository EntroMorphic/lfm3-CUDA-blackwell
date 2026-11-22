#!/usr/bin/env bash
# validation/tcgen05-v2, development evidence for amendment 2: lfm3-tcgen05-gemm-speed (each engine GEMM shape and M,
# tcgen05 on and off) on a quiet Thor, the service stopped. Built from build/tc2-src with tc2-on's gemm_tcgen05.o.
# Usage (Thor host, copied to build/tc2-speedmap/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=tc2-speedmap; S=tc2-src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[t]cgen05-gemm"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include \
  -DLFM_GEMM_TCGEN05 -o lfm3-tcgen05-gemm-speed $C/$S/native/tests/lfm3_tcgen05_gemm_speed.cu $C/tc2-on/gemm_tcgen05.o > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build exit $?"; sha256sum lfm3-tcgen05-gemm-speed $B/tc2-on/gemm_tcgen05.o | tee binaries.sha256
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 $C/$O/lfm3-tcgen05-gemm-speed > speed.jsonl 2> speed.err
echo "speed exit $?"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
)
python3 - <<'PY'
import json, collections
rows = [json.loads(l) for l in open('speed.jsonl') if l.startswith('{') and 'shape' in l]
by = collections.defaultdict(list)
for r in rows: by[r['shape']].append(r)
for s, rs in by.items():
    print(f"{s:13} n {rs[0]['n']:5d} k {rs[0]['k']:5d}  " + "  ".join(f"{r['m']}:{r['off_over_on']:.2f}{'' if r['tcgen05_launched'] else '*'}" for r in rs))
print("(off/on: above 1 means tcgen05 faster; * tcgen05 did not launch)")
PY
echo "TC2-SPEEDMAP-FINISHED $(date +%T)"
