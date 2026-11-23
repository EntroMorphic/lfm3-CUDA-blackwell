#!/usr/bin/env bash
# validation/alloc-effect-v1 stage 4 (a diagnostic, rules in the README): alloc-extent-probe, mma.sync and tcgen05 builds,
# processes mma.sync, tcgen05, mma.sync, tcgen05 back to back after 60 s idle; the over-current counters, VDD_GPU current and VIN power at 10 Hz.
# Quiet Thor, the service stopped for the window.
# Usage (Thor host, copied to build/alloc-s4/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=alloc-s4; S=alloc4-src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[a]lloc-placement|[a]lloc-state|[a]lloc-onset|[a]lloc-extent"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && \
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -DLFM3_NATIVE_GEMM -o alloc-extent-probe-mma \
    $C/$S/native/tests/alloc_extent_probe.cu $C/$S/native/src/safetensors.cpp > build-mma.log 2>&1 && \
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -DLFM3_NATIVE_GEMM -DLFM_GEMM_TCGEN05 -o alloc-extent-probe-tc \
    $C/$S/native/tests/alloc_extent_probe.cu $C/$S/native/src/safetensors.cpp $C/tc2-on/gemm_tcgen05.o > build-tc.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build exit $?"; sha256sum alloc-extent-probe-mma alloc-extent-probe-tc $B/tc2-on/gemm_tcgen05.o | tee binaries.sha256
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""; HW=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ -n "$HW" ] && kill -TERM $HW 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
H=/sys/class/hwmon
( while true; do echo "$(date +%s.%N) $(cat $H/hwmon4/oc2_event_cnt) $(cat $H/hwmon4/oc3_event_cnt) $(cat $H/hwmon6/curr1_input) $(cat $H/hwmon5/power1_input)"; sleep 0.1; done ) > hw.log 2>&1 &
HW=$!
sleep 60
for run in mma:M1 tc:T1 mma:M2 tc:T2; do b=${run%%:*}; tag=${run#*:}
  echo "== $tag $(date +%T)"; $X $C/$O/alloc-extent-probe-$b > $tag.jsonl 2> $tag.err; echo "$tag exit $?"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
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
hw = [l.split() for l in open('hw.log') if len(l.split()) == 5]
print(f"hw: {len(hw)} samples; oc2 moved {int(hw[-1][1]) - int(hw[0][1])}, oc3 moved {int(hw[-1][2]) - int(hw[0][2])}; "
      f"VDD_GPU max {max(int(x[3]) for x in hw)} mA; VIN max {max(int(x[4]) for x in hw) / 1e6:.1f} W")
def load(tags): return [json.loads(l) for t in tags for l in open(f'{t}.jsonl') if l.startswith('{')]
M, T = load(('M1', 'M2')), load(('T1', 'T2'))
print("paths:", [x for x in M + T if 'path' in x])
print("== size (mma.sync): post / pre per trial")
for cp in ('h2d_4k', 'h2d_64k', 'h2d_1m', 'h2d_8m', 'h2d_30m', 'h2d_async_64k', 'd2h_4k', 'd2h_30m'):
    r = [x['post'] / x['pre'] for x in M if x.get('copy') == cp]
    print(f"   {cp:14} {' '.join(f'{v:.2f}' for v in r)}{'  INDUCES' if r and all(v >= 1.25 for v in r) else ''}")
print("== restore (mma.sync): post / pre per trial")
for g in (0, 1, 10, 100, 300, 1000):
    r = [x['post'] / x['pre'] for x in M if x.get('part') == 'restore' and x['gap_ms'] == g]
    print(f"   {g:5d} ms  {' '.join(f'{v:.2f}' for v in r)}{'  RESTORED' if r and all(v <= 1.10 for v in r) else ''}")
for t in ('M1', 'M2', 'T1', 'T2'):
    p = [x for x in load((t,)) if x.get('part') == 'persist'][0]
    print(f"== persist {t}: pre {p['pre']:.3f}; per 100: {' '.join(f'{v:.3f}' for v in p['by_hundred'])}")
for name, L in (('mma.sync', M), ('tcgen05 (deployed dispatch)', T)):
    print(f"== shapes, {name}: fast ms, slow / fast per trial")
    works = list(dict.fromkeys(x['work'] for x in L if x.get('part') == 'shapes'))
    for w in works:
        r = [x for x in L if x.get('work') == w]
        q = [x['slow'] / x['fast'] for x in r]
        print(f"   {w:20} {statistics.median(x['fast'] for x in r):8.4f}  {' '.join(f'{v:.2f}' for v in q)}{'  AFFECTED' if q and all(v >= 1.25 for v in q) else ''}")
PY
echo "ALLOC-S4-FINISHED $(date +%T)"
