#!/usr/bin/env bash
# validation/alloc-effect-v1 stage 3 (a diagnostic, rules in the README): alloc-onset-probe, mma.sync and tcgen05 builds,
# every ingredient twice per binary, 32 processes back to back after 60 s idle; the over-current counters, VDD_GPU current and VIN power at 10 Hz.
# Quiet Thor, the service stopped for the window.
# Usage (Thor host, copied to build/alloc-s3/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=alloc-s3; S=alloc3-src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[a]lloc-placement|[a]lloc-state|[a]lloc-onset"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && \
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -DLFM3_NATIVE_GEMM -o alloc-onset-probe-mma \
    $C/$S/native/tests/alloc_onset_probe.cu $C/$S/native/src/safetensors.cpp > build-mma.log 2>&1 && \
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I $C/$S/native/include -DLFM3_NATIVE_GEMM -DLFM_GEMM_TCGEN05 -o alloc-onset-probe-tc \
    $C/$S/native/tests/alloc_onset_probe.cu $C/$S/native/src/safetensors.cpp $C/tc2-on/gemm_tcgen05.o > build-tc.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"
echo "build exit $?"; sha256sum alloc-onset-probe-mma alloc-onset-probe-tc $B/tc2-on/gemm_tcgen05.o | tee binaries.sha256
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
n=0
for ing in malloc_h2d none gemm_other h2d stage1 malloc_fill h2d_pinned malloc stage1 h2d malloc none malloc_fill gemm_other h2d_pinned malloc_h2d; do
  for b in mma tc; do n=$((n+1))
    echo "== $n $b $ing $(date +%T)"; $X $C/$O/alloc-onset-probe-$b $ing >> $b.jsonl 2>> $b.err; echo "$n exit $?"
  done
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
ings = ['none', 'malloc', 'malloc_fill', 'h2d', 'h2d_pinned', 'malloc_h2d', 'gemm_other', 'stage1']
for binary in ('mma', 'tc'):
    L = [json.loads(l) for l in open(f'{binary}.jsonl') if l.startswith('{')]
    print(f"== {binary}: {len(L)} processes")
    induce = {}
    for g in ings:
        rs = [x for x in L if x['ingredient'] == g]
        flags = [x['after'] >= 1.25 * x['baseline'] for x in rs]
        induce[g] = len(rs) == 2 and all(flags)
        for x, f in zip(rs, flags):
            onset = next((i + 1 for i, v in enumerate(x['steps']) if v >= 1.25 * x['baseline']), None)
            rest = ' '.join(f"{k}={v:.3f}{'*' if f and v <= 0.80 * x['after'] else ''}" for k, v in x['restore'].items())
            print(f"   {g:11} baseline {x['baseline']:.3f} after {x['after']:.3f} ({x['after'] / x['baseline']:.2f}x){' INDUCES' if f else ''}"
                  f"{f' onset step {onset}' if f else ''}; steps {' '.join(f'{v:.3f}' for v in x['steps'])}; {rest}")
    base = {g: statistics.median(x['baseline'] for x in L if x['ingredient'] == g) for g in ('none', 'gemm_other')}
    print(f"   start footprint: gemm_other baseline {base['gemm_other']:.3f} against none {base['none']:.3f} ms")
    nonerows = [x for x in L if x['ingredient'] == 'none']
    if any(x['after'] >= 1.25 * x['baseline'] for x in nonerows): print(f"   VOID ({binary}): none induces"); continue
    print(f"   READING ({binary}): induces: {', '.join(g for g in ings if induce[g]) or 'NONE'}  (* = restorer at or under 0.80x after)")
PY
echo "ALLOC-S3-FINISHED $(date +%T)"
