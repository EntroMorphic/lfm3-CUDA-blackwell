#!/usr/bin/env bash
# validation/scratch-keep-v1 rules 3 and 4, on a quiet Thor (after keep-exact). A = build/merge6-srv (main 8120c19, deployed
# as 0012be944a78); B = build/keep-on.
#   builds (before the window): A's and B's trees with -DLFM3_TTFT_TRACE, tcgen05 ON (keep-trace-a, keep-trace-b);
#   rule 3: measure_lfm3_audio_ttft.py on tcgen05-v2's five clips (1 untimed + 5 timed), A1 B1 A2 B2;
#   rule 4: measure_lfm3_audio_device_memory.py (the short clip, the 90 s clip twice, then cherry once), A1 B1 A2 B2; then
#           B twice with LFM3_AUDIO_SCRATCH_KEEP_MIB=256 (R1 R2, the release path);
#   reported: trace_lfm3_audio_ttft.py on both trace builds (TA, TB), splitting the encoder from the language model's call.
# Usage (Thor host, copied to build/keep-quiet/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=keep-quiet; S=keep-src
M=/workspace/models/LFM3-Omni-2B; T=$C/$S/tools; A=merge6-srv; ON=keep-on
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[t]race_lfm3|[a]udio-mel-check|[a]udio-handler-bench|[l]fm3-vision-check|[a]lloc-|[g]emm-wrap-check"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
cp $B/tc2-quiet/clips.json clips.json
read SHORT LONG < <(python3 -c "import json; c = {x['name']: x['wav'] for x in json.load(open('clips.json'))}; print(c['short'], c['joined90'])")
until grep -q "^KEEP-EXACT-FINISHED" $B/keep-exact/run.log 2>/dev/null; do sleep 60; done
echo "== trace builds $(date +%T)"
for v in a:merge6-src b:$S; do IFS=: read t src <<< "$v"
  mkdir -p $B/keep-trace-$t
  docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd $C/keep-trace-$t
    cmake -S $C/$src/native -B cmake -DLIQUID_LFM3_NATIVE_GEMM=ON -DLIQUID_LFM3_TCGEN05=ON \
      -DCMAKE_CUDA_FLAGS=-DLFM3_TTFT_TRACE -DCMAKE_CXX_FLAGS=-DLFM3_TTFT_TRACE > cmake.log 2>&1
    cmake --build cmake --target liquid-lfm3-server -j4 > cmake-build.log 2>&1
    cp cmake/liquid-lfm3-server .
    ! grep -iE ' error|warning' cmake-build.log
    chown -R 1000:1000 $C/keep-trace-$t"; echo "trace build $t exit $?"
done
(cd $B && sha256sum $A/liquid-lfm3-server $ON/liquid-lfm3-server keep-trace-a/liquid-lfm3-server keep-trace-b/liquid-lfm3-server) | tee binaries.sha256
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
pw() { bash $R/tools/thor/power_state.sh power-$1.json power-start.json > /dev/null || echo "POWER DIFFERS before $1"; }
srv() { [ ${1:0:1} = A ] && echo $C/$A/liquid-lfm3-server || echo $C/$ON/liquid-lfm3-server; }
for tag in A1 B1 A2 B2; do
  pw $tag; echo "== audio $tag $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $(srv $tag) $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "audio $tag exit $?"
done
pw memory; echo "== device memory, ABAB $(date +%T)"
$X env LFM3_MEMORY_IMAGE=$C/vimg-e2e-a1/req/cherry.json python3 $T/measure_lfm3_audio_device_memory.py $SHORT $LONG $M $C/$O/device-memory.json \
  A1=$(srv A1) B1=$(srv B1) A2=$(srv A2) B2=$(srv B2) > device-memory.log 2>&1; echo "device memory exit $?"
pw release; echo "== device memory, the release path $(date +%T)"
$X env LFM3_MEMORY_IMAGE=$C/vimg-e2e-a1/req/cherry.json LFM3_AUDIO_SCRATCH_KEEP_MIB=256 python3 $T/measure_lfm3_audio_device_memory.py \
  $SHORT $LONG $M $C/$O/device-memory-release.json R1=$(srv B1) R2=$(srv B2) > device-memory-release.log 2>&1; echo "release path exit $?"
for t in a b; do
  pw trace-$t; echo "== trace $t $(date +%T)"
  $X env LFM3_TTFT_TRACE=$C/$O/trace-$t.txt python3 $T/trace_lfm3_audio_ttft.py $C/keep-trace-$t/liquid-lfm3-server $M $C/$O/clips.json \
    $C/$O/T$t.json 5 > T$t.log 2>&1; echo "trace $t exit $?"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
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
clean = bool(n) and not foreign
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
ok = clean
print(f"{'clip':10} {'A1':>8} {'B1':>8} {'A2':>8} {'B2':>8} {'saved':>8} {'B/A':>6} texts   (time to first token from the file, s)")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; b = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    texts = len({m[t][c]['text'] for t in m}) == 1 and all(m[t][c]['texts_identical'] for t in m)
    saved = statistics.mean(a) - statistics.mean(b)
    good = saved > 0 and texts and (c != 'joined90' or saved >= 0.005); ok &= good
    print(f"{c:10} {a[0]:8.4f} {b[0]:8.4f} {a[1]:8.4f} {b[1]:8.4f} {saved:8.4f} {statistics.mean(b) / statistics.mean(a):6.3f} {texts}{'' if good else '  FAIL'}")
print("RULE 3:", "PASS" if ok else "FAIL")
MIB = 2**20
d = json.load(open('device-memory.json')); r = json.load(open('device-memory-release.json'))
dev = lambda x, k: x[k]['device'] / MIB
a90 = statistics.mean(dev(d[t], 'after_90') for t in ('A1', 'A2')); b90 = statistics.mean(dev(d[t], 'after_90') for t in ('B1', 'B2'))
r90 = statistics.mean(dev(r[t], 'after_90') for t in ('R1', 'R2'))
deployed = statistics.mean((7741, 7729))  # merge5-srv after the 90 s clip, encoder-wrap-v1's quiet run (A1, A2)
r4a = 550 <= b90 - a90 <= 700; r4b = b90 <= deployed + 128; r4c = abs(r90 - a90) <= 32
for t in ('A1', 'B1', 'A2', 'B2'):
    print(f"{t} " + " ".join(f"{k} {dev(d[t], k):.0f}" for k in ('started', 'short', 'after_90', 'after_image')) + " MiB")
for t in ('R1', 'R2'):
    print(f"{t} (B, cap 256 MiB) " + " ".join(f"{k} {dev(r[t], k):.0f}" for k in ('started', 'short', 'after_90', 'after_image')) + " MiB")
print(f"4a: after 90 s B - A = {b90 - a90:.0f} MiB (550..700): {'PASS' if r4a else 'FAIL'}")
print(f"4b: after 90 s B = {b90:.0f} MiB, the old deploy {deployed:.0f} + 128: {'PASS' if r4b else 'FAIL'}")
print(f"4c: after 90 s, B with the 256 MiB cap - A = {r90 - a90:.0f} MiB (within 32): {'PASS' if r4c else 'FAIL'}")
print("RULE 4:", "PASS" if r4a and r4b and r4c and clean else "FAIL")
t = {k: json.load(open(f'T{k}.json'))['medians'] for k in ('a', 'b')}
print("traces (medians, ms): A's tree / B's tree")
for k in ('log_mel', 'encoder', 'lm_forward_call', 'total'):
    print(f"  {k:16} " + " ".join(f"{1000 * t['a'][c].get(k, float('nan')):7.2f}/{1000 * t['b'][c].get(k, float('nan')):<7.2f}" for c in t['a']))
PY
echo "KEEP-QUIET-FINISHED $(date +%T)"
