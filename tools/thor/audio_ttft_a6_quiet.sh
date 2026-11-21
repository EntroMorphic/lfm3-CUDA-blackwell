#!/usr/bin/env bash
# validation/audio-ttft-v1 amendment 6, the timing and memory rules, one quiet window (then the exactness script):
#   rule 4 of the amendment: time to first token on the five clips (tools/measure_lfm3_audio_ttft.py, 1 untimed + 5
#     timed), merge-srv A1, a6-srv B1, merge-srv A2, a6-srv B2, then ttft-srv once (the kept scratch, as reference);
#     a6-srv's median at most merge-srv's (the mean of A1, A2) at every clip;
#   rule 3 (re-measured, recorded): audio-encoder-bench a6-bench (--repeat 5, log_mel F2 4 threads) and HF BF16
#     (tools/time_lfm3_audio_hf.py, 1 untimed + 5 timed, 32 new tokens): the encoder at most HF's at 35 and 90 s;
#   rule 3 of the amendment (5b): tools/measure_lfm3_audio_device_memory.py, merge-1, a6-1, merge-2, a6-2: the mean of
#     a6's plain in-use after the 90 s clip minus the mean of merge's at most 1 GiB;
#   amendment 3's leftover: the batch-server check on a6-srv with LFM3_AUDIO_ENCODER=fp32 (timing sub-checks).
# The service stopped for the window and started again however it ends; the 1 s sampler; the power state per step.
# Usage (Thor host, copied to build/a6-run/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=a6-run
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; S=ttft-src-a6b; T=$C/$S/tools
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
H="docker exec -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tmp/cuda-13.0-ptxas/ptxas lfm3-profile-v1 /workspace/venvs/hf-5.20.2/bin/python"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
echo "== builds $(date +%T)"
(cd $B/a6-exact && BUILD_ONLY=1 bash run.sh > builds.log 2>&1); tail -6 $B/a6-exact/builds.log
cp $B/ttft-s1q/clips.json clips.json; cp $B/ttft-s1q/clip-args.txt clip-args.txt; cp $B/$S/tools/time_lfm3_audio_hf.py hf_tool.py
python3 -c "import json; c={x['name']: x['wav'] for x in json.load(open('clips.json'))}; print(c['short'], c['joined90'])" > clips.txt
read SHORT LONG < clips.txt
(cd $B && sha256sum merge-srv/liquid-lfm3-server a6-srv/liquid-lfm3-server ttft-srv/liquid-lfm3-server a6-bench/audio-encoder-bench) | tee binaries.sha256
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
for run in merge-srv:A1 a6-srv:B1 merge-srv:A2 a6-srv:B2 ttft-srv:R; do srv=${run%%:*}; tag=${run#*:}
  pw $tag; echo "== $tag $srv $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $C/$srv/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
pw bench; echo "== the bench $(date +%T)"
$X $C/a6-bench/audio-encoder-bench $M $C/$O/bench.json --repeat 5 --mel-threads 4 $(cat clip-args.txt) > bench.log 2>&1; echo "bench exit $?"
pw hf; echo "== HF $(date +%T)"
$H $C/$O/hf_tool.py $M $C/$O/hf.jsonl 5 32 $(cat clip-args.txt) > hf.log 2>&1; echo "HF exit $?"
pw mem; echo "== device memory, ABAB $(date +%T)"
$X python3 $T/measure_lfm3_audio_device_memory.py $SHORT $LONG $M $C/$O/device-memory.json merge-1=$C/merge-srv/liquid-lfm3-server \
  a6-1=$C/a6-srv/liquid-lfm3-server merge-2=$C/merge-srv/liquid-lfm3-server a6-2=$C/a6-srv/liquid-lfm3-server > device-memory.log 2>&1
echo "device memory exit $?"
pw batch; echo "== the batch server, LFM3_AUDIO_ENCODER=fp32 $(date +%T)"
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_ENCODER=fp32 lfm3-profile-v1 \
  python3 $T/check_lfm3_speech_batch_server.py $C/a6-srv/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L \
  $C/$O/batch-server-fp32.json > batch-server-fp32.log 2>&1; echo "batch server fp32 exit $?: $(grep -E 'CHECK (PASS|FAIL)' batch-server-fp32.log)"
pw end
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
python3 - <<'PY'
import collections, json, statistics
rows = collections.defaultdict(list)
for line in open('samples.log'):
    p = line.rstrip('\n').split(' ', 2)
    if len(p) >= 2: rows[p[0]].append(p[1:])
n = foreign = defunct = 0
for t, r in rows.items():
    if not any(x[0] == 'T' for x in r): continue
    n += 1
    procs = [x[1] for x in r if x[0] == 'P' and len(x) > 1]
    dock = [x[1] for x in r if x[0] == 'D' and len(x) > 1]
    defunct += any('<defunct>' in q for q in procs)
    foreign += any(d not in ('lfm3-profile-v1', 'lfm3-omni-dev') for d in dock) or any(
        ('kin_' in q or 'nsys' in q or 'compute-sanitizer' in q or 'lfm3-speech' in q or '/.local/lib/liquid-lfm3/' in q) for q in procs)
print(f"sampler: {n} samples; foreign in {foreign}; defunct (allowed) in {defunct}")
print("MONITOR:", "CLEAN" if n and not foreign else "NOT CLEAN")
m = {t: json.load(open(f'{t}.json')) for t in ('A1', 'B1', 'A2', 'B2', 'R')}
med = {t: m[t]['medians'] for t in m}
print(f"{'clip':9} {'merge A1':>9} {'a6 B1':>9} {'merge A2':>9} {'a6 B2':>9} {'ttft R':>9} {'saved':>8} | prefill merge / a6 / ttft | texts")
ok4 = True
for c in med['A1']:
    a = [med[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; b = [med[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    pa = statistics.mean(med[t][c]['prefill_seconds'] for t in ('A1', 'A2')); pb = statistics.mean(med[t][c]['prefill_seconds'] for t in ('B1', 'B2'))
    ok = max(b) <= statistics.mean(a); ok4 &= ok
    texts = len({med[t][c]['text'] for t in m}) == 1
    print(f"{c:9} {a[0]:9.4f} {b[0]:9.4f} {a[1]:9.4f} {b[1]:9.4f} {med['R'][c]['from_file_seconds']:9.4f} {statistics.mean(a) - statistics.mean(b):8.4f} | "
          f"{pa:.4f} / {pb:.4f} / {med['R'][c]['prefill_seconds']:.4f} | all identical {texts}{'' if ok else '  OVER'}")
print("AMENDMENT 6 RULE 4 (a6 at most merge at every clip):", "PASS" if ok4 else "FAIL")
bench = {c['name']: c for c in json.load(open('bench.json'))['clips']}
hf = [r for r in (json.loads(l) for l in open('hf.jsonl')) if 'name' in r]  # as audio_ttft_stage1_quiet.sh: the rows' median
r3 = {}
for c in ('typical', 'long', 'joined90'):
    h = [r['encode_seconds'] for r in hf if r['name'] == c]
    if c in bench and h: r3[c] = (round(bench[c]['encode_wall_seconds'], 4), round(statistics.median(h), 4))
print("RULE 3 (encoder at most HF's at 35 and 90 s):", "PASS" if all(r3[c][0] <= r3[c][1] for c in ('long', 'joined90') if c in r3) else "FAIL", r3)
d = json.load(open('device-memory.json'))
a6 = [d[k]['after_90']['in_use'] for k in ('a6-1', 'a6-2')]; mg = [d[k]['after_90']['in_use'] for k in ('merge-1', 'merge-2')]
diff = statistics.mean(a6) - statistics.mean(mg)
print(f"AMENDMENT 6 RULE 3 (5b): a6 {[round(x / 2**20) for x in a6]} MiB, merge {[round(x / 2**20) for x in mg]} MiB after the 90 s clip; "
      f"difference of means {diff / 2**20:.0f} MiB (bound 1024): {'PASS' if diff <= 2**30 else 'FAIL'}; pairs "
      f"{[round((a6[i] - mg[i]) / 2**20) for i in range(2)]} MiB; baselines {[round(d[k]['before']['in_use'] / 2**20) for k in ('merge-1', 'a6-1', 'merge-2', 'a6-2')]} MiB")
PY
echo "A6-QUIET-FINISHED $(date +%T)"
echo "== amendment 6's exactness script $(date +%T)"
(cd $B/a6-exact && bash run.sh > run.log 2>&1)
echo "A6-ALL-FINISHED $(date +%T)"
