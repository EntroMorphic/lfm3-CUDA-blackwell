#!/usr/bin/env bash
# validation/tcgen05-v2 rules 2 (the quiet part), 4b and 5, one quiet window after build/tc2-exact finishes.
#   prep: audio-encoder-bench ON (linked with tc2-on's gemm_tcgen05.o) and OFF, from build/tc2-src;
#   quiet (the service stopped, the 1 s sampler, the power state):
#     rule 2: the stream check on the ON server; the batch-server check, ON against OFF as the reference server;
#     rule 5: text prefill (lfm3-speech, 512 / 1,149 / 4,096 tokens, each 4 times per process) and audio time to first
#       token (the server, five clips), OFF and ON alternating ABAB; the audio encoder alone ON and OFF (reported);
#     rule 4b: tcgen05-gemm-v1's soak (check_lfm3_tcgen05_soak.py, 3,600 s, 12 clients, text share 0.25), ON then OFF,
#       the host sampler beside each, then tcgen05_4b_pair_verdict.py (amendments 2 and 2a).
# Usage (Thor host, copied to build/tc2-quiet/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=tc2-quiet; S=tc2-src
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav; T=$C/$S/tools; TH=$R/tools/thor
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
until grep -q "^TC2-EXACT-FINISHED" $B/tc2-exact/run.log 2>/dev/null; do sleep 60; done
echo "== prep $(date +%T)"
for m in on off; do
  extra=""; [ $m = on ] && extra="-DLFM_GEMM_TCGEN05 $C/tc2-on/gemm_tcgen05.o"
  docker run --rm --network none -v $W:/workspace $IMG bash -c "mkdir -p $C/$O/bench-$m && cd $C/$O/bench-$m && nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 \
    -I $C/$S/native/include -DLFM3_NATIVE_GEMM -o audio-encoder-bench $C/$S/native/tests/audio_encoder_bench.cu $C/$S/native/src/safetensors.cpp $extra \
    > build.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"; echo "bench $m exit $?"
done
cp $B/rest-s1/clips.json clips.json
python3 -c "import json; print(' '.join(f\"{c['name']}={c['wav']}\" for c in json.load(open('clips.json'))))" > clip-args.txt
for v in off-1 on-1 off-2 on-2; do mkdir -p text-$v
  for k in 1 2 3 4; do for n in 512 1149 4096; do echo "$C/alloc-text/req/L$n.req.json $C/$O/text-$v/L$n-r$k"; done; done > list-$v.txt
done
(cd $B && sha256sum tc2-on/liquid-lfm3-server tc2-off/liquid-lfm3-server tc2-on/lfm3-speech tc2-off/lfm3-speech $O/bench-on/audio-encoder-bench \
  $O/bench-off/audio-encoder-bench $S/tools/check_lfm3_tcgen05_soak.py $S/tools/tcgen05_4b_pair_verdict.py) | tee binaries.sha256
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
pw stream; echo "== rule 2: the stream check, ON $(date +%T)"
$X python3 $T/check_lfm3_speech_stream.py $C/tc2-on/liquid-lfm3-server $M $C $L $C/$O/stream-check.json > stream-check.log 2>&1; echo "stream check exit $?: $(tail -1 stream-check.log | cut -c1-150)"
pw batch; echo "== rule 2: the batch server, ON against OFF $(date +%T)"
$X python3 $T/check_lfm3_speech_batch_server.py $C/tc2-on/liquid-lfm3-server $C/tc2-off/liquid-lfm3-server $M $C $L $C/$O/batch-server.json \
  > batch-server.log 2>&1; echo "batch server exit $?: $(grep -E 'CHECK (PASS|FAIL)' batch-server.log)"
for v in off-1 on-1 off-2 on-2; do m=${v%%-*}
  pw text-$v; echo "== rule 5: text prefill $v $(date +%T)"
  $X $C/tc2-$m/lfm3-speech $M --batch $C/$O/list-$v.txt > text-$v.log 2>&1; echo "text $v exit $?"
done
for run in off:A1 on:B1 off:A2 on:B2; do m=${run%%:*}; tag=${run#*:}
  pw $tag; echo "== rule 5: time to first token $tag ($m) $(date +%T)"
  $X python3 $T/measure_lfm3_audio_ttft.py $C/tc2-$m/liquid-lfm3-server $M $C/$O/clips.json $C/$O/$tag.json 5 > $tag.log 2>&1; echo "$tag exit $?"
done
for m in off on; do
  pw bench-$m; $X $C/$O/bench-$m/audio-encoder-bench $M $C/$O/bench-$m.json --repeat 5 --mel-threads 4 $(cat clip-args.txt) > bench-$m.log 2>&1; echo "bench $m exit $?"
done
for side in on off; do
  d=$B/$O/4b-$side; mkdir -p $d
  pw 4b-$side; echo "== rule 4b: the soak, $side $(date +%T)"
  rm -f $d/stop; python3 $TH/host_memory_sampler.py $d/host-memory.jsonl $d/stop 30 &
  SP=$!
  $X python3 $T/check_lfm3_tcgen05_soak.py $C/tc2-$side/liquid-lfm3-server $M $C $L $C/vimg-e2e-a4 $C/$O/4b-$side/soak-4b.json 3600 12 0.25 \
    > $d/soak-4b.log 2>&1; s=$?
  sleep 60; touch $d/stop; wait $SP
  echo "4b $side: harness exit $s; $(tail -2 $d/soak-4b.log | tr '\n' ' ' | cut -c1-300)"
done
pw end
kill -TERM $SAMPLER 2>/dev/null; SAMPLER=""
)
mkdir -p pair && ln -sfn ../4b-on pair/on && ln -sfn ../4b-off pair/off
python3 $B/$S/tools/tcgen05_4b_pair_verdict.py $B/$O/pair > verdict.log 2>&1; echo "4b verdict exit $?: $(tail -1 verdict.log | cut -c1-300)"
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
ok = True
print("text prefill (s): request 1 | median of requests 2-4")
for L in (512, 1149, 4096):
    cells = {}
    for v in ('off-1', 'on-1', 'off-2', 'on-2'):
        t = [json.load(open(f'text-{v}/L{L}-r{k}.json'))['prefill_seconds'] for k in (1, 2, 3, 4)]
        cells[v] = (t[0], statistics.median(t[1:]))
    off = statistics.mean(cells[v][1] for v in ('off-1', 'off-2')); on = max(cells[v][1] for v in ('on-1', 'on-2'))
    ok &= on <= off
    print(f"  {L:5d}: " + "  ".join(f"{v} {cells[v][0]:.4f}|{cells[v][1]:.4f}" for v in cells) + f"  ON/OFF {on / off:.3f}{'' if on <= off else '  OVER'}")
m = {t: json.load(open(f'{t}.json'))['medians'] for t in ('A1', 'B1', 'A2', 'B2')}
print("time to first token (s): OFF A1, ON B1, OFF A2, ON B2, saved, texts")
for c in m['A1']:
    a = [m[t][c]['from_file_seconds'] for t in ('A1', 'A2')]; b = [m[t][c]['from_file_seconds'] for t in ('B1', 'B2')]
    texts = len({m[t][c]['text'] for t in m}) == 1; good = max(b) <= statistics.mean(a) and texts; ok &= good
    print(f"  {c:9} {a[0]:.4f} {b[0]:.4f} {a[1]:.4f} {b[1]:.4f} {statistics.mean(a) - statistics.mean(b):.4f} {texts}{'' if good else '  OVER'}")
bo = {c['name']: c for c in json.load(open('bench-off.json'))['clips']}; bn = {c['name']: c for c in json.load(open('bench-on.json'))['clips']}
print("the encoder alone (reported), OFF / ON (s):", {c: (round(bo[c]['encode_wall_seconds'], 4), round(bn[c]['encode_wall_seconds'], 4), bo[c]['encoded_fnv1a'] == bn[c]['encoded_fnv1a']) for c in bo})
print("RULE 5:", "PASS" if ok else "FAIL")
PY
echo "TC2-QUIET-FINISHED $(date +%T)"
