#!/usr/bin/env bash
# validation/audio-ttft-v1 stage 0 (measure first, reported): on a quiet Thor, the five clips of audio-speed-v1 rule 0.
#   A. the deployed server (build/merge-srv, a681eab4...) fresh in lfm3-profile-v1: tools/measure_lfm3_audio_ttft.py,
#      1 untimed + 5 timed passes (time to first token from the file, and from the first byte);
#   B. audio-encoder-bench built from the merged tree (build/ttft-bench): log_mel alone (F2, 4 threads) and the encoder
#      alone, --repeat 5;
#   C. HF BF16 (tools/time_lfm3_audio_hf.py): 1 untimed + 5 timed passes, 32 new tokens;
#   D. the server again (A2), for drift across the window.
# The bench is built before the quiet wait. The 1 s sampler and the power state before every step; the always-on
# service stopped for the window and started again however it ends. Waits for 15 quiet minutes.
# Usage (Thor host, copied to build/ttft-s0/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=ttft-s0
M=/workspace/models/LFM3-Omni-2B; S=merge-src; T=$C/$S/tools; SRV=merge-srv; BENCH=ttft-bench
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
H="docker exec -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tmp/cuda-13.0-ptxas/ptxas lfm3-profile-v1 /workspace/venvs/hf-5.20.2/bin/python"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
fin() { echo "TTFT-S0-FINISHED $(date +%T)"; }
cd $B/$O || exit 1
[ -f $B/quiet-w3-audio/clips.json ] && cp $B/quiet-w3-audio/clips.json clips.json || { echo "NO CLIPS"; fin; exit 1; }
python3 -c "import json; print(' '.join(f\"{c['name']}={c['wav']}\" for c in json.load(open('clips.json'))))" > clip-args.txt
[ -f $B/$S/tools/measure_lfm3_audio_ttft.py ] || { echo "NO TOOL in $S"; fin; exit 1; }
cp $B/$S/tools/time_lfm3_audio_hf.py hf_tool.py

echo "== build the bench from $S $(date +%T)"
[ -x $B/$BENCH/audio-encoder-bench ] || LFM3_WORKSPACE=$W bash $B/$S/tools/thor/build_audio_bench.sh $BENCH build/$S > build-bench.log 2>&1 \
  || { echo "BENCH BUILD FAILED: $(tail -2 build-bench.log)"; fin; exit 1; }
(cd $B && sha256sum $SRV/liquid-lfm3-server $BENCH/audio-encoder-bench) | tee binaries.sha256

wait_quiet 2880 || { echo "NOT QUIET after 24 h"; fin; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
SAMPLER=""
trap '[ -n "$SAMPLER" ] && kill -TERM $SAMPLER 2>/dev/null; [ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; fin; exit 1; }
echo "== quiet $(date +%T)"
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; fin; exit 1; }
( while true; do t=$(date +%s.%N); pgrep -af "$PAT" | sed "s|^|$t P |"; docker ps --format "{{.Names}}" | sed "s|^|$t D |"; echo "$t T"; sleep 1; done ) > samples.log 2>&1 &
SAMPLER=$!
power_bad=0; failed=0
pw() { bash $R/tools/thor/power_state.sh power-$1.json power-start.json > /dev/null || { power_bad=1; echo "POWER DIFFERS before $1"; }; }

pw A; echo "== A. the server $(date +%T)"
$X python3 $T/measure_lfm3_audio_ttft.py $C/$SRV/liquid-lfm3-server $M $C/$O/clips.json $C/$O/server-A.json 5 > server-A.log 2>&1 || failed=1
echo "server A exit $?"; grep "time to first token" server-A.log
pw B; echo "== B. the bench $(date +%T)"
$X $C/$BENCH/audio-encoder-bench $M $C/$O/bench.json --repeat 5 --mel-threads 4 $(cat clip-args.txt) > bench.log 2>&1 || failed=1
echo "bench exit $?"
pw C; echo "== C. HF $(date +%T)"
$H $C/$O/hf_tool.py $M $C/$O/hf.jsonl 5 32 $(cat clip-args.txt) > hf.log 2>&1 || failed=1
echo "HF exit $?"
pw D; echo "== D. the server again $(date +%T)"
$X python3 $T/measure_lfm3_audio_ttft.py $C/$SRV/liquid-lfm3-server $M $C/$O/clips.json $C/$O/server-D.json 5 > server-D.log 2>&1 || failed=1
echo "server D exit $?"; grep "time to first token" server-D.log
kill -TERM $SAMPLER; wait $SAMPLER 2>/dev/null; SAMPLER=""
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || { power_bad=1; echo "POWER DIFFERS at the end"; }
echo "power: $([ $power_bad = 0 ] && echo "unchanged before every step and at the end" || echo "CHANGED: void")"
echo "steps: $([ $failed = 0 ] && echo "all exited 0" || echo "A STEP FAILED: void")"
docker exec lfm3-profile-v1 chown -R 1000:1000 $C/$O 2>/dev/null

python3 - "$C" "$SRV" <<'PY'
import collections, sys
C, srv = sys.argv[1], sys.argv[2]
ticks, foreign, defunct, servers = set(), [], set(), collections.Counter()
for line in open("samples.log"):
    parts = line.rstrip("\n").split(" ", 2)
    if len(parts) < 2: continue
    t, kind = parts[0], parts[1]; ticks.add(t)
    if kind == "T": continue
    rest = parts[2] if len(parts) > 2 else ""
    if kind == "D":
        if rest not in ("lfm3-profile-v1", "lfm3-omni-dev"): foreign.append((t, "container " + rest))
        continue
    cmd = rest.split(" ", 1)[1] if " " in rest else rest
    if cmd.endswith("<defunct>"): defunct.add(t); continue
    if "ttft-s0" in cmd: continue
    exe = cmd.split(" ", 1)[0]
    if exe.startswith(f"{C}/{srv}/") and exe.endswith(("liquid-lfm3-server", "liquid-lfm3-worker")):
        if exe.endswith("server"): servers[t] += 1
        continue
    foreign.append((t, cmd[:160]))
many = [t for t, n in servers.items() if n > 1]
print(f"sampler: {len(ticks)} samples; foreign in {len({t for t, _ in foreign})}; more than one server in {len(many)}; defunct (allowed) in {len(defunct)}")
for t, w in foreign[:10]: print("  foreign", t, w)
print("MONITOR:", "CLEAN" if ticks and not foreign and not many else "VOID")
PY

python3 - <<'PY'
import json, statistics as st
names = ["short", "median", "typical", "long", "joined90"]
A, D = (json.load(open(f"server-{k}.json"))["medians"] for k in "AD")
bench = {c["name"]: c for c in json.load(open("bench.json"))["clips"]}
hf = [r for r in (json.loads(l) for l in open("hf.jsonl")) if "name" in r]  # the first line is the environment; rows are timed passes
rows = {}
for n in names:
    h = [r for r in hf if r["name"] == n]
    nat = st.median([A[n]["from_file_seconds"], D[n]["from_file_seconds"]])
    rows[n] = dict(audio_seconds=bench[n]["audio_seconds"], server_A=A[n]["from_file_seconds"], server_D=D[n]["from_file_seconds"],
                   server_from_send_A=A[n]["from_send_seconds"], server_prefill_A=A[n]["prefill_seconds"],
                   log_mel=bench[n]["log_mel_seconds"], encoder=bench[n]["encode_wall_seconds"],
                   hf_ttft=st.median(r["time_to_first_token_seconds"] for r in h),
                   hf_features=st.median(r["feature_extractor_seconds"] for r in h),
                   hf_encoder=st.median(r["encode_seconds"] for r in h),
                   texts_identical=A[n]["texts_identical"] and D[n]["texts_identical"] and A[n]["text"] == D[n]["text"])
    r = rows[n]
    r["native_mean_AD"] = (r["server_A"] + r["server_D"]) / 2
    r["ratio"] = r["native_mean_AD"] / r["hf_ttft"]
    r["rest"] = r["native_mean_AD"] - r["log_mel"] - r["encoder"]
json.dump(rows, open("summary.json", "w"), indent=1)
print(f"{'clip':9s} {'audio':>6s} {'srv A':>7s} {'srv D':>7s} {'HF':>7s} {'ratio':>6s} {'log_mel':>8s} {'encoder':>8s} {'rest':>7s} {'HF feat':>8s} {'HF enc':>7s}")
for n, r in rows.items():
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else "   -  "
    print(f"{n:9s} {r['audio_seconds']:6.2f} {r['server_A']:7.4f} {r['server_D']:7.4f} {r['hf_ttft']:7.4f} {r['ratio']:6.2f} "
          f"{r['log_mel']:8.4f} {r['encoder']:8.4f} {r['rest']:7.4f} {f(r['hf_features']):>8s} {f(r['hf_encoder']):>7s}  texts identical {r['texts_identical']}")
PY
fin
