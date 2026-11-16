#!/usr/bin/env bash
# validation/audio-speed-v1 rule 0 (also audio-frontend-v1's baseline): where an audio-in request's time to first token
# goes, on a quiet Thor in MAXN, the power state recorded at the start and checked at the end. Five ASR clips: short
# (4446-2275-0004, 2.11 s), the test-clean median length, typical (1089-134686-0000, 10.4 s), long (the longest
# test-clean utterance, 4507-16021-0047, 34.96 s) and the 90-second joined recording.
#   1. audio-encoder-bench (BENCH_TAG): host log_mel and the GPU encoder alone, 1 untimed + 5 timed runs each, and the
#      GPU's FP64/FP32 FMA rates (encode-only);
#   2. lfm3-speech (SPEECH_TAG, the current engine): the same requests, record_audio off, 1 untimed + 5 timed passes:
#      encode_seconds (log_mel + encoder), prefill_seconds (that + the language model's prefill) and the decode
#      (end to end per request; time to first token = prefill + one mean decode step);
#   3. HF BF16 (tools/time_lfm3_audio_hf.py), 1 untimed + 5 timed passes;
#   4. nsys (host) on audio-encoder-bench, the typical and 90-second clips, 1 run: the encoder's kernel shares.
# Usage (Thor host): audio_speed_rule0.sh BENCH_TAG SPEECH_TAG OUT_TAG   -> build/OUT_TAG/{summary.json, ...}
set -uo pipefail
BT=${1:?BENCH_TAG}; ST=${2:?SPEECH_TAG}; O=${3:?OUT_TAG}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE to the workspace (the directory holding liquid-CUDA)}
B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)  # this script's tools/ (a worktree copy or the repo's)
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
H="docker exec -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tmp/cuda-13.0-ptxas/ptxas lfm3-profile-v1 /workspace/venvs/hf-5.20.2/bin/python"
mkdir -p $B/$O/native $B/$O/req && cd $B/$O
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3|[a]udio-encoder-bench" | head -3)
[ -n "$busy" ] && { echo "NOT QUIET: $busy"; exit 1; }
bash $TOOLS/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power (not MAXN)"; exit 1; }
cp $TOOLS/time_lfm3_audio_hf.py hf_tool.py  # the tool as run, inside the container's view
# the clips and their native requests (record_audio off: dumps would add host copies to the timing)
python3 - $B $O <<'PY' || { echo "FAIL requests"; exit 1; }
import json, sys
from pathlib import Path
b, o = Path(sys.argv[1]), sys.argv[2]
cases = json.load(open(b / "audioin-v2/cases.json"))["cases"]
secs = sorted(c["seconds"] for c in cases)
median = secs[len(secs) // 2]
mid = min(cases, key=lambda c: (abs(c["seconds"] - median), c["case"]))
pick = [("short", "4446-2275-0004"), ("median", mid["case"]), ("typical", "1089-134686-0000"),
        ("long", "4507-16021-0047")]
clips = []
for name, case in pick:
    r = json.load(open(b / f"audioin-v2/full/requests/{case}.req.json"))
    r["record_audio"] = False
    (b / o / "req" / f"{name}.req.json").write_text(json.dumps(r) + "\n")
    clips.append(dict(name=name, case=case, wav=r["audio"][0]))
j = json.load(open(b / "server-stream-v3-check/work/asr-joined-90s.json"))
wav = "/workspace/liquid-CUDA/build/server-stream-v3-check/work/joined-90s.wav"
r = dict(prompt_ids=j["prompt"], audio=[wav], mode="text", max_new_tokens=2048, seed=0, record_audio=False)
(b / o / "req" / "joined90.req.json").write_text(json.dumps(r) + "\n")
clips.append(dict(name="joined90", case="asr-joined-90s", wav=wav))
(b / o / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
(b / o / "clip-args.txt").write_text(" ".join(f"{c['name']}={c['wav']}" for c in clips) + "\n")
PY
args=$(cat clip-args.txt)
echo "== bench $(date +%T)"
$X $C/$BT/audio-encoder-bench $M $C/$O/bench.json --repeat 5 --fp64 $args > bench.log 2>&1 || { echo "bench FAILED"; tail -3 bench.log; }
echo "== native $(date +%T)"
for p in warm p0 p1 p2 p3 p4; do for n in short median typical long joined90; do echo "$C/$O/req/$n.req.json $C/$O/native/$p-$n"; done; done > native-list.txt
$X $C/$ST/lfm3-speech $M --batch $C/$O/native-list.txt > native.log 2>&1 || { echo "native FAILED"; tail -3 native.log; }
echo "== HF $(date +%T)"
$H $C/$O/hf_tool.py $M $C/$O/hf.jsonl 5 2048 $args > hf.log 2>&1 || { echo "HF FAILED"; tail -3 hf.log; }
bash $TOOLS/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID: power state changed"
echo "== nsys (host) $(date +%T)"
if command -v nsys > /dev/null; then
  hargs=$(python3 -c "import json; print(' '.join(c['name'] + '=' + c['wav'].replace('/workspace', '$W') for c in json.load(open('clips.json')) if c['name'] in ('typical', 'joined90')))")
  nsys profile -o nsys-encoder --force-overwrite true -t cuda $B/$BT/audio-encoder-bench $W/models/LFM3-Omni-2B nsys-bench.json --repeat 1 $hargs > nsys.log 2>&1 \
    && nsys stats -r cuda_gpu_kern_sum -f csv -o nsys-kernels nsys-encoder.nsys-rep >> nsys.log 2>&1 || echo "nsys FAILED (see nsys.log)"
else
  echo "no nsys on the host" | tee nsys.log
fi
python3 - $B/$O <<'PY'
import json, statistics, sys
from pathlib import Path
o = Path(sys.argv[1])
med = statistics.median
bench = {c["name"]: c for c in json.load(open(o / "bench.json"))["clips"]}
hf = [json.loads(l) for l in open(o / "hf.jsonl")][1:]
rows = {}
for name, c in bench.items():
    nat = [json.load(open(o / "native" / f"p{k}-{name}.json")) for k in range(5)]
    step = med([j["lm_step_seconds"] / max(1, j["decode_steps"]) for j in nat])
    enc, pre, gen = (med([j[k] for j in nat]) for k in ("encode_seconds", "prefill_seconds", "generate_seconds"))
    h = [r for r in hf if r["name"] == name]
    ttft = pre + step
    rows[name] = dict(
        audio_seconds=c["audio_seconds"], tokens=c["tokens"],
        bench=dict(log_mel=c["log_mel_seconds"], encoder=c["encode_wall_seconds"], encoder_gpu=c["encode_gpu_seconds"]),
        native=dict(encode=enc, prefill=pre, lm_prefill=pre - enc, step=step, time_to_first_token=ttft, decode=gen,
                    end_to_end=pre + gen, new_tokens=len(nat[0]["text_tokens"])),
        hf=dict(processor=med([r["processor_seconds"] for r in h]), feature_extractor=med([r["feature_extractor_seconds"] for r in h]),
                encode=med([r["encode_seconds"] for r in h]), time_to_first_token=med([r["time_to_first_token_seconds"] for r in h]),
                end_to_end=med([r["end_to_end_seconds"] for r in h]), new_tokens=h[0]["new_tokens"]) if h else None,
        shares=dict(log_mel=c["log_mel_seconds"] / ttft, encoder=c["encode_wall_seconds"] / ttft))
fma = json.load(open(o / "bench.json")).get("fma_flops")
go = any(rows[n]["shares"]["encoder"] >= 0.15 for n in ("typical", "long") if n in rows)
out = dict(rows=rows, fma_flops=fma, rule0_encoder_share_ge_15pct_at_typical_or_long=go,
           note="encode-only: bench; end to end: native time_to_first_token = prefill + one mean step (the request "
                "pre-tokenized, the WAV read outside the timing); HF's from the file read to the first new token")
(o / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"{'clip':9s} {'audio s':>7s} {'log_mel':>8s} {'encoder':>8s} {'LM pre':>7s} {'nat TTFT':>8s} {'HF TTFT':>8s} | {'mel %':>6s} {'enc %':>6s}")
for n, r in rows.items():
    h = r["hf"]["time_to_first_token"] if r["hf"] else float("nan")
    print(f"{n:9s} {r['audio_seconds']:7.2f} {r['bench']['log_mel']:8.4f} {r['bench']['encoder']:8.4f} {r['native']['lm_prefill']:7.4f} "
          f"{r['native']['time_to_first_token']:8.4f} {h:8.4f} | {100 * r['shares']['log_mel']:5.1f}% {100 * r['shares']['encoder']:5.1f}%")
if fma:
    print(f"FMA: FP64 {fma['fp64'] / 1e9:.1f} GFLOP/s, FP32 {fma['fp32'] / 1e9:.1f} GFLOP/s")
print("rule 0:", "GO (encoder >= 15% of time to first token at typical or long)" if go else "STOP (encoder < 15% at typical and long)")
PY
echo "AUDIO RULE 0 DONE $(date +%T)"
