#!/usr/bin/env bash
# Quiet window 3: validation/audio-frontend-v1's timing rules (amended rule 2; amendment 2's F1 against F2; the threads
# rule), on a quiet Thor in MAXN, unattended, no arguments. Results under build/quiet-w3-audio/; exits nonzero if a
# measurement cannot be made (not quiet, not MAXN, a run failing); the rules' verdicts are in summary.json.
# The front ends: today's DFT (dft), F1 (exact-f1), F2 (exact-f2), chosen by LFM3_AUDIO_FEATURES. Clips (rule 0's): short
# 4446-2275-0004 (2.11 s), the test-clean median, typical 1089-134686-0000 (10.4 s), long 4507-16021-0047 (34.96 s),
# and the 90-second joined recording.
#   1. audio-encoder-bench (build/audio-bench-a2): log_mel alone (front end only) and the encoder alone (encode-only),
#      each front end at 1 and 4 threads, 1 untimed + 5 timed runs per clip;
#   2. lfm3-speech (build/audio-fe-speech-a3, 4 threads) in each mode, 1 untimed + 5 timed passes over the five requests:
#      time to first token = prefill + one mean decode step (end to end, the request pre-tokenized);
#   3. HF BF16 (tools/time_lfm3_audio_hf.py), 1 untimed + 5 timed passes: its time to first token.
# The verdicts (amended rule 2): time to first token at most HF's at 10.4, 35 and 90 s for each front end; F2 or F1 by
# the lower 90 s front-end time; 4 threads kept only if they cut the 90 s front-end time by at least 20% against 1.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$(cd "$HERE/.." && pwd); W=$(cd "$HERE/../../../../.." && pwd)
B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
BENCH=audio-bench-a2; SPEECH=audio-fe-speech-a3; O=quiet-w3-audio
fail() { echo "FAIL: $*"; exit 1; }
[ -x $B/$BENCH/audio-encoder-bench ] || fail "no $B/$BENCH/audio-encoder-bench"
[ -x $B/$SPEECH/lfm3-speech ] || fail "no $B/$SPEECH/lfm3-speech"
X() { docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=$1 lfm3-profile-v1 "${@:2}"; }
H="docker exec -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tmp/cuda-13.0-ptxas/ptxas lfm3-profile-v1 /workspace/venvs/hf-5.20.2/bin/python"
mkdir -p $B/$O/req $B/$O/native && cd $B/$O || fail "cannot make $B/$O"
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3|[a]udio-encoder-bench|[a]udio-mel-" | head -3)
[ -n "$busy" ] && fail "not quiet: $busy"
bash $TOOLS/thor/power_state.sh power-start.json > /dev/null || fail "power state (not MAXN)"
(cd $B && sha256sum $BENCH/audio-encoder-bench $SPEECH/lfm3-speech) > binaries.sha256
cp $TOOLS/time_lfm3_audio_hf.py hf_tool.py
python3 - $B $O <<'PY' || fail "requests"
import json, sys
from pathlib import Path
b, o = Path(sys.argv[1]), sys.argv[2]
cases = json.load(open(b / "audioin-v2/cases.json"))["cases"]
secs = sorted(c["seconds"] for c in cases)
median = secs[len(secs) // 2]
mid = min(cases, key=lambda c: (abs(c["seconds"] - median), c["case"]))
clips = []
for name, case in [("short", "4446-2275-0004"), ("median", mid["case"]), ("typical", "1089-134686-0000"),
                   ("long", "4507-16021-0047")]:
    r = json.load(open(b / f"audioin-v2/full/requests/{case}.req.json"))
    r["record_audio"] = False
    (b / o / "req" / f"{name}.req.json").write_text(json.dumps(r) + "\n")
    clips.append(dict(name=name, case=case, wav=r["audio"][0]))
j = json.load(open(b / "server-stream-v3-check/work/asr-joined-90s.json"))
wav = "/workspace/liquid-CUDA/build/server-stream-v3-check/work/joined-90s.wav"
(b / o / "req" / "joined90.req.json").write_text(json.dumps(dict(prompt_ids=j["prompt"], audio=[wav], mode="text",
                                                                 max_new_tokens=2048, seed=0, record_audio=False)) + "\n")
clips.append(dict(name="joined90", case="asr-joined-90s", wav=wav))
(b / o / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
(b / o / "clip-args.txt").write_text(" ".join(f"{c['name']}={c['wav']}" for c in clips) + "\n")
PY
args=$(cat clip-args.txt)
MODES="dft exact-f1 exact-f2"
for mode in $MODES; do for th in 1 4; do
  echo "== bench $mode, $th threads $(date +%T)"
  X $mode $C/$BENCH/audio-encoder-bench $M $C/$O/bench-$mode-t$th.json --repeat 5 --mel-threads $th $args \
    > bench-$mode-t$th.log 2>&1 || fail "bench $mode t$th: $(tail -1 bench-$mode-t$th.log)"
done; done
for mode in $MODES; do
  echo "== lfm3-speech $mode $(date +%T)"
  mkdir -p native/$mode
  for p in warm p0 p1 p2 p3 p4; do for n in short median typical long joined90; do
    echo "$C/$O/req/$n.req.json $C/$O/native/$mode/$p-$n"; done; done > native-$mode-list.txt
  X $mode $C/$SPEECH/lfm3-speech $M --batch $C/$O/native-$mode-list.txt > native-$mode.log 2>&1 \
    || fail "lfm3-speech $mode: $(tail -1 native-$mode.log)"
done
echo "== HF $(date +%T)"
$H $C/$O/hf_tool.py $M $C/$O/hf.jsonl 5 2048 $args > hf.log 2>&1 || fail "HF: $(tail -1 hf.log)"
bash $TOOLS/thor/power_state.sh power-end.json power-start.json > /dev/null || fail "VOID: the power state changed during the run"
python3 - $B/$O "$MODES" <<'PY' || fail "summary"
import json, statistics, sys
from pathlib import Path
o, modes = Path(sys.argv[1]), sys.argv[2].split()
med = statistics.median
names = ["short", "median", "typical", "long", "joined90"]
hf = [json.loads(l) for l in open(o / "hf.jsonl")][1:]
rows = {}
for mode in modes:
    bench = {t: {c["name"]: c for c in json.load(open(o / f"bench-{mode}-t{t}.json"))["clips"]} for t in (1, 4)}
    rows[mode] = {}
    for n in names:
        nat = [json.load(open(o / "native" / mode / f"p{k}-{n}.json")) for k in range(5)]
        step = med([j["lm_step_seconds"] / max(1, j["decode_steps"]) for j in nat])
        pre = med([j["prefill_seconds"] for j in nat])
        enc = med([j["encode_seconds"] for j in nat])
        rows[mode][n] = dict(audio_seconds=bench[1][n]["audio_seconds"],
                             log_mel_t1=bench[1][n]["log_mel_seconds"], log_mel_t4=bench[4][n]["log_mel_seconds"],
                             encoder=bench[4][n]["encode_wall_seconds"], native_encode=enc, native_prefill=pre,
                             native_ttft=pre + step,
                             hf_ttft=med([r["time_to_first_token_seconds"] for r in hf if r["name"] == n]))
verdict = {}
for mode in modes:
    v = {n: rows[mode][n]["native_ttft"] <= rows[mode][n]["hf_ttft"] for n in ("typical", "long", "joined90")}
    verdict[mode] = dict(ttft_at_most_hf=v, rule2_pass=all(v.values()))
f1, f2 = rows["exact-f1"]["joined90"]["log_mel_t4"], rows["exact-f2"]["joined90"]["log_mel_t4"]
ships = "exact-f2" if f2 <= f1 else "exact-f1"
t1, t4 = rows[ships]["joined90"]["log_mel_t1"], rows[ships]["joined90"]["log_mel_t4"]
out = dict(rows=rows, verdict=verdict, ships=ships, threads=dict(t1=t1, t4=t4, keep_4=t4 <= 0.8 * t1),
           note="front end only: log_mel_t1/t4 (bench); encode-only: encoder (bench); end to end: native_ttft = prefill + "
                "one mean step, hf_ttft from the file read to the first new token")
(o / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
for mode in modes:
    print(f"-- {mode}: rule 2 {'PASS' if verdict[mode]['rule2_pass'] else 'FAIL'}")
    print(f"   {'clip':9s} {'audio s':>7s} {'mel t1':>8s} {'mel t4':>8s} {'encoder':>8s} {'nat TTFT':>9s} {'HF TTFT':>8s}")
    for n in names:
        r = rows[mode][n]
        print(f"   {n:9s} {r['audio_seconds']:7.2f} {r['log_mel_t1']:8.4f} {r['log_mel_t4']:8.4f} {r['encoder']:8.4f} "
              f"{r['native_ttft']:9.4f} {r['hf_ttft']:8.4f}")
print(f"ships: {ships} (90 s front end, 4 threads: F1 {f1:.4f} s, F2 {f2:.4f} s); 4 threads kept: {t4 <= 0.8 * t1} "
      f"({t1:.4f} s at 1, {t4:.4f} s at 4)")
PY
echo "QUIET AUDIO FRONTEND DONE $(date +%T)"
