#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 5 (speed), for an unattended quiet window: the build with the tcgen05 path (tgemm-on1)
# against the same sources without it (tgemm-off1). Three rounds; in each, ON and OFF alternate in order:
#   1. vision encode at 8,192 patches: lfm3-vision-check on the maximum-size case, --repeat 3 (warm median of 3);
#   2. LM prefill at 4,096 and 32,768 tokens: lfm3-speech text requests (1 new token), one untimed pass then 3 timed,
#      prefill_seconds;
#   3. batch-32 aggregate decode tok/s: lfm3-batch, batch_round.sh's rate-32 request (greedy, 256 new tokens, warm-up);
#   4. image requests end to end against HF: tools/thor/vision_speed_round.sh on the ON build (11 stage-0 requests).
# Bars (README rule 5, a direction counts in 2 of 3 rounds): encode and both prefills faster ON than OFF; native end to
# end at most HF's on all 11 requests; batch-32 decode tok/s ON >= 0.99 x OFF.
# No arguments. Assumes a quiet MAXN machine; refuses to start otherwise. Results: build/quiet-w3-tcgen05/.
# Exit 0: every measurement completed (verdicts in summary.txt); 1: not quiet or not MAXN; 2: a measurement failed.
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
TOOLS=$W/liquid-CUDA/tools/thor   # the main checkout's (unchanged) power and vision speed scripts
ON=tgemm-on1; OFF=tgemm-off1; O=quiet-w3-tcgen05; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 2
log() { echo "$(date +%T) $*" | tee -a run.log; }
fail() { log "FAIL: $*"; exit 2; }
# Quiet: no engine, sanitizer, probe or HF process (bracketed patterns: never this script or the window's "quiet_" runner).
busy=$(pgrep -af "[l]fm3-speech|[l]fm3-batch|[l]fm3-vision-check|[l]iquid-lfm3|[c]ompute-sanitizer|[t]cgen05-gemm-probe|[t]cgen05-gemm-check|[r]un_lfm3|[c]heck_lfm3|[t]ime_lfm3" | head -3)
[ -n "$busy" ] && { log "NOT QUIET: $busy"; exit 1; }
bash $TOOLS/power_state.sh power-start.json > /dev/null || { log "FAIL power (not MAXN)"; exit 1; }
(cd $B && sha256sum $ON/lfm3-speech $ON/lfm3-vision-check $ON/lfm3-batch $OFF/lfm3-speech $OFF/lfm3-vision-check $OFF/lfm3-batch) > binaries.sha256 || fail "binaries missing"
# Requests: text prompts of 4,096 and 32,768 tokens (a real token stream repeated) and the batch-32 rate request.
python3 - $B $B/$O <<'PY' || fail "requests"
import json, sys
B, O = sys.argv[1:]
src = json.load(open(f"{B}/speech-e2e-v1/requests/il-fridge-s1.req.json"))["prompt_ids"]
# The model's context limit is 32,768 positions: the "32,768" case is 32,767 prompt tokens + 1 new in capacity 32,768.
for n, prompt, cap in ((4096, 4096, 4352), (32768, 32767, 32768)):
    ids = (src * (prompt // len(src) + 1))[:prompt]
    json.dump({"prompt_ids": ids, "mode": "text", "max_new_tokens": 1, "seed": 1, "capacity": cap}, open(f"{O}/prefill-{n}.req.json", "w"))
p = json.load(open(f"{B}/cli-demo/out2.json"))["input_ids"]
json.dump(dict(capacity=4096, prompts=[p], batch=32, max_new_tokens=256, temperature=0, stop=[], warmup=True), open(f"{O}/rate-32.json", "w"))
PY
# The maximum-size image (8,192 patches), made by tools/thor/vision_sanitizers.sh
MAXCASE=$C/vision-v1/san/cases/max
test -f $B/vision-v1/san/cases/max/pixel_values.npy || fail "no maximum-size vision case"
for r in 1 2 3; do
  if [ $r = 2 ]; then order="$OFF $ON"; else order="$ON $OFF"; fi
  for b in $order; do
    side=$([ $b = $ON ] && echo on || echo off); d=r$r-$side; mkdir -p $d
    log "round $r $side: vision encode"
    $X $C/$b/lfm3-vision-check $M $MAXCASE $C/$O/$d/vision --linear x2 --attention x2 --repeat 3 > $d/vision.log 2>&1 || fail "vision $d"
    log "round $r $side: prefill"
    for n in 4096 32768; do
      for p in warm t1 t2 t3; do echo "$C/$O/prefill-$n.req.json $C/$O/$d/prefill-$n-$p"; done
    done > $d/prefill-list.txt
    $X $C/$b/lfm3-speech $M --batch $C/$O/$d/prefill-list.txt > $d/prefill.log 2>&1 || fail "prefill $d"
    log "round $r $side: batch-32 decode"
    $X $C/$b/lfm3-batch $M $C/$O/rate-32.json $C/$O/$d/rate-32.out.json > $d/rate-32.log 2>&1 || fail "batch $d"
  done
  log "round $r: image requests against HF (ON build)"
  bash $TOOLS/vision_speed_round.sh $ON $O/image-r$r > image-r$r.log 2>&1 || fail "image round $r"
  grep -q "VISION SPEED DONE" image-r$r.log || fail "image round $r incomplete"
done
bash $TOOLS/power_state.sh power-end.json power-start.json > /dev/null || log "VOID: power state changed during the window"
python3 - $B/$O <<'PY' | tee summary.txt
import json, statistics, sys
from pathlib import Path
o = Path(sys.argv[1]); med = statistics.median
def vision(d): return json.load(open(o / d / "vision" / "stages.json"))["seconds"][0]
def prefill(d, n): return med(json.load(open(o / d / f"prefill-{n}-t{i}.json"))["prefill_seconds"] for i in (1, 2, 3))
def rate(d): return json.load(open(o / d / "rate-32.out.json"))["decode_tokens_per_second"]
rows = []
def judge(name, on, off, better):  # better(on, off) per round; passes in >= 2 of 3
    wins = [better(a, b) for a, b in zip(on, off)]
    ok = sum(wins) >= 2
    rows.append(ok)
    print(f"{name:28s} ON {['%.4f' % x for x in on]}  OFF {['%.4f' % x for x in off]}  "
          f"ON/OFF median {med(on) / med(off):.3f}  {'PASS' if ok else 'FAIL'} ({sum(wins)} of 3)")
R = (1, 2, 3)
judge("vision encode 8192 (s)", [vision(f"r{r}-on") for r in R], [vision(f"r{r}-off") for r in R], lambda a, b: a < b)
for n in (4096, 32768):
    judge(f"prefill {n} (s)", [prefill(f"r{r}-on", n) for r in R], [prefill(f"r{r}-off", n) for r in R], lambda a, b: a < b)
judge("batch-32 decode (tok/s)", [rate(f"r{r}-on") for r in R], [rate(f"r{r}-off") for r in R], lambda a, b: a >= 0.99 * b)
img = []
for r in R:
    s = json.load(open(o / f"image-r{r}" / "summary.json"))
    faster = all(v["native"]["end_to_end"] <= v["hf"]["end_to_end"] for v in s.values())
    img.append(faster)
    print(f"image round {r}: native at most HF end to end on {sum(v['native']['end_to_end'] <= v['hf']['end_to_end'] for v in s.values())} of {len(s)}")
ok = sum(img) >= 2; rows.append(ok)
print(f"image requests vs HF: {'PASS' if ok else 'FAIL'} ({sum(img)} of 3 rounds on all requests)")
print("RULE 5:", "PASS" if all(rows) else "FAIL")
PY
[ ${PIPESTATUS[0]} = 0 ] || fail "summary"
log "QUIET TCGEN05 DONE"
