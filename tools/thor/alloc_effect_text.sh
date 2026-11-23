#!/usr/bin/env bash
# validation/alloc-effect-v1 stage 0 (a diagnostic): does a text-only prefill run in the slow state that audio-ttft-v1
# found (the language model's large GEMMs ~1.8x slower when no cudaMalloc/cudaFree came before them)? lfm3-speech from
# main (build/merge4-src) with a diagnostic switch added in a throwaway copy (not committed to the engine):
# LFM3_DIAG_TOUCH set -> one 2 MiB cudaMalloc + cudaFree right before each request's prefill. Text-only prompts of
# 128, 512, 1,149 and 4,096 tokens (the 90 s request's prompt with its audio span replaced by ordinary token ids),
# 8 new tokens; each process runs every length 4 times; processes off, on, off, on. Quiet Thor.
# Usage (Thor host, copied to build/alloc-text/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=alloc-text
M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3|[a]udio-mel-check|[a]udio-handler-bench"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
echo "== the diagnostic source and build $(date +%T)"
D=$B/alloc-diag-src; mkdir -p $D && cp -r $B/merge4-src/native $B/merge4-src/tools $D/ 2>/dev/null
python3 - $D/native/src/lfm3_speech.cuh <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "    const auto t0 = Clock::now();\n    runner_.reset();\n"
assert s.count(a) == 1
s = s.replace(a, "    { static const bool touch = std::getenv(\"LFM3_DIAG_TOUCH\") != nullptr;  // alloc-effect-v1 DIAGNOSTIC ONLY\n"
                 "      if (touch) { void* t = nullptr; cuda_check(cudaMalloc(&t, size_t(2) << 20)); cuda_check(cudaFree(t)); } }\n" + a)
open(p, "w").write(s)
PY
grep -c "alloc-effect-v1 DIAGNOSTIC" $D/native/src/lfm3_speech.cuh
LFM3_WORKSPACE=$W bash $B/merge4-src/tools/thor/build_speech.sh alloc-diag-speech build/alloc-diag-src > build.log 2>&1; echo "build exit $?"
sha256sum $B/alloc-diag-speech/lfm3-speech | tee binaries.sha256
mkdir -p req
python3 - $B/quiet-w3-audio/req/joined90.req.json req <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); p = r["prompt_ids"]
head = [t for t in p[:15] if t not in (125009, 125010, 125031)]
tail = [t for t in p[-10:] if t not in (125009, 125010, 125031)]
for n in (128, 512, 1149, 4096):
    filler = [1000 + (i * 37) % 20000 for i in range(n - len(head) - len(tail))]
    q = {"prompt_ids": head + filler + tail, "mode": "text", "max_new_tokens": 8, "seed": 0, "record_audio": False}
    assert len(q["prompt_ids"]) == n
    json.dump(q, open(f"{sys.argv[2]}/L{n}.req.json", "w"))
print("requests written")
PY
for v in off-1 on-1 off-2 on-2; do mkdir -p out-$v
  for k in 1 2 3 4; do for n in 128 512 1149 4096; do echo "$C/$O/req/L$n.req.json $C/$O/out-$v/L$n-r$k"; done; done > list-$v.txt
done
(
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "NOT MAXN"; exit 1; }
for v in off-1 on-1 off-2 on-2; do
  env=""; [ ${v%%-*} = on ] && env="-e LFM3_DIAG_TOUCH=1"
  echo "== $v $(date +%T)"
  $X $env lfm3-profile-v1 $C/alloc-diag-speech/lfm3-speech $M --batch $C/$O/list-$v.txt > $v.log 2>&1; echo "$v exit $?"
done
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null && echo "power: unchanged" || echo "POWER DIFFERS"
)
python3 - <<'PY'
import json, statistics
print(f"{'tokens':>7} " + " ".join(f"{v:>26}" for v in ('off-1', 'on-1', 'off-2', 'on-2')) + "   (prefill s: request 1 | median of 2-4)")
for n in (128, 512, 1149, 4096):
    row = []
    for v in ('off-1', 'on-1', 'off-2', 'on-2'):
        t = [json.load(open(f'out-{v}/L{n}-r{k}.json'))['prefill_seconds'] for k in (1, 2, 3, 4)]
        row.append(f"{t[0]:.4f} | {statistics.median(t[1:]):.4f}")
    print(f"{n:7d} " + " ".join(f"{x:>26}" for x in row))
PY
echo "ALLOC-TEXT-FINISHED $(date +%T)"
