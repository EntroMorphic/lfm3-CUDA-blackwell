#!/usr/bin/env bash
# validation/audio-ttft-v1, a diagnostic (not a rule): the language model's prefill GEMMs slow down after the first
# audio request when the encoder keeps its scratch (build/a5-diag: fp32 mode too; merge-srv, which frees it, never).
# lfm3-speech variants of build/ttft-src (6ec2358's engine), each run on four 90 s requests in one process (--batch):
#   keep:    as built (ttft-speech-a2's code, rebuilt here as diag-keep);
#   release: kKeepBytes = 0 (the scratch freed after every clip, as the deployed encoder does);
#   touch:   the scratch kept, and a 2 MiB cudaMalloc + cudaFree after every encode (does any allocation call restore it?).
# Per request: encode_seconds and prefill_seconds; prefill - encode is the language model's share. Quiet Thor.
# Usage (Thor host, copied to build/keep-diag/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=keep-diag
M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
PAT="[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[n]sys|[n]vcc|[c]make --build|[n]inja|[k]in_|[v]envs/hf-|[c]heck_lfm3|[a]udio-encoder-bench|[m]easure_lfm3"
busy() { { pgrep -af "$PAT" | grep -v "$O" | grep -v "/.local/lib/liquid-lfm3/"; docker ps --format "{{.Names}}" | grep -v -x -E "lfm3-profile-v1|lfm3-omni-dev"; } | head -3; }
wait_quiet() { local quiet=0 i; for i in $(seq 1 $1); do if [ -z "$(busy)" ]; then quiet=$((quiet+1)); [ $quiet -ge 30 ] && return 0; else quiet=0; fi; [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30; done; [ -z "$(busy)" ]; }
cd $B/$O || exit 1
echo "== sources and builds $(date +%T)"
for v in keep release touch; do
  rm -rf $B/diag-src-$v && mkdir -p $B/diag-src-$v && cp -r $B/ttft-src/native $B/ttft-src/tools $B/diag-src-$v/
  H=$B/diag-src-$v/native/include/lfm/audio_encoder.cuh
  case $v in
    release) sed -i 's|static constexpr size_t kKeepBytes = size_t(1) << 30;|static constexpr size_t kKeepBytes = 0;|' $H
             grep -q "kKeepBytes = 0;" $H || echo "EDIT FAILED release";;
    touch) python3 - $H <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "      linear_.release();\n    }\n    if (dumps) {"
assert s.count(a) == 1
s = s.replace(a, "      linear_.release();\n    }\n    { void* t = nullptr; cuda_check(cudaMalloc(&t, size_t(2) << 20)); cuda_check(cudaFree(t)); }  // keep-diag: touch\n    if (dumps) {")
open(p, "w").write(s)
PY
           grep -q "keep-diag: touch" $H || echo "EDIT FAILED touch";;
  esac
  LFM3_WORKSPACE=$W bash $B/ttft-src/tools/thor/build_speech.sh diag-$v build/diag-src-$v > build-$v.log 2>&1; echo "$v build exit $?"
done
(cd $B && sha256sum diag-keep/lfm3-speech diag-release/lfm3-speech diag-touch/lfm3-speech ttft-speech-a2/lfm3-speech) | tee binaries.sha256
wait_quiet 2880 || { echo "NOT QUIET after 24 h"; exit 1; }
SVC=0; systemctl --user is-active --quiet liquid-lfm3 && systemctl --user stop liquid-lfm3 && SVC=1 && echo "stopped liquid-lfm3 $(date +%T)"
trap '[ $SVC = 1 ] && systemctl --user start liquid-lfm3 && echo "started liquid-lfm3 again $(date +%T)"' EXIT
sleep 30; [ -z "$(busy)" ] || { echo "NOT QUIET after stopping the service: $(busy)"; exit 1; }
for v in keep release touch keep; do
  mkdir -p out-$v; for i in 1 2 3 4; do echo "$C/quiet-w3-audio/req/joined90.req.json $C/$O/out-$v/r$i"; done > list-$v.txt
  echo "== $v $(date +%T)"
  $X $C/diag-$v/lfm3-speech $M --batch $C/$O/list-$v.txt > $v.log 2>&1; echo "$v exit $?"
  python3 -c "
import json
for i in range(1, 5):
    d = json.load(open('out-$v/r%d.json' % i))
    print('  $v request %d: encode %.4f s, prefill %.4f s, language model %.4f s' % (i, d['encode_seconds'], d['prefill_seconds'], d['prefill_seconds'] - d['encode_seconds']))"
done
echo "KEEP-DIAG-FINISHED $(date +%T)"
