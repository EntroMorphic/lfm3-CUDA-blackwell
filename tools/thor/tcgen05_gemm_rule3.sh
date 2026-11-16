#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 3: compute-sanitizer 2026.4.0 (the 13.5.25 archive) on the build with the tcgen05
# path: memcheck (leak check), racecheck, synccheck, initcheck
#   (a) lfm3-vision-check's small, two-image and maximum-size cases (tools/thor/vision_sanitizers.sh, VISION_TAG=A);
#   (b) lfm3-speech, a 4,096-token text prompt (one prefill of 4,096 rows: the LM's projections on tcgen05), 4 new
#       tokens; every output file byte-identical to a plain run;
#   (c) the probe's two positive controls (probe build PROBE_TAG) on the same sanitizer: must be reported.
# Usage (Thor host): tcgen05_gemm_rule3.sh A OUT_TAG [PROBE_TAG]   (PROBE_TAG default tcgen05-b4)
set -uo pipefail
A=${1:?A}; O=${2:?OUT_TAG}; PT=${3:-tcgen05-b4}
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; E=/workspace/liquid-CUDA/build
M=/workspace/models/LFM3-Omni-2B
SAN=$E/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O/text && cd $B/$O
sha256sum $B/$A/lfm3-speech $B/$A/lfm3-vision-check $B/$PT/tcgen05-gemm-probe > binaries.sha256
$X $SAN --version > sanitizer-version.txt 2>&1
echo "== (a) vision $(date +%T)"
VISION_TAG=$A bash $W/liquid-CUDA/tools/thor/vision_sanitizers.sh > vision.log 2>&1; grep -vE '^$' vision.log
echo "== (b) text prefill, 4,096 tokens $(date +%T)"
# 4,096 prompt ids: the speech-e2e fridge prompt's ids repeated (a real token stream; the content does not matter here)
python3 - $B/speech-e2e-v1/requests/il-fridge-s1.req.json $B/$O/text/text4096.req.json <<'PY'
import json, sys
src = json.load(open(sys.argv[1]))["prompt_ids"]
ids = (src * (4096 // len(src) + 1))[:4096]
json.dump({"prompt_ids": ids, "mode": "text", "max_new_tokens": 4, "seed": 1}, open(sys.argv[2], "w"))
PY
run() {  # NAME [sanitizer args...]
  local n=$1; shift
  $X "$@" $E/$A/lfm3-speech $M $E/$O/text/text4096.req.json $E/$O/text/$n > text/$n.log 2>&1
}
run plain; echo "plain exit $?"
for t in memcheck racecheck synccheck initcheck; do
  pre=(--tool $t --error-exitcode 99 --print-limit 50); [ $t = memcheck ] && pre+=(--leak-check full)
  t0=$(date +%s); run $t $SAN "${pre[@]}"; s=$?
  same=identical; n=0
  for f in text/plain.*; do
    case $f in *.log) continue;; esac
    n=$((n+1)); suf=${f#text/plain}
    if [[ $suf == .json ]]; then
      python3 -c "import json,sys; a,b=(json.load(open(x)) for x in sys.argv[1:]); sys.exit(a.get('text_tokens')!=b.get('text_tokens'))" $f text/$t$suf || same=DIFFER
    else cmp -s $f text/$t$suf || same=DIFFER; fi
  done
  echo "text $t: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY' text/$t.log | tr '\n' ' '); $n outputs $same"
done
echo "== (c) positive controls (probe $PT) $(date +%T)"
mkdir -p controls/unwritten controls/oob
for ctl in "initcheck unwritten" "memcheck oob"; do
  set -- $ctl
  $X $SAN --tool $1 --error-exitcode 86 --log-file $E/$O/controls/$2/sanitizer.log $E/$PT/tcgen05-gemm-probe sanitize 3 \
    $E/$O/controls/$2 --m 100 --data int --control $2 --timeout-ms 600000 > controls/$2/stdout.log 2>&1
  s=$?; echo "control $2 ($1): exit $s $([ $s = 86 ] && echo DETECTED || echo 'NOT DETECTED'); $(grep 'ERROR SUMMARY' controls/$2/sanitizer.log | tr '\n' ' ')"
done
$X chown -R 1000:1000 $E/$O
echo "RULE3 DONE $(date +%T)"
