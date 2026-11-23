#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 2: every byte-identity regression on a build with the tcgen05 path (build tag A,
# from build_tcgen05_engine.sh: lfm3-speech, lfm3-vision-check, server and worker in one directory), each against
# the recorded validated outputs. The checks are the established scripts of the main checkout's tools/thor:
#   speech_regression_check.sh (36), vision_rule6_speech.sh (audio-in 360 + 93, e2e 90, split 90),
#   vision_accuracy_round.sh x2 x2 (then all 713 .f32 stage files against vspeed-rule1-a2, byte for byte),
#   vision_e2e_round.sh (the 11 stage-0 cases against vimg-e2e-a4: tokens, pixel values, every array),
#   vision_rule6_server.sh (concurrent engine exactness, batch, stream, audio in, TTS, text gate, images via server).
# Usage (Thor host): tcgen05_gemm_rule2.sh A OUT_PREFIX [STEPS]   STEPS: any of "regression speech vision e2e server"
set -uo pipefail
A=${1:?A}; P=${2:?OUT_PREFIX}; STEPS=${3:-regression speech vision e2e server}
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; T=$W/liquid-CUDA/tools/thor
mkdir -p $B/$P && cd $B/$P
sha256sum $B/$A/lfm3-speech $B/$A/lfm3-vision-check $B/$A/liquid-lfm3-server $B/$A/liquid-lfm3-worker > binaries.sha256
step() { [[ " $STEPS " == *" $1 "* ]]; }
if step regression; then
  echo "== speech regression $(date +%T)"
  bash $T/speech_regression_check.sh $A > regression.log 2>&1
  echo "regression: same $(grep -c '^SAME' regression.log); $(grep -cE 'DIFFER|RUN FAILED' regression.log) differ/failed"
fi
if step speech; then
  echo "== audio in, e2e, split $(date +%T)"
  bash $T/vision_rule6_speech.sh $A > speech.log 2>&1
  grep -E "identical|DIFFER|exit [1-9]" speech.log | head -20
fi
if step vision; then
  echo "== vision stages $(date +%T)"
  bash $T/vision_accuracy_round.sh $A $P-vision x2 x2 > vision.log 2>&1; tail -2 vision.log
  n=0; d=0
  for f in $(cd $B/vspeed-rule1-a2 && find . -mindepth 2 -name '*.f32' | sort); do
    n=$((n+1)); cmp -s $B/vspeed-rule1-a2/$f $B/$P-vision/$f || { d=$((d+1)); echo "  DIFFER vision $f"; }
  done
  echo "vision stage files: $((n-d)) of $n identical to vspeed-rule1-a2"
fi
if step e2e; then
  echo "== image stage-0 $(date +%T)"
  bash $T/vision_e2e_round.sh $A $P-e2e > e2e.log 2>&1; echo "e2e exit $?"
  n=0; d=0
  for f in $(cd $B/vimg-e2e-a4/out && ls | grep -vE '\.json$'); do
    n=$((n+1)); cmp -s $B/vimg-e2e-a4/out/$f $B/$P-e2e/out/$f || { d=$((d+1)); echo "  DIFFER e2e $f"; }
  done
  for j in $B/vimg-e2e-a4/out/*.json; do
    n=$((n+1)); f=$(basename $j)
    python3 -c "import json,sys; a,b=(json.load(open(x)) for x in sys.argv[1:]); sys.exit(a['text_tokens']!=b['text_tokens'])" $j $B/$P-e2e/out/$f \
      || { d=$((d+1)); echo "  DIFFER e2e tokens $f"; }
  done
  echo "image stage-0: $((n-d)) of $n identical to vimg-e2e-a4 (arrays byte for byte, reply tokens)"
fi
if step server; then
  echo "== server checks $(date +%T)"
  E2E_ROUND=$P-e2e bash $T/vision_rule6_server.sh $A $A $P-server > server.log 2>&1; cat server.log | grep -vE '^$' | tail -25
fi
echo "RULE2 DONE $(date +%T)"
