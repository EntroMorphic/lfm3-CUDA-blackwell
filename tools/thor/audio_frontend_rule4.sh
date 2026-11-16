#!/usr/bin/env bash
# validation/audio-frontend-v1 rules 4 and 6, the lfm3-speech part (exactness: may share the GPU). lfm3-speech
# SPEECH_TAG (the new front end, 4 threads by default) against the recorded runs:
#   1. the 40 dumped clips and the 9 edge cases: every output file (stages, tokens) equals audioin-v2's and
#      server-stream-v1-check's (audio-assurance-v1 rule 8's comparison);
#   2. speech-e2e-v1 and the split long-form runs: 90 + 90 identical (rule 6: speech out unchanged);
#   3. all 2,620 LibriSpeech utterances alone: every token sequence equals audioin-v2's native run (assurance rule 2);
#   4. the same in 8 staggered rows: every file equals step 3's (assurance rule 3).
# A step that would start after NOT_AFTER (HH:MM, default 14:00) is skipped and says so (Thor's quiet windows).
# Usage (Thor host): LFM3_WORKSPACE=... audio_frontend_rule4.sh SPEECH_TAG OUT_TAG [NOT_AFTER]
set -uo pipefail
T=${1:?SPEECH_TAG}; O=${2:?OUT_TAG}; LATEST=${3:-14:00}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; E=/workspace/liquid-CUDA/build
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=$E/$T/lfm3-speech; M=/workspace/models/LFM3-Omni-2B; C=$E/$O
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia ${LFM3_AUDIO_FEATURES:+-e LFM3_AUDIO_FEATURES=$LFM3_AUDIO_FEATURES} lfm3-profile-v1"  # the front end: LFM3_AUDIO_FEATURES passed through
mkdir -p $B/$O && cd $B/$O
sha256sum $B/$T/lfm3-speech > binary.sha256
late() { [ "$(date +%H:%M)" \> "$LATEST" ] && { echo "SKIPPED $1: past $LATEST"; return 0; }; return 1; }
same3() {  # REF_DIR NEW_DIR LABEL: raw.i32 frames.i32 f32 of every stem in REF_DIR
  local n=0 d=0
  for f in $1/*.raw.i32; do s=$(basename $f .raw.i32)
    for x in raw.i32 frames.i32 f32; do n=$((n+1)); cmp -s $1/$s.$x $2/$s.$x || { d=$((d+1)); echo "  DIFFER $3 $s.$x"; }; done
  done; echo "$3: $((n-d)) of $n identical"; }
echo "== 1. dumped clips and edge cases $(date +%T)"
if ! late dumps; then
  rm -rf dump edge && mkdir -p dump edge
  sed "s#$E/audioin-v2/native-dump/#$C/dump/#" $B/audioin-v2/list-dump.txt > list-dump.txt
  sed "s#$E/server-stream-v1-check/edge/native/#$C/edge/#" $B/server-stream-v1-check/edge/list.txt > list-edge.txt
  $EXEC $BIN $M --batch $C/list-dump.txt > dump.log 2>&1; echo "dump exit $?"
  $EXEC $BIN $M --batch $C/list-edge.txt > edge.log 2>&1; echo "edge exit $?"
  python3 $TOOLS/compare_speech_outputs.py $B/audioin-v2/native-dump dump dump.json | tail -2
  python3 $TOOLS/compare_speech_outputs.py $B/server-stream-v1-check/edge/native edge edge.json | tail -2
fi
echo "== 2. e2e and split $(date +%T)"
if ! late e2e; then
  rm -rf e2e split && mkdir -p e2e split
  for r in $B/speech-e2e-v1/requests/*.req.json; do n=$(basename $r .req.json); echo "$E/speech-e2e-v1/requests/$n.req.json $C/e2e/$n"; done > list-e2e.txt
  for r in $B/speech-chunk-native-v1/requests/*.req.json; do n=$(basename $r .req.json); echo "$E/speech-chunk-native-v1/requests/$n.req.json $C/split/$n"; done > list-split.txt
  $EXEC $BIN $M --batch $C/list-e2e.txt > e2e.log 2>&1; echo "e2e exit $?"
  $EXEC $BIN $M --batch $C/list-split.txt > split.log 2>&1; echo "split exit $?"
  same3 $B/speech-e2e-v1/native e2e e2e | tee e2e-split.txt
  same3 $B/speech-chunk-native-v1/native split split | tee -a e2e-split.txt
fi
echo "== 3. the corpus alone $(date +%T)"
if ! late corpus; then
  rm -rf alone && mkdir -p alone
  sed "s#$E/audioin-v2/native-full/#$C/alone/#" $B/audioin-v2/list-full.txt > list-alone.txt
  $EXEC $BIN $M --batch $C/list-alone.txt > alone.log 2>&1; echo "alone exit $?"
  python3 $TOOLS/compare_speech_outputs.py $B/audioin-v2/native-full alone corpus-slots.json --slots-only | tail -2
fi
echo "== 4. the corpus in 8 rows $(date +%T)"
if ! late rows8; then
  rm -rf rows8 && mkdir -p rows8
  sed "s#$E/audioin-v2/native-full/#$C/rows8/#" $B/audioin-v2/list-full.txt > list-rows8.txt
  $EXEC $BIN $M --concurrent 8 $C/list-rows8.txt 7 > rows8.log 2>&1; echo "rows8 exit $?"
  python3 $TOOLS/compare_speech_outputs.py alone rows8 rows8-files.json | tail -2
fi
echo "AUDIO FRONTEND RULE 4 DONE $(date +%T)"
