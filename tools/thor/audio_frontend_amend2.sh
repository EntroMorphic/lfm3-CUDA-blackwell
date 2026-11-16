#!/usr/bin/env bash
# validation/audio-frontend-v1 amendment 2, the GPU exactness checks (shared Thor is fine; nothing here is timed), one
# step at a time (lfm3-profile-v1 is limited to 48 GB):
#   1. rules 3-4 and 6 with the correctly rounded front end (LFM3_AUDIO_FEATURES=exact-f2): the 40 dumps and 9 edge
#      cases against the recorded runs, e2e and split, the 2,620 alone against audioin-v2's tokens, and in 8 rows;
#   2. rule 5, the same with LFM3_AUDIO_FEATURES=dft (today's bytes);
#   3. the server (SERVER_TAG): the audio-in check in both modes, and 8 and 32 requests at once (exact-f2) against
#      step 1's corpus outputs;
#   4. compute-sanitizer, all four tools, on lfm3-speech with exact-f2.
# Usage (Thor host): LFM3_WORKSPACE=... audio_frontend_amend2.sh SPEECH_TAG SERVER_TAG OUT_PREFIX
set -uo pipefail
T=${1:?SPEECH_TAG}; S=${2:?SERVER_TAG}; O=${3:?OUT_PREFIX}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; E=/workspace/liquid-CUDA/build
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); CT=/workspace/liquid-CUDA/${TOOLS#$W/liquid-CUDA/}
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
echo "== 1. exact-f2 $(date +%T)"
LFM3_AUDIO_FEATURES=exact-f2 bash $TOOLS/thor/audio_frontend_rule4.sh $T $O-f2 23:59
echo "== 2. dft $(date +%T)"
LFM3_AUDIO_FEATURES=dft bash $TOOLS/thor/audio_frontend_rule4.sh $T $O-dft 23:59
echo "== 3. server $(date +%T)"
mkdir -p $B/$O-server && cd $B/$O-server
for mode in dft exact-f2; do
  X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=$mode lfm3-profile-v1"
  mkdir -p $B/$O-server/$mode
  $X python3 $CT/check_lfm3_audio_in_server.py $E/$S/liquid-lfm3-server $E/$T/lfm3-speech $M $L $E/$O-server/$mode/work \
    $E/$O-server/$mode/audio-in-server.json > $mode-audio-in.log 2>&1; echo "audio-in server ($mode): exit $?; $(tail -1 $mode-audio-in.log)"
done
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=exact-f2 lfm3-profile-v1"
for n in 8 32; do
  $X python3 $CT/check_lfm3_audio_frontend_server.py $E/$S/liquid-lfm3-server $M $E/audioin-v2/cases.json $E/$O-f2/alone $n \
    $E/$O-server/at-once-$n.json --queue-size 40 --handlers 48 > at-once-$n.log 2>&1; echo "at once $n: exit $?; $(tail -1 at-once-$n.log)"
done
echo "== 4. sanitizers $(date +%T)"
LFM3_AUDIO_FEATURES=exact-f2 bash $TOOLS/thor/audio_frontend_sanitizers.sh $T $O-san
echo "AMENDMENT 2 GPU CHECKS DONE $(date +%T)"
