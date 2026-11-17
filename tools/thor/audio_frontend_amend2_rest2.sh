#!/usr/bin/env bash
# validation/audio-frontend-v1 amendment 2: audio_frontend_amend2_rest.sh from its step 1 on, without the sanitizers
# (racecheck would run past quiet window 3; the sanitizers run after it, audio_frontend_sanitizers.sh):
#   1. wait for the running 8-row corpus pass (ROWS8_PID in lfm3-profile-v1), then compare it with the run alone;
#   2. rule 5: today's bytes with LFM3_AUDIO_FEATURES=dft: the features of all 2,620 + 40 + 12 clips (audio-mel-check,
#      CPU) and the 360 + 93 stage files (lfm3-speech);
#   3. the server: the audio-in check in both modes, and 8 and 32 requests at once (exact-f2);
# Usage (Thor host): LFM3_WORKSPACE=... audio_frontend_amend2_rest2.sh SPEECH_TAG SERVER_TAG MEL_TAG OUT_PREFIX ROWS8_PID
set -uo pipefail
T=${1:?SPEECH_TAG}; S=${2:?SERVER_TAG}; MT=${3:?MEL_TAG}; O=${4:?OUT_PREFIX}; PID=${5:?ROWS8_PID}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; E=/workspace/liquid-CUDA/build
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); CT=/workspace/liquid-CUDA/${TOOLS#$W/liquid-CUDA/}
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
run() { docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -e LFM3_AUDIO_FEATURES=$1 lfm3-profile-v1 "${@:2}"; }
echo "== 1. waiting for the 8-row corpus pass (container pid $PID) $(date +%T)"
while docker exec lfm3-profile-v1 kill -0 $PID 2> /dev/null; do sleep 30; done
cd $B/$O-f2
python3 $TOOLS/compare_speech_outputs.py alone rows8 rows8-files.json | tail -1
echo "== 2. rule 5 (dft) $(date +%T)"
mkdir -p $B/$O-dft && cd $B/$O-dft
for l in all2620 dump40 edge; do
  ref=$E/audio-fe-r1/ref; [ $l = all2620 ] || ref=-
  run dft $E/$MT/audio-mel-check --threads 4 --compare $ref $E/audio-fe-r1/$l.txt $E/$O-dft/features-$l.json 2>&1 | tail -1
done
rm -rf dump edge && mkdir -p dump edge
sed "s#$E/audioin-v2/native-dump/#$E/$O-dft/dump/#" $B/audioin-v2/list-dump.txt > list-dump.txt
sed "s#$E/server-stream-v1-check/edge/native/#$E/$O-dft/edge/#" $B/server-stream-v1-check/edge/list.txt > list-edge.txt
run dft $E/$T/lfm3-speech $M --batch $E/$O-dft/list-dump.txt > dump.log 2>&1; echo "dump exit $?"
run dft $E/$T/lfm3-speech $M --batch $E/$O-dft/list-edge.txt > edge.log 2>&1; echo "edge exit $?"
python3 $TOOLS/compare_speech_outputs.py $B/audioin-v2/native-dump dump dump.json | tail -1
python3 $TOOLS/compare_speech_outputs.py $B/server-stream-v1-check/edge/native edge edge.json | tail -1
echo "== 3. server $(date +%T)"
mkdir -p $B/$O-server && cd $B/$O-server
for mode in dft exact-f2; do
  mkdir -p $mode
  run $mode python3 $CT/check_lfm3_audio_in_server.py $E/$S/liquid-lfm3-server $E/$T/lfm3-speech $M $L $E/$O-server/$mode/work \
    $E/$O-server/$mode/audio-in-server.json > $mode-audio-in.log 2>&1; echo "audio-in server ($mode): exit $?; $(tail -1 $mode-audio-in.log)"
done
for n in 8 32; do
  run exact-f2 python3 $CT/check_lfm3_audio_frontend_server.py $E/$S/liquid-lfm3-server $M $E/audioin-v2/cases.json $E/$O-f2/alone $n \
    $E/$O-server/at-once-$n.json --queue-size 40 --handlers 48 > at-once-$n.log 2>&1; echo "at once $n: exit $?; $(tail -1 at-once-$n.log)"
done
echo "AMENDMENT 2 GPU CHECKS (BUT THE SANITIZERS) DONE $(date +%T)"
