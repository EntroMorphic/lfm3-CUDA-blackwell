#!/usr/bin/env bash
# validation/mel-tiered-v1 rule 1 on Thor (aarch64; not timing, may overlap other work):
#   - audio-mel-check from main as deployed (build/merge2-src, 717cca6) writes F2's features for all 2,620 utterances,
#     the 40 dumped clips and the 12 edge clips: the reference;
#   - audio-mel-check from this round (build/mel3-rule-src) compares: F3 at 1 and 4 threads (--repeat 2), F2 and F1 at 4
#     threads (the refactor changed neither), and dft at 4 threads against audio-frontend-v1's original bytes;
#   - amendment 1's empirical check: audio-input-tool fft-bound on 2,000 frames of the 90 s clip and on every edge clip.
# Usage (Thor host, copied to build/mel3-rule1/run.sh): bash run.sh > run.log 2>&1
set -uo pipefail
W=$HOME/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build; O=mel3-rule1; S=mel3-rule-src
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec lfm3-profile-v1"
cd $B/$O || exit 1
echo "== builds $(date +%T)"
LFM3_WORKSPACE=$W bash $B/$S/tools/thor/build_mel_check.sh mel3-main build/merge2-src > build-main.log 2>&1; echo "main mel check exit $?"
LFM3_WORKSPACE=$W bash $B/$S/tools/thor/build_mel_check.sh mel3-round build/$S > build-round.log 2>&1; echo "round mel check exit $?"
docker run --rm --network none -v $W:/workspace $IMG bash -c "cd $C/$O && g++ -O3 -std=c++20 -I $C/$S/native/include -o audio-input-tool \
  $C/$S/native/tests/audio_input_tool.cpp -lpthread > build-tool.log 2>&1; rc=\$?; chown -R 1000:1000 .; exit \$rc"; echo "audio-input-tool exit $?"
(cd $B && sha256sum mel3-main/audio-mel-check mel3-round/audio-mel-check $O/audio-input-tool) | tee binaries.sha256
for l in all2620 dump40 edge; do
  awk '{print $1, $2}' $B/audio-fe-r1/$l.txt > $l.txt
  mkdir -p ref-$l
  $X $C/mel3-main/audio-mel-check --threads 4 --transform f2 --write $C/$O/ref-$l $C/$O/$l.txt $C/$O/main-f2-$l.json 2>&1 | tail -1 | sed "s/^/  main f2 write $l: /"
  for v in f3:1 f3:4 f2:4 f1:4; do tr=${v%%:*}; th=${v#*:}
    $X $C/mel3-round/audio-mel-check --threads $th --transform $tr --repeat 2 --compare $C/$O/ref-$l $C/$O/$l.txt $C/$O/$tr-$th-$l.json 2>&1 | tail -1 | sed "s/^/  $tr threads $th $l: /"
  done
  ref=$C/audio-fe-r1/ref; [ $l = all2620 ] || ref=-
  $X $C/mel3-round/audio-mel-check --threads 4 --compare $ref $C/audio-fe-r1/$l.txt $C/$O/dft-$l.json 2>&1 | tail -1 | sed "s/^/  dft (original bytes) $l: /"
done
echo "== amendment 1: the bounds against quad $(date +%T)"
echo -n "  90 s clip, 2,000 frames: "; ./audio-input-tool fft-bound $B/server-stream-v3-check/work/joined-90s.wav 2000
while read name wav rest; do w=${wav/\/workspace\//$W/}; echo -n "  edge $name: "; ./audio-input-tool fft-bound $w 200; done < edge.txt
echo "MEL3-RULE1-FINISHED $(date +%T)"
