#!/usr/bin/env bash
# validation/audio-frontend-v1 rule 5 (the part that does not need the server change): compute-sanitizer memcheck
# (leak check on), initcheck, synccheck and racecheck on lfm3-speech with the 1-sample edge clip (cut-1: one encoder
# token), a typical clip (1089-134686-0000), the 90-second joined recording and a two-clip request, all four in one
# process, every stage dumped (record_audio) and decoding capped at 16 tokens (the front end and encoder are the
# subject; stage files do not depend on the reply's length). Pass: 0 errors, and every output file of every tool's run
# byte-identical to the plain run's. Also: the plain run's stage files (mel, down, encoded, features) of cut-1, the
# two-clip request and 1089-134686-0000 equal their recorded runs' (server-stream-v1-check/edge, audioin-v2).
# Usage (Thor host): LFM3_WORKSPACE=... audio_frontend_sanitizers.sh SPEECH_TAG OUT_TAG [TOOLS...]
set -uo pipefail
T=${1:?SPEECH_TAG}; O=${2:?OUT_TAG}; shift 2
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
SAN=$C/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia ${LFM3_AUDIO_FEATURES:+-e LFM3_AUDIO_FEATURES=$LFM3_AUDIO_FEATURES} lfm3-profile-v1"  # the front end: LFM3_AUDIO_FEATURES passed through
M=/workspace/models/LFM3-Omni-2B; TOOLS=${*:-memcheck initcheck synccheck racecheck}
mkdir -p $B/$O/req && cd $B/$O
sha256sum $B/$T/lfm3-speech > binary.sha256
python3 - $B $O <<'PY'
import json, sys
from pathlib import Path
b, o = Path(sys.argv[1]), Path(sys.argv[1]) / sys.argv[2]
reqs = {"cut1": json.load(open(b / "server-stream-v1-check/edge/requests/cut-1.req.json")),
        "two": json.load(open(b / "server-stream-v1-check/edge/requests/two-clips.req.json")),
        "typical": json.load(open(b / "audioin-v2/dump/requests/1089-134686-0000.req.json"))}
j = json.load(open(b / "server-stream-v3-check/work/asr-joined-90s.json"))
reqs["joined90"] = dict(prompt_ids=j["prompt"], audio=["/workspace/liquid-CUDA/build/server-stream-v3-check/work/joined-90s.wav"],
                        mode="text", seed=0)
for name, r in reqs.items():
    r.update(max_new_tokens=16, record_audio=True)
    (o / "req" / f"{name}.req.json").write_text(json.dumps(r) + "\n")
PY
run() {  # run TOOL_PREFIX DIR
  mkdir -p $B/$O/$2
  for n in cut1 two typical joined90; do echo "$C/$O/req/$n.req.json $C/$O/$2/$n"; done > $B/$O/$2.list
  $EXEC $1 $C/$T/lfm3-speech $M --batch $C/$O/$2.list > $B/$O/$2.log 2>&1
}
[ -f plain/joined90.json ] || { run "" plain; echo "plain: exit $?"; }
# the plain run against the recorded stage files
for pair in "cut1 server-stream-v1-check/edge/native/cut-1" "two server-stream-v1-check/edge/native/two-clips" \
            "typical audioin-v2/native-dump/1089-134686-0000"; do
  set -- $pair; same=identical; n=0
  for f in $B/$2.audio*.{mel,down,encoded,features}.f32; do
    [ -f "$f" ] || continue; n=$((n + 1)); g=plain/$1.${f#$B/$2.}
    cmp -s $f $g || { same=DIFFER; echo "  differs: $g"; }
  done
  echo "plain $1 against the recorded run: $n stage files $same"
done
for t in $TOOLS; do
  pre="$SAN --tool $t --error-exitcode 99 --print-limit 50"; [ $t = memcheck ] && pre="$pre --leak-check full"
  t0=$(date +%s); run "$pre" $t; s=$?
  same=identical; for f in plain/*.f32 plain/*.i32; do cmp -s $f $t/$(basename $f) || same=DIFFER; done
  for f in plain/*.json; do  # the reply files, timings aside
    python3 -c "
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
drop = lambda d: {k: v for k, v in d.items() if not k.endswith('_seconds') and k not in ('real_time_factor', 'real_time_factor_end_to_end')}
sys.exit(drop(a) != drop(b))" $f $t/$(basename $f) || same=DIFFER
  done
  echo "$t: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY' $t.log | tr '\n' ' '); outputs $same"
done
$EXEC chown -R 1000:1000 $C/$O
echo AUDIO FRONTEND SANITIZERS DONE
