#!/usr/bin/env bash
# validation/audio-frontend-v1, the rules left for a quiet window (amendment 3): rule 0's second part and rule 3 (the
# stall at 8 and 16 speech rows), the memory bound (16 x 90 s at once), and rule 6 (lfm3-speech's regression, the
# audio-in files, e2e and split, the concurrent engine, the server's batch, stream, audio-in and TTS checks, the text
# gate, images through the server). No arguments; a quiet Thor in MAXN. Results under build/quiet-audio-rest/.
# Exit: 0 every gated rule passed; 1 the run was not possible (not quiet, not MAXN, a build or a tool without a result,
# the power state changed); 2 a gated rule failed (summary.json says which).
# Sources: this script's tree (build/audio-src-b: the round's code) and build/audio-src-base (main at a350659, the
# server before this round: rule 0's "current server").
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$(cd "$HERE/.." && pwd); SRC=$(cd "$HERE/../.." && pwd); W=$(cd "$HERE/../../../../.." && pwd)
export LFM3_WORKSPACE=$W
B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
CT=/workspace/liquid-CUDA/${TOOLS#$W/liquid-CUDA/}; SRC_REL=${SRC#$W/liquid-CUDA/}; BASE_REL=build/audio-src-base
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
O=${QAR_OUT:-quiet-audio-rest}; P=${QAR_TAG:-audio-fe-w}  # the output and build names (audio-handles-v1 runs it with its own)
SP=$P-speech; SV=$P-srv; BS=${QAR_BASE_TAG:-audio-fe-w-base-srv}; BE=$P-bench
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || { echo "NOT RUNNABLE: cannot make $B/$O"; exit 1; }
nr() { echo "NOT RUNNABLE: $*"; exit 1; }
FAILS=()
gate() { if [ "$1" -eq 0 ]; then echo "PASS $2"; else echo "FAIL $2"; FAILS+=("$2"); fi; }
[ -f $W/liquid-CUDA/$BASE_REL/native/src/lfm3_server.cu ] || nr "no base sources in $BASE_REL"
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3|[a]udio-encoder-bench|[a]udio-mel-" | head -3)
[ -n "$busy" ] && nr "not quiet: $busy"

echo "== builds $(date +%T)"
# Safe to rerun: a build is reused when its binary exists and its recorded source tree (tree.sha256: every file under
# native/, hashed) equals the current one; otherwise it is built and the record written. FORCE_BUILD=1 rebuilds all.
tree() { (cd $W/liquid-CUDA/$1 && find native -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
SRC_TREE=$(tree $SRC_REL) && BASE_TREE=$(tree $BASE_REL) || nr "cannot hash the sources"
reuse() {  # reuse DIR BINARY TREE: true when DIR/BINARY exists and DIR's record equals TREE
  [ "${FORCE_BUILD:-0}" != 1 ] && [ -x $B/$1/$2 ] && [ "$(cat $B/$1/tree.sha256 2>/dev/null)" = "$3" ] && echo "reused $1"
}
reuse $SP lfm3-speech $SRC_TREE || {
  bash $HERE/build_speech.sh $SP $SRC_REL > build-speech.log 2>&1 || nr "lfm3-speech build: $(tail -2 build-speech.log)"
  echo $SRC_TREE > $B/$SP/tree.sha256; }
reuse $BE audio-encoder-bench $SRC_TREE || {
  bash $HERE/build_audio_bench.sh $BE $SRC_REL > build-bench.log 2>&1 || nr "bench build: $(tail -2 build-bench.log)"
  echo $SRC_TREE > $B/$BE/tree.sha256; }
for triple in "$SV $SRC_REL $SRC_TREE" "$BS $BASE_REL $BASE_TREE"; do set -- $triple
  reuse $1 liquid-lfm3-server $3 && continue
  rm -rf "${B:?}/${1:?}"  # a fresh CMake tree (a previous partial build is not reused)
  docker run --rm --network none -v $W/liquid-CUDA/$2/native:/src/native:ro -v $W:/workspace $IMG bash -c \
    "bash $C/../$SRC_REL/tools/thor/build_native.sh $C/$1; rc=\$?; chown -R 1000:1000 $C/$1 $C/$1.cmake.log $C/$1.build.log; exit \$rc" \
    > build-$1.log 2>&1 || nr "server build $1: $(tail -2 build-$1.log)"
  echo $3 > $B/$1/tree.sha256
done
(cd $B && sha256sum $SP/lfm3-speech $BE/audio-encoder-bench $SV/liquid-lfm3-server $SV/liquid-lfm3-worker $BS/liquid-lfm3-server) > binaries.sha256
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3" | head -3)
[ -n "$busy" ] && nr "not quiet after the builds: $busy"
bash $HERE/power_state.sh power-start.json > /dev/null || nr "power state (not MAXN)"

# rule 0's five clips and their requests (as tools/thor/quiet_audio_frontend.sh)
python3 - $B $O <<'PY' || nr "clips"
import json, sys
from pathlib import Path
b, o = Path(sys.argv[1]), sys.argv[2]
(b / o / "req").mkdir(exist_ok=True)
cases = json.load(open(b / "audioin-v2/cases.json"))["cases"]
secs = sorted(c["seconds"] for c in cases)
mid = min(cases, key=lambda c: (abs(c["seconds"] - secs[len(secs) // 2]), c["case"]))
clips = []
for name, case in [("short", "4446-2275-0004"), ("median", mid["case"]), ("typical", "1089-134686-0000"),
                   ("long", "4507-16021-0047")]:
    r = json.load(open(b / f"audioin-v2/full/requests/{case}.req.json"))
    r["record_audio"] = False
    (b / o / "req" / f"{name}.req.json").write_text(json.dumps(r) + "\n")
    clips.append(dict(name=name, case=case, wav=r["audio"][0]))
j = json.load(open(b / "server-stream-v3-check/work/asr-joined-90s.json"))
wav = "/workspace/liquid-CUDA/build/server-stream-v3-check/work/joined-90s.wav"
(b / o / "req" / "joined90.req.json").write_text(json.dumps(dict(prompt_ids=j["prompt"], audio=[wav], mode="text",
                                                                 max_new_tokens=2048, seed=0, record_audio=False)) + "\n")
clips.append(dict(name="joined90", case="asr-joined-90s", wav=wav))
(b / o / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
(b / o / "clip-args.txt").write_text(" ".join(f"{c['name']}={c['wav']}" for c in clips) + "\n")
PY

echo "== rule 3's bar: the encoder alone and the language-model prefill per clip $(date +%T)"
$X $C/$BE/audio-encoder-bench $M $C/$O/bench.json --repeat 5 $(cat clip-args.txt) > bench.log 2>&1 || nr "bench: $(tail -1 bench.log)"
mkdir -p native
for p in warm p0 p1 p2 p3 p4; do for n in short median typical long joined90; do
  echo "$C/$O/req/$n.req.json $C/$O/native/$p-$n"; done; done > native-list.txt
$X $C/$SP/lfm3-speech $M --batch $C/$O/native-list.txt > native.log 2>&1 || nr "lfm3-speech: $(tail -1 native.log)"

echo "== the stall (rule 0 on the base server; rule 3 on the new) $(date +%T)"
for srv in $BS $SV; do for rows in 8 16; do
  $X python3 $CT/measure_lfm3_audio_stall.py $C/$srv/liquid-lfm3-server $M $C/$O/clips.json $rows $C/$O/stall-$srv-$rows.json 3 \
    > stall-$srv-$rows.log 2>&1 || nr "stall $srv $rows: $(tail -2 stall-$srv-$rows.log)"
  cat stall-$srv-$rows.log
done; done

echo "== the memory bound $(date +%T)"
$X python3 $CT/measure_lfm3_audio_memory.py $C/$SV/liquid-lfm3-server $M \
  /workspace/liquid-CUDA/build/server-stream-v3-check/work/joined-90s.wav $C/$O/memory.json > memory.log 2>&1; s=$?
tail -1 memory.log; [ $s -eq 1 ] && nr "memory tool"; gate $s "memory bound"

echo "== rule 6: lfm3-speech $(date +%T)"
bash $HERE/speech_regression_check.sh $SP > speech-regression.log 2>&1
same=$(grep -c '^SAME' speech-regression.log); bad=$(grep -cE '^DIFFER|RUN FAILED|STALE hidden left' speech-regression.log)
refuse_bad=$(grep '^REFUSE' speech-regression.log | grep -vE 'status [1-9][0-9]*, (outputs left 0|request kept yes)' | wc -l)
echo "speech regression: $same SAME, $bad differ or failed, $refuse_bad unclean refusals"
ok=1; [ "$same" -eq 36 ] && [ "$bad" -eq 0 ] && [ "$refuse_bad" -eq 0 ] && ok=0
gate "$ok" "lfm3-speech regression (36)"
bash $HERE/vision_rule6_speech.sh $SP > rule6-speech.log 2>&1
python3 - $B/$SP-rule6 <<'PY' > rule6-speech-verdict.txt; s=$?
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
expected = {"8455-210777-0002.audio0.mel.f32", "8555-292519-0007.audio0.mel.f32", "8555-292519-0007.audio0.down.f32",
            "8555-292519-0007.audio0.encoded.f32", "8555-292519-0007.audio0.features.f32"}  # F2's changes (rule 4)
dump, edge = json.load(open(d / "dump.json")), json.load(open(d / "edge.json"))
ok = set(dump["differ"]) <= expected and not dump["missing"] and edge["identical"] == edge["files"] == 93
print("dump differ:", dump["differ"], "| edge", edge["identical"], "of", edge["files"])
sys.exit(0 if ok else 1)
PY
cat rule6-speech-verdict.txt; gate $s "audio-in files (360 + 93; F2's 5 expected)"
for k in e2e split; do
  line=$(grep -E "^$k: " rule6-speech.log | tail -1); echo "$line"
  ok=1; [ -n "$line" ] && echo "$line" | grep -qE "^$k: ([0-9]+) of \1 identical" && ok=0
  gate "$ok" "$k"
done
rm -rf $B/$O/conc
$X python3 $CT/check_lfm3_speech_concurrent.py $C/$SP/lfm3-speech $M $C $L $C/$O/conc $C/$O/engine-exact.json $C/speed-final2/conc/alone \
  > engine-exact.log 2>&1; gate $? "concurrent engine"

echo "== rule 6: the server $(date +%T)"
$X python3 $CT/check_lfm3_speech_batch_server.py $C/$SV/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L \
  $C/$O/batch-server.json > batch-server.log 2>&1; gate $? "batch server"
$X python3 $CT/check_lfm3_speech_stream.py $C/$SV/liquid-lfm3-server $M $C $L $C/$O/stream-check.json > stream-check.log 2>&1
gate $? "stream"
$X python3 $CT/check_lfm3_audio_in_server.py $C/$SV/liquid-lfm3-server $C/$SP/lfm3-speech $M $L $C/$O/audio-in-work \
  $C/$O/audio-in-server.json > audio-in-server.log 2>&1; gate $? "audio-in server"
$X python3 $CT/check_lfm3_speech_server.py $C/$SV/liquid-lfm3-server $M $C $C/$O/speech-check.json > speech-check.log 2>&1
gate $? "TTS server"
rm -rf $B/$O/text-gate
$X python3 $CT/check_lfm3_native_server.py --checkpoint $M --server $C/$SV/liquid-lfm3-server --worker $C/$SV/liquid-lfm3-worker \
  --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/$O/text-gate > text-gate.log 2>&1
gate $? "text gate"
# images-speed-v1 (the vision encoder on the tensor cores) made vimg-e2e-a4 the reference; a1 is stale (quiet window 12)
$X python3 $CT/check_lfm3_image_server.py $C/$SV/liquid-lfm3-server $M $C/vimg-e2e-a4 $C/$O/image-server.json > image-server.log 2>&1
gate $? "images through the server"

bash $HERE/power_state.sh power-end.json power-start.json > /dev/null || nr "VOID: the power state changed during the run"

echo "== rule 3 and the summary $(date +%T)"
python3 - $B/$O $SV $BS <<'PY'; s=$?
import json, statistics, sys
from pathlib import Path
o, sv, bs = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
med = statistics.median
bench = {c["name"]: c for c in json.load(open(o / "bench.json"))["clips"]}
bar = {}
for n, c in bench.items():
    nat = [json.load(open(o / "native" / f"p{k}-{n}.json")) for k in range(5)]
    prefill = med([j["prefill_seconds"] - j["encode_seconds"] for j in nat])
    bar[n] = dict(encoder=c["encode_wall_seconds"], lm_prefill=prefill, bar=c["encode_wall_seconds"] + prefill + 0.010)
stall = {f"{srv}-{rows}": json.load(open(o / f"stall-{srv}-{rows}.json")) for srv in (bs, sv) for rows in (8, 16)}
rule3 = {}
for rows in (8, 16):
    for n, b in bar.items():
        st = stall[f"{sv}-{rows}"]["clips"][n]["stall_seconds"]
        rule3[f"{rows}-{n}"] = dict(stall=st, bar=b["bar"], pass_=st <= b["bar"],
                                    base_stall=stall[f"{bs}-{rows}"]["clips"][n]["stall_seconds"])
ok = all(r["pass_"] for r in rule3.values())
(o / "rule3.json").write_text(json.dumps(dict(bar=bar, rule3=rule3, pass_=ok), indent=1) + "\n")
print(f"{'rows-clip':16s} {'base stall':>10s} {'new stall':>10s} {'bar':>8s}")
for k, r in rule3.items():
    print(f"{k:16s} {r['base_stall']:10.4f} {r['stall']:10.4f} {r['bar']:8.4f} {'PASS' if r['pass_'] else 'FAIL'}")
print("rule 0's record: the base server's 90 s stall at 8 rows", round(rule3["8-joined90"]["base_stall"], 4), "s")
sys.exit(0 if ok else 1)
PY
gate $s "rule 3 (the stall)"
python3 -c "import json,sys; json.dump(dict(failed=sys.argv[1:]), open('summary.json','w'))" "${FAILS[@]}"
echo "QUIET AUDIO REST DONE $(date +%T): ${#FAILS[@]} gated rule(s) failed${FAILS:+: ${FAILS[*]}}"
[ ${#FAILS[@]} -eq 0 ] && exit 0 || exit 2
