#!/usr/bin/env bash
# G4 red-team verification on Thor (round 4): builds, bit-identity of the KV-group prefill split (perplexity NLL
# and traces), the FP32-anchored spread with hardened tools, exact-memory http-batch, and all gates.
set -uo pipefail
W=/home/ztflynn/Projects/000-research/lfm3-omni-2B; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
D=cmake-g4-default-v6; N=cmake-g4-native-v6; S=score-g4-v6; P=ppl-g4-v6
for o in $D $N $S $P spread-v2 metrics-native-v6-http-batch g4-native-v6-gates; do test ! -e $B/$o || { echo "exists $o"; exit 1; }; done
step(){ echo "== $1 $(date +%T)"; }
build(){ docker run --rm --network none -v $R/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$1 ${2:-}; rc=\$?; chown -R 1000:1000 $C/$1 $C/$1.cmake.log $C/$1.build.log 2>/dev/null; exit \$rc"; }
step build-default; build $D || { echo FAIL; exit 1; }; echo "warnings: $(grep -ci warning $B/$D.build.log)"
step build-native; build $N -DLIQUID_LFM3_NATIVE_GEMM=ON || { echo FAIL; exit 1; }; echo "warnings: $(grep -ci warning $B/$N.build.log)"
step compare-default
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle lfm3-decode-batch-probe; do
  objcopy --strip-all $B/cmake-server-v3/$f /tmp/g4f-a; objcopy --strip-all $B/$D/$f /tmp/g4f-b
  echo "$f stripped:$(cmp -s /tmp/g4f-a /tmp/g4f-b && echo same || echo DIFF)"
done
step build-scorer; mkdir -p $B/$S
docker run --rm --network none -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_scorer.sh $C/$S native-gemm; rc=\$?; chown -R 1000:1000 $C/$S; exit \$rc" || { echo FAIL scorer; exit 1; }
EXE="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
step ppl-bit-identity; mkdir -p $B/$P
cp -p $B/ppl-v5/windows-*.i32 $B/$P/
for L in 256 257 4096 32768; do
  $EXE $C/$S/lfm3-score /workspace/models/LFM3-Omni-2B $C/$P/windows-$L.i32 $L $C/$P/nll-native-gemm-$L.f32 >> $B/$P.native.log 2>&1 || { echo "FAIL score $L"; exit 1; }
  echo "$L: $(cmp -s $B/$P/nll-native-gemm-$L.f32 $B/ppl-g4-v3/nll-native-gemm-$L.f32 && echo 'NLL bitwise identical to attempt 3' || echo 'NLL DIFFERS from attempt 3')"
done
CK=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools
step spread-references; mkdir -p $B/spread-v2
ref(){ name=$1; c=$2; tokens=$3; shift 3; $EXE /opt/venv/bin/python3 $T/lfm3_reference.py --checkpoint $CK --output $C/spread-v2/$name/$c --tokens $tokens "$@" > $B/spread-v2/$name-$c.log 2>&1 || { echo "FAIL ref $name $c"; tail -3 $B/spread-v2/$name-$c.log; }; }
for spec in eager:--attention:eager sdpa:--attention:sdpa fp32:--dtype:float32; do
  name=${spec%%:*}; rest=${spec#*:}; flag=${rest%%:*}; value=${rest#*:}
  ref $name baseline 16 $flag $value; ref $name arithmetic 16 $flag $value --prompt "What is 17 plus 25? Answer briefly."
  ref $name window257 4 $flag $value --length 257; ref $name single 4 $flag $value --length 1
done
step spread-native
for c in baseline arithmetic window257 single; do
  $EXE $C/$N/liquid-lfm3 --experimental $CK $C/spread-v2/eager/$c/request.json $C/spread-v2/eager/$c/native.json > $B/spread-v2/native-$c.log 2>&1 || { echo "FAIL native $c"; tail -3 $B/spread-v2/native-$c.log; }
  # The KV-group prefill split must leave every trace byte unchanged (v5 traces from round 3).
  diff -rq $B/spread-eager-v1/$c/native $B/spread-v2/eager/$c/native > /dev/null && echo "$c: native traces byte-identical to v5" || { echo "$c: native traces DIFFER from v5"; diff -rq $B/spread-eager-v1/$c/native $B/spread-v2/eager/$c/native | head -5; }
done
step spread-compare
python3 $R/tools/compare_trace_spread.py $B/spread-v2/fp32 $B/spread-v2/rank.json --cases baseline,arithmetic,window257,single \
  --candidate eager $B/spread-v2/eager/reference $B/spread-v2/eager --candidate sdpa $B/spread-v2/sdpa/reference $B/spread-v2/sdpa \
  --candidate native $B/spread-v2/eager/native $B/spread-v2/eager
python3 -c "import json;[print(c['case'],json.load(open('$B/spread-v2/fp32/'+c['case']+'/reference.json'))['precision']) for c in json.load(open('$B/spread-v2/rank.json'))['cases']]"
CTL_RES=/workspace/tmp/control-codecfree-remediated-001/output.json
step http-batch-metrics
$EXE -e LFM_ALLOCATION_METRICS=1 lfm3-profile-v1 /opt/venv/bin/python3 $T/check_lfm3_http_batch.py --checkpoint $CK --worker $C/$N/liquid-lfm3-worker --control-result $CTL_RES --output $C/metrics-native-v6-http-batch > $B/metrics-native-v6-http-batch.log 2>&1
echo "rc=$?"
python3 - $B/metrics-native-v6-http-batch/summary.json <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); e=j["empty_session_gpu_baseline"]; p=j.get("post_close_gpu",{}); s=[x["gpu"] for x in j.get("samples",[])]
live=sorted({x.get("app_live_bytes") for x in s})
print("empty app_live", e.get("app_live_bytes"), "post-close", p.get("app_live_bytes"), "soak distinct", live, "samples", len(s), "status", j.get("status"))
PY
step gates-native
KEEP_GOING=1 FORWARD_ARGS=--activation-tolerance bash $R/tools/thor/run_native_gates.sh $N g4-native-v6; echo "gates rc=$?"
cat $B/g4-native-v6-gates/failures 2>/dev/null; cat $B/g4-native-v6-gates/status.json
echo "ALL DONE $(date +%T)"
