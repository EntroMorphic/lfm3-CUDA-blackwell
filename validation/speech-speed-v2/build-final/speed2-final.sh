#!/usr/bin/env bash
# speech-speed-v2: rebuild the references (8f87d69: speed-a4, speed-s2, text tools) from nothing, build the new engine,
# then every check of validation/speech-speed-v2/README.md's rules.
set -uo pipefail
W=/home/ztflynn/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
L=/workspace/data/librispeech/wav; REF=$W/ref-8f87d69
RA=speed-a4r; RS=speed-s2r; RT=speed-t4r; T=speed2-a2; S=speed2-s1; TT=speed2-t2
O=$B/speed2-final; CO=$C/speed2-final
F="-O3 --fmad=false -std=c++20 -arch=sm_110 -I /workspace/liquid-CUDA/native/include"
rm -rf $B/speed2-final; echo "== reference build $(date +%T)"
mkdir -p $B/$RA $B/$RT
docker run --rm --network none -v $W:/workspace -v $REF/native:/workspace/liquid-CUDA/native:ro $IMG bash -c "set -e; N=/workspace/liquid-CUDA/native
  cd $C/$RA; nvcc $F -DLFM3_NATIVE_GEMM -o lfm3-speech \$N/tests/lfm3_speech.cu \$N/src/safetensors.cpp > build.log 2>&1
  cd $C/$RT; nvcc $F -DLFM3_NATIVE_GEMM -o lfm3-batch \$N/tests/lfm3_batch.cu \$N/src/safetensors.cpp > build.log 2>&1
  chown -R 1000:1000 $C/$RA $C/$RT" || { echo "FAIL reference build"; exit 1; }
docker run --rm --network none -v $REF/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$RS -DLIQUID_LFM3_NATIVE_GEMM=ON > /dev/null; chown -R 1000:1000 $C/$RS $C/$RS.*" || { echo "FAIL reference server"; exit 1; }
echo "== build $(date +%T)"
bash $W/liquid-CUDA/tools/thor/build_speech.sh $T || { echo "FAIL build speech"; exit 1; }
bash $W/liquid-CUDA/tools/thor/build_batch_tools.sh $TT > /dev/null || { echo "FAIL build text tools"; exit 1; }
docker run --rm --network none -v $W/liquid-CUDA/native:/src/native:ro -v $W:/workspace $IMG bash -c "bash /workspace/liquid-CUDA/tools/thor/build_native.sh $C/$S -DLIQUID_LFM3_NATIVE_GEMM=ON && (cd /workspace/liquid-CUDA && sha256sum native/include/lfm/*.hpp native/include/lfm/*.cuh native/src/*.cu native/src/*.cuh native/src/*.cpp) > $C/$S/sources.sha256; chown -R 1000:1000 $C/$S $C/$S.*" || { echo "FAIL build server"; exit 1; }
echo "server build warnings: $(grep -ciE 'warning' $B/$S.build.log)"
docker run --rm --network none -v $W:/workspace $IMG bash -c "set -e; cd /workspace/liquid-CUDA
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -DLFM3_NATIVE_GEMM -I native/include -o build/$T/lfm3-batch-capacity-check native/tests/lfm3_batch_capacity_check.cu native/src/safetensors.cpp
  nvcc -O3 --fmad=false -std=c++20 -arch=sm_110 -I native/include -o build/$T/fp32-linear-check native/tests/fp32_linear_staged_check.cu
  chown -R 1000:1000 build/$T" > $B/$T/tools-build.log 2>&1 || echo "FAIL tools build"
mkdir -p $O
sha256sum $B/$RA/lfm3-speech $B/$RS/liquid-lfm3-server $B/$RS/liquid-lfm3-worker $B/$RT/lfm3-batch $B/$T/lfm3-speech $B/$T/fp32-linear-check $B/$T/lfm3-batch-capacity-check $B/$TT/lfm3-batch $B/$TT/tile-sweep $B/$S/liquid-lfm3-server $B/$S/liquid-lfm3-worker > $O/binaries.sha256
cat $O/binaries.sha256
# nvcc names its temporary file by a process counter (tmpxft_000000NN), kept in the symbol string table only: a rebuild
# from the same sources differs there by a byte. The references are compared with their symbol tables stripped
# against speech-speed-v1's recorded binaries (whose raw hashes the recorded ones are).
for p in speed-a4/lfm3-speech:$RA/lfm3-speech speed-s2/liquid-lfm3-server:$RS/liquid-lfm3-server speed-s2/liquid-lfm3-worker:$RS/liquid-lfm3-worker; do
  r=${p%%:*}; n=${p#*:}
  objcopy --strip-all $B/$r /tmp/s2r-a && objcopy --strip-all $B/$n /tmp/s2r-b
  echo "$(sha256sum $B/$r | cut -c1-16) recorded $r; stripped $(sha256sum < /tmp/s2r-a | cut -c1-16) vs rebuilt $n $(sha256sum < /tmp/s2r-b | cut -c1-16)" | tee -a $O/reference-check.txt
  cmp -s /tmp/s2r-a /tmp/s2r-b || { echo "FAIL reference $n"; exit 1; }
done
grep -q "^efd4f4de4deecfec" <(sha256sum $B/speed-a4/lfm3-speech) && grep -q "^e66b8fcab511c05f" <(sha256sum $B/speed-s2/liquid-lfm3-server) && grep -q "^6789ab216e2ef5ba" <(sha256sum $B/speed-s2/liquid-lfm3-worker) || { echo "FAIL recorded binaries changed"; exit 1; }
echo "references equal the recorded binaries, symbol tables stripped"
bash $W/liquid-CUDA/tools/thor/power_state.sh $O/power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
echo "== tile sweep $(date +%T)"
$X $C/$TT/tile-sweep > $O/tile-sweep.log 2>&1; tail -1 $O/tile-sweep.log
echo "== fp32 linear $(date +%T)"
$X $C/$T/fp32-linear-check > $O/fp32-linear-check.log 2>&1; tail -1 $O/fp32-linear-check.log
echo "== capacity check $(date +%T)"
$X $C/$T/lfm3-batch-capacity-check $M > $O/capacity-check.json 2>&1; echo "capacity check exit $? $(cat $O/capacity-check.json)"
echo "== speech regression $(date +%T)"
bash $W/liquid-CUDA/tools/thor/speech_regression_check.sh $T > $O/speech-regression.log 2>&1
echo "same $(grep -c '^SAME' $O/speech-regression.log)"; grep -E "DIFFER|RUN FAILED" $O/speech-regression.log
echo "== engine exactness $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_concurrent.py $C/$T/lfm3-speech $M $C $L $CO/conc $CO/engine-exact.json $C/speed-final2/conc/alone > $O/engine-exact.log 2>&1; grep -E "reference|CONCURRENT" $O/engine-exact.log
echo "== text batch, same and not slower $(date +%T)"
mkdir -p $O/text
python3 - "$B" "$O/text" <<'P'
import json, sys
B, O = sys.argv[1:]
names = ("sky", "code", "arith", "poem", "memory")
prompts = [json.load(open(f"{B}/g4-coh-v2/native-{n}-512.json"))["input_ids"] for n in names]
prompts.append(json.load(open(f"{B}/cli-demo/out2.json"))["input_ids"])
for mode, t in (("greedy", 0), ("sampled", 0.528)):
    json.dump(dict(capacity=4096, prompts=prompts, batch=6, max_new_tokens=512, temperature=t, seed=42, serial=False), open(f"{O}/rows-{mode}.json", "w"))
for n in (1, 8, 16):
    json.dump(dict(capacity=4096, prompts=[prompts[5]], batch=n, max_new_tokens=256, temperature=0, stop=[], warmup=True), open(f"{O}/rate-{n}.json", "w"))
P
for b in $RT $TT; do
  for f in rows-greedy rows-sampled; do $X $C/$b/lfm3-batch $M $CO/text/$f.json $CO/text/$f-$b.out.json > /dev/null || echo "FAIL $b $f"; done
done
for i in 1 2; do for n in 1 8 16; do for b in $RT $TT; do
  printf "  $b batch $n run $i: "; $X $C/$b/lfm3-batch $M $CO/text/rate-$n.json $CO/text/rate-$n-$b-$i.out.json || echo "FAIL $b rate $n"
done; done; done
python3 - "$O/text" $RT $TT > $O/text-batch.json <<'P'
import json, sys, statistics
O, a, b = sys.argv[1:]
ids = lambda f: [r["generated_ids"] for r in json.load(open(f))["rows"]]
out = {"same": {}, "rate": {}}
for f in ("rows-greedy", "rows-sampled"): out["same"][f] = ids(f"{O}/{f}-{a}.out.json") == ids(f"{O}/{f}-{b}.out.json")
for n in (1, 8, 16):
    out["same"][f"rate-{n}"] = all(ids(f"{O}/rate-{n}-{a}-{i}.out.json") == ids(f"{O}/rate-{n}-{b}-{i}.out.json") for i in (1, 2))
    r = {k: statistics.median(json.load(open(f"{O}/rate-{n}-{k}-{i}.out.json"))["decode_tokens_per_second"] for i in (1, 2)) for k in (a, b)}
    out["rate"][n] = dict(old=r[a], new=r[b], ratio=r[b] / r[a])
out["pass_"] = all(out["same"].values()) and all(v["ratio"] >= 0.98 for v in out["rate"].values())
print(json.dumps(out, indent=1))
P
cat $O/text-batch.json | tr -d '\n' | head -c 600; echo
echo "== throughput, paired $(date +%T)"
for k in 1 2; do
  for b in $RA $T; do
    $X python3 /workspace/liquid-CUDA/tools/measure_lfm3_speech_batch_throughput.py $C/$b/lfm3-speech $M $C $CO/tp-$b-$k $CO/tp-$b-$k.json > $O/tp-$b-$k.log 2>&1; echo "$b run $k: $(tail -1 $O/tp-$b-$k.log)"
  done
done
echo "== batch server $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_batch_server.py $C/$S/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L $CO/batch-server.json > $O/batch-server.log 2>&1; tail -1 $O/batch-server.log
echo "== stream check $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_stream.py $C/$S/liquid-lfm3-server $M $C $L $CO/stream-check.json > $O/stream-check.log 2>&1; tail -1 $O/stream-check.log
echo "== audio-in server $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_audio_in_server.py $C/$S/liquid-lfm3-server $C/$T/lfm3-speech $M $L $CO/work $CO/audio-in-server.json > $O/audio-in-server.log 2>&1; tail -1 $O/audio-in-server.log
echo "== TTS server $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_server.py $C/$S/liquid-lfm3-server $M $C $CO/speech-check.json > $O/speech-check.log 2>&1; tail -1 $O/speech-check.log
echo "== text gate $(date +%T)"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_native_server.py --checkpoint $M --server $C/$S/liquid-lfm3-server --worker $C/$S/liquid-lfm3-worker --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $CO/text-gate > $O/text-gate.log 2>&1; echo "text gate exit $?"
bash $W/liquid-CUDA/tools/thor/power_state.sh $O/power-end.json $O/power-start.json > /dev/null || echo "WARN power changed"
echo "SPEED2FINAL DONE $(date +%T)"
