#!/usr/bin/env bash
# Run every gate that drives liquid-lfm3 / liquid-lfm3-worker (forward, context, 32K, sampling, resident
# worker, HTTP server, native + HTTP cancellation, grouped worker, HTTP grouped batching) against one
# build, plus the tokenizer-parse startup cost on Thor's CPU. Thor host. Stops at the first failure.
# Writes OUT/status.json per stage. Not run: the standalone resource soak (unchanged by token choice;
# HTTP grouped batching already holds a 10-minute soak).
# Usage: run_native_gates.sh BUILD_DIR_NAME OUT_TAG
# KEEP_GOING=1 runs every stage and lists the failing ones in OUT/failures (inventory mode); the exit
# status is then 1 if any stage failed. Both modes run the native (single-binary) server gate last.
# FORWARD_ARGS passes extra flags to the forward gate (--activation-tolerance for the native-GEMM build).
set -uo pipefail
build=$1; tag=$2
case "${KEEP_GOING:-0}" in 0|1) ;; *) echo "KEEP_GOING must be 0 or 1"; exit 2 ;; esac
W=/home/ztflynn/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
out="$B/$tag-gates"; test ! -e "$out" || { echo "exists: $out"; exit 1; }; mkdir "$out"
status(){ printf '{"stage":"%s","rc":%s,"unix":%s}\n' "$1" "$2" "$(date +%s)" > "$out/status.json"; }
bins="liquid-lfm3 liquid-lfm3-worker text-sampling-oracle liquid-lfm3-server"
for f in $bins; do test -x "$B/$build/$f" || { echo "missing $B/$build/$f"; exit 1; }; done
python3 -c 'import json,sys; print(json.dumps(dict(build=sys.argv[1], keep_going=sys.argv[2] == "1", forward_args=sys.argv[3])))' \
  "$build" "${KEEP_GOING:-0}" "${FORWARD_ARGS:-}" > "$out/mode.json"
(cd "$B/$build" && sha256sum $bins) > "$out/binaries.sha256" \
  || { echo "cannot hash binaries"; exit 1; }
status parse-timing null
g++ -std=c++20 -O2 -I $W/liquid-CUDA/native/include $W/liquid-CUDA/native/tests/tokenizer_count_probe.cpp -o "$out/tokcount" || { status parse-timing 1; exit 1; }
for _ in 1 2 3; do
  r=$( { /usr/bin/time -f "%e s %M KB" "$out/tokcount" $W/models/LFM3-Omni-2B/tokenizer.json 128000; } 2>&1 ); echo "$r" >> "$out/parse-timing.log"
  [ "$(echo "$r" | head -1)" = 125033 ] || { status parse-timing 1; exit 1; }
done
EX="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 /opt/venv/bin/python3"
failed=0
run(){ name=$1; shift; status "$name" null; $EX "$@" > "$out/$name.log" 2>&1; rc=$?; [ $rc = 0 ] && return 0
  status "$name" $rc; [ "${KEEP_GOING:-0}" = 1 ] || exit $rc; echo "$name rc=$rc" >> "$out/failures"; failed=1; }
CK=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools; N=$C/$build; O=$C/$tag
REQ=$C/forward-bounded-prefill-v1/baseline/generation-request.json   # accepted tokenized request (sampling/worker gates)
# Accepted 128-token native control for the HTTP/cancellation gates (the reflective-recursion prompt, 86
# tokens to <|im_end|>). Both files are pinned by the sha256 values recorded in
# validation/frontend-remediation/post-remediation-native/summary.json (artifacts_sha256), the output also in
# validation/http-group-benchmark-v1/summary.json (control_sha256).
CTL_REQ=/workspace/tmp/control-codecfree-remediated-001/request.json; CTL_RES=/workspace/tmp/control-codecfree-remediated-001/output.json
for pin in "request.json:5651a375a4dd1144c6f49e72a666cf2423ef3062b5502821381a6936ed1c4946" \
           "output.json:5d1a4ecbd9be9f3de6d85fa888d640610c32e33caa1729c4f7a1fd8f5fff230b"; do
  f=${pin%%:*}; want=${pin#*:}
  [ "$(sha256sum "$W/tmp/control-codecfree-remediated-001/$f" | cut -c1-64)" = "$want" ] \
    || { status control-pin 1; echo "control $f differs from the accepted one"; exit 1; }
done
LONG=$C/context32k-bounded-prefill-v1/decode_into32768/request.json
run forward        $T/check_lfm3_forward.py --checkpoint $CK --binary $N/liquid-lfm3 --output $O-forward ${FORWARD_ARGS:-}
run context        $T/check_lfm3_context.py --checkpoint $CK --binary $N/liquid-lfm3 --suite full --reuse-reference $C/context-full-bounded-release-v2 --timeout 180 --output $O-context
run sampling       $T/check_lfm3_sampling.py --checkpoint $CK --binary $N/liquid-lfm3 --request $REQ --oracle $N/text-sampling-oracle --output $O-sampling
run worker         $T/check_lfm3_worker.py --checkpoint $CK --worker $N/liquid-lfm3-worker --single $N/liquid-lfm3 --request $REQ --long-request $LONG --oracle $N/text-sampling-oracle --output $O-worker
run grouped-worker $T/check_lfm3_batch_worker.py --checkpoint $CK --worker $N/liquid-lfm3-worker --request $REQ --oracle $N/text-sampling-oracle --long-request $LONG --output $O-grouped-worker
run server         $T/check_lfm3_server.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --concurrency-rounds 3 --output $O-server
run cancellation   $T/check_lfm3_cancellation.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-request $CTL_REQ --control-result $CTL_RES --output $O-cancellation
run http-cancellation $T/check_lfm3_http_cancellation.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O-http-cancellation
run 32k            $T/check_lfm3_32k.py --checkpoint $CK --binary $N/liquid-lfm3 --output $O-32k
run http-batch     $T/check_lfm3_http_batch.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O-http-batch
run native-server $T/check_lfm3_native_server.py --checkpoint $CK --server $N/liquid-lfm3-server --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O-native-server
[ $failed = 0 ] || { status "done-with-failures" 1; exit 1; }
status "done" 0
