#!/usr/bin/env bash
# G6 verification on Thor host: forward (6890), context suite, grouped worker with oracle replay of the
# live-token bound, plus the tokenizer-parse startup cost on Thor's CPU. Status file per stage.
set -uo pipefail
W=/home/ztflynn/Projects/000-research/lfm3-omni-2B; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
out=$B/g6-gates-v1; test ! -e $out || { echo "exists: $out"; exit 1; }; mkdir $out
status(){ printf '{"stage":"%s","rc":%s,"unix":%s}\n' "$1" "$2" "$(date +%s)" > $out/status.json; }
status parse-timing null
g++ -std=c++20 -O2 -I $W/liquid-CUDA/native/include $W/liquid-CUDA/native/tests/tokenizer_count_probe.cpp -o $out/tokcount || { status parse-timing 1; exit 1; }
for i in 1 2 3; do /usr/bin/time -f "%e s %M KB" $out/tokcount $W/models/LFM3-Omni-2B/tokenizer.json 128000; done > $out/parse-timing.log 2>&1
EX="docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 /opt/venv/bin/python3"
run(){ name=$1; shift; status $name null; $EX "$@" > $out/$name.log 2>&1; rc=$?; [ $rc = 0 ] || { status $name $rc; exit $rc; }; }
CK=/workspace/models/LFM3-Omni-2B; T=/workspace/liquid-CUDA/tools; N=$C/cmake-g6-v1
run forward $T/check_lfm3_forward.py --checkpoint $CK --binary $N/liquid-lfm3 --output $C/g6-forward-v1
run context $T/check_lfm3_context.py --checkpoint $CK --binary $N/liquid-lfm3 --suite full --reuse-reference $C/context-full-bounded-release-v2 --timeout 180 --output $C/g6-context-v1
run grouped-worker $T/check_lfm3_batch_worker.py --checkpoint $CK --worker $N/liquid-lfm3-worker --request $C/forward-bounded-prefill-v1/baseline/generation-request.json --oracle $N/text-sampling-oracle --long-request $C/context32k-bounded-prefill-v1/decode_into32768/request.json --output $C/g6-grouped-worker-v1
status done 0
