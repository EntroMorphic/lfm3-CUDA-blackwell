#!/usr/bin/env bash
# Full perplexity evaluation, reference then native. Runs on the Thor host and drives both containers:
# the HF reference in lfm3-omni-dev, the native scorer in lfm3-profile-v1 (pinned runtime).
# Writes OUT.status.json after every stage (liveness for watchers).
# Usage (Thor host): run_ppl.sh OUT_NAME
set -uo pipefail
name=$1; B=/home/ztflynn/Projects/000-research/lfm3-omni-2B/liquid-CUDA/build; out=$B/$name; C=/workspace/liquid-CUDA/build/$name
status(){ printf '{"stage":"%s","rc":%s,"unix":%s}\n' "$1" "$2" "$(date +%s)" > $out.status.json; }
test ! -e $out || { echo "exists: $out"; exit 1; }
status reference null
docker exec -e TRITON_PTXAS_PATH=/workspace/tools/ptxas -e TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas lfm3-omni-dev \
  python /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt --output $C \
  --lengths 256 257 4096 32768 > $out.reference.log 2>&1
rc=$?; [ $rc = 0 ] || { status reference $rc; exit $rc; }
for L in 256 257 4096 32768; do
  status native-$L null
  docker exec lfm3-profile-v1 /workspace/liquid-CUDA/build/score-v1/lfm3-score /workspace/models/LFM3-Omni-2B $C/windows-$L.i32 $L $C/nll-native-$L.f32 >> $out.native.log 2>&1
  rc=$?; [ $rc = 0 ] || { status native-$L $rc; exit $rc; }
done
status done 0
