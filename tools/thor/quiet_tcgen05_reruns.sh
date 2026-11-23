#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 2, the two server checks to rerun on a quiet machine (the first run, on a shared GPU,
# stopped in the batch-server check's reference phase with 503 worker_unavailable, and failed only the stream check's
# timing rule): the batch-server check, then the stream check, on the build with the tcgen05 path (tgemm-on1), as
# tools/thor/vision_rule6_server.sh runs them.
# No arguments. Assumes a quiet MAXN machine; refuses to start otherwise. Results: build/tgemm-r2-quiet/.
# Exit 0: both pass; 1: not quiet or not MAXN; 2: a check failed.
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
TOOLS=$W/liquid-CUDA/tools/thor; S=tgemm-on1; O=tgemm-r2-quiet; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 2
log() { echo "$(date +%T) $*" | tee -a run.log; }
busy=$(pgrep -af "[l]fm3-speech|[l]fm3-batch|[l]fm3-vision-check|[l]iquid-lfm3|[c]ompute-sanitizer|[t]cgen05-gemm-probe|[l]fm3-tcgen05|[r]un_lfm3|[c]heck_lfm3|[t]ime_lfm3" | head -3)
[ -n "$busy" ] && { log "NOT QUIET: $busy"; exit 1; }
bash $TOOLS/power_state.sh power-start.json > /dev/null || { log "FAIL power (not MAXN)"; exit 1; }
sha256sum $B/$S/liquid-lfm3-server $B/$S/liquid-lfm3-worker $B/server-stream-v3/liquid-lfm3-server > binaries.sha256
rc=0
log "batch server"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_batch_server.py $C/$S/liquid-lfm3-server $C/server-stream-v3/liquid-lfm3-server $M $C $L $C/$O/batch-server.json > batch-server.log 2>&1 || rc=2
tail -1 batch-server.log | tee -a run.log
log "stream check"
$X python3 /workspace/liquid-CUDA/tools/check_lfm3_speech_stream.py $C/$S/liquid-lfm3-server $M $C $L $C/$O/stream-check.json > stream-check.log 2>&1 || rc=2
tail -1 stream-check.log | tee -a run.log
bash $TOOLS/power_state.sh power-end.json power-start.json > /dev/null || log "VOID: power state changed"
log "RERUNS DONE: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
