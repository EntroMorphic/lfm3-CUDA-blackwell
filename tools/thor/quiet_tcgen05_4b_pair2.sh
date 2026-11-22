#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 4b under amendment 2: a fresh soak pair in reverse order, ON (tgemm-on1) first, then
# OFF (tgemm-off1); the soak unchanged (tools/check_lfm3_tcgen05_soak.py: 3,600 s, 12 clients, text share 0.25, images
# and long prompts; every memory reading recorded with free, buff/cache and the container cgroup), with
# tools/thor/host_memory_sampler.py every 30 s beside each run. Then the amended verdict
# (tools/tcgen05_4b_pair_verdict.py: server RSS and cgroup anon growth gated, ON <= max(OFF, 0) + 32 MiB, the harness's
# non-memory clauses; host figure and 429 retries per reply reported).
# No arguments; no quiet check of its own (called by a quiet-window runner). Results: build/quiet-tcgen05-4b-pair2/.
# Exit 0: rule 4b passes under amendment 2; 1: binaries missing or not MAXN; 2: rule 4b fails or a run did not finish.
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
TOOLS=$W/liquid-CUDA/tools/thor; SRC=$C/src-tcgen05; O=quiet-tcgen05-4b-pair2
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 1
log() { echo "$(date +%T) $*" | tee -a run.log; }
bash $TOOLS/power_state.sh power-start.json > /dev/null || { log "FAIL power (not MAXN)"; exit 1; }
(cd $B && sha256sum tgemm-on1/liquid-lfm3-server tgemm-on1/liquid-lfm3-worker tgemm-off1/liquid-lfm3-server \
  tgemm-off1/liquid-lfm3-worker src-tcgen05/tools/check_lfm3_tcgen05_soak.py src-tcgen05/tools/check_lfm3_audio_soak.py \
  src-tcgen05/tools/tcgen05_4b_pair_verdict.py) > binaries.sha256 || { log "FAIL: binaries missing"; exit 1; }
for side in on off; do
  T=tgemm-${side}1; d=$B/$O/$side; mkdir -p $d
  log "4b on $T (3,600 s, 12 clients, text share 0.25)"
  rm -f $d/stop; python3 $TOOLS/host_memory_sampler.py $d/host-memory.jsonl $d/stop 30 &
  SP=$!
  $X python3 $SRC/tools/check_lfm3_tcgen05_soak.py $C/$T/liquid-lfm3-server $M $C $L $C/vimg-e2e-a4 $C/$O/$side/soak-4b.json 3600 12 0.25 \
    > $d/soak-4b.log 2>&1; s=$?
  sleep 60; touch $d/stop; wait $SP
  tail -2 $d/soak-4b.log | tee -a run.log
  log "4b $side: harness exit $s (its own memory verdict is reported only under amendment 2)"
  [ -f $d/soak-4b.json ] && [ -f $d/soak-4b.readings.jsonl ] || { log "FAIL: $side run left no records"; exit 2; }
done
bash $TOOLS/power_state.sh power-end.json power-start.json > /dev/null || log "VOID: power state changed during the pair"
python3 $B/src-tcgen05/tools/tcgen05_4b_pair_verdict.py $B/$O > verdict.log 2>&1; v=$?
tail -1 verdict.log | tee -a run.log
log "QUIET TCGEN05 4B PAIR2 DONE: $([ $v = 0 ] && echo PASS || echo FAIL)"
[ $v = 0 ] && exit 0 || exit 2
