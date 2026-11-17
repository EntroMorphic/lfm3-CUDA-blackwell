#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 4b, the off/on pair (README "Rule 4b off/on pair", read as set there before the data):
# the window-4 server soak (tools/check_lfm3_tcgen05_soak.py: 3,600 s, 12 clients, text share 0.25, images and long
# prompts added; its pass rule unchanged) first on tgemm-off1 (the same sources without the tcgen05 path), then on
# tgemm-on1. Each run also records every host-memory reading with free, buff/cache and the container's memory cgroup
# (OUT.readings.jsonl), with tools/thor/host_memory_sampler.py every 30 s beside it.
# No arguments; no quiet check of its own (called by a quiet-window runner). Results: build/quiet-w5-tcgen05-4b/{off,on}/.
# Exit 0: both runs pass; 1: binaries missing or not MAXN; 2: a run failed (the pair is still read per the README).
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
TOOLS=$W/liquid-CUDA/tools/thor; SRC=$C/src-tcgen05; O=quiet-w5-tcgen05-4b
M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 1
log() { echo "$(date +%T) $*" | tee -a run.log; }
bash $TOOLS/power_state.sh power-start.json > /dev/null || { log "FAIL power (not MAXN)"; exit 1; }
(cd $B && sha256sum tgemm-off1/liquid-lfm3-server tgemm-off1/liquid-lfm3-worker tgemm-on1/liquid-lfm3-server \
  tgemm-on1/liquid-lfm3-worker src-tcgen05/tools/check_lfm3_tcgen05_soak.py src-tcgen05/tools/check_lfm3_audio_soak.py) \
  > binaries.sha256 || { log "FAIL: binaries missing"; exit 1; }
rc=0
for side in off on; do
  T=tgemm-${side}1; d=$B/$O/$side; mkdir -p $d
  log "4b on $T (3,600 s, 12 clients, text share 0.25)"
  rm -f $d/stop; python3 $TOOLS/host_memory_sampler.py $d/host-memory.jsonl $d/stop 30 &
  SP=$!
  $X python3 $SRC/tools/check_lfm3_tcgen05_soak.py $C/$T/liquid-lfm3-server $M $C $L $C/vimg-e2e-a4 $C/$O/$side/soak-4b.json 3600 12 0.25 \
    > $d/soak-4b.log 2>&1; s=$?
  sleep 60; touch $d/stop; wait $SP
  tail -2 $d/soak-4b.log | tee -a run.log
  log "4b $side: exit $s $([ $s = 0 ] && echo PASS || echo FAIL)"
  [ $s = 0 ] || rc=2
done
bash $TOOLS/power_state.sh power-end.json power-start.json > /dev/null || log "VOID: power state changed during the pair"
log "QUIET TCGEN05 4B OFF/ON DONE: $([ $rc = 0 ] && echo 'both PASS' || echo 'at least one FAIL')"
exit $rc
