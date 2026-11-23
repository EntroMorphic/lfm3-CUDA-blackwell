#!/usr/bin/env bash
# validation/tcgen05-gemm-v1 rule 4 (tensor-memory soak), for an unattended quiet window, on the build with the
# tcgen05 path (tgemm-on1):
#   4a. lfm3-tcgen05-soak, 1,200 s: 4 streams of tcgen05, mma.sync and attention GEMMs; bitwise against solo, no
#       error, every call within 10x the same problem's median under the soak (README amendment 1; 10x solo reported);
#   4b. tools/check_lfm3_tcgen05_soak.py, 3,600 s, 12 clients, text share 0.25: the audio soak of audio-assurance-v1 /
#       text-wait-v1 with the 11 image stage-0 requests and three long text prompts (~300, 1,500, 3,900 tokens) added;
#       its pass rule (every reply identical to its reference, no 5xx or transport error, processes alive, memory flat
#       at the idle checkpoints, the crowd identical, exit 0); host memory sampled every 30 s beside it.
# No arguments. Assumes a quiet MAXN machine; refuses to start otherwise. Results: build/quiet-w4-tcgen05/.
# Exit 0: both pass; 1: not quiet or not MAXN; 2: a soak failed or did not run.
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
TOOLS=$W/liquid-CUDA/tools/thor; SRC=$C/src-tcgen05
ON=tgemm-on1; O=quiet-w4-tcgen05; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 2
log() { echo "$(date +%T) $*" | tee -a run.log; }
busy=$(pgrep -af "[l]fm3-speech|[l]fm3-batch|[l]fm3-vision-check|[l]iquid-lfm3|[c]ompute-sanitizer|[t]cgen05-gemm-probe|[l]fm3-tcgen05|[r]un_lfm3|[c]heck_lfm3|[t]ime_lfm3" | head -3)
[ -n "$busy" ] && { log "NOT QUIET: $busy"; exit 1; }
bash $TOOLS/power_state.sh power-start.json > /dev/null || { log "FAIL power (not MAXN)"; exit 1; }
(cd $B && sha256sum $ON/lfm3-tcgen05-soak $ON/liquid-lfm3-server $ON/liquid-lfm3-worker src-tcgen05/tools/check_lfm3_tcgen05_soak.py) > binaries.sha256 \
  || { log "FAIL: binaries missing"; exit 2; }
rc=0
log "4a synthetic soak (1,200 s)"
$X $C/$ON/lfm3-tcgen05-soak 1200 > soak-4a.jsonl 2> soak-4a.err; a=$?
tail -1 soak-4a.jsonl | tee -a run.log
[ $a = 0 ] || { log "4a FAIL (exit $a)"; rc=2; }
log "4b server soak (3,600 s, 12 clients, text share 0.25)"
rm -f stop; python3 $TOOLS/host_memory_sampler.py $B/$O/host-memory.jsonl $B/$O/stop 30 &
SP=$!
$X python3 $SRC/tools/check_lfm3_tcgen05_soak.py $C/$ON/liquid-lfm3-server $M $C $L $C/vimg-e2e-a4 $C/$O/soak-4b.json 3600 12 0.25 > soak-4b.log 2>&1; s=$?
sleep 60; touch stop; wait $SP
tail -3 soak-4b.log | tee -a run.log
[ $s = 0 ] || { log "4b FAIL (exit $s)"; rc=2; }
bash $TOOLS/power_state.sh power-end.json power-start.json > /dev/null || log "VOID: power state changed during the window"
log "QUIET TCGEN05 SOAK DONE: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
