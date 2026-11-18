#!/usr/bin/env bash
# Quiet window 18 (2026-11-17), after window 17: always-on-v1 rule 1 as amended after window 15. Window 15's build
# (build/always-on-srv, main d0a5e80), unchanged; the tool from main (build/always-on-src2: --image-third). Part c only:
# 3 fresh --warmup servers, each kind's 1st, 2nd and 3rd request; the 3rd image is quiet-w10-c3c5's c3-31.png (3840 x
# 2160, the cherry photograph perturbed, never sent to these servers), so it is a miss, as the 1st is. Timing: quiet
# Thor, in the container (whose stall caveat is in prefill-stall-v1's window 14).
# Usage (Thor host): quiet_session_v18.sh   (results: build/quiet-w18-always-on/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; R=$W/liquid-CUDA; B=$R/build; C=/workspace/liquid-CUDA/build
O=quiet-w18-always-on; M=/workspace/models/LFM3-Omni-2B; S=always-on-src2; SRV=always-on-srv
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
THIRD=quiet-w10-c3c5/images/fresh/c3-31.png

missing=0
for f in $B/$SRV/liquid-lfm3-server $B/$S/tools/measure_lfm3_first_request.py $B/$THIRD; do
  [ -e $f ] || { echo "MISSING $f"; missing=1; }
done
grep -q -- "--image-third" $B/$S/tools/measure_lfm3_first_request.py || { echo "MISSING --image-third in build/$S"; missing=1; }
[ $missing = 0 ] || { echo "NOT STARTED: inputs missing"; echo "QUIET-V18-FINISHED $(date +%T)"; exit 1; }

until grep -q "^QUIET-V17-FINISHED\|^NOT QUIET" $B/quiet-w17.log 2>/dev/null; do sleep 60; done
busy() { pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[l]fm3-vision-check|[a]udio-encoder-bench|[v]envs/hf-|[n]sys|[t]cgen05_gemm|[t]cgen05-soak|[t]gemm_host_race|[f]rontend-|[n]vcc|[c]make --build|[n]inja" | grep -v "quiet_" | head -3; }
quiet=0
for i in $(seq 1 1440); do
  if [ -z "$(busy)" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && break; else quiet=0; fi
  [ $i = 1 ] && echo "waiting for quiet $(date +%T): $(busy | cut -c1-150)"; sleep 30
done
[ -n "$(busy)" ] && { echo "NOT QUIET after 12 h"; exit 1; }
echo "== quiet $(date +%T)"
mkdir -p $B/$O && cd $B/$O || exit 1
(cd $B && cat $S/SOURCE_COMMIT 2>/dev/null; sha256sum $SRV/liquid-lfm3-server $S/tools/measure_lfm3_first_request.py $THIRD) > sources.txt
bash $R/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power"; exit 1; }
echo "== rule 1 (amended): part c, the 3rd image a miss $(date +%T)"
$X python3 $C/$S/tools/measure_lfm3_first_request.py $C/$SRV/liquid-lfm3-server $M $C $C/$O/rule1.json --parts c \
  --image-third $C/$THIRD > rule1.log 2>&1
echo "exit $?"; grep "^image\|^rule" rule1.log
python3 -c "
import json; d=json.load(open('rule1.json'))['rule1']
for k,v in d['per_kind'].items(): print(k, 'first', round(v['first_median'],4), 'third', round(v['third_median'],4), 'bound', round(v['bound'],4), 'PASS' if v['passed'] else 'FAIL')
print('RULE 1', 'PASS' if d['passed'] else 'FAIL')"
bash $R/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID power changed"
echo "QUIET-V18-FINISHED $(date +%T)"
