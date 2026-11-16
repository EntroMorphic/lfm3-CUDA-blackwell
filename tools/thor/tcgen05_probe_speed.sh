#!/usr/bin/env bash
# validation/tcgen05-probe-v1, speed rules S1-S3 (method in its README, set before any run): one round = the Peak
# microbenchmark, then every shape at every M in its listed output type, variants 1-3 against lfm::gemm and cuBLAS
# (3 warm-ups, median of 20 CUDA-event timings each). Quiet Thor, MAXN, clocks locked; power state recorded at the
# start and checked at the end. Run three rounds, then tools/tcgen05_probe_speed_summary.py on the three.
# Usage (Thor host): tcgen05_probe_speed.sh BUILD_TAG OUT_TAG        e.g. b3 tcgen05-speed-r1
set -uo pipefail
T=${1:?BUILD_TAG}; O=${2:?OUT_TAG}
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
P=$C/tcgen05-$T/tcgen05-gemm-probe
mkdir -p $B/$O && cd $B/$O
busy=$(pgrep -af "[l]fm3-speech|[l]iquid-lfm3|[c]ompute-sanitizer|[r]un_lfm3|[c]heck_lfm3|[t]cgen05-gemm-probe|[p]ython.*torch" | head -3)
[ -n "$busy" ] && { echo "NOT QUIET: $busy"; exit 1; }
bash $W/liquid-CUDA/tools/thor/power_state.sh power-start.json > /dev/null || { echo "FAIL power (not MAXN)"; exit 1; }
sha256sum $B/tcgen05-$T/tcgen05-gemm-probe > binary.sha256
echo "== peak $(date +%T)"
$X $P peak 100000 > peak.jsonl 2> peak.err || { echo "peak FAILED"; tail -3 peak.err; }
echo "== speed $(date +%T)"
$X $P speed 1,2,3 --reps 20 > speed.jsonl 2> speed.err || { echo "speed FAILED"; tail -3 speed.err; }
bash $W/liquid-CUDA/tools/thor/power_state.sh power-end.json power-start.json > /dev/null || echo "VOID: power state changed"
echo "TCGEN05 SPEED DONE $(date +%T): $(wc -l < speed.jsonl) cases"
