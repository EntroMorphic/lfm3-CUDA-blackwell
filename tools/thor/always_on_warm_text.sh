#!/usr/bin/env bash
# always-on-v1 rule 2, the text part (2026-11-17, after window 18): the soak's three text bodies, each the first request
# on a fresh server with --warmup and without it (tools/check_lfm3_warm_text.py), window 15's build. Exactness only.
# Usage (Thor host): always_on_warm_text.sh   (results: build/always-on-warm-text/)
set -uo pipefail
W=${LFM3_WORKSPACE:-$HOME/Projects/000-research/lfm3-omni-2B}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build
O=always-on-warm-text; S=always-on-src2; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O && cd $B/$O || exit 1
(cd $B && cat $S/SOURCE_COMMIT; sha256sum always-on-srv/liquid-lfm3-server $S/tools/check_lfm3_warm_text.py) > sources.txt
$X python3 $C/$S/tools/check_lfm3_warm_text.py $C/always-on-srv/liquid-lfm3-server $M $C/$O/warm-text.json > warm-text.log 2>&1
echo "exit $?"; cat warm-text.log
echo "ALWAYS-ON-WARM-TEXT-FINISHED $(date +%T)"
