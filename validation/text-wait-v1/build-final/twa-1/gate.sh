#!/usr/bin/env bash
# text-wait-v1 rule 1: the text gate on twa-1, after the round (it bounds contention with timeouts: run it alone).
cd ~/Projects/000-research/lfm3-omni-2B/liquid-CUDA
until grep -q "TEXTWAIT ROUND DONE" build/textwait/twa-1/round.log; do sleep 60; done
C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
rm -rf build/textwait/twa-1/text-gate
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 /workspace/liquid-CUDA/tools/check_lfm3_native_server.py --checkpoint $M --server $C/twa-1/liquid-lfm3-server --worker $C/twa-1/liquid-lfm3-worker --control-result /workspace/tmp/control-codecfree-remediated-001/output.json --output $C/textwait/twa-1/text-gate > build/textwait/twa-1/text-gate.log 2>&1
echo "TEXT GATE exit $? $(date +%T)"
