#!/usr/bin/env bash
# text-wait-v1 rule 3, attempt 2 (load rows kept busy to 10 arrivals), alone on a quiet Thor after the round and gate.
cd ~/Projects/000-research/lfm3-omni-2B/liquid-CUDA
until grep -q "TEXT GATE exit" build/textwait/twa-1/gate.log; do sleep 60; done
if ps -eo args | grep -E "[c]ompute-sanitizer|[l]fm3-speech |[l]iquid-lfm3-server" | grep -v defunct; then echo "NOT QUIET"; exit 1; fi
C=/workspace/liquid-CUDA/build
docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 /workspace/liquid-CUDA/tools/check_lfm3_text_wait.py $C/twa-1/liquid-lfm3-server /workspace/models/LFM3-Omni-2B $C /workspace/data/librispeech/wav $C/textwait/twa-1/rule3-attempt2.json rule3 > build/textwait/twa-1/rule3-attempt2.log 2>&1
echo "RULE3 DONE $(date +%T)"; tail -2 build/textwait/twa-1/rule3-attempt2.log
