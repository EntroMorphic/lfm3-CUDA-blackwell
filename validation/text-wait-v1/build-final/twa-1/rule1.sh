#!/usr/bin/env bash
# text-wait-v1 rule 1 over HTTP: twa-1 against assure-s4, as is and with CUDA_LAUNCH_BLOCKING=1.
C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B; L=/workspace/data/librispeech/wav
for mode in normal blocking; do
  E=""; [ $mode = blocking ] && E="-e CUDA_LAUNCH_BLOCKING=1"
  echo "== $mode $(date +%T)"
  docker exec $E -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1 python3 /workspace/liquid-CUDA/tools/check_lfm3_text_wait_exact.py $C/twa-1/liquid-lfm3-server $C/assure-s4/liquid-lfm3-server $M $C $L $C/textwait/twa-1/rule1-$mode.json
done
echo "RULE1 DONE $(date +%T)"
