#!/usr/bin/env bash
# images-v1 resize and pixel_values run on Thor (2026-11-15). From ~/Projects/000-research/lfm3-omni-2B.
# Usage: thor_run.sh FMA_MODE
set -euo pipefail
MODE=${1:?FMA_MODE}
W=$PWD; L=liquid-CUDA/build/images-v1; O=$L/preprocess
IMG=sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f
EX="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia -w /workspace lfm3-profile-v1"
HFPY=/workspace/venvs/hf-5.20.2/bin/python
# HF: the noise sweep through the processor, and torch's resize alone (random cases, and the given ones)
$EX $HFPY -I liquid-CUDA/tools/hash_lfm3_image_front.py models/LFM3-Omni-2B $O/hf-noise.jsonl noise=noise:500:2026 | tail -1
$EX $HFPY -I liquid-CUDA/tools/hash_torch_resize.py 3000 2026 $O/resize-cases.txt $O/torch-resize.jsonl
$EX $HFPY -I liquid-CUDA/tools/hash_torch_resize.py --cases $O/diffcases.txt $O/torch-diffcases.jsonl
$EX $HFPY -I liquid-CUDA/tools/hash_torch_resize.py --cases $O/fmacases.txt $O/torch-fmacases.jsonl
$EX /opt/venv/bin/python -I -c "
import json, re
with open('$O/noise-sizes.txt', 'w') as f:
    for line in open('$O/hf-noise.jsonl'):
        r = json.loads(line)
        if 'name' in r:
            w, h = re.search(r'-(\d+)x(\d+)$', r['name']).groups()
            f.write(f'{w} {h}\n')
"
$EX chown -R 1000:1000 /workspace/$O
docker run --rm --network none --memory 24g -v $W:/workspace -w /workspace/liquid-CUDA $IMG bash -c \
  "bash tools/build_image_decode_check.sh build/images-v1/decode-aarch64-pre -DLFM_RESIZE_FMA_MODE=$MODE 2>&1 | tail -1 && \
   bash tools/build_image_decode_check.sh build/images-v1/decode-aarch64-pre-asan -DLFM_RESIZE_FMA_MODE=$MODE -fsanitize=address,undefined -fno-sanitize-recover=all 2>&1 | tail -1 && \
   cd /workspace && B=liquid-CUDA/build/images-v1 && C=\$B/decode-aarch64-pre/lfm3-image-decode-check && A=\$B/decode-aarch64-pre-asan/lfm3-image-decode-check && \
   \$C --preprocess cases=liquid-CUDA/build/vision-ref/hf-5.20.2/images variants=data/vision-image-variants bench=data/vision-bench/images > \$B/preprocess/native-files.jsonl && \
   \$C --preprocess noise=noise:2026:\$B/preprocess/noise-sizes.txt > \$B/preprocess/native-noise.jsonl && \
   \$C --resize \$B/preprocess/resize-cases.txt > \$B/preprocess/native-resize.jsonl && \
   \$C --resize \$B/preprocess/diffcases.txt > \$B/preprocess/native-diffcases.jsonl && \
   \$C --resize \$B/preprocess/fmacases.txt > \$B/preprocess/native-fmacases.jsonl && \
   (\$A --preprocess noise=noise:2026:\$B/preprocess/noise-sizes.txt cases=liquid-CUDA/build/vision-ref/hf-5.20.2/images > \$B/preprocess/asan.jsonl 2> \$B/preprocess/asan.stderr; echo asan exit \$?) && \
   sha256sum \$C \$A && chown -R 1000:1000 \$B"
echo PREPROCESS RUN DONE
