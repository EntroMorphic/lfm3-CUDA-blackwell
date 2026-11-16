# The vision encoder on the tensor cores: meet or beat HF end to end, keep the accuracy lead

## What it will be (Tripp, 2026-11-16: "Do it.")
images-v1's speed round found native 30% slower than HF end to end on 4K photographs, and faster on every other
stage-0 request (`validation/images-v1/README.md`, "Speed against HF"). nsys traced it to the code: the encoder's
matrix multiplies run `fp32::linear_staged`, a few-row kernel that streams every weight matrix 512 times at 8,192
patches (memory-bound, ~31% of the FP32 cores' peak), and its attention holds one query per thread with a
shared-memory load per FMA (~30%). Even a perfect FP32 kernel (~0.55 s) cannot reach HF's 0.148 s encode: the tensor
cores are required. This round moves the encoder's multiplies and attention onto them with the engine's existing
`mma.sync` path, keeping its FP32 residual stream, norms, softmax and nonlinearities. (`tcgen05`/TMEM/TMA is a
separate, later round.)

## Design
- **Multiplies**: `lfm::gemm` (`wide`: `mma.sync` m16n8k16, BF16 in, FP32 accumulate), FP32 output, then one kernel
  adding the bias (and, split, the halves). The weights are BF16 already, exactly. Activations, two variants:
  - `x1`: the FP32 activation rounded to BF16 (one GEMM);
  - `x2`: split, `hi = bf16(x)`, `lo = bf16(x - hi)`: `x·W = hi·W + lo·W` to ~2^-16 of x (a batch of 2 GEMMs
    sharing W, summed in FP32).
- **Attention**: a new fused kernel (flash-attention style, head dimension 64, all keys: the encoder is
  bidirectional): a block's warps own 16 queries each; per key tile, S = Q·Kᵀ on the tensor cores, the online softmax
  in FP32 registers, O += P·V on the tensor cores. Variants:
  - `x1`: Q, K, V and P in BF16;
  - `x2`: each split into hi + lo, three products each (hi·hi + hi·lo + lo·hi; lo·lo is below FP32's resolution here).
- Everything else unchanged: the patchifier and projector use the same multiply path; norms, gate, spatial conv,
  rotary (FP64 tables), GELU (FP64), residual adds stay as they are. The output is rounded to BF16 as before.
- The variant is chosen by the rules below, in this order: (linear `x1`, attention `x1`), then the split for
  whichever part a failure points to, then both split. The fastest variant that passes rule 1 ships. No other
  variant is added without amending these rules first.

## Rules, set before any code
1. **Accuracy, on all 31 images of images-v1 rule 2** (11 stage-0, 20 held-out MMStar; the same HF FP32 and BF16
   dumps; `tools/compare_lfm3_vision_stages.py`): on every image, image_features' relative L2 error to HF FP32 is at
   most **0.15 x HF BF16's**, and the minimum per-token cosine to FP32 is at least **0.9999**. (images-v1: ratios
   0.023-0.112, cosines >= 0.9999983; the output's BF16 rounding alone is 1.66e-3.)
2. **Speed** (`tools/thor/vision_speed_round.sh`, quiet, MAXN, the method of images-v1): native's end to end is at
   most HF's on **every** stage-0 request, the two 4K photographs included. Also reported: the encode alone at
   7,920 patches, and nsys's kernel shares.
3. **Benchmarks** (the full MMStar and POPE runs of images-v1 rule 3, the new build): MMStar answers agree with HF
   FP32's at least as often as HF BF16's do (rule 3b's measure: >= 93.87%); POPE accuracy against HF BF16 with a
   paired-bootstrap lower bound >= -0.5 points.
4. **Coherency**: I read the 11 stage-0 replies and every benchmark item whose answer changed from images-v1's
   native run.
5. **Sanitizers**: memcheck (leaks), racecheck, synccheck and initcheck on `lfm3-vision-check`'s small, two-image and
   largest cases: 0 errors, outputs byte-identical to the plain run.
6. **Nothing else changes**: lfm3-speech's regression (36 outputs), the audio-in replies (360 + 93 files), e2e and
   split (90 + 90), and the server checks (batch, stream, audio in, TTS, text gate with control `7ea05068...`) all
   identical; images through the server identical to lfm3-speech; the host suites pass.
