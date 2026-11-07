# G4 stage A, attempt 2: declared before the run

Attempt 1 (`validation/g4-native-gemm-v1/`) failed the §9 bar at 257 and 32,768 tokens. Tripp chose
option 1: make one principled precision change, declare it before running, and run it once.

## The change (under `LFM3_NATIVE_GEMM` only; the default build is untouched)

Attention scores stay in FP32 from the QKᵀ accumulator through the softmax:

1. `lfm::gemm` gains an optional FP32 output, `Problem::c_f32`. When it is set, every kernel stores
   its FP32 accumulator without rounding to BF16. The accumulation order is unchanged.
2. QKᵀ writes FP32 scores.
3. The softmax kernels read FP32 scores and apply the 1/√d scale in FP32, with no BF16 rounding,
   either of the raw score or of the scaled score.
4. Probabilities are still rounded to BF16, because they are the BF16 operand of P·V. This is what
   SDPA's flash and memory-efficient kernels do as well.

Nothing else changes from attempt 1: the projections, P·V, padding and traces are the same.

**Justification:** §9 item 4 allows "keep FP32 between fused ops" as a strict improvement. Attempt 1
rounds the scores twice, at the GEMM output and after scaling. That reproduces HF eager, while SDPA,
the reference the bar is measured against, keeps the scores in FP32.

**Cost:** the score scratch doubles in bytes. At capacity 4,096 that is 512 MB → 1 GB of device
memory; at capacity 32,768 it is 128 MB → 256 MB.

## The single run that decides it

- The perplexity harness is the same as attempt 1: `build_scorer.sh OUT native-gemm`, WikiText-2 test
  at 256, 257, 4,096 and 32,768 tokens, compared with `tools/ppl_compare.py` against `g6-v2/ppl-v5`
  SDPA. ε = 10⁻⁵ and the 95% paired bootstrap CI are unchanged.
- The run happens once, and its result is reported whatever it is.
- **If it passes at all four lengths,** the variant becomes the native-GEMM build. The option stays
  default OFF until the remaining gates are dealt with.
- **If it fails at any length,** G4 stage A stays parked. No further variants will be tried
  without Tripp's say-so, because trying variants until one passes would make the CI meaningless.
