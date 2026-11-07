# G4 stage A, attempt 3 (FP32 logits): the §9 bar PASSES at all four lengths

This is the single run declared in `journal/g4-257_synth.md` (`737de02`). The bias that it removes
was measured before the run (`logit-bias/`, `8ff0415`): **8.0×10⁻⁵ nats/token**, from the convexity of
NLL in the logits.

**What changed from attempt 2** (under `LFM3_NATIVE_GEMM` only; FP32 scores are kept):
- lm_head stores its FP32 accumulator unrounded (`Runner::lm_head`, through `Problem::c_f32`).
- The engine returns FP32 logits to the sampler, in both `forward()` and the two-lane batch.
- The scorer's log-softmax reads FP32 logits.
- Logits traces keep their `.bf16` format, as a rounded copy.
- Nothing upstream of lm_head changes.

## Results (Thor; `run.log`; hashes in `sha256.txt`, which match the committed sources)

| Check | Result |
|---|---|
| Builds | 0 warnings in both configurations |
| Default build unchanged | `liquid-lfm3`, worker and server are identical to `cmake-server-v3` after `objcopy --strip-all` |
| Native build links no cuBLAS | NEEDED: `libcudart.so.13` plus the C and C++ runtime libraries. Sizes are 2.47, 2.55 and 2.73 MB |
| Host suite (x86) | 1,566 passed, 2 skipped |
| Perplexity against §9 (`ppl/compare-native-gemm.json`) | **PASS at every length** |

| Length | PPL, SDPA | PPL, native cuBLAS | Attempt 1 | Attempt 2 | **Attempt 3** | Attempt 3 − SDPA, 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| 256 | 26.5821 | 26.5812 | 26.5809 | 26.5805 | **26.5780** | [−2.78, −0.28]×10⁻⁴ | **better**, PASS |
| 257 | 26.5045 | 26.5066 | 26.5087 | 26.5086 | **26.5067** | [−0.35, +2.01]×10⁻⁴ | equal, PASS |
| 4,096 | 13.0596 | 13.0590 | 13.0590 | 13.0599 | **13.0587** | [−2.02, +0.60]×10⁻⁴ | equal, PASS |
| 32,768 | 12.5771 | 12.5776 | 12.5788 | 12.5767 | **12.5762** | [−1.83, +0.45]×10⁻⁴ | equal, PASS |

## Was the hypothesis right? (attempt 3 − attempt 2, which isolates the logit rounding)

| Length | Declared prediction | Measured, with 95% CI over windows | Windows lower (predicted) | |
|---|---|---|---|---|
| 256 | −(0.6 to 1.2)×10⁻⁴ | **−9.5×10⁻⁵** [−1.41, −0.49]×10⁻⁴ | 635 of 1,160, 55% (about 55%) | in range |
| 257 | −(0.6 to 1.2)×10⁻⁴ | **−7.3×10⁻⁵** [−1.17, −0.30]×10⁻⁴ | 628 of 1,155, 54% (about 55%) | in range |
| 4,096 | −(0.6 to 1.2)×10⁻⁴ | **−9.0×10⁻⁵** [−1.29, −0.47]×10⁻⁴ | 56 of 72, 78% (about 70%) | in range |
| 32,768 | −(0.6 to 1.2)×10⁻⁴ | **−3.9×10⁻⁵** [−0.72, −0.06]×10⁻⁴ | 7 of 9 (7 to 9 of 9) | **below range** |

- **Sign:** negative at all four lengths, and every CI excludes zero. Removing logit rounding lowers
  NLL wherever it was measured, as convexity requires.
- **Size:** three lengths fall within the declared range and match the 8.0×10⁻⁵ measured on one window.
- **The miss at 32,768:** −3.9×10⁻⁵ falls short of the range, and is a hair outside the success
  criterion of "within a factor of two of 8×10⁻⁵". Its CI reaches −7.2×10⁻⁵, and there are only 9
  windows. This run cannot say whether long-context logits are lower-entropy, or whether it was
  sampling. The verdict at 32K did not depend on it: the attempt-2 result there was already a PASS.
- **The verdict at 257 is not a coin flip:** the native engine moved from +1.57×10⁻⁴ to +8.3×10⁻⁵
  against SDPA. That is now the same as the cuBLAS engine (+4.3×10⁻⁶ against it) and HF eager, both of
  which pass.

## What this establishes

- **The quality condition for G4 stage A is met.** The engine with no cuBLAS is equal or better than
  HF SDPA at every §9 length, and better at 256.
- **The 257 failure in attempts 1 and 2 was not a GEMM defect.** It was cuBLAS-vs-native noise
  (+7.8×10⁻⁵, 52% of windows) at the length where SDPA changes backend. Removing a known, measured bias
  that every BF16 stack carries lifted the engine above it.
- **What is not established yet:**
  - The option stays default OFF, as declared.
  - The native build's runtime gates (server, worker, batch, sampling distribution) have not been
    run on it.
  - Its sampler now sees FP32 logits, so gates that pin bitwise agreement with the cuBLAS engine's
    BF16 logits will need re-baselining rather than equality.
  - Prefill remains about 3× cuBLAS until a `tcgen05` kernel exists (stage B).
