# G4 stage A, attempt 2 (FP32 scores): the §9 bar FAILS at 257. G4 stage A is parked

This is the single run declared in `PLAN.md`, which was committed before the code changed (`77c5fc1`). It
fails at 257 tokens. As declared, G4 stage A is parked, and no further variants will be tried without
Tripp's say-so.

**What changed from attempt 1** (under `LFM3_NATIVE_GEMM` only):
- `lfm::gemm` has an optional FP32 output, `Problem::c_f32`, with the same accumulation order.
- QKᵀ stores FP32 scores, and the softmax applies the 1/√d scale in FP32 with no BF16 rounding.
- Probabilities are still BF16 for P·V.

## Results (Thor; `run.log`; hashes in `sha256.txt`, sources matching the commit)

| Check | Result |
|---|---|
| Builds | 0 warnings in both configurations |
| Default build unchanged | `liquid-lfm3`, worker and server are identical to `cmake-server-v3` after `objcopy --strip-all` |
| Native build links no cuBLAS | NEEDED: `libcudart.so.13` plus the C and C++ runtime libraries. Sizes are 2.47, 2.55 and 2.73 MB |
| Host suite (x86) | 1,566 passed, 2 skipped |
| Perplexity against §9 (`ppl/compare-native-gemm.json`) | **FAIL at 257** |

| Length | PPL, SDPA | PPL, native cuBLAS | PPL, attempt 1 | PPL, attempt 2 | Attempt 2 − SDPA, 95% CI | Verdict |
|---|---|---|---|---|---|---|
| 256 | 26.5821 | 26.5812 | 26.5809 | 26.5805 | [−1.87, +0.70]×10⁻⁴ | equal, PASS |
| 257 | 26.5045 | 26.5066 | 26.5087 | 26.5086 | [+0.28, +2.85]×10⁻⁴ | **worse, FAIL** |
| 4,096 | 13.0596 | 13.0590 | 13.0590 | 13.0599 | [−1.24, +1.58]×10⁻⁴ | equal, PASS |
| 32,768 | 12.5771 | 12.5776 | 12.5788 | 12.5767 | [−1.41, +0.74]×10⁻⁴ | equal, PASS |

## What the paired numbers show (mean ΔNLL per window, nats/token)

| Length | Attempt 2 − attempt 1 | Attempt 2 − native cuBLAS | Native cuBLAS − SDPA |
|---|---|---|---|
| 256 | −1.3×10⁻⁵ (572 of 1,160 windows higher) | −2.7×10⁻⁵ | −3.2×10⁻⁵ |
| 257 | **−1.2×10⁻⁶** (564 of 1,155 higher) | **+7.8×10⁻⁵** (597 of 1,155 higher) | +7.9×10⁻⁵ |
| 4,096 | +6.4×10⁻⁵ (41 of 72 higher) | +6.4×10⁻⁵ | −4.7×10⁻⁵ |
| 32,768 | −1.7×10⁻⁴ (2 of 9 higher) | −7.1×10⁻⁵ | +3.7×10⁻⁵ |

- **FP32 scores fixed 32K.** That length moved from +9.9×10⁻⁵ to −7.1×10⁻⁵ against cuBLAS, and passes.
- **FP32 scores did nothing at 257.** Attempt 2 − attempt 1 is −1.2×10⁻⁶ there.
- **The 257 shift lies elsewhere.** Against cuBLAS it is about +7.8×10⁻⁵ in both attempts, so it does
  not come from score rounding. The remaining differences from cuBLAS are the projection GEMMs, P·V, and
  the padding to 264 keys, all at M = 257. This run cannot tell which one.
- **Most of the gap to SDPA was already there.** The cuBLAS engine already sat +7.9×10⁻⁵ above SDPA at
  257, with a CI of [−0.6, +2.2]×10⁻⁴, so it passed narrowly. The native GEMM adds about as much again,
  and the windows split close to 50/50, which is the signature of rounding noise rather than bias.
  Under the bar as decided, that is still a FAIL.
