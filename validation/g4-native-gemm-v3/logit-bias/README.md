# The NLL bias from rounding logits to BF16, measured before attempt 3

Measured on Thor on 2026-11-07 with the default (cuBLAS) `liquid-lfm3` from `cmake-server-v3`, so this
evidence is independent of the native-GEMM build and of the §9 bar data it predicts.

**Input:** window 0 of `ppl-v5/windows-257.i32`. Positions 0–64 prefilled, positions 65–256 teacher-
forced with `trace_logits_only`, giving 193 logit steps (`build/lmm-logits-v1/` on Thor). The last one predicts past
the window's end, so 192 positions are scored.

**Why there is a bias:** NLL = logsumexp(z) − z_y is convex in z. If the logits carry a zero-mean
rounding error ε, E[NLL] rises by ½ Σ p_i(1−p_i) σ_i² to second order (the −ε_y term has mean zero).
A BF16 value of magnitude 16–32 has ulp 1/8; at 8–16, 1/16. The error is uniform in ±ulp/2, so
σ² = ulp²/12.

**Result** (`bias.py TRACE ppl-v5/windows-257.i32 257 0 65`; `bias.txt` is its verbatim output):

| Quantity | Value |
|---|---|
| Max |logit| per position | median 20.8, max 24.9 |
| ulp at the argmax logit | median 1/16 |
| p_max | median 0.508 |
| Mean NLL of the 192 scored positions | 2.650 |
| Bias, second-order formula | **7.99e-5** nats/token (median 9.5e-5, range 3e-8 to 2.3e-4) |
| Bias, Monte Carlo (64 draws per position, control variate ε·p) | **8.04e-5**, standard error over positions 3.3e-6 |

Both the native scorer (`row_nll` on bf16 logits) and the HF reference (`lm_head(hidden).float()` after
a bf16 matmul, `tools/ppl_reference.py:65`) carry this bias. Storing the lm_head accumulator in FP32
removes it from the native engine. The prediction for attempt 3 is in `journal/g4-257_synth.md`.

**What the Monte Carlo checks:** it draws errors under the same model (independent, uniform in ±ulp/2) and so
validates the second-order truncation, not the model. Deterministic round-to-nearest-even on synthetic FP32
logits gave 9.5e-5 against the formula's 1.03e-4 (an independent check by review, about 8% apart). At exact
powers of two the true rounding cell is asymmetric, which the ±ulp/2 model ignores (about 1 value in 128).
