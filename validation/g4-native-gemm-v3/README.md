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
| 256 | −(0.6 to 1.2)×10⁻⁴ | **−9.5×10⁻⁵** [−1.40, −0.50]×10⁻⁴ | 635 of 1,160, 55% (about 55%) | in range |
| 257 | −(0.6 to 1.2)×10⁻⁴ | **−7.3×10⁻⁵** [−1.17, −0.29]×10⁻⁴ | 628 of 1,155, 54% (about 55%) | in range |
| 4,096 | −(0.6 to 1.2)×10⁻⁴ | **−9.0×10⁻⁵** [−1.29, −0.49]×10⁻⁴ | 56 of 72, 78% (about 70%) | in range |
| 32,768 | −(0.6 to 1.2)×10⁻⁴ | **−3.9×10⁻⁵** [−0.72, −0.05]×10⁻⁴ | 7 of 9 (7 to 9 of 9) | **below range** |

The CIs in this table are percentile bootstraps over windows (10,000 resamples, from `tools/g4_paired_stats.py`;
the first version of this table used an ad hoc 4,000-resample bootstrap and differed in the last digit). With 9 windows at 32,768 that interval is too
narrow; the Student-t interval there is [−8.1, +0.4]×10⁻⁵ and includes zero
(`../g4-native-gates-v1/paired-stats.json`, from `tools/g4_paired_stats.py`).

**The declared success criteria (`journal/g4-257_synth.md`), scored as declared:**

| Criterion | Result |
|---|---|
| Default build identical after stripping; native build links no cuBLAS | **Met** |
| Attempt 3 − attempt 2 negative at every length, within a factor of two of 8×10⁻⁵ (4–16×10⁻⁵) | **Not met at 32,768:** −3.85×10⁻⁵ is below 4×10⁻⁵. Met at 256, 257, 4,096 |
| §9 PASS at all four lengths | **Met** |
| Host suite 1,566 passed | **Met** (observed locally; the log was not committed) |

**The other predictions:**

| Prediction | Result |
|---|---|
| Shift −(0.6 to 1.2)×10⁻⁴ at each length | Met at 256, 257, 4,096; **missed at 32,768** |
| 257's CI lower bound moves from +2.8×10⁻⁵ to about −5×10⁻⁵ | It moved to −3.5×10⁻⁵: right direction, smaller than predicted |
| Corpus bias at or above the one-window 8×10⁻⁵ (Node 10) | **Missed:** 7.3×10⁻⁵ at 257 and 3.9×10⁻⁵ at 32,768 |

- **Sign:** the shift is negative at all four lengths. The bootstrap CIs exclude zero at all four; the
  t-interval excludes it at 256, 257 and 4,096 and not at 32,768.
- **Size:** three lengths fall within the declared range, consistent with the 8.0×10⁻⁵ measured on one window.
  At 32,768 the bias came out about half the size; this run cannot say whether long-context logits are
  lower-entropy or whether it is sampling (9 windows).

## What this establishes, and what it does not

- **The §9 bar, as written, passes at all four lengths.** It passes because of a strict improvement at the
  output stage (§9 item 4, "keep FP32 between fused ops"): the logits are no longer rounded to BF16 before the
  log-softmax. Tripp approved this run ("Let's experiment and find the bedrock.", 2026-11-07, after the
  declaration in `journal/g4-257_synth.md`).
- **Everything upstream of lm_head is bit for bit attempt 2.** At 257 the native − cuBLAS difference upstream is
  unchanged (+7.8×10⁻⁵ in attempts 1, 2 and 3, each CI including zero). The attempt-3 engine's +4.3×10⁻⁶ against
  cuBLAS at 257 is that difference **offset** by removing a bias of about the same size, not agreement in the
  computation.
- **The comparison is asymmetric.** SDPA still rounds its logits to BF16 (`tools/ppl_reference.py:65`). Compared
  like for like, with both sides rounding, the native engine is attempt 2, which fails at 257 (CI
  [+0.27, +2.85]×10⁻⁴). The intervention was chosen because its predicted size would clear that margin, so the
  PASS was the expected outcome of the change, not independent evidence about the GEMMs.
- **Whether the native GEMMs differ systematically from cuBLAS at 257 is open.** No 257-row shape was probed. The
  paired data cannot tell noise from a small systematic difference of +7.8×10⁻⁵.
- **Multiple comparisons:** this is the third run against the same bar. The pre-registration protects the size
  prediction of this run, not the bar verdict across attempts.
- **Not established here** (later evidence: `../g4-native-gates-v1/`): the runtime gates on the native build,
  re-baselining gates that pin bitwise agreement with the cuBLAS engine, and a default-ON decision. Prefill remains
  slower than cuBLAS until a `tcgen05` kernel exists (stage B).

## Corrections (2026-11-07, red-team)

- The sections above were rewritten after an independent review. The earlier text said the 257 failure "was not
  a GEMM defect" and that the 257 verdict "is not a coin flip" because attempt 3 matched cuBLAS; both are
  withdrawn for the reasons given above. It also said every CI excludes zero, which is false at 32,768 with a
  t-interval, and it did not score the declared criteria as a table.
- `logit-bias/bias.txt` is now the verbatim output of `logit-bias/bias.py` (fixed to stop at the end of window 0:
  192 scored positions). The earlier file came from interactive passes; its numbers are reproduced.
- `sha256.txt` hashes the scorer's sources; `native/src/lfm3_decode_batch.cuh`, also changed, is not part of the
  scorer and is not hashed there.
- The default-build identity check covered `liquid-lfm3`, worker and server, not `lfm3-decode-batch-probe` (round 4 of
  `../g4-native-gates-v1/` compared it: stripped identical to `cmake-server-v3`).
