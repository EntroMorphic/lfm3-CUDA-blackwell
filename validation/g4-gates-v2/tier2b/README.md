# G4 tier 2b: the native engine's NLL is closer to the FP32 model's than HF SDPA's, because of its FP32 logits — PASS

Declared in `../PLAN.md` (`c4e001d`), anchor amended before the run (`ba2c4a7`; evidence behind the amendment in
`../amendment-evidence/`, committed after the run), run once with `tools/thor/g4_tier2b.sh` (log `run.log`, report
`tier2b.json`).

**Anchor:** the model in FP32, eager attention computed 2,048 query rows at a time, TF32 disabled (`summary.json`:
precision), over the §9 windows (hashes identical to `ppl-v5`). NLL hashes: `anchor-sha256.txt`; metadata:
`nll-hf-eager_blocked-fp32-*.meta.json`. **Engine:** attempt 3's native NLL (`../../g4-native-gemm-v3/`), matched to
the v6 build's hashes (`../../g4-native-gates-v1/round4/sha256.txt`) by `--expect-sha256`.

**What it measures:** the mean absolute difference between an engine's NLL at the target token and the FP32
model's, per token, paired over windows against HF SDPA's. It is not a measure of the whole output distribution.

## The gate (`tier2b.json`)

| Length | Windows | Mean \|native − FP32\| | Mean \|SDPA − FP32\| | D = native − SDPA, 95% CI | Windows native closer | Verdict |
|---|---|---|---|---|---|---|
| 256 | 1,160 | 0.01674 | 0.01940 | [−2.74, −2.57]×10⁻³ | 1,113 | closer than SDPA |
| 257 | 1,155 | 0.01677 | 0.01948 | [−2.79, −2.62]×10⁻³ | 1,118 | closer than SDPA |
| 4,096 | 72 | 0.01501 | 0.01753 | [−2.60, −2.43]×10⁻³ | 72 | closer than SDPA |
| 32,768 | 9 | 0.01585 | 0.01818 | [−2.43, −2.20]×10⁻³ | 9 | closer than SDPA |

Native's deviation is 12.8–14.4% smaller than SDPA's. Against HF eager (information, ≤ 4,096): D = −4.61, −4.59,
−4.73 ×10⁻³, all CIs below zero. At 32,768 the CI is a percentile bootstrap over 9 windows, which is too narrow
there; native is closer in all 9 windows, and the Student-t interval is [−2.47, −2.18]×10⁻³ (`tier2b-recomputed.json`,
the gate's inputs recomputed by `tools/thor/g4_tier2b_diagnostics.sh` with t-intervals added; identical D and bootstrap
CIs).

## Where the difference comes from: the logits (`diagnostic-attempt2-bf16-logits.json`)

The comparison above is asymmetric: native's logits are FP32 (attempt 3), while SDPA's and eager's are rounded to BF16
before the log-softmax (`tools/ppl_reference.py`, `lm_head(...).float()` on a BF16 model). Attempt 2 is bit for bit
attempt 3 upstream of lm_head but rounds its logits to BF16 like SDPA, so it is the like-for-like engine. Run after the
gate, as a diagnostic, with the same anchor (`tools/thor/g4_tier2b_diagnostics.sh`, log `diagnostics.log`; engine files
matched to `../../g4-native-gemm-v2/sha256.txt`):

| Length | Mean \|attempt 2 − FP32\| | Mean \|SDPA − FP32\| | D, 95% CI | Windows attempt 2 closer | vs eager, D (CI) |
|---|---|---|---|---|---|
| 256 | 0.01941 | 0.01940 | +1.4×10⁻⁵ [−7.1, +10.1]×10⁻⁵ | 579 of 1,160 | −1.93×10⁻³ (below zero) |
| 257 | 0.01946 | 0.01948 | −1.7×10⁻⁵ [−10.4, +7.4]×10⁻⁵ | 576 of 1,155 | −1.91×10⁻³ (below zero) |
| 4,096 | 0.01749 | 0.01753 | −3.5×10⁻⁵ [−11.3, +4.5]×10⁻⁵ | 39 of 72 | −2.25×10⁻³ (below zero) |
| 32,768 | 0.01824 | 0.01818 | +6.1×10⁻⁵ [−4.4, +15.6]×10⁻⁵ (t: [−6.6, +18.9]×10⁻⁵) | 3 of 9 | — |

- **Upstream of the logits, the native engine is not distinguishable from SDPA** in closeness to the FP32 model: every
  CI includes zero, at a resolution of about ±10⁻⁴ (ten times ε). It is closer than HF eager at every length where
  eager exists.
- **The 12.8–14.4% advantage over SDPA comes from keeping the logits in FP32**, i.e. from not rounding the output,
  not from the GEMMs or attention. It is real for the engine as shipped (its output is FP32), but it is an output
  effect.
- So tier 2b does not answer whether native and cuBLAS differ upstream at 257 tokens (`../README.md`).

## Robustness: against the anchor as originally declared (`robust-ppl-g4-v3.json`, `robust-ppl-g4-v2.json`, same script)

The declared anchor, FP32 with SDPA attention (`build/ppl-fp32-v1` on Thor), completed 256, 257 and 4,096 before
Thor hung at 32,768 (`../amendment-evidence/`). Against it, at those three lengths, both attempt 3 and attempt 2 give
the same D, CIs and verdicts as against the amended anchor, to the digits shown; the anchors differ by a mean of
4.7×10⁻⁶ per token on the one window measured. The declared anchor has no run-level `summary.json` (the run stopped at
32,768), so its windows are taken from `ppl-v5` (each NLL file's metadata binds to them) and its TF32 setting is the one
`ppl_reference.py --dtype float32` applies, not a recorded field.

## Relation to the fixture measurements

The four forward fixtures measured full-distribution logit KL from FP32 (`../../g4-native-gates-v1/`), where native
was farther than SDPA on three small fixtures (2–16 steps). Tier 2b answers a related, powered question (target-token
NLL only), not that one; the KL question stays open.
