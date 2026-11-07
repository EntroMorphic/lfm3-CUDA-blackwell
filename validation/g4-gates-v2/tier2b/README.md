# G4 tier 2b: the native engine's NLL is closer to the FP32 model's than HF SDPA's — PASS

Declared in `../PLAN.md` (`c4e001d`), anchor amended before the run (`ba2c4a7`), run once with
`tools/thor/g4_tier2b.sh` (log `run.log`, report `tier2b.json`).

**Anchor:** the model in FP32, eager attention computed 2,048 query rows at a time (bitwise identical to plain FP32
eager on a 4,096-token window), TF32 disabled (`summary.json`: precision), over the §9 windows (hashes identical
to `ppl-v5`). NLL hashes: `anchor-sha256.txt`; metadata: `nll-hf-eager_blocked-fp32-*.meta.json`.
**Engine:** attempt 3's native NLL (`../../g4-native-gemm-v3/`), matched to the v6 build's hashes
(`../../g4-native-gates-v1/round4/sha256.txt`) by `--expect-sha256`.

| Length | Windows | Mean \|native − FP32\| | Mean \|SDPA − FP32\| | D = native − SDPA, 95% CI | Verdict |
|---|---|---|---|---|---|
| 256 | 1,160 | 0.01674 | 0.01940 | [−2.74, −2.57]×10⁻³ | closer than SDPA |
| 257 | 1,155 | 0.01677 | 0.01948 | [−2.79, −2.62]×10⁻³ | closer than SDPA |
| 4,096 | 72 | 0.01501 | 0.01753 | [−2.60, −2.43]×10⁻³ | closer than SDPA |
| 32,768 | 9 | 0.01585 | 0.01818 | [−2.43, −2.20]×10⁻³ | closer than SDPA |

Against HF eager (information, ≤ 4,096): D = −4.61, −4.59, −4.73 ×10⁻³, all CIs below zero.

- Per token, native's NLL is about 13–14% closer to the FP32 model's than SDPA's is, at every length; the CIs lie
  far below −ε (ε = 10⁻⁵). At 32,768 the bootstrap has 9 windows; the margin is about 200 times ε.
- What this measures: closeness of the per-token NLL at the target token, not of the full distribution.
- This is the powered version of the question the four forward fixtures could not settle (there, native's logit KL
  was farther than SDPA's on three small fixtures with 2–16 steps).
