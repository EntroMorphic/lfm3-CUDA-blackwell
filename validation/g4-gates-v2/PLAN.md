# G4: the gates that replace bitwise-eager for the native-GEMM engine — declared before implementation

Tripp approved three tiers and an exact memory check on 2026-11-07 (`../g4-native-gates-v1/README.md`). This file
fixes how each is measured, before any of it is built or run. The powered fidelity test (2b) runs once; its result is
reported whatever it is.

## Tier 1 — regression: bitwise against the native engine's own pinned output

- `forward` (4 fixtures: baseline, arithmetic, window257, single) and `context` (its 9 cases and their free-running
  generations): a committed manifest of SHA-256 per native trace file (forward: every activation and logit file;
  context: every `.logits.bf16` and `.logits.f32`), plus the generated ids.
- PASS iff the native run produces exactly the pinned file set with identical hashes and identical ids.
- The pin is made from the native build that passes tiers 2 and 3, with `--pin`; re-pinning is a deliberate,
  committed act that must cite a new tier-2/3 pass.

## Tier 2 — fidelity against an FP32 anchor

**2a, activations and tokens (forward fixtures).** Anchor: the model in FP32, eager attention, TF32 disabled
(`lfm3_reference.py --dtype float32`). In the same gate run: HF eager BF16 and HF SDPA BF16 references.
- PASS iff, per fixture, native's median activation relative RMS from the anchor, over the files every candidate
  traces at steps where all tokens agree, is ≤ the larger of eager's and SDPA's (`compare_trace_spread.py`);
- and teacher-forced argmax and free-running greedy equal HF eager's exactly (forward), teacher argmax equals
  eager's exactly (context) — as the gates already require.

**2b, logits, powered (the §9 windows).** Anchor: the model in FP32 with SDPA attention (eager cannot run 32K),
TF32 disabled, scored by `ppl_reference.py` over the same windows as `ppl-v5`.
- Per token: A = |NLL_native − NLL_fp32|, S = |NLL_sdpa − NLL_fp32|. Paired over windows: D = mean(A) − mean(S).
- Read like §9: 95% paired bootstrap CI over windows (ppl_compare's), ε = 1e-5 nats/token.
  **FAIL iff the whole CI of D lies above +ε** at any of 256, 257, 4,096, 32,768. Native NLL: the attempt-3 files
  (bitwise identical to the v6 build's, `../g4-native-gates-v1/round4/sha256.txt`).
- Also reported, not part of the verdict: the same with HF eager in place of SDPA (≤ 4,096), and mean(A), mean(S).
- If 2b fails, the default stays cuBLAS and the result goes to Tripp.

## Tier 3 — quality

§9 perplexity as decided (already PASS: `../g4-native-gemm-v3/`; the v6 files are bitwise the same).

## Memory — exact engine-buffer accounting

`http-batch` with `LFM_ALLOCATION_METRICS=1` on a native build: PASS iff the worker's `app_live_bytes` after close
equals its value with sessions empty, and no soak sample exceeds the first warm sample. Device-wide readings are
recorded, not judged. (A default build has no accounting; it keeps the device-wide check.)

## Then

Default the CMake option to the native GEMM (cuBLAS remains an explicit option for A/B), make `build_scorer.sh`
default to `native-gemm`, let `benchmark_lfm3.py` accept the native labels for speed while recording that the
eager-parity claim does not apply, and run every gate on the final default build.
