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

## Amendment (2026-11-07, before tier 2b ran): the 2b anchor is exact FP32 eager attention, computed in blocks

The declared 2b anchor, FP32 with SDPA attention, cannot be used as declared:
- **It does not fit at 32,768.** In FP32, SDPA falls back to an unfused kernel; one 32,768-token window exceeded
  96 GB of Thor's unified memory and hung the machine twice (once fully; once stopped by a watchdog, after which
  the driver held the memory until a reboot).
- **It is not exact FP32.** Its memory-efficient attention uses 3xTF32 GEMMs regardless of the TF32 flags. On one
  4,096-token window it differs from plain FP32 eager by up to 5.05e-5 nats per token (mean 4.7e-6), which is the
  size of ε.

Instead the anchor is `ppl_reference.py --configs eager_blocked --dtype float32`: the model's own eager attention
math (scores, additive eager mask, FP32 softmax, P·V), computed 2,048 query rows at a time, TF32 disabled. On one
4,096-token window it is **bitwise identical** to plain FP32 eager; at 32,768 it runs within a few GB. It is used at
all four lengths, so every length has the same anchor. Engine name: `hf-eager_blocked-fp32`. Nothing else in 2b
changes: the statistic, the CI, ε, the FAIL rule, and the single run.

## Notes after the run (2026-11-07, independent review); the declaration and amendment above are unchanged

- The amendment's figures are backed by `amendment-evidence/`, committed after the run: the memory figures are
  transcribed from the session (they were printed, not written to files); the 4,096 comparison is reproduced by
  `amendment-evidence/compare.py`.
- "exceeded 96 GB": it reached about 96 GB (107 → 11 GB available) when the watchdog stopped it. "Hung the machine
  twice": once it hung; the second time a SIGKILL watchdog stopped it and the driver kept the memory until a reboot.
- "at 32,768 it runs within a few GB" is not supported as stated: the run kept MemAvailable at or above 73 GB, and
  MemAvailable may not capture every GPU allocation on Thor.
- "bitwise identical to plain FP32 eager" was measured on one 4,096-token window on Thor (identical NLL files). Blocks
  smaller than the sequence change matmul shapes, so in general they agree to FP32 rounding (a CPU test,
  `tests/test_ppl_reference_eager_blocked.py`, bounds that); at 256 and 257 one block covers the sequence.
- The declared anchor (FP32 SDPA) had completed 256, 257 and 4,096 before the amendment; against those lengths 2b gives
  the same result (`tier2b/robust-*.json`).
- 2b's margin comes from the engine's FP32 logits; with BF16 logits the engine is as close to FP32 as SDPA
  (`tier2b/README.md`).

