# G4 stage A: the runtime gates on the native-GEMM build, and what the failing ones measure

Native-GEMM builds `cmake-g4-native-v3` (round 1) through `-v6` (round 4, the remediated build). Thor,
`lfm3-profile-v1`. Each default build is identical to `cmake-server-v3` after `objcopy --strip-all`
(`liquid-lfm3`, worker, server; round 4 also `lfm3-decode-batch-probe`). Round logs: `round2/run.log`,
`round3/run.log`, `round4/run.log` and its script `round4/g4f.sh`; round 1 was run directly with
`tools/thor/run_native_gates.sh` and has no separate log (its status, failures and hashes are in `round1/`).
Round 4 used the runner and spread tool as they were before the second verification pass (its `mode.json` has
the older format). Its http-batch step failed to start (a script bug, `rc=127` in `round4/run.log`); the v6
http-batch, speed and FP32-rank evidence is in `evidence-v1/`, produced by `tools/thor/g4_native_evidence.sh`.
`round4/http-batch-memory.json`'s v6 entry came from a by-hand rerun and is superseded by `evidence-v1/`.

## The gates

| Gate | Round 1 (v3) | Round 4 (v6) | What it pins |
|---|---|---|---|
| server, cancellation, http-cancellation | PASS | PASS | The pinned 128-token cuBLAS greedy control replays exactly on native |
| native-server (G7 single binary) | PASS | PASS | Serving, cancellation, batching, soak, 22-request differential, shutdown |
| grouped-worker | PASS | PASS | Serial vs. grouped decode: every logit bit |
| 32k | PASS | PASS | Argmax against SDPA, with tolerance |
| worker | FAIL | **PASS** | Worker logits vs. single-shot trace, exact. Failed because the trace held rounded logits; fixed by the exact `.logits.f32` sidecar |
| sampling | FAIL | **PASS** | Oracle replays the sampler from the trace. Same cause, same fix |
| http-batch | FAIL | **PASS** | Device-wide free memory (see below) |
| forward | FAIL | FAIL | Bitwise BF16 activations against HF eager; with `--activation-tolerance`, relative RMS ≤ 0.02 |
| context | FAIL | FAIL | Bitwise BF16 logits against HF eager; every teacher argmax is exact |

`forward` and `context` pin bitwise agreement with HF eager on cuBLAS, which an engine without cuBLAS cannot
meet. Their replacement (tiers below) was approved by Tripp on 2026-11-07 and is not yet implemented.

## http-batch: no leak of engine buffers (`round3/http-batch-memory.json`, `evidence-v1/http-batch-summary.json`)

The gate reads device-wide free memory, which on Thor's unified memory also moves with host activity: in an A/B
the same hour, the default build's post-close reading was +529 MiB and the native build's +29 MiB; in the full
runs the native build read −139 MiB (a soak sample, round 1) and −48 MiB (post-close, round 2), failing the
128 MiB and 16 MiB allowances.

The native build now counts its own engine buffers (`Buffer`, `LFM_ALLOCATION_METRICS=1`). What it does **not**
count: the model weights (plain `cudaMalloc` in `safetensors.cpp`), the CUDA context and lazily loaded modules,
and allocator rounding. So it rules out a leak of engine buffers only. In a full soak on v5: 0 bytes live with
sessions empty, 166,705,408 bytes at all 656 soak samples (one distinct value), 0 bytes after close. On v6, with
the KV-group split (`evidence-v1/http-batch-summary.json`, from `tools/thor/g4_native_evidence.sh`): 0, then
128,956,672 bytes at all 623 samples, then 0 after close:
exactly 36 MiB less for the gate's two capacity-512 sessions, the predicted 18 MiB of score and probability
scratch per session. The failing runs were not instrumented.

## FP32 score scratch: a quarter of the memory, the same bits (round 4)

Prefill attention now runs one KV-head group (4 query heads) at a time; decode runs all heads in one call.
`lfm::gemm` picks its kernel from M, N, K and the B layout, never from the batch count, so the split changes no
arithmetic. Score scratch at capacity 4,096: 1 GiB → 256 MiB. Measured on v6 (`round4/run.log`):
- Perplexity NLL **bitwise identical** to attempt 3 at 256, 257, 4,096 and 32,768 (`round4/sha256.txt`).
- Native activation traces **byte-identical** to v5 on all four forward fixtures.

## Speed (`evidence-v1/decode-rate/`, v6 builds, hashes in its `binaries.sha256`)

Produced by `tools/thor/g4_native_evidence.sh` (log `evidence-v1/run.log`): the prompt is L tokens of WikiText-2, then
128 teacher-forced decode steps; one warm-up in process; median of three repetitions per build, alternating builds.
The raw results are committed; `decode-rate/summarize.py` reproduces `summary.json` from them.

| Prompt | Prefill, cuBLAS | Prefill, native | Decode, cuBLAS | Decode, native | Same argmax |
|---|---|---|---|---|---|
| 86 | 0.032 s | 0.031 s | 38.8 tok/s | **59.9 tok/s** | 129/129 |
| 1,024 | 0.068 s | 0.133 s | 38.9 | **58.8** | 128/129 |
| 4,096 | 0.551 s | 0.869 s | 37.4 | **57.4** | 126/129 |
| 32,000 | 6.65 s | 9.52 s | 30.5 | **49.0** | 125/129 |

- **Decode:** native is 1.51–1.61× faster. Absolute rates vary between runs (an earlier v5 run gave 30–39 and
  47–51 tok/s); the ratio held.
- **Prefill:** native takes 1.00–1.96× cuBLAS's time. The KV-group split costs prefill time: on v5 (all heads per
  call) native prefill was 1.42× cuBLAS's at 4,096 and 1.21× at 32,000 (that run, `build/decode-rate-v1` on Thor,
  is not committed; its numbers are quoted from its output); on v6 it is 1.58× and 1.43×. Memory against prefill
  time is a trade-off; stage B (`tcgen05`) targets prefill.
- **Same argmax:** native and cuBLAS teacher-forced argmax agree at 125–129 of 129 steps; whether the
  disagreements are near-ties has not been checked.

## Distance from the true computation: an FP32 anchor (`evidence-v1/fp32-rank.json`)

Anchor: the same model in FP32 with eager attention and TF32 disabled (`lfm3_reference.py --dtype float32`;
precision flags recorded in each fixture's `reference.json`), logits kept exact. Candidates: HF eager BF16, HF SDPA
BF16, native (its exact FP32 logits). All candidates were fed the anchor's prompt, all tokens agree, and all are
compared on the **same files**: SDPA does not trace attention internals, so 704/88/176/176 anchor files per fixture
are excluded for everyone.

| Fixture (logit steps) | Median activation rel. RMS: eager | SDPA | native | Mean logit KL (nats): eager | SDPA | native |
|---|---|---|---|---|---|---|
| baseline (16) | 0.0104 | **0.0089** | 0.0092 | 4.43e-4 | **2.11e-4** | 2.25e-4 |
| arithmetic (2) | 0.0104 | 0.0093 | **0.0092** | 1.00e-6 | **6.6e-7** | 1.43e-6 |
| window257 (4) | 0.0109 | **0.0102** | 0.0103 | 6.1e-7 | **5.3e-7** | 7.0e-7 |
| single (4) | 0.0139 | 0.0139 | **0.0136** | 3.02e-4 | **2.82e-4** | 3.80e-4 |

- **Activations:** eager is the farthest from FP32 in 3 of 4 fixtures and ties SDPA in the fourth. Native is within
  the reference stack's spread (at or below the worse of eager and SDPA) in all four and closest in two.
- **Logits:** native is about half of eager's distance on baseline and close to SDPA, but farther than both on the
  three small fixtures. A prediction made in conversation before the first FP32 run, that native's FP32 logits
  would make it closest in all four, **failed**; it was not recorded in the repo. These fixtures have 2 to 16
  logit steps and no confidence interval. The powered version of this question is tier 2 below.
- **Against eager instead of FP32** (`round3/eager-anchored-spread.json`, unequal file sets, superseded for
  ranking): SDPA's activations are 0.025–0.041 relative RMS from eager, above the `forward` gate's 0.02. Native
  is 0.029–0.052, and farther from eager than SDPA in 57–85% of the files both trace.
- `round4/fp32-rank.json` has the same numbers; `evidence-v1/fp32-rank.json` was recomputed with the current
  tool, which also checks each run's exact logit traces (present and consistent for the anchor and native).
- `round3/fp32-rank.json` is superseded: it compared candidates over different file sets, and it was produced by
  commands run by hand that are not in `round3/run.log`.

## The replacement for forward and context (approved 2026-11-07, not yet implemented)

| Tier | Gate |
|---|---|
| 1. Regression | Bitwise equality against the native engine's own pinned traces and logits on the existing fixtures |
| 2. Fidelity | Against the FP32 anchor: native's median activation distance per fixture ≤ the worse of eager and SDPA; teacher argmax and greedy exact against eager; for logits, per-token NLL against the FP32 model over the WikiText windows with a bootstrap CI, judged like §9 |
| 3. Quality | §9 perplexity, as it stands |
| Memory | `http-batch` checks the worker's exact engine-buffer bytes; the device-wide reading stays as information |

## Known consequences of the native build

- `tools/benchmark_lfm3.py` requires the native result to report `attention: eager-bf16`; the native build now
  reports `eager-fp32-scores` (with `gemm: lfm-gemm`, `logits: fp32`), so that benchmark, an eager-parity claim,
  rejects it. To be resolved when the default is switched.
- The sampling oracle's source changed (exact traces), so `text-sampling-oracle` differs from the v3 build in
  both configurations; its BF16-only replay is unchanged and tested.

## Paired statistics

All attempt-to-attempt window counts, means and CIs cited in `../g4-native-gemm-v{1,2,3}/` reproduce from
`tools/g4_paired_stats.py` over the NLL files hashed in those directories; output: `paired-stats.json`.
