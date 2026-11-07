# G4 stage A: the runtime gates on the native-GEMM build, and what the failing ones measure

Native-GEMM builds `cmake-g4-native-v3` (round 1), `-v4` (round 2), `-v5` (round 3). Each default build is
identical to `cmake-server-v3` after `objcopy --strip-all` (round logs). Thor, `lfm3-profile-v1`.

## The gates

| Gate | Round 1 (v3) | After the fixes (v4/v5) | What it pins |
|---|---|---|---|
| server, cancellation, http-cancellation | PASS | PASS | The pinned 128-token cuBLAS greedy control replays exactly on native |
| native-server (G7 single binary) | PASS | PASS | Serving, cancellation, batching, soak, 22-request differential, shutdown |
| grouped-worker | PASS | PASS | Serial vs. grouped decode: every logit bit (73 steps) |
| 32k | PASS | PASS | Argmax against SDPA, with tolerance |
| worker | FAIL | **PASS** | Worker logits vs. single-shot trace, exact. Failed because the trace held rounded logits; fixed by the exact `.logits.f32` sidecar |
| sampling | FAIL | **PASS** | Oracle replays the sampler from the trace. Same cause, same fix |
| http-batch | FAIL | PASS alone, FAIL in the full run | Device-wide free memory. Not the engine (below) |
| forward | FAIL | FAIL | Bitwise BF16 activations against HF eager; with `--activation-tolerance`, relative RMS ≤ 0.02 |
| context | FAIL | FAIL (bitwise); every teacher argmax exact | Bitwise BF16 logits against HF eager |

## http-batch: not a leak (`round3/http-batch-memory.json`)

The gate reads device-wide free memory, which on Thor's unified memory moves with host activity. The worker
now counts its own engine buffers (`LFM_ALLOCATION_METRICS=1`, native build only). In a full soak:
**0 bytes live with sessions empty, 166,705,408 bytes at all 656 soak samples (one distinct value), 0 bytes
after close.** Device-wide free memory moved by 300 MiB in the same run, and by +529 MiB on the default
build in an A/B the same hour. The failing readings (−139 MiB in a soak sample, −48 MiB after close) are
host-side noise.

## forward and context: distance from eager, against the reference stack's own spread

`context` with no tolerance (`round3/context-measure.json`): all 9 cases pass teacher argmax exactly
(windows 255, 256, 257, 1024, 4096 prefill and final); max logit difference 0.125–0.28, 0.65 at `conv7`.

HF SDPA traces on the four `forward` fixtures (`round3/eager-anchored-spread.json`): SDPA itself fails the
`forward` gate's 0.02 relative-RMS default (0.031–0.041 from eager). Native is at 0.029–0.052, about the
same: median per-file ratio to SDPA 1.02–1.37.

## Which is closer to the true computation? (`round3/fp32-rank.json`)

Anchor: the same model in FP32 (`lfm3_reference.py --dtype float32`, eager attention), logits kept exact.
Candidates: HF eager BF16, HF SDPA BF16, native (its exact FP32 logits). All tokens agree in all fixtures.

| Fixture (logit steps) | Median activation rel. RMS: eager | SDPA | native | Mean logit KL (nats): eager | SDPA | native |
|---|---|---|---|---|---|---|
| baseline (16) | 0.0104 | 0.0089 | 0.0091 | 4.43e-4 | 2.11e-4 | 2.25e-4 |
| arithmetic (2) | 0.0104 | 0.0093 | **0.0090** | 1.00e-6 | 6.6e-7 | 1.43e-6 |
| window257 (4) | 0.0112 | 0.0102 | 0.0105 | 6.1e-7 | 5.3e-7 | 7.0e-7 |
| single (4) | 0.0138 | 0.0139 | **0.0135** | 3.02e-4 | 2.82e-4 | 3.80e-4 |

- **Eager is the farthest from FP32 in median activations in every fixture.** The bitwise gates pin the
  engine to the least accurate of the three references.
- **Activations:** native is inside the reference stack's spread, best in 2 of 4.
- **Logits:** native is half of eager's distance on baseline and about equal to SDPA, but marginally
  farther than both on the three small fixtures. A prediction recorded before the run, that native's FP32
  logits would make it closest in all four, **failed**. These fixtures have 2–16 logit steps and no
  confidence interval; §9's 295,000-token perplexity is the powered measure of the same question.
- A first pass at this comparison (`spread-fp32-v1` on Thor) rounded the anchor's and native's logits to
  BF16 on save, which erases exactly the difference being measured; it is superseded by this one.
