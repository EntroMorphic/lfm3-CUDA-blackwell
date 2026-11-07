# G4 prefill: heads per call by a 256 MiB budget — verified; and a speed claim withdrawn

`tools/thor/g4_prefill_budget_round.sh` (log `run.log`) on the default build with the budget rule (`90ae1d4`): the most
query heads per prefill attention call whose FP32 score scratch fits 256 MiB (16 at capacity 32,768 and at ≤ 2,048;
4 at 4,096). Measurement behind the rule: `../g4-prefill-heads-v1/`.

## Verification (no quality evidence needs redoing: the setting is bit-invariant)

| Check | Result |
|---|---|
| Build | 0 warnings; no binary links cuBLAS (`default-binaries.sha256`) |
| Perplexity NLL | **bitwise identical** to attempt 3 at 256, 257, 4,096 and 32,768 |
| Every gate against the unchanged pins (`TIERS=golden`) | **PASS** (`golden/status.json`); forward and context report "matches" against `../g4-gates-v2/pins/` |
| Exact memory (`http-batch-summary.json`) | 0 bytes with sessions empty, 166,705,408 at all 667 soak samples, 0 after close. The gate's capacity-512 sessions now use 16 heads (16 MiB of score scratch), the same footprint as v5 |
| FP32-anchored spread (`fp32-rank.json`) | identical to `../g4-gates-v2/` |

## Speed in this session (`decode-rate/`; power state `power-state-after.txt`: 120 W mode)

| Prompt (capacity) | Prefill, cuBLAS | Prefill, native | Decode, cuBLAS | Decode, native | Same argmax |
|---|---|---|---|---|---|
| 86 (214) | 0.022 s | 0.025 s (1.12×) | 57.9 tok/s | 61.1 tok/s (1.06×) | 129/129 |
| 1,024 (1,152) | 0.062 s | 0.120 s (1.94×) | 57.1 | 60.7 (1.06×) | 128/129 |
| 4,096 (32,768) | 0.513 s | 0.737 s (1.44×) | 55.0 | 58.9 (1.07×) | 126/129 |
| 32,000 (32,768) | 5.69 s | 7.61 s (1.34×) | 40.7 | 50.8 (1.25×) | 125/129 |

- **The budget rule did what it was for:** native 32,000-token prefill is 7.61 s, against 9.37–9.52 s with 4 heads per
  call (`../g4-native-gates-v1/`) and 7.62 s in the heads measurement.
- **The decode comparison does not reproduce.** The cuBLAS engine here is the same binary as in `../g4-native-gates-v1/`
  (identical to `cmake-server-v3` after stripping), and decodes at 55–58 tok/s here against 30–39 tok/s there, at
  4,096 and below. The native engine moved little (59–61 against 58–60). Both sessions alternated the builds, so the
  difference is in the conditions, which affected the two engines differently. The earlier sessions' power mode and
  clocks were not recorded (Thor rebooted in between; the kernel log does not show the mode). **The claim that native
  decodes 1.51–1.61× faster than cuBLAS is withdrawn**; in this session it is 1.06–1.25×. Speed runs now record the
  power mode and GPU clock limits (`tools/thor/g4_native_evidence.sh`).
