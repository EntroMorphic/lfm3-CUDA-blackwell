# G4 prefill: heads per call by a 256 MiB budget — verified; and a speed claim withdrawn

`tools/thor/g4_prefill_budget_round.sh` (log `run.log`) on the default build with the budget rule (`90ae1d4`): the most
query heads per prefill attention call whose FP32 score buffer fits 256 MiB. Measurement behind the rule:
`../g4-prefill-heads-v1/`. What the rule chooses (the formula enumerated at every capacity 1–32,768; corrected
2026-11-07, this README first said "16 at ≤ 2,048, 4 at 4,096" and omitted the 8-head band):

| Capacity | Heads per call | Largest FP32 score buffer |
|---|---|---|
| 1–2,048 | 16 | 256 MiB |
| 2,049–2,896 | 8 | 255.95 MiB |
| 2,897–4,096 | 4 | 256 MiB |
| 4,097–32,768 | 16 | 256 MiB |

The BF16 probability buffer has the same element count, so attention scratch is at most 384 MiB per session (the
cuBLAS engine: two 512 MiB BF16 buffers at 4,096). Only capacity 512 (the http-batch gate) and the four speed cases
below were run; the 8-head band is bit-invariant by the same construction but was not run.

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

- **The budget rule did what it was for, in native time:** native 32,000-token prefill is 7.61 s here. The
  same-session comparison is `../g4-prefill-heads-v1/`: 7.62 s with 16 heads against 9.10 s with 4. (The 9.37–9.52 s
  of `../g4-native-gates-v1/` is from another session and is not a valid comparison; see the next point.)
- **Not met: v5's ratio against cuBLAS.** The plan's criterion was 32K prefill back to v5's 1.21× cuBLAS's time.
  Here it is 1.34×: native's time equals v5's (7.61 s), but the cuBLAS engine prefilled in 5.69 s against about
  6.3 s in v5's session. Native prefill is not slower than in v5, but what the ratio "should" be is open: it moved
  between sessions, like decode.
- **The decode comparison does not reproduce.** The cuBLAS engine here is the same binary as in `../g4-native-gates-v1/`
  (identical to `cmake-server-v3` after stripping), and decodes at 55–58 tok/s here against 30–39 tok/s there, at
  4,096 and below. The native engine moved little (59–61 against 58–60). Both sessions alternated the builds, so the
  difference is in the conditions, which affected the two engines differently. The earlier sessions' power mode and
  clocks were not recorded (Thor rebooted in between; the kernel log does not show the mode). **The claim that native
  decodes 1.51–1.61× faster than cuBLAS is withdrawn**; in this session it is 1.06–1.25×. Speed runs now record the
  power mode and GPU clock limits (`tools/thor/g4_native_evidence.sh`).

## Red-team remediation (2026-11-07, `remediation/run.log`, `tools/thor/g4_prefill_parse_check.sh`)

- **The budget table above was incomplete** (the 8-head band at 2,049–2,896); corrected. The 256 MiB bound was
  checked by enumerating the rule over every capacity 1–32,768, not only the capacities run.
- **`LFM3_PREFILL_HEADS` was parsed with `atoi`:** `abc` silently meant the default and `16x` meant 16. It is now
  parsed strictly; `abc`, `16x`, the empty string, `0`, `-4` and `99999999999` are refused as not a positive integer,
  and `3` and `12` as not a multiple of the KV group dividing the head count (all eight checked on Thor).
- **The scripts' cuBLAS-link checks used `readelf | grep -q` under `pipefail`**, the SIGPIPE pattern that gave a false
  negative before, and treated a failed `readelf` as clean. Fixed in `g4_prefill_budget_round.sh` and
  `g4_prefill_heads.sh`, and checked against stub `readelf`s (links, clean, readelf fails).
- **Claims:** the 9.37–9.52 s cross-session comparison replaced by the same-session one; v5's 1.21× ratio reported as
  not met; the P·V-grid explanation marked as not profiled; scratch figures labelled as the FP32 score buffer.
- **Verification of the rebuilt engine (strict parsing, comments):** 0 warnings, no binary links cuBLAS; perplexity
  NLL bitwise identical to attempt 3 at 256, 257, 4,096 and 32,768; logits of 129 traced steps identical for the
  default and 4, 8 and 16 heads per call at capacity 2,560 (the 8-head band, not run before) and 32,768 (that hash
  equals the 4,096-prompt hash in `../g4-prefill-heads-v1/run.log`). The tier-1 golden gates were not rerun: the
  default path's logic is unchanged and its NLL and traces are bit-identical.
