# G4 stage A done: the default engine runs without cuBLAS and passes every gate

The gates that replace bitwise-eager for the native-GEMM engine were declared in `PLAN.md` (`c4e001d`), built and
reviewed (`dccf25a`, `9a26d0c`), and run on Thor with committed scripts (`tools/thor/g4_tier2b.sh`,
`tools/thor/g4_final_round.sh`). One exception to "reviewed before running": the 2b anchor as declared (FP32 SDPA) was
started before `dccf25a`, with the `--dtype float32` change that `dccf25a` then committed, and ran 256, 257 and 4,096
before Thor hung at 32,768; the anchor was then amended
(`ba2c4a7`, evidence in `amendment-evidence/`, committed after the run) before 2b ran. Against the declared anchor's
three lengths the 2b result is the same (`tier2b/README.md`). The commit message of `0a2f3a8` says it adds the
final-round script; that script was committed in `9a26d0c`, also before it ran.

## The default build (`final/run.log`, hashes `final/default-binaries.sha256`)

| Check | Result |
|---|---|
| `LIQUID_LFM3_NATIVE_GEMM` default | ON: `liquid-lfm3`, worker, server and decode-batch probe are identical to `cmake-g4-native-v6` after stripping |
| Linkage | `libcudart.so.13` plus the C and C++ runtime; no cuBLAS. Server 2.73 MB, worker 2.56 MB, `liquid-lfm3` 2.48 MB |
| cuBLAS option (`-DLIQUID_LFM3_NATIVE_GEMM=OFF`) | identical to `cmake-server-v3` after stripping, kept for A/B |
| Builds | 0 warnings in both configurations |

A first attempt of this round (`final/run-attempt1-unsynced.log`) stopped before any gate: Thor's tree had not been
synced after the default flip, so its default build still linked cuBLAS, and the gate runner refused to pin it. Nothing
records which source tree ran in the attempts before this one; from now on `g4_final_round.sh` and `g4_tier2b.sh` log
the commit and the hashes of the sources they use.

## The tiers

| Tier | Gate | Result |
|---|---|---|
| 3. Quality | §9 perplexity | The default scorer's NLL is **bitwise identical** to attempt 3 at 256, 257, 4,096 and 32,768 (`final/run.log`), so `../g4-native-gemm-v3/`'s PASS (better at 256, equal elsewhere) is this build's. As that README says: the PASS comes from FP32 logits, a change at the output that SDPA does not share (like for like, attempt 2 fails at 257), and it was the third attempt against the same bar |
| 2b. Fidelity, powered | mean \|NLL − NLL_FP32\| at the target token, against SDPA's, §9 windows | **PASS**: native's deviation is 12.8–14.4% smaller than SDPA's at every length (`tier2b/`). The difference comes from the FP32 logits: with BF16 logits (attempt 2) native is not distinguishable from SDPA (resolution about ±10⁻⁴) and closer than eager |
| 2a. Fidelity, fixtures | median activation distance from FP32 ≤ worse of eager and SDPA; tokens exact against eager | **PASS** on all four fixtures (`final/*/forward-summary.json`): native 0.0092 / 0.0092 / 0.0103 / 0.0136 against eager 0.0104 / 0.0104 / 0.0109 / 0.0139 and SDPA 0.0089 / 0.0093 / 0.0102 / 0.0139 |
| 1. Regression | bitwise against the engine's own pinned traces and ids | Pinned by a strict run (`final/pin/mode.json`: keep_going false) in which every gate passed; the next run matched the pins bit for bit (`final/golden/*-summary.json`: "matches"). Pins: `pins/forward.json` (4 fixtures, every trace file, ids), `pins/context.json` (9 cases, every logit trace; free-running ids for the 3 cases that run greedy), each with the binary's hash |
| Memory | exact engine-buffer bytes | **PASS** in both runs: 0 with sessions empty, one value (128,956,672 bytes) across every soak sample (666 and 654), 0 after close (`final/*/http-batch-summary.json`, the full reports; `http-batch-memory.json` is extracted from them by `final/http-batch-memory.py`) |

Every other gate passed in both runs: server, cancellation, http-cancellation, native-server (the single binary),
sampling, worker, grouped-worker, 32k, http-batch. The evidence for these is each run's `status.json`, which the
runner writes as "done" only if every stage passed; their own reports are not committed.

## Speed and memory against the cuBLAS engine (`../g4-native-gates-v1/`)

Decode 1.51–1.61× faster; prefill takes 1.00–1.96× cuBLAS's time (the stage B `tcgen05` kernel targets prefill); FP32 score
scratch per session 256 MiB at capacity 4,096 (the cuBLAS engine's BF16 scratch is 512 MiB).

## What remains open

- Whether native and cuBLAS differ systematically at 257 tokens upstream of the logits (+7.8×10⁻⁵, CI including
  zero; `../g4-native-gemm-v3/`). Tier 2b cannot answer it: its margin comes from the FP32 logits.
- Full-distribution closeness to FP32: on the four small fixtures native's logit KL was farther than SDPA's on three
  (`../g4-native-gates-v1/`); no powered measurement of KL exists.
- The trade-off of the KV-group prefill split (4× less score scratch, prefill 1.43–1.58× instead of 1.21–1.42×
  cuBLAS's time at 32K/4K) awaits Tripp's call.
- Stage B: a `tcgen05` prefill GEMM.
