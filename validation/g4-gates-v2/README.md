# G4 stage A done: the default engine runs without cuBLAS and passes every gate

The gates that replace bitwise-eager for the native-GEMM engine were declared in `PLAN.md` (`c4e001d`, anchor
amended in `ba2c4a7`), built and reviewed before running (`dccf25a`, `9a26d0c`), and run on Thor with committed
scripts (`tools/thor/g4_tier2b.sh`, `tools/thor/g4_final_round.sh`).

## The default build (`final/run.log`, hashes `final/default-binaries.sha256`)

| Check | Result |
|---|---|
| `LIQUID_LFM3_NATIVE_GEMM` default | ON: `liquid-lfm3`, worker, server and decode-batch probe are identical to `cmake-g4-native-v6` after stripping |
| Linkage | `libcudart.so.13` plus the C and C++ runtime; no cuBLAS. Server 2.73 MB, worker 2.56 MB, `liquid-lfm3` 2.48 MB |
| cuBLAS option (`-DLIQUID_LFM3_NATIVE_GEMM=OFF`) | identical to `cmake-server-v3` after stripping, kept for A/B |
| Builds | 0 warnings in both configurations |

A first attempt of this round (`g4-final-v1` on Thor) stopped before any gate: Thor's tree had not been synced
after the default flip, so its default build still linked cuBLAS, and the gate runner refused to pin it.

## The tiers

| Tier | Gate | Result |
|---|---|---|
| 3. Quality | §9 perplexity | The default scorer's NLL is **bitwise identical** to attempt 3 at 256, 257, 4,096 and 32,768 (`final/run.log`), so `../g4-native-gemm-v3/`'s PASS (better at 256, equal elsewhere) is this build's |
| 2b. Fidelity, powered | per-token \|NLL − NLL_FP32\| against SDPA's, §9 windows | **PASS**, native 13–14% closer to FP32 than SDPA at every length (`tier2b/`) |
| 2a. Fidelity, fixtures | median activation distance from FP32 ≤ worse of eager and SDPA; tokens exact against eager | **PASS** on all four fixtures (`final/*/forward-summary.json`): native 0.0092 / 0.0092 / 0.0103 / 0.0136 against eager 0.0104 / 0.0104 / 0.0109 / 0.0139 and SDPA 0.0089 / 0.0093 / 0.0102 / 0.0139 |
| 1. Regression | bitwise against the engine's own pinned traces and ids | Pinned by a strict run in which every gate passed (`final/pin/`); the next run matched the pins bit for bit (`final/golden/`). Pins: `pins/forward.json` (4 fixtures, every trace file), `pins/context.json` (9 cases, every logit trace and the free-running ids), each with the binary's hash |
| Memory | exact engine-buffer bytes | **PASS** in both runs: 0 with sessions empty, 128,956,672 bytes at every soak sample (666 and 654), 0 after close (`final/*/http-batch-memory.json`) |

Every other gate passed in both runs: server, cancellation, http-cancellation, native-server (the single binary),
sampling, worker, grouped-worker, 32k, http-batch.

## Speed and memory against the cuBLAS engine (`../g4-native-gates-v1/`)

Decode 1.51–1.61× faster; prefill 1.00–1.96× slower (the stage B `tcgen05` kernel targets prefill); FP32 score
scratch per session 256 MiB at capacity 4,096 (the cuBLAS engine's BF16 scratch is 512 MiB).

## What remains open

- Whether native and cuBLAS differ systematically at 257 tokens upstream of the logits (+7.8×10⁻⁵, CI including
  zero; `../g4-native-gemm-v3/`). Tier 2b says that, wherever native lands, it is closer to FP32 than SDPA.
- The trade-off of the KV-group prefill split (4× less score scratch, prefill 1.43–1.58× instead of 1.21–1.42×
  cuBLAS's time at 32K/4K) awaits Tripp's call.
- Stage B: a `tcgen05` prefill GEMM.
