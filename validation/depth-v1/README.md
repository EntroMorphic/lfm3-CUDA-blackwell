# #2 step 3, part 1: native depth decoder (A1, B1): PASS

The native depth decoder is `native/include/lfm/depth.cuh`, over the shared FP32 operators of
`native/include/lfm/fp32_ops.cuh`. The tool `liquid-depth` (`native/src/depth_main.cu`) runs it on the reference's
own recorded passes, with codes forced and no RNG. Quality bar: `journal/lfm3-tts-quality-bar.md`, A1 and B1.

Runs on Thor (120 W):
- **Anchors:** `tools/lfm3_depth_anchor.py`, logs in `anchor/`.
- **Native and check:** `tools/thor/depth_round.sh` → `tools/lfm3_depth_check.py`, results in `native/`, log in
  `run.log`.
- **Data:** 6 cases, 1,021 frames, 3 passes each. Every frame's 12 codebooks are sampled once, at the pass that
  revealed them, giving 12,252 positions. The logits stay on Thor (`thor-only-files.sha256.txt`).

## Design

- **Precision:** FP32 activations over the stored BF16 weights; rotary in FP64. This is the codec's recipe.
- **`depth_linear` runs once per frame.** HF recomputes it every pass from the same input.
- **Trunk:** 3 bidirectional layers, using the shared GEMM, RMSNorm, SwiGLU and rotary, plus a warp-per-(slot, head)
  attention over the 12 slots.
- **Heads:** every codebook's norm in one launch, then one GEMV per requested codebook against its tied embedding.
  `pass()` computes only the codebooks asked for; the A1 tool asks for all 12.
- **Shared GEMM** (`lfm::fp32::linear`, new):
  - One warp per 4 output columns, over groups of 16 rows.
  - Each output's reduction order depends only on K: each lane sums its k ascending, then a fixed shuffle tree
    adds the lanes. So results do not depend on how rows are batched.
  - It replaces the codec's 64×64-tiled kernel (see below).

## The anchors (`anchor/anchor.json`)

| Variant | How it runs |
|---|---|
| HF BF16 SDPA | one pass per call, as in generation |
| HF BF16 eager | one pass per call |
| HF FP32 | GPU, TF32 off, all passes batched |
| HF FP64 | CPU, batched; both RMSNorm conventions, the rotary tables and their application widened to FP64 |

Self-checks, all six cases:
- SDPA reproduces the trace's logits **bit for bit**, for all 1,021 frames.
- Eager differs from SDPA.
- The widening leaves BF16 unchanged.

## A1: logit fidelity

Relative L2 distance of each sampled position's 2,017 logits (`native/a1b1.json`):

| Variant | Median vs FP32 | p99 vs FP32 | Median vs FP64 | Max vs FP64 |
|---|---|---|---|---|
| **native** | 7.4e-7 | 1.6e-6 | **1.6e-7** | 6.7e-7 |
| HF FP32 | — | — | 7.2e-7 | 3.4e-6 |
| HF BF16 SDPA | 4.4e-3 | 1.0e-2 | 4.4e-3 | 2.2e-2 |
| HF BF16 eager | 4.9e-3 | 1.2e-2 | 4.9e-3 | 2.6e-2 |

**PASS:** native 7.4e-7 against a bar of 4.9e-3 (the worse of SDPA and eager).
- Native is closer to FP64 than HF's own FP32 is. Its median distance from FP32 is therefore mostly FP32's error.
- It is about 28,000× closer to FP64 than HF's shipped BF16.
- It holds in every case: native's median is 5.5–7.9e-7 per case, SDPA's 4.3–4.8e-3.

## B1: distribution fidelity

Measured at the sampling distribution: the repetition penalty 1.2 over the codebook's codes in the previous 64
accepted frames (EOS frames excluded, as HF), then temperature 0.4, exactly as HF samples. These are the same
12,252 positions. (The first version took the previous 64 raw frames. Every trace's only EOS frame is its last, so
the two agree bit for bit here; `native/a1b1.json` is the rerun, `rerun-step3-redteam.log` its log.)

| Variant | Mean KL(p_FP64 ‖ p) |
|---|---|
| **native** | **1.5e-12** nats |
| HF FP32 | 9.0e-11 |
| HF BF16 SDPA | 1.9e-3 |
| HF BF16 eager | 2.1e-3 |

Paired per position, with a 95% bootstrap CI and ε = 1e-5 as in §9:

| Pair | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| native − SDPA | −1.94e-3 | [−1.97e-3, −1.91e-3] | better |
| native − eager | −2.08e-3 | [−2.11e-3, −2.04e-3] | better |

**PASS.** The verdict holds in every case: native 1.3–1.9e-12 per case, SDPA 1.7–2.5e-3.

### B1 as written in the bar is biased, so the verdict uses KL instead

The bar as approved scores the NLL of the reference's own sampled codes, native minus SDPA. But HF SDPA sampled
those codes. For any model m, the expected NLL_m − NLL_SDPA is therefore KL(p_SDPA ‖ p_m) ≥ 0: SDPA is favoured by
construction. The run shows it:

| NLL minus SDPA's | Mean | 95% CI | Verdict |
|---|---|---|---|
| native | +1.20e-3 | [9.1e-5, 2.3e-3] | "worse" |
| **FP64 itself** | +1.20e-3 | [9.1e-5, 2.3e-3] | "worse" |
| FP32 | +1.20e-3 | [9.1e-5, 2.3e-3] | "worse" |
| eager | +2.30e-3 | [8.3e-4, 3.8e-3] | "worse" |

The true model fails the as-written test, so that test cannot judge fidelity. The KL from FP64 at the same
positions is exact (summed over all 2,017 codes, no sampling noise), and it is unbiased. It is the expected NLL
penalty a model pays against the true distribution. **This replaces the bar's B1 measure, approved by
Tripp 2026-11-07** (`journal/lfm3-tts-quality-bar.md`).

## Regression tripwire

A1 and B1 compare against HF BF16, so a regression to BF16-level error (cos/sin, activations, eps) would still pass
them. `tools/lfm3_depth_check.py` therefore also fails unless native stays at the level it achieved:

| | Limit | Native |
|---|---|---|
| Median relative L2 from FP64 | 1e-6 | 1.6e-7 |
| Max relative L2 from FP64 | 1e-5 | 6.7e-7 |
| Mean KL from FP64 | 1e-9 nats | 1.5e-12 |

## Sanitizers (step-3 red-team)

`compute-sanitizer` memcheck, racecheck, initcheck and synccheck on `liquid-depth` (6 frames of tts-fox-female):
0 errors, 0 hazards, and output bit-identical to the plain run. The codec (40 frames of tts-fox-female): memcheck,
initcheck and synccheck on the whole decode with the reset-and-streaming self-test, and on 1-frame and 7-frame
chunks; racecheck on the whole decode with the self-test. All 0 errors, every output bit-identical to the plain
whole decode. Summary lines and every output's SHA-256: `sanitizers.txt`.

## Cost (`native/runs.jsonl`), unoptimized

- **Time:** 7.1–10.9 ms per frame for 3 passes with all 12 heads scored, against a frame budget of 80 ms.
  Generation needs only the 4 sampled heads per pass. Not profiled: about 65 launches per pass and a synchronous
  state copy.
- **Memory:** 0.67 MB of buffers. The weights are the main checkpoint's, shared with the text engine in the single
  binary.

## Open

- After that, end to end: native LM hidden states instead of HF's, so EOS placement under a native LM (tts-trace-v1
  Q5) can be studied.
- The depth decoder in the single binary.
