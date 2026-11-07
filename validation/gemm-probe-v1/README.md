# G4 stage 1: native BF16 GEMM against cuBLAS, every production shape

G4 removes cuBLAS from the engine: `libcublasLt` (630 MB) and `libcublas` (62 MB), against a 1.46 MB
server binary and a 0.7 MB `libcudart`. This stage builds the replacement, `native/include/lfm/gemm.cuh`,
and measures it in isolation. Integration into the engine and the perplexity bar (journal §9) are the
next stage.

## What `lfm::gemm` is

- FP32 accumulation, BF16 output rounded to nearest even.
- Deterministic: a fixed reduction order and no atomics.
- It does not reproduce cuBLAS's kernel choice or reduction order. Journal §9 judges the forward pass by
  perplexity within the 95% CI, so bitwise identity with cuBLAS is not required.

| Kernel | Used for | How |
|---|---|---|
| `thin` | M ≤ 8 with B stored N×K: decode projections and lm_head | 4 columns per warp, streaming 16-byte `__ldcs` loads, butterfly reduction. Bound by weight bandwidth |
| `thin_k128` | M ≤ 8 with K = 128: decode QKᵀ | 8 lanes per key row |
| `split_kn` | M ≤ 8 with B stored K×N: decode P·V | Splits of 256 keys into an FP32 workspace, then summed in a fixed order |
| `wide` | Everything larger: prefill | `mma.sync` m16n8k16 tensor cores, a 3-stage `cp.async` pipeline through XOR-swizzled shared memory, `ldmatrix`/`ldmatrix.trans`. Tiles are 128, 64 or 32 × 128 × 64, with 2×2 warps |

## Evidence (Thor, CUDA 13.0.88, pinned compiler image; hashes in `sha256.txt`)

**Probe** (`native/tests/lfm3_gemm_probe.cu` → `probe.jsonl`):
- Every production linear shape and every attention product is run three ways: natively, through cuBLAS
  called exactly as production calls it, and against an FP64 reference over sampled outputs.
- Decode-sized cases (M ≤ 8) are timed with cold weights, as in decode.
- Builds have 0 warnings and 0 register spills.

| | Result |
|---|---|
| Accuracy against FP64 | Native max and mean ulp equal cuBLAS in 61 of 62 cases. The exception is w2 at M = 34: max 0.504 against 0.499 ulp, with the same mean. 99.6–100% of outputs are bit-identical to cuBLAS |
| Decode P·V at 32,768 keys | Native max error is **0.50 ulp**; cuBLAS reaches **34.96 ulp** |
| Decode projections, M = 1–2 | **0.66–0.93×** cuBLAS's time. lm_head takes 3.13 ms against 3.80 ms |
| Decode attention | Faster at 300 keys (0.61× and 0.87×). At 4,096 keys QKᵀ takes 1.68× and P·V 1.35×; at 32,768 keys 1.91× and 1.58× |
| Prefill projections | 2.2–3.2× cuBLAS's time at 4,096 and 32,768 rows. For example, w1/w3 at 32,768 rows takes 28.7 ms against 9.7 ms |
| Prefill attention | 0.98–1.04× at 4,096 and 32,768 keys |

**Tile sweep** (`native/tests/lfm3_gemm_sweep.cu` → `sweep.jsonl`):
- 11 tile configurations on 4 prefill shapes. All 44 runs are **bitwise identical** to the reference
  configuration, as the fixed k16 accumulation order requires.
- The best tile is 128×128×64 with 2×2 warps and 3 stages: 38.8–41.6 TFLOPS on the K = 2048 shapes and
  33.4 TFLOPS on w2 (K = 7680).

**Why prefill is slower:**
- A register-only microbenchmark of the `mma.sync` m16n8k16 BF16 ceiling on Thor's 20 SMs
  (`native/tests/lfm3_mma_peak.cu` → `mma-peak.txt`) measured **56.7 TFLOPS**. The sweep's best tile
  reaches about 70% of that.
- cuBLAS runs w1/w3 at 32,768 rows in 9.74 ms, which is **105.8 TFLOPS**. That is above the `mma.sync`
  ceiling, so cuBLAS must be using the Blackwell `tcgen05` tensor-memory path.
- Matching cuBLAS prefill therefore needs a `tcgen05` kernel; tuning `mma.sync` cannot get there.

## Plan

- **Stage A:** integrate `lfm::gemm` behind a build option, prove the §9 perplexity bar at 256, 257,
  4,096 and 32,768 tokens, and drop the cuBLAS link. Decode projections get faster and long-context
  decode attention slower; the end-to-end decode rate is to be measured. Prefill is about 3× slower.
- **Stage B:** a `tcgen05` prefill kernel to close the prefill gap.
