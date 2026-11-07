# G4 stage A, attempt 1: the engine on lfm::gemm. The §9 bar FAILS at 257 and 32,768

The CMake option `LIQUID_LFM3_NATIVE_GEMM=ON` (default OFF) builds `liquid-lfm3`, `liquid-lfm3-worker`,
`liquid-lfm3-server` and `lfm3-decode-batch-probe` on `native/include/lfm/gemm.cuh`, with no cuBLAS. The
perplexity scorer builds the same way with `tools/thor/build_scorer.sh OUT native-gemm`.

**What changes under the option** (`native/src/lfm3_main.cu`, `native/src/lfm3_decode_batch.cuh`):
- `Runner::linear` calls `lfm::gemm::linear`.
- Attention QKᵀ and P·V run as single strided, grouped `lfm::gemm` problems: 16 query heads over 4 KV
  heads, with keys at `key_start`. There are no pointer arrays.
- Score and probability rows are padded to a multiple of 8 keys, with the padding zeroed. The scratch is
  sized for round8(capacity).
- P·V reads value rows only below `end` (`b_rows`). Decode P·V uses a split-K workspace.
- Probability traces are written compacted to the unpadded layout, so trace files keep their format.

## Results (Thor; run log `run.log`; hashes `sha256.txt`)

| Check | Result |
|---|---|
| Builds | 0 warnings in both configurations |
| Default build unchanged | `liquid-lfm3`, worker and server are identical to `cmake-server-v3` after `objcopy --strip-all`. The full files differ only in nvcc's temporary filenames in the symbol table. Every existing gate therefore still applies to the default build |
| Native build links no cuBLAS | `readelf -d` NEEDED for all three: `libcudart.so.13`, `libstdc++`, `libm`, `libgcc_s`, `libc` and `ld-linux`. Sizes are 2.28, 2.36 and 2.60 MB. The scorer links cudart statically and needs no CUDA library |
| Perplexity against §9 (`ppl/compare-native-gemm.json`) | **FAIL** (below) |

| Length | PPL, SDPA | PPL, native cuBLAS (ppl-v5) | PPL, native GEMM | GEMM − SDPA, 95% CI | Verdict |
|---|---|---|---|---|---|
| 256 | 26.5821 | 26.5812 | 26.5809 | [−1.95, +0.97]×10⁻⁴ | equal, PASS |
| 257 | 26.5045 | 26.5066 | 26.5087 | [+0.16, +2.98]×10⁻⁴ | **worse, FAIL** |
| 4,096 | 13.0596 | 13.0590 | 13.0590 | [−2.09, +1.21]×10⁻⁴ | equal, PASS |
| 32,768 | 12.5771 | 12.5776 | 12.5788 | [+0.26, +2.50]×10⁻⁴ | **worse, FAIL** |

## What the failure is, and what it is not

Paired against the native cuBLAS engine (`ppl-v5/nll-native-*`):

| Length | GEMM − cuBLAS, mean NLL | Windows with GEMM higher |
|---|---|---|
| 4,096 | 0: the NLL file is **bitwise identical**. Every GEMM in the model matched cuBLAS bit for bit at this length | — |
| 256 | −1.4×10⁻⁵ | 560 of 1,160 |
| 257 | +7.9×10⁻⁵. The CI includes zero | 583 of 1,155 |
| 32,768 | +9.9×10⁻⁵ | 7 of 9 |

**Evidence that it is not a defect:**
- The 4,096 result is bitwise identical to cuBLAS.
- At 256 and 257, per-token differences start at zero and grow with position, which is ordinary rounding
  divergence.
- At 257 the windows split close to 50/50.

**Where the differences come from:** they appear where cuBLAS and lfm::gemm round differently. That
includes attention spans that are not a multiple of 8 keys, such as 257 tokens and the 32K sliding tiles
of 4096 + 255 keys. There the native kernel pads K with zeros and accumulates in a different order.

**Why the bar still fails:** at 257, half of the GEMM − SDPA shift (+7.9×10⁻⁵ of +1.58×10⁻⁴) was already
present in the cuBLAS baseline. The native GEMM's noise-level change moves the CI's lower bound to
+1.6×10⁻⁵, just past ε = 10⁻⁵. At 32K there are only 9 windows.

The bar is §9's as decided, and this run fails it. It has not been reinterpreted.

## Corrections (2026-11-07, red-team)

- **"Evidence that it is not a defect" is weaker than stated.** The 50/50 window split has almost no power:
  a real shift of this size (about 0.1 of the per-window standard deviation) also splits close to 50/50.
  The paired statistics (`../g4-native-gates-v1/paired-stats.json`, from `tools/g4_paired_stats.py`) give
  native − cuBLAS at 257 = +7.9×10⁻⁵ with a 95% CI of [−4.5, +20.3]×10⁻⁵: consistent with noise, and
  equally consistent with a small systematic difference. No M = 257 shape was ever probed, so the source
  is not established.
- **The 32K verdict rests on 9 windows.** The percentile bootstrap used by the §9 bar is too narrow at n = 9;
  a Student-t interval for native − cuBLAS at 32K is [−0.6, +20.4]×10⁻⁵, which includes zero, while the
  bootstrap's excludes it. The bar is read as decided, with the bootstrap.
- `sha256.txt` hashes the scorer's sources; it omits `native/src/lfm3_decode_batch.cuh`, which this attempt also
  changed (it is not part of the scorer).
- **Provenance of the bitwise-identical 4,096 NLL file** (checked 2026-11-07, `scorer-provenance.txt`): the scorer
  `score-g4-v1` was built in `native-gemm` mode, has 173 `lfm::gemm` symbols and no cuBLAS symbol or NEEDED entry, and
  its sources hash to this attempt's commit. Its 256 NLL file differs from cuBLAS while the 4,096 file is identical, consistent
  with the probe's 100% identical outputs on the 4,096-row shapes.
- The default-build identity check covered `liquid-lfm3`, worker and server, not `lfm3-decode-batch-probe`. This
  attempt's default probe (74138f…) was never compared; round 4 of `../g4-native-gates-v1/` compared only the v6 probe.
