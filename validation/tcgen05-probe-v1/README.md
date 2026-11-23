# tcgen05 probe: can a native BF16 GEMM on Thor's tensor memory reach well past the mma.sync ceiling?

## The question
`lfm::gemm`'s `wide` path (`mma.sync` m16n8k16, `cp.async`, `ldmatrix`) is capped by the `mma.sync` ceiling
measured on Thor's 20 SMs, **56.7 TFLOPS** (`validation/gemm-probe-v1`, `mma-peak.txt`). It reaches 33-42 TFLOPS on
the language model's prefill shapes and ~43 TFLOPS on the vision encoder's split GEMMs (`validation/images-speed-v1`,
nsys). cuBLAS runs w1/w3 at 32,768 rows at **105.8 TFLOPS**, so it uses Blackwell's `tcgen05` path. This probe asks
whether a native kernel (no cuBLAS, no CUTLASS) built on `tcgen05.mma` with TMEM accumulators and TMA loads gets
there, and stays exact. It is a probe: nothing in the engine changes in this round. A YES feeds later rounds (the
vision and audio encoders, the language model's prefill).

## Feasibility, established before these rules (2026-11-16, `build/tcgen05-feas/`, not a result)
- The pinned build image (nvcc 13.0.88) compiles `tcgen05.alloc/dealloc/relinquish_alloc_permit`, `tcgen05.ld/st`,
  `tcgen05.mma.cta_group::1.kind::f16`, `tcgen05.mma.cta_group::2.kind::f16` (in a `__cluster_dims__(2,1,1)` kernel),
  `tcgen05.commit ... mbarrier::arrive::one`, `cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes` and
  `mbarrier.try_wait.parity` to SASS (`UTCHMMA`, `UTCHMMA.2CTA`, `LDTM`, `UTMALDG.2D`, `UTCATOMSWS`, `SYNCS.*`) with
  `-gencode arch=compute_110a,code=sm_110a`, and the same with `compute_110f,code=sm_110f`.
- `-arch=sm_110` (the engine's flag) rejects every `tcgen05` instruction; so does the shorthand `-arch=sm_110a`, because
  it also emits `compute_110` PTX. The probe's translation unit must use the explicit `-gencode` above.
- One kernel mixing `cta_group::1` and `cta_group::2` is refused by ptxas: 1-CTA and 2-CTA are separate kernels.
- On Thor (driver R38.2), a kernel allocating 32 TMEM columns, writing them with `tcgen05.st`, reading them back with
  `tcgen05.ld` and deallocating ran with no error and 128 of 128 lanes correct.
- cuBLAS's choice, for orientation (torch 2.10 / CUDA 13.0 in `lfm3-profile-v1`): `nvjet_sm110_tst_*_2cta_*` kernels on
  every shape tried, tiles 128x192 / 128x256 / 128x128 x k64, 6-8 stages, 2-CTA throughout.
- Not available: CUTLASS/CuTe headers (none on Thor or in the image). The descriptors are written from the PTX ISA's
  tables and proven by the unit tests below.
- Thor: 20 SMs, 227 KB dynamic shared memory per block (opt-in), 32 MB L2, 512 TMEM columns x 128 lanes per SM.

## Design
**Contract** (a subset of `lfm::gemm::Problem`): `C (M x N) = A (M x K, K contiguous) · Bᵀ` with B stored N x K
(projections: `X·Wᵀ`, and the vision encoder's split `[hi | lo] · [W | W]ᵀ`), batch 1; FP32 output (`c_f32`, the
vision encoder) or BF16 rounded to nearest even (`c`, the language model's prefill). M any value >= 1; N a multiple
of 256 and K a multiple of 64 (every real shape below satisfies both); 16-byte aligned operands. Anything else is
refused, not run. P·V (B stored K x N) and M <= 8 are out of scope.

**Kernel** (`native/probes/tcgen05_gemm_probe.cu`, its own `-gencode arch=compute_110a,code=sm_110a`):
- TMA (`cp.async.bulk.tensor.2d`, `CUtensorMap` with `SWIZZLE_128B`, rows of 64 BF16 = 128 bytes) loads A and B tiles
  into a ring of shared-memory stages; rows past M are zero-filled by TMA, not read.
- One thread issues `tcgen05.mma.kind::f16` (BF16 in, FP32 accumulate in TMEM), K = 16 per instruction, four per
  64-wide stage; `tcgen05.commit` arrives on the stage's "empty" mbarrier to hand it back to the loader, and on an
  "accumulator full" mbarrier after the last K stage.
- Warp-specialised: a TMA producer warp, an MMA warp, four epilogue warps (one per 32-lane quarter of TMEM) that
  `tcgen05.ld` the accumulator, convert, and write C with **ordinary** global stores (no TMA store: initcheck does not
  track TMA tensor stores, `audit/TMA_STORE_FINDINGS.md`).
- Fixed reduction order: each output's K is accumulated in increasing K by one CTA's MMA chain; no split-K, no atomics.
  The output is deterministic run to run (rule C4), though not bitwise equal to `lfm::gemm` (different hardware
  accumulation order).
- Every mbarrier wait in the probe build is a bounded spin: past ~2 s of `globaltimer` it records the barrier and
  `__trap()`s, so a protocol bug ends as a launch error instead of a hung GPU.

**Variants**, built in this order; each must pass the correctness rules before it is timed; no other variant without
amending this file first:
- **V1**: 1-CTA, tile 128 x 256 x 64, as many stages as fit (4 at 48 KB a stage), one tile per CTA, accumulator 256
  TMEM columns.
- **V2**: V1 made persistent (grid = 20 SMs, static tile order) with two TMEM accumulators (2 x 256 = 512 columns),
  so one tile's epilogue overlaps the next tile's MMAs.
- **V3**: 2-CTA (`cta_group::2`, cluster of 2, MMA M = 256 across the pair, each CTA loading half the B tile), on V2's
  schedule: cuBLAS's choice.
- **Peak**: a microbenchmark of `tcgen05.mma` alone on shared-memory-resident tiles (no loads), 1-CTA and 2-CTA: the
  ceiling, as `lfm3_mma_peak` measured `mma.sync`'s. Reported, no bar.

**Build-up** (unit tests before any full GEMM): one 128 x 256 x 16 MMA from shared memory against a CPU product
(the instruction and shared-memory descriptors, swizzle included); then one K = 64 stage; then a full tile; each
bit-exact with integer inputs (as rule C1) before the next step.

## Shapes
N x K, with K doubled where the vision encoder doubles it (A = [hi | lo], B = [W | W]):
| source | (N, K) |
|---|---|
| vision, split (FP32 out) | patchifier (1024, 3072); conv in_proj (3072, 2048); q, conv out_proj, attention out (1024, 2048); k, v (512, 2048); w1, w3 (2560, 2048); w2 (1024, 5120); projector 1 (4096, 8192); projector 2 (2048, 8192) |
| language model prefill (BF16 out) | conv in_proj (6144, 2048); q, o, conv out (2048, 2048); k, v (512, 2048); w1, w3 (7680, 2048); w2 (2048, 7680) |
M in {64, 100, 128, 257, 1000, 1980, 4096, 7920, 8192} for every shape (100, 257, 1000, 1980 and 7920 are not
multiples of the tile; 7920 is a real 4K image's patch count). The undoubled vision shapes (1024x1024 and so on) are
the `x1` path, which does not ship (`images-speed-v1` rule 1); K = 2048 above covers their kernel geometry. 13 (N, K)
x 9 M = **117 cases** in their listed output type; the correctness rules also run each in the other output type
(**234 cases**). The audio encoder's shapes (width 768: N or K of 768 or 2304) do not meet the N contract and are left
to a later round.

## Rules, set before any probe code
**Correctness, every case, every variant timed:**
- **C1 exactness**: A and B filled with random integers in [-4, 4] (every partial sum is an integer below 2^24, so
  any FP32 accumulation order is exact). Every output element is **bit-identical** to `lfm::gemm`'s output in the same
  output type. The exactness premise is itself checked: on 4,096 sampled elements per case `lfm::gemm` equals a CPU
  FP64 product exactly.
- **C2 accuracy**: A and B random BF16 (normal, unit scale, with per-row and per-column scales drawn log-uniformly
  over 2^-4..2^4, as activations have outliers), and for the vision shapes A = [hi | lo] split from FP32 normals with
  B = [W | W]. Reference: an FP64 GPU kernel (every element, with S = Σ|a_k b_k|), itself checked against a CPU
  long-double product on 4,096 sampled elements per case to 1e-12 relative. Then, on **every element**:
  - FP32 out: |c - ref| <= K · 2^-23 · S (twice the FP32 recursive-summation bound, any order, either rounding mode);
  - BF16 out: |c - ref| <= K · 2^-23 · S + one half BF16 ulp of the FP32 value;
  - and per case: the probe's max and mean of |c - ref| / S are at most **2x** and **1.25x** `lfm::gemm`'s on the same
    inputs (FP32 out); for BF16 out, max ulp against the FP64 reference at most max(0.55, 1.1 x `lfm::gemm`'s), and
    every element within 1 BF16 ulp of `lfm::gemm`'s.
- **C3 coverage**: C is filled with a NaN pattern before each run, with 64 KB guard regions before and after it:
  every element of C is written, and the guards are unchanged.
- **C4 determinism**: every C1 and C2 case run 5 times: all outputs bit-identical across runs.
- **C5 no trap**: no watchdog trap, no launch error, in any of the above.

**Sanitizers** (compute-sanitizer 13.5.25, as earlier rounds), on V-final, cases {M 100, 257, 1000} x {vision w2
(1024, 5120) FP32, LM w1/w3 (7680, 2048) BF16} with C1's and C2's data: memcheck (with leak check), racecheck,
synccheck and initcheck each report **0 errors**, and the outputs are byte-identical to the plain run. Known coverage
limits, recorded not waived: NVIDIA lists some `tensormap`/`tcgen05` global accesses as possible initcheck false
positives (the probe has no `tcgen05` global access and no TMA store; a report on the TMA loads would be investigated
with an ordinary-load control as in `audit/TMA_STORE_FINDINGS.md` before any conclusion); racecheck does not model the
async proxy or TMEM, so a clean racecheck does not attest the mbarrier protocol: C1 and C4 do.

**Speed** (Tripp runs it, quiet Thor, MAXN, GPU clock locked at 1.575 GHz; three rounds; per case the median of 20
timed runs after 3 warm-ups, CUDA events, back-to-back on the same buffers as `gemm-probe-v1`'s prefill method; a case
counts in a round's direction only if it does so in at least 2 of 3 rounds). Compared: the variant, `lfm::gemm`, and
cuBLAS called as production called it (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, default tensor op; FP32 C for FP32 out).
Only the listed output type of each shape is timed.
- **S1 (the question)**: on the 13 shapes at M in {4096, 7920, 8192} (39 cases), the fastest variant passing every
  correctness rule is above **56.7 TFLOPS** on every case, and its geometric-mean speedup over `lfm::gemm` is at least
  **1.5x**. Both hold: **YES**, a native tcgen05 kernel reaches well past the `mma.sync` ceiling. Otherwise **NO**,
  with the numbers.
- **S2**: its geometric mean against cuBLAS on the same 39 cases is reported; at least 0.9x is called "parity".
- **S3**: for M below 4096, the ratio to `lfm::gemm` per case is reported, and the smallest M at which the variant is
  faster on every shape is recorded for a later dispatch rule. No bar.
- The Peak microbenchmark is reported beside S1 as the ceiling.
- Any speed number seen while the GPU is shared is labelled "indicative, contended" and is not a result.

## What would end the probe early (a valid result, recorded)
- The descriptors cannot be made bit-exact on the one-MMA unit test after a bounded effort: the toolchain path is
  recorded as a dead end with the failing encodings.
- A watchdog trap or hang that cannot be explained: SIGTERM, recorded, Tripp told.
- The Peak microbenchmark below ~70 TFLOPS: Thor's tcgen05 rate would not allow S1 with margin; recorded and reported
  before V2/V3 are built.

## Risks
- Descriptor bit layouts (shared-memory matrix descriptor, instruction descriptor, SWIZZLE_128B addressing) come from
  the PTX ISA without CUTLASS to copy; mistakes give wrong numbers or hangs. Mitigated by the build-up unit tests and
  the watchdog.
- `sm_110a` code runs only on sm_110: integrating it later means an arch-specific translation unit (or `sm_110f`) in
  an engine built for `sm_110`.
- Sanitizer coverage of TMA/TMEM/the async proxy is partial (above).
- Small M (64-1000) is bandwidth- or occupancy-bound (M = 64 fills half a 128-row MMA, few tiles for 20 SMs):
  `lfm::gemm` may stay faster there; S3 measures it.
- Thor's clocks and shared GPU: no speed result until Tripp's quiet run.

## Results (2026-11-16): C1-C5 and sanitizers PASS for V1, V2 and V3 (b4); S1 YES: V3 2.38x `lfm::gemm`, cuBLAS parity
Code: `native/probes/tcgen05_gemm.cuh` (the kernels; notes 1-6 document every encoding), `tcgen05_gemm_probe.cu`
(unit, peak, correct, sanitize, speed), `tcgen05_probe_engine_gemm.cu` (`lfm::gemm` built apart with `-arch=sm_110`,
as the engine builds it). Build: `tools/thor/build_tcgen05_probe.sh` in the pinned image, 0 warnings, no spills in
the probe's kernels (`lfm::gemm`'s decode `thin_k128<3>` spills 16 bytes, as it does in the engine; not used here).
Final binary **b4** (`cd1271c4...`); `evidence/` holds the records, `sources.sha256` the source hashes per build.

**Build-up unit tests** (`evidence/unit-b4.jsonl`, also passed on b1-b3): bit-exact on the first build, no encoding
needed a second try.
- TMA with `SWIZZLE_128B` writes exactly `sw128_offset` (note 1), rows past the tensor zero-filled (72 of 128 in).
- One `tcgen05.mma` 128 x 256 x 16 and one K = 64 stage, operands placed by threads in the documented layout,
  instruction descriptor `0x08400490`: 0 mismatches in 32,768 outputs each.
- 2-CTA: one 256 x 256 x 16 MMA across a CTA pair and one K = 64 stage, descriptor `0x10400490`: 0 mismatches in 65,536.

**C1-C5** (`tools/thor/tcgen05_probe_correct.sh`, summarised by `tools/tcgen05_probe_summary.py`): 612 cases per
variant: C1's integer data on all 13 shapes x 9 M x 2 output types (234), C2's normal data the same (234), and C2's
split data on the 8 vision shapes (144); each probe output run 5 times. (The design's "234 cases" counted one data
kind.)
| variant | build | cases passing C1-C5 | every element bitwise equal to `lfm::gemm` |
|---|---|---|---|
| V1 | b2 (`31a14671...`), b3 (`3faa0e10...`) | 612/612, 612/612 | 612/612 cases |
| V2 | b3 | 612/612 | 612/612 cases |
| V3 | b3 (superseded, see the racecheck finding) | 612/612 | 612/612 cases |
| V3 | b4 (`cd1271c4...`) | **612/612** (rerun 14:53-15:04; a first run was stopped by SIGTERM at 13:29 for the quiet window after 214/214, `correct-v3-b4-interrupted.jsonl`) | 612/612 cases |
V1's and V2's kernel (`gemm_1cta`) has the same SASS in b3 and b4 (7,476 lines compared), so their b3 records stand
for b4.
- **A finding beyond the rules:** every output element of every variant, FP32 and BF16, on all three data kinds, is
  **bitwise identical** to `lfm::gemm`'s (`mma.sync` m16n8k16). The FP32 accumulation of `tcgen05.mma` over K = 16
  steps in increasing K gives the same bits as `mma.sync`'s over the same K order on this hardware. C2's ratios are
  therefore exactly 1.0 (max and mean), and the probe inherits `lfm::gemm`'s accuracy record unchanged. A later
  integration could be judged bitwise against the engine's current path, not by perplexity.
- C3: no element left unwritten, no guard byte changed; C4: no run differed; C5: no trap, no launch error.
- C2's reference: GPU FP64 on every element; its CPU long-double check on 4,096 samples per case agreed within
  1e-12 x S everywhere (the rule wrote "relative"; S = sum |a b| is the scale used, since the result itself can be 0).

**Sanitizers** (`tools/thor/tcgen05_probe_sanitize.sh`, compute-sanitizer 2026.4.0 from the 13.5.25 archive; all three
variants, not only the final one; 15 cases each: M in {100, 257, 1000}, vision w2 FP32 with int/normal/split data,
LM w1/w3 BF16 with int/normal data):
- b3: **two failures, both recorded.** (1) memcheck reported 1 error in every variant: a 4-byte leak, the host-mapped
  watchdog flag never freed (harness, not kernel). (2) **racecheck reported 15 hazards in V3**: a race in the 2-CTA
  TMEM allocation. Its SASS (`UTCATOMSWS.2CTA`, then a remote `SYNCS.ARRIVE` and `STAS` into the peer CTA) touches the
  peer's reserved shared memory, and the kernel allocated before any cluster barrier, so the leader could reach the
  peer before the peer had initialised it (the write is reported at a PC before the kernel's entry). C1-C5 did not
  catch it (612/612): a real ordering bug found only by the sanitizer. Fixed in b4: a cluster barrier before every
  2-CTA allocation (`gemm_2cta`, the 2-CTA unit test and peak kernels; note 6), and the flag freed.
- b4: memcheck (leak check), racecheck, synccheck, initcheck: **0 errors** for V1, V2 and V3; all 45 outputs
  byte-identical to the plain run.
- **Positive controls** (added in b4, same kernels): initcheck on a C row the probe never writes reports it (128
  errors); memcheck on the probe writing one row past C's allocation reports it (5 errors). The tools see this
  kernel's ordinary stores; a clean report is not blindness. (Racecheck has no control: it does not model TMA, TMEM or
  the async proxy, so C1 and C4 attest the mbarrier pipeline; it did see the allocation race above.)

**Peak and speed: S1 YES** (Tripp's quiet round, 2026-11-16 14:52-14:53, `tools/thor/tcgen05_probe_speed.sh b4
tcgen05-speed-r{1,2,3}`; MAXN, GPU clocks locked at 1.575 GHz at the start and end of every round; binary b4;
`evidence/speed-r{1,2,3}/`, `evidence/speed-summary.txt` from `tools/tcgen05_probe_speed_summary.py`):
- **Peak** (`tcgen05.mma` alone): **258 TFLOPS**, 1-CTA and 2-CTA alike, in all three rounds: 4.5x the `mma.sync`
  ceiling (56.7). The early-stop rule (< 70) does not apply.
- Geometric-mean TFLOPS over the 39 cases at M >= 4096: V1 92.5, V2 99.8, **V3 106.8** (the fastest; it passes every
  correctness rule on b4, below).
- **S1: YES.** V3 is above 56.7 TFLOPS on 39 of 39 cases (in at least 2 of 3 rounds; its lowest, the projector's
  4096 x 8192, ~80) and **2.38x** `lfm::gemm` (geometric mean; the rule asked 1.5x).
- **S2: parity**, V3 at **0.98x** cuBLAS (geometric mean). cuBLAS's 105.8 TFLOPS, the number that started this, is
  matched: V3 runs the LM's w1/w3 at 8,192 rows at 115.6 TFLOPS (cuBLAS 113.3 in the same rounds).
- **S3:** V3 is faster than `lfm::gemm` on every shape at every M from **257** (ratios 1.29-3.30). At M <= 128 its
  ratio is 0.97-2.23: never more than 3% slower (the projector shapes at M = 64 and 128 are level, both
  weight-bandwidth bound). cuBLAS leads at M <= 128 on most shapes (not on the LM's w2 at M = 64 or the projector's at
  M = 64).
- Not the best everywhere: on the projector's 4096 x 8192 at M >= 4096, V1 (91) beats V3 (81); at M <= 128, V1/V2
  beat V3 on most shapes. A per-shape choice is left to a later round.
- The contended check of 13:00 (`evidence/indicative-contended-b3/`, Peak 79) was the GPU shared, as labelled.
