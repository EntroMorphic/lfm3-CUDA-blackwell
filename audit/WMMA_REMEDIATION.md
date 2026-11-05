# Native ordinary-store GEMM remediation candidate

## Current disposition

The isolated runtime candidate replaces the measured NVJET linear path with
native BF16 WMMA and ordinary global stores. It passes all **6,890 unchanged eager
checks**, all **nine exact context fixtures through4096**, and integrated initcheck,
unexcluded synccheck and memcheck on the original four-token single fixture.
**Production is not switched and full acceptance remains false.** Upper-context,
paired/worker, resource, performance and full32K racecheck acceptance are separate.

Candidate single SHA256:
`d6ed2fefc5e8691cb574b4468e91e9e1cd5361c605f66f5cdac50fb8450d288d`.
The original single/worker/paired source paths and binaries remain unchanged.

## Failed alternatives retained

1. Explicit workspace budgets0 and16KiB preserve the small fixture's numbers but
   still report6,528 initcheck errors and internal synccheck failures.
2. Disallowing reduced-precision reductions preserves the small fixture's numbers
   but still reports12,288 initcheck errors and internal synccheck failures.
3. The first real-weight Lt screen correctly rejected an ambiguous tensor suffix:
   audio, language and vision each have a layer0 convolution projection. The
   corrected extractor selects the complete language-model key and verifies shape.
4. Initial Lt enumeration executes nine configurations per real fixture. Only two
   match all6,144 decode outputs; both report12,288 initcheck errors and internal
   synccheck failures. The other tested decode defaults differ by2–3 BF16 values.
   The first34-row instrumented prefill candidate times out at60s; no pass is
   credited. Expanded split-K enumeration executes107 decode configurations
   (24 exact) and80 prefill configurations (7 exact). Numerically exact decode
   reduction-scheme variants1/4 still reproduce both checker failures.

Evidence: `blas-policy-screen-v1`, `lt-policy-screen-v1`, `lt-policy-screen-v2`,
`lt-policy-split-scan-v3`, `lt-policy-reductions-v4` under `validation/`.
Enumeration is a bounded candidate search, not a complete proof about every
possible library algorithm or configuration.

## Native arithmetic and stores

`native/tests/lfm3_wmma_policy_probe.cu` uses one warp per16x16 output tile, BF16
WMMA in16-element K steps, FP32 partial accumulators, an ordinary-store FP32
scratch matrix, and a native reduction/materialization kernel. Shared-memory
zero padding covers only input tails; scratch/output is never preinitialized.
Producer and consumer outputs are checked in full against real checkpoint traces.

For the actual layer0 projection:

| Fixture | One partition | Two partitions | Four partitions |
| --- | --- | --- | --- |
| Decode,6,144 values |3 different |**0 different** |2 different |
| Prefill,208,896 values |**0 different** |192 different |205 different |

Both exact choices independently pass initcheck, memcheck and synccheck without
exclusions. This demonstrates why blindly changing reduction order is not a fix.
See `wmma-policy-screen-v1`. All12 standalone observations, including rejected
partition choices, are independently replayed from complete producer/consumer
bytes and original reference outputs in `validation/wmma-policy-replay-v1.json`.
Checkpoint weight bytes remain remote and are not copied into this repository.

`tools/prepare_lfm3_wmma_runtime.py` creates an isolated source mirror and build.
Only `Runner::linear` dispatch changes. Its generated header incorporates the
exact standalone WMMA kernel source, checked by host tests and recorded hashes.
It queries cuBLASLt's first heuristic with the original8,519,680-byte budget to
obtain the partition count. In this pinned experiment, algorithm ID66 selects
the native path; other IDs retain the original cuBLAS call. Unsupported partition
geometry throws rather than silently changing arithmetic. Both attention batched
GEMMs, all normalization/sampling/cache code and all numerical gates are unchanged.

This ID-based heuristic mapping is **candidate policy for the pinned environment**,
not a portable vendor guarantee. It must not be generalized across library/GPU
versions or presented as an accepted deployment policy without further checks.
The native scratch is distinct, owned and fully written by each producer; it is
not an initialization workaround for the original vendor scratch.

To regenerate a fresh candidate (not a production switch):

```sh
python3 tools/prepare_lfm3_wmma_runtime.py --output build/wmma-runtime-candidate-NEW
cmake -S build/wmma-runtime-candidate-NEW -B build/wmma-runtime-candidate-NEW/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/wmma-runtime-candidate-NEW/build --target liquid-lfm3 liquid-lfm3-worker lfm3-decode-batch-probe -j4
```

Build on Thor in compiler image
`sha256:3c2b5b8e0857ada02326ee34175d52c35f6c909d89f49df6ae12f5ac5109052f`;
execute in the original pinned runtime/library environment, not compiler-image
cuBLAS. `validation/wmma-runtime-source-v1/` preserves the complete emitted source
mirror and binary identities. The candidate single's static SASS contains HMMA
and no `UTMASTG`; disassembly metadata binds the actual binary in
`validation/wmma-runtime-disassembly-v1/`. This is not a dynamic instruction count
or a claim about external library modules.

## Numerical and integrated checks

- Original `tools/check_lfm3_forward.py`: all6,890 exact activation/logit checks,
  greedy/teacher IDs, suppression/stopping and reset pass. The regenerated
  reference files also match **all6,890 original reference hashes**, independently
  replayed; no goldens or tolerances changed.
- Original full `tools/check_lfm3_context.py`, reusing its pinned original
  references: nine fixtures,19 complete vocabulary logit steps, zero tolerances,
  reset and12 pre-weight rejections pass.
- The original single-token-prompt/four-generated-token request, with reset and
  expected original application output, passes integrated initcheck (16.205s),
  unexcluded synccheck (2.727s), and memcheck (18.553s). These are scoped elapsed
  tool observations, not serving speed benchmarks.

Evidence: `wmma-runtime-forward-v2`, `wmma-runtime-context-v1`, and
`wmma-runtime-sanitizers-v1`. The first forward attempt failed during reference
Triton assembly because it omitted the documented Thor assembler variables; it
never reached native comparison. That failure is retained as
`wmma-runtime-forward-v1`. The successful retry uses the existing documented
`/workspace/tools/ptxas` path for both Triton assembler variables, with its hash
pinned; no system tool replacement occurred.

Actual generation responses are copied to project-root
`../tmp/wmma-runtime-candidate-v1/`, including decoded text linked to raw response
hashes. The current host suite passes1267 tests
(`validation/host-wmma-candidate-v2.log`), including fail-closed host models for
missing/duplicate checker footers, internal checker errors, incorrect application
IDs, missing output and timeout.

## Still required

No production promotion, full-runtime/upper-context sanitizer certificate or full
racecheck completion is implied. Full32K independent SDPA, paired matrices,
integrated grouped worker, legacy serving/cancellation, balanced performance,
resource soak and upper sanitizer/racecheck gates must retain their original
assertions and complete on the selected final binary. There are no suppressions,
checker kernel exclusions, scratch seeds, pedantic math, system library changes,
vendor submissions or waivers in this remediation.
