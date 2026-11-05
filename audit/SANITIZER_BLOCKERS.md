# Sanitizer blocker follow-up

Neither a timeout nor correct sampled output is a full sanitizer pass. No output
preinitialization, instruction suppression, selected-block coverage reduction or
disabled checking was used.

## Initcheck sanitizer-only version/shape matrix

`tools/check_lfm3_initcheck_matrix.py` holds the checkpoint-free probe binary,
resolved cuBLAS/runtime libraries and driver fixed, changing only sanitizer and
shape. It tests257x2048,1x128,257x128 and1x2048 with CUDA-fill, intentionally
missing-write, and GEMM producers. The unchanged probe checks **128 sampled
values**, not every output. Nine host classification counterexamples prevent
numeric failure, timeout or an unrelated library error from being mislabeled as
the established downstream tracking reproduction.

Official standalone packages were verified against redistribution manifests:

-13.4.92 from CUDA13.4.2 (2026-09-16), product Compute Sanitizer2026.3.0.0.
-13.5.25 from CUDA13.5.0 (2026-11-04), product Compute Sanitizer2026.4.0.0.

Only standalone tools were extracted under `build/`; no driver, toolkit or
serving-library replacement was made. Archive and manifest hashes are preserved
in `validation/initcheck-version-shapes-v2/`. Official manifests:
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.2.json
and https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json.

Measured versions2026.2/2026.3/2026.4 yield:

-12/12 CUDA-fill controls clean.
-12/12 intentional missing-write controls detected.
-9/12 GEMM cases reproduce uninitialized reads in downstream `read_output` despite
  correct sampled values (three shapes per tool).
-3/12 GEMM cases (1x2048, all tools) report an uninitialized read inside
  `cublasLt::splitKreduce_kernel`. This is a **distinct unresolved diagnostic**, not
  automatically the same tracking bug, and the strict matrix remains failed.

cuBLAS reports130100. Resolved serving libraries are pinned by SHA256 in the
matrix summary: cudart `7bdba2b5...`, cuBLAS `7761d260...`, cuBLASLt `3839e12c...`.
All GEMM probes printed correct128-sample results; that does not waive either
diagnostic. The raw failed summary is preserved, not changed to a pass.

NVIDIA documents some unsupported `tensormap`/`tcgen05` initcheck accesses, making
an instrumentation gap plausible but **not a confirmed diagnosis here**:
https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations.
The separately documented cuBLAS issue6196059 concerns other async-copy/OOB
behavior; do not equate it to these uninitialized-read reports without confirmation.

Next vendor reproduction should contain the existing checkpoint-free projection
probe, build command, three controls, this shape/version matrix, exact libraries,
Thor/BSP/R580 details and128-sample scope. Ask for a supported Thor tooling pair
and tracking ID, not a suppression. No external report has been published.

## Integrated32K racecheck

The original1200-second monolithic timeout remains open. Its log contains only
the tool banner, so there was no evidence locating the slow launch. The60 small
operator fixtures passing four tools do not replace integrated coverage.

Compute Sanitizer2026.4 `--help` confirms `--dump-kernel-launches`,
`--print-session-details`, `--racecheck-num-workers`, `--launch-skip` and
`--launch-count`. Next diagnostic run must keep the full accepted32K request and
binary, enable launch/session detail, and retain elapsed/CPU/RSS observations.
No launch filtering or force-blocking mode is permitted in the full-run gate.
If partitioning is later required, its launch ledger must cover the union of
**every** original launch and be reported as partitioned coverage, not a
monolithic pass. A single slow launch still requires direct investigation.
