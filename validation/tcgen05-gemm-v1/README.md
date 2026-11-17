# tcgen05 in `lfm::gemm`: the engine's large projections on tensor memory, every output byte unchanged

## The question
`validation/tcgen05-probe-v1` found V3, a 2-CTA `tcgen05` kernel, **2.38x** `lfm::gemm` at M >= 4096 (geometric
mean, 39 cases), at cuBLAS parity (0.98x), and faster on every engine shape from M = 257. On all 612 cases per variant,
every output element was **bitwise identical** to `lfm::gemm`'s. This round puts V3 behind `lfm::gemm::run` for the
shapes it covers, and asks: does every engine output stay byte for byte the same, with no sanitizer error and no stall
under concurrent load, while prefill and image encoding get faster?

## Design (set before any code)
**Build split.**
- The kernel moves from `native/probes/tcgen05_gemm.cuh` to `native/include/lfm/gemm_tcgen05.cuh` unchanged (V3 as in
  probe build b4, SASS compared), plus a host entry in its own translation unit, `native/src/gemm_tcgen05.cu`, compiled
  only with `-gencode arch=compute_110a,code=sm_110a`. Every other unit keeps `-arch=sm_110`.
- `gemm.cuh` stays header-only. With the CMake option `LIQUID_LFM3_TCGEN05` (default OFF in this round, ON only if
  every rule passes) it declares `bool lfm::gemm::tcgen05::try_run(const Problem&, cudaStream_t)`, and `run()` calls
  it first; without the option nothing changes (the OFF build is the A/B baseline).
- Tensor maps are encoded per call on the host by `cuTensorMapEncodeTiled`, reached through
  `cudaGetDriverEntryPointByVersion` (no `libcuda` link). The bounded mbarrier waits stay, with a 10 s timeout: a
  trap is a launch error the engine already reports, never a hang.

**Dispatch** (from the probe's S3; one rule for all shapes, no per-shape table in this round):
- V3 when all hold: `b_k_contiguous`, `batch == 1`, `group == 1`, `n % 256 == 0`, `k % 64 == 0`, `lda == k`,
  `ldc == n`, A, B and C 16-byte aligned, and **`m >= 257`**. FP32 (`c_f32`) and BF16 out both.
- Everything else keeps today's kernel: M <= 8 (thin/split-K), attention Q·Kᵀ and P·V (batched, `group > 1`, K x N),
  `gate_proj` (N = 16), lossless-packed weights, and M <= 256.
- `lfm3_batch.cuh`'s `linear_tiled` launches tile kernels directly, not through `run()`. Its `rows > 64` branch
  (`Wide128`) gains the same dispatch when `splits == 1`; the split-K path (strided, batched A) stays as it is.
- Probe facts this rests on: all of `lfm::gemm`'s tile configurations are bitwise identical to one another (gemm-probe
  -v1 sweep, the fixed k16 order), and V3 is bitwise identical to them (tcgen05-probe-v1). Shape choice therefore
  cannot change a result; only a bug can.

**Fallback.**
- At first use, per device: compute capability exactly 11.0, the kernel's attributes readable
  (`cudaFuncGetAttributes`), the shared-memory opt-in set, a driver entry point for the tensor-map encoder. Any
  failure: the path is off for that device, recorded once on stderr, and every call takes today's kernel.
- `LFM3_TCGEN05=0` forces it off at run time (for A/B and for any field problem).
- No silent retry: a launch error from the tcgen05 path is reported like any other kernel's.

## Rules, set before any code
1. **GEMM exactness at the engine's entry point** (`lfm3-tcgen05-gemm-check`, a test binary): `lfm::gemm::run` with
   the path on against off, bitwise, on every `run()` and `linear_tiled` call shape the engine makes (the probe's 13
   shapes, the LM head N = 128,000, and the M <= 256 / >= 257 boundary: M in {256, 257, 258, 4096, 32768}) with C1's
   integer and C2's normal and split data, both output types; plus the refused shapes (`gate_proj`, Q·Kᵀ, P·V,
   M <= 256, a misaligned operand) shown to take the old kernel (a dispatch counter). 0 differing bytes.
2. **Every byte-identity regression**, the `LIQUID_LFM3_TCGEN05=ON` build against today's recorded outputs:
   - lfm3-speech's regression, 36 outputs (`tools/thor/speech_regression_check.sh`);
   - the audio-in replies, 360 + 93 files; e2e and split, 90 + 90 (`validation/audio-in-v1`, `speech-e2e-v1`,
     `speech-split-v1` methods);
   - the vision encoder: all 713 stage files of the 31 images of images-v1 rule 2 (`tools/thor/vision_accuracy_round.sh`);
   - the image stage-0 replies (11, `vision_e2e_round.sh`), and images through the server identical to lfm3-speech;
   - the server checks: batch, stream, audio in, TTS, and the text gate with control `7ea05068...`
     (`vision_rule6_server.sh`, `vision_rule6_speech.sh`);
   - the host suites pass.
   All identical, 0 bytes differing. Any difference fails the round; it is investigated, not explained away.
3. **Sanitizers** (compute-sanitizer 2026.4.0, the 13.5.25 archive): memcheck (leak check), racecheck, synccheck and
   initcheck on `lfm3-vision-check`'s small, two-image and largest cases and on a 4,096-token text prefill: 0 errors,
   outputs byte-identical to the plain run; the probe's two positive controls rerun on the same sanitizer and binary
   (must be reported).
4. **Tensor-memory soak** (the risk: `tcgen05.alloc` blocks while another kernel holds TMEM on the SM; the server runs
   speech, text and images on concurrent streams):
   - (a) synthetic: 4 host threads on 4 streams, each launching V3 GEMMs interleaved with `mma.sync` GEMMs and the
     engine's attention, for 20 minutes; every output bitwise equal to the same call run alone; no trap, no launch
     error; every GEMM's latency at most 10x its solo median (a stall shows as orders of magnitude, not percent).
   - (b) the server under mixed load for 60 minutes (the soak harness of `validation/text-wait-v1`, host memory
     sampled): concurrent image requests (including the 4K photographs), text prompts of 300-4,096 tokens and speech
     streams; every reply identical to the same request served alone; no request over its timeout, no trap; host and
     device memory flat after warm-up (text-wait-v1's allocator criteria).
   Soak and timing runs only on a quiet Thor.
5. **Speed** (Tripp runs it, quiet Thor, MAXN, clocks locked; three rounds, a direction counts in 2 of 3):
   - vision encode at 8,192 patches (`lfm3-vision-check`, warm median of 3): faster than the OFF build;
   - LM prefill at 4,096 and 32,768 tokens: faster than the OFF build;
   - image requests end to end (`vision_speed_round.sh`): native still at most HF on all 11 stage-0 requests;
   - **batch-32 aggregate decode tok/s** (the P0 measure): not slower than the OFF build by more than 1%.
   Expected (not a bar): the encoder's GEMMs were ~28% of encode at ~43 TFLOPS, so encode ~0.85x.
6. **Coherency**: outputs identical means the readings of images-v1, audio-in-v1 and speech-e2e-v1 stand; I still read
   the 11 image stage-0 replies and 10 audio-in replies from the ON build end to end.

## Not in this round
- A per-shape choice (V1 is faster than V3 on the projector's 4096 x 8192 at M >= 4096: 91 against 81 TFLOPS) and
  M <= 256 kernels: a later round, from the probe's per-case table.
- The audio encoder (FP32 `linear_staged`, widths 768/2304: outside the N contract), attention on tcgen05, split-K.

## Risks
- `sm_110a` code runs on sm_110 only; the fallback covers any other device.
- Tensor maps are kernel parameters: a CUDA graph capturing a tcgen05 call fixes its pointers like any other kernel's.
  The engine's decode graphs (M <= 64) never take this path; recorded, not tested here.
- The 10 s watchdog could fire on a preempted kernel only under extreme time-slicing; the soak watches for it.

## Results so far (2026-11-16)
Builds (`tools/thor/build_tcgen05_engine.sh`, the same sources ON and OFF; `lfm3-batch` and `lfm3-tcgen05-soak` added
later from unchanged sources): `tgemm-on1` (lfm3-speech `7e3da591...`, server `0193b11b...`, worker `eb71d809...`,
lfm3-vision-check `0cd39878...`), `tgemm-off1`. The engine's `gemm_2cta` SASS is identical to the probe's b4 (2,292
and 2,228 instruction lines, BF16 and FP32 out). Host suite on the workstation: 1,646 passed, 703 skipped, 0 failed.

**Rule 1: PASS.** `lfm3-tcgen05-gemm-check`: 353 of 353 cases, 0 elements differing with the path on against off.
Every engine case at M >= 257 made exactly one tcgen05 launch; M = 256 and the five refused problems (gate_proj
N = 16, ldc != n, M = 256, batched Q·Kᵀ, P·V K x N) made none. Recorded deviation: the LM head (N = 128,000) ran at
M in {256, 257, 258, 4096}, not 32,768 (its FP32 output pair would be 33.5 GB).

**Rule 2: every byte identical; two checks to rerun quiet for their non-exactness parts.**
(`tools/thor/tcgen05_gemm_rule2.sh tgemm-on1 tgemm-r2`, shared GPU, 15:19-17:37)
- speech regression 36 of 36 (and again 36 of 36 in the server phase); audio-in 360 of 360 and 93 of 93; e2e 90 of
  90; split 90 of 90; vision stage files 713 of 713 against `vspeed-rule1-a2` (rule-1 thresholds pass as before:
  ratios 0.023-0.112, cosine 0.9999966); image stage-0 67 of 67 files against `vimg-e2e-a4` (arrays byte for byte,
  reply tokens); concurrent engine exactness 41 of 41; audio-in through the server PASS; TTS through the server PASS;
  text gate passed, control `7ea05068...`; images through the server PASS (alone, streamed, concurrent).
- **Batch-server check: did not complete.** It stopped in its first phase, on the *reference* server
  (`server-stream-v3`, an old build without tcgen05), with `503 worker_unavailable` for `budget-truncated-s1`; the
  build under test was never reached. Most likely the shared GPU (images-speed-v1 ran these on a quiet machine). Rerun
  quiet, after window 3.
- **Stream check: exactness PASS, timing FAIL.** Every case exact (rule 1), rule 2 pass; its rule 3 (the cost of
  streaming, median streamed / plain) 1.083 against its bar, with plain runs swinging 10.1-17.3 s on the shared GPU: a
  timing rule measured on a contended machine. Rerun quiet with the batch-server check.
- Coverage: most speech requests have short prompts (< 257 tokens) and never reach tcgen05; the vision encoder (every
  image), image prompts, the text gate's longer prompts and the server's image checks do.
