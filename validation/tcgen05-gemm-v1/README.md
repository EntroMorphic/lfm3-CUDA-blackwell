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

## Amendment 1 (2026-11-16, before window 4; Tripp approved): rule 4a's latency gate
- **Original bar, kept as a reported figure:** every call's latency at most 10x the problem's solo median.
- **The gate becomes:** every call's latency at most **10x the same problem's median under the soak** (the 4 streams
  sharing the GPU), per problem. The ratio to the solo median is still reported (count over 10x, worst ratio).
- **What informed it:** a 30-second functional dry run of `lfm3-tcgen05-soak` on a **shared** GPU (another engineer's
  work running; 17:40), not the soak itself. 10,459 calls, 0 outputs differing, no error or trap. Each problem's
  median under the soak was ~4x its solo time (four streams queueing for the SMs), its p99 11-13x; **278 calls
  (2.7%) were over 10x solo, worst 19.3x**; against each problem's own soak median the worst call was **4.75x**. A
  blocked tensor-memory allocation, the risk this rule exists for, would show orders of magnitude beyond either; the
  solo bar measured ordinary queueing. The soak's problems were resized before this run (each ~1 ms alone; the LM head
  at 300 rows, 5.6 ms alone, removed) after a first dry run had shown it as the long pole.

**Rule 2, the quiet reruns (2026-11-16 18:41-19:12, after window 3; MAXN, clocks locked;
`tools/thor/quiet_tcgen05_reruns.sh` -> `build/tgemm-r2-quiet/`): both PASS. Rule 2 PASSES.**
- Batch-server check: **PASS** (exactness rule 1 and real-time rule 3; up to 16 concurrent TTS streams with no gap; 7
  TTS streams with 5 spoken questions joining: 0 stalls; server exit 0).
- Stream check: **PASS** (every case exact, rules 1-3; the cost ratio streamed / plain 1.005, against 1.083 on the
  shared GPU).
- Rule 5 (Tripp's window 3, `quiet_tcgen05_gemm.sh`): RULE 5 PASS; image requests at most HF end to end on 3 of 3
  rounds (the full numbers are in `build/quiet-w3-tcgen05/summary.txt`, to be copied here).

## Rule 4 results (window 4, 2026-11-16 21:11-22:48, quiet, unattended; `build/quiet-w4-tcgen05/`)
**4a, synthetic soak: PASS.** 1,200 s, 4 streams, 685,356 calls (342,063 on tcgen05): 0 outputs differing, no error
or trap; the gate (amendment 1): 0 calls over 10x their problem's soak median, worst 5.72x (`v.w2.m8192`). Reported
(the original bar): 2 calls over 10x solo, worst 19.2x.

**4b, server soak: FAIL**, on one clause: the idle memory figure `host_used_less_client` grew **83.0 MiB** from the
first idle checkpoint to the last (bar 64 MiB). Every other clause passed: 3,474 replies, all status 200, **0
differ, 0 errors** (1,144 unstreamed, 1,290 streamed, 153 dropped by design, 871 text; 52,036 429s retried, as the
harness does under 12 clients with 4K images in the mix), the processes alive throughout, server RSS idle growth
+19.3 MiB (pass), the crowd 56 of 56 identical unstreamed and streamed, exit 0. (`client_load`, the client's own
RSS, +155.5 MiB, is reported by the harness, not gated.) The bar is not changed.

Analysis (from `soak-4b.json`, `soak-4b.samples.jsonl` and the 30 s `host-memory.jsonl` sampler beside it):
- The four idle checkpoints read 13,907.5 / 13,983.9 / 14,008.8 / 13,990.4 MiB. The growth is all in the first
  interval: checkpoint 2 to 4 is +6.5 MiB, 3 to 4 is -18.4 MiB. The first two checkpoints' 10 idle readings spread
  by **503 and 496 MiB** (the last two: 0.0 and 9.9), so their medians were taken on a figure still moving by half
  a gigabyte at idle; the 83 MiB is smaller than that within-checkpoint spread.
- The image path's one-time GPU allocations are not it: the sampler's `residual` (on Thor, GPU memory from system
  memory) stepped +1.36 GB once at 21:40:34, during the harness's reference phase (each image request alone, before
  the timed soak, ~21:45), which matches the vision encoder's kept `[W | W]` cache and scratch (~1.4 GB,
  images-speed-v1's red-team), and was flat (9.5-9.9 GB) from there to the end.
- The soak ran alone in the window (the runner is sequential: the frontend checks before it, the stall catcher after).
- **Is it tcgen05?** Not shown either way by experiment: this soak (images and long prompts added to the audio soak)
  has never run on the OFF build, so whether the same clause fails there is unknown. By construction the tcgen05 path
  allocates nothing (tensor maps on the host stack, the accumulator in on-chip tensor memory, no device or host
  allocation), so it cannot itself hold memory; the figure's movement sits in the early checkpoints and the harness's
  host figure has drifted before without tcgen05 (text-wait-v1 rule 7: "the earlier passes were luck", four soaks).
  These make the server/harness more likely, but do not prove it.
- To settle it (proposed, not run): the same 4b soak on `tgemm-off1` in a quiet window (60 min + references), and,
  if it passes there, once more on `tgemm-on1`; reported beside it, the 10 readings of each checkpoint (the harness
  keeps only median and spread) to see what moves by 500 MiB at idle early in the soak.

## Rule 4b off/on pair: how it will be read (set 2026-11-16, before the data; Tripp's window 5)
`tools/thor/quiet_tcgen05_4b_offon.sh`: the window-4 soak unchanged (same harness, cases, 3,600 s, 12 clients, text
share 0.25, same pass rule and 64 MiB bar), first on `tgemm-off1` (the same sources built without the tcgen05 path),
then on `tgemm-on1`, quiet, one after the other. Added, recorded only: every host-memory reading the harness takes
(the 10 idle readings of each checkpoint and the 30 s load samples) with free, buff/cache (Buffers + Cached +
SReclaimable) and the container's memory cgroup (`memory.current`, `memory.max`, anon, file, pgscan, pgsteal) beside
it (`OUT.readings.jsonl`); context from Tripp: lfm3-profile-v1's cgroup sits at its 96 GiB limit with ~43 GB of page
cache charged, so in-cgroup reclaim may move the host figure at idle.
Reading of the pair (each run passes or fails by the harness's own rule):
| OFF | ON | reading |
|---|---|---|
| FAIL | FAIL | the harness or the server, not tcgen05; rule 4b stays FAIL for this build as measured, attributed to them |
| PASS | FAIL | tcgen05 is implicated: rule 4b FAIL, attributed to the tcgen05 path, investigated before any integration |
| FAIL | PASS | not tcgen05: the clause fails without it; ON's pass is recorded; rule 4b PASS on the rerun, with the window-4 failure and OFF's kept in the record |
| PASS | PASS | the window-4 failure did not reproduce on either build; rule 4b PASS on the rerun, window 4's failure kept in the record |
Whatever the outcome, I also report, from the per-reading records: whether each checkpoint's within-checkpoint spread
moves with free / buff-cache / the cgroup's file pages (reclaim) rather than with anon memory, and the same for the
growth between checkpoints. A single pair is one sample each: when one run passes and the other fails only on the
idle-memory clause and their `host_used_less_client` growth figures differ by less than 64 MiB (the bar's own
size), the pair is reported as **not decisive** instead of being read by the table, and the rule-4b verdict stays
FAIL. Order effect (OFF first) noted.

## Rule 3 results (sanitizers, 2026-11-16 23:12 - ; `build/tgemm-r3/`, `tgemm-r3b/`)
First run (`tools/thor/tcgen05_gemm_rule3.sh tgemm-on1 tgemm-r3`, sharing the GPU and the container with the audio
round's sanitizers):
- Vision (`lfm3-vision-check`, small / two-image / 8,192-patch cases): memcheck (0 bytes leaked), initcheck and
  synccheck 0 errors on all three, outputs identical; racecheck small and two-image 0 hazards, outputs identical.
- **Racecheck, maximum case: not completed.** After 3,525 s the process was killed by the kernel's **memory-cgroup
  OOM killer** (lfm3-profile-v1's cgroup at its 96 GiB limit, 00:19:16: it killed a 34.6 GB `lfm3-speech`, the other
  round's sanitizer run, and reaped `lfm3-vision-check`); the sanitizer reported 0 hazards up to then and "process
  didn't terminate successfully" (exit 9), its outputs incomplete. Not a kernel or tcgen05 fault; rerun alone
  (`tgemm-r3b`). Its logs are kept in `tgemm-r3/oom-killed/`.
- **The 4,096-token text case did not run: my script's error.** The request had no `capacity`, so the default 4,096
  could not hold 4,096 prompt tokens + 4 new; even the plain run refused it ("prompt (4096) + max_new_tokens exceeds
  capacity"), and the sanitized runs had nothing to compare (the script printed "0 outputs identical", a false
  label, now "NOT COMPARED"). Fixed (capacity 4,352) and rerun in `tgemm-r3b`.
- Positive controls on the same sanitizer: initcheck's unwritten row reported (128 errors), memcheck's out-of-bounds
  row reported (5 errors).

Rerun (`tgemm-r3b`, 00:20-01:12, the container otherwise empty at the start; `tgemm-r3c`, 01:13-01:20):
- Racecheck, vision: small, two-image and **maximum (8,192 patches, every projection on tcgen05): 0 hazards**,
  outputs identical (65, 199 and 2,144 s).
- Text prefill, 4,096 tokens (capacity fixed): memcheck (0 bytes leaked), synccheck and initcheck **0 errors, all 5
  outputs identical**.
- **Racecheck on the 4,096-token prefill: cannot run in this container.** After 614 s the memory-cgroup OOM killer
  killed it (01:11:28): `lfm3-speech` under racecheck had reached **100 GB of anonymous memory**, past
  lfm3-profile-v1's 96 GiB limit, alone in the container. Racecheck's host-side tracking for a 4,096-row prefill does
  not fit; the rule as written cannot be met for this one tool in this container.
- Recorded as information, not as a substitute until Tripp decides: racecheck on a **1,024-token** prefill
  (M = 1,024 >= 257, so the LM's projections take tcgen05): **0 hazards, all 5 outputs identical** to its plain run
  (`tgemm-r3c`, 416 s).
**Rule 3 status:** every check run is clean and its outputs identical, the controls detected; the one cell not met as
written is racecheck x 4,096-token prefill (infeasible: 100 GB at the kill). Pending Tripp's decision on the 1,024-
token racecheck as its stand-in.

**Rule 3, the 4,096-token racecheck (Tripp, 2026-11-17: no stand-in; run it on the host).** The 1,024-token result
above stays as information only. `tools/thor/tcgen05_host_racecheck.sh` runs it on the Thor host (125 GB, no memory
cgroup) with `build/racecheck-pinned-libs-v1` (as `build/stall-v4/run.sh`), a plain host run of the same request
first; it waits until the audio round's racecheck has finished and no compute-sanitizer runs, logs free memory every
minute, and stops the run with SIGTERM (recorded NOT COMPLETED) if MemAvailable drops under 8 GB. Waiting on Thor
since 01:21 (`build/tgemm-r3-host/`). Result: pending.

## Rule 6 results (coherency, 2026-11-17): read, no engine fault
From the ON build's own outputs (`tgemm-r2-e2e/out`, `tgemm-on1-rule6/dump`), which rule 2 found byte-identical to
the validated builds', read end to end:
- **The 11 image stage-0 replies.** Seven describe their images accurately and fluently: cherry (white blossoms on a
  branch, stamens, blurred background), china (the red-and-green pagoda over water), flower (a peach dahlia, 79
  tokens, a complete answer), gray (the same pavilion in black and white), mirror (a sunset lake as a mirror), rgba
  (the dahlia in detail), two (a structured comparison of the pavilion and the dahlia). Four are fluent but unrelated to
  their image, encyclopedia-style text with invented facts: noise (a UAE football season), small (a "Jade Pagoda" in
  Nanjing), tall (a fabricated 2020 Senate race), wide (the "Tian Tan Buddha" in Beijing). Those four test images are
  degenerate: their prompts carry 0 (noise) or 1-2 (small, tall, wide) image tokens, and HF FP32 replies to `small`
  with the same "Jade Pagoda" text. The model's behavior on near-empty images, not the engine's. No repetition loop, no
  broken text, no language switch.
- **10 audio-in replies** (the first 10 of the 40 dumped clips, "Perform ASR in English"): each a complete, coherent
  LibriSpeech transcription ending at `<|im_end|>`, e.g. "He hoped there would be stew for dinner, turnips and carrots
  and bruised potatoes and fat mutton pieces..." and the Sonnet 115 passage ("But reckoning time, whose million
  accidents..."), with ordinary ASR slips ("Carton" for Cotton, "dam soul" for damsel, "twisted" for 'twixt, "the tree
  kind" for three-wire).
**Rule 6: PASS** (no coherency fault; the off-image replies are the model's, identical to the validated build's and,
where HF's FP32 reply is recorded, to HF's).

## Rule 4b off/on pair results (window 5, 2026-11-17 02:31-05:04, quiet; `build/quiet-w5-tcgen05-4b/`)
| | OFF (`tgemm-off1`) | ON (`tgemm-on1`) |
|---|---|---|
| harness verdict | **PASS** (exit 0) | **FAIL** (exit 1) |
| replies | 3,440, all 200, 0 differ, 0 errors | 3,420, all 200, 0 differ, 0 errors |
| crowd after | 56 / 56 identical, both kinds | 56 / 56 identical, both kinds |
| server RSS, idle growth | -29.1 MiB (pass) | +7.4 MiB (pass) |
| `host_used_less_client`, idle checkpoints (MiB) | 13,787.5 / 13,826.8 / 13,838.4 / 13,844.8 | 13,775.3 / 13,914.8 / 13,853.1 / 13,845.8 |
| its growth (bar 64) | **+57.3 (pass)** | **+70.5 (FAIL)** |
| within-checkpoint spreads (MiB) | 5.0 / **502.7** / 3.8 / 3.2 | 14.4 / 5.7 / 2.3 / 0.2 |
| checkpoint drains (s) | 45.2 / 22.2 / 64.4 / 4.8 | 43.8 / 83.4 / 93.8 / 92.8 |
| 429 retries | 58,825 | 71,145 (+21%) |
| client RSS (reported, not gated) | +69.5 MiB | +61.8 MiB |
The clause that failed on ON: the idle host-memory figure (`host_used_less_client`), +70.5 MiB against 64; every
other clause passed.

**Reading, by the table set before the data:** OFF PASS / ON FAIL would implicate tcgen05, but the not-decisive clause
applies exactly: one run passes and the other fails *only* on the idle-memory clause, and their growth figures
differ by **13.2 MiB** (70.5 - 57.3), less than 64. **The pair is not decisive; rule 4b stays FAIL.**

Per-reading data (`{off,on}/soak-4b.readings.jsonl`; summary by `readings_summary.py`):
- **The ~500 MiB idle swing is not reclaim, and is outside the container's cgroup.** In OFF's checkpoint 2, 7 readings
  sit at 14,350 MiB used and the last 3 at 13,848: `MemFree` steps up by 502 MiB at one reading while buff/cache
  (19,380 MiB), AnonPages, Shmem, and the container cgroup's `memory.current`, anon and file pages stay constant and
  pgscan / pgsteal do not move (no reclaim in either run; the cgroup is far below its 96 GiB limit, 6.3-6.6 GB).
  Memory that is not page cache, not process anonymous memory and not charged to the container was released at idle:
  on Thor, driver (GPU) memory taken from system memory. Window 4's first two checkpoints showed the same signature (503
  and 496 MiB). It occurs on the OFF build too.
- **The growth between checkpoints has the same form on both builds:** the cgroup's anon rises ~170 MiB from the first
  checkpoint to the second (OFF 432 -> 598, ON 421 -> 602) and then holds (OFF 645, 601; ON 590, 591), with host-wide
  AnonPages moving alike (OFF 1,968 -> 2,135; ON 1,956 -> 2,138).
- **The end states are equal:** the last checkpoints read 13,844.8 (OFF) and 13,845.8 MiB (ON), 1.0 MiB apart; ON's
  larger growth figure comes from its first checkpoint sitting 12.2 MiB lower and a high second checkpoint (13,914.8).
- 429 retries, +21% on ON with 20 fewer replies: noted, not explained by this data. The counts track how long the 12
  clients spend retrying a full queue; ON's checkpoint drains were longer (83-94 s against 5-64 s), so long requests
  (the 130 s budget-truncated TTS case, the 4K images) were in flight at different times. One sample per build.
**Status:** rule 4b FAIL as measured (window 4: ON +83.0 MiB; window 5: OFF +57.3 pass, ON +70.5 fail, not decisive).
The per-reading data show nothing specific to tcgen05; I do not claim more than the table allows.

## Amendment 2 (2026-11-17, Tripp's decision; set before any new data): rule 4b's memory gate, for new runs only
Windows 4 and 5 stay recorded as **FAIL** (window 5's pair not decisive). For runs made after this amendment:
- **Gated (each bar 64 MiB, per run):**
  1. the server process's RSS growth over the idle checkpoints (the harness's `memory_idle.server_rss`, first to last
     checkpoint; the clause and its bar unchanged);
  2. the container memory cgroup's **anon** growth from **checkpoint 2 to checkpoint 4** (each checkpoint's value the
     median of its 10 idle readings, from `soak-4b.readings.jsonl`), after the first-call warm-up that windows 4-5
     showed on both builds (~170 MiB of anon between checkpoints 1 and 2, then flat).
  Both are figures the engine controls.
- **Equivalence (gated):** for each gated figure, ON's growth minus OFF's growth is under **32 MiB** (ON may be lower by
  any amount).
- **Every other clause of the harness still gates**, unchanged: every reply identical to its reference, no 5xx or
  transport error, both processes alive, the crowd identical, exit 0.
- **Reported, not gating:** `host_used_less_client` (the harness's own memory verdict), because on Thor it includes
  driver memory that moves ~500 MiB at idle outside the container (window 5: MemFree +502 MiB at one reading with page
  cache, process memory and the cgroup constant) and OFF alone grew 57.3 MiB against 64; and the client's RSS.
- **Retries, reported:** 429 retries per completed reply for each build. If ON's ratio exceeds OFF's by more than 25%,
  the cause is investigated before integration; it does not gate.
- **The run:** a fresh pair in reverse order, ON first then OFF (`tools/thor/quiet_tcgen05_4b_pair2.sh`), same harness,
  cases, duration (3,600 s), clients (12) and text share (0.25); the verdict computed by
  `tools/tcgen05_4b_pair_verdict.py` from the two runs' records. Rule 4b passes if both runs pass every gated item
  above and the equivalence holds.

## Amendment 2a (2026-11-17, Tripp's decision; before the reverse-order pair runs)
- **Equivalence, restated:** for each gated figure (server RSS growth; cgroup anon growth, checkpoints 2 to 4), ON's
  growth must be **at most max(OFF's growth, 0) + 32 MiB**. A shrinking OFF no longer counts against ON.
- **What prompted it, plainly:** after writing amendment 2 I ran `tools/tcgen05_4b_pair_verdict.py` on window 5's
  records, as a check that it reads every field correctly. This was a field check, not a judgment of those runs:
  amendment 2 applies only to new runs, and windows 4 and 5 stay recorded as FAIL. The check showed that the one-sided
  "ON minus OFF < 32" would have failed on server RSS only because OFF's RSS shrank (ON +7.4 MiB, OFF -29.1 MiB,
  difference 36.5), not because ON grew. 2a is the only change that check prompted; every other part of amendment 2
  is unchanged.

**Rule 3, the 4,096-token racecheck on the host (2026-11-17 07:12-07:25, `build/tgemm-r3-host/`): NOT COMPLETED.**
Started after the audio round's racecheck (07:12:15), alone; the plain host run of the request passed (exit 0). The
racecheck run's memory grew about linearly, ~8 GiB a minute with no plateau: MemAvailable 118.3 GiB at the start,
95.4 at 2 min, 70.3 at 5, 46.0 at 8, 21.8 at 11, **6.9 GiB at 13 min (781 s)**, when the guard stopped it with
SIGTERM as set (under 8 GB). It had used over 110 GiB and was still growing; in the container it was OOM-killed at
100 GB. Racecheck's host-side tracking of a 4,096-row prefill does not fit on Thor (125 GB). No hazard was reported
before the stop (the log ends at the sanitizer banner).
**Rule 3 status: every check run is clean with outputs identical and the controls detected; the racecheck x 4,096-
token prefill cell is not completed (infeasible on this machine: >110 GiB and growing).** The 1,024-token racecheck
(clean) stays information only, per Tripp's decision; the cell is left for Tripp.

## Amendment 3 (2026-11-17, Tripp's decision): rule 3's text-prefill racecheck at 2,048 tokens
- **The 4,096-token racecheck is recorded NOT COMPLETED, infeasible on Thor:** in lfm3-profile-v1 the memory cgroup's
  OOM killer stopped it at 100 GB of anonymous memory (96 GiB limit); on the host (125 GB, no cgroup) the 8 GB guard
  stopped it after 781 s, MemAvailable 118.3 -> 95.4 -> 70.3 -> 46.0 -> 21.8 -> 6.9 GiB at 0/2/5/8/11/13 minutes
  (over 110 GiB used, ~8 GiB a minute, no plateau).
- **The racecheck gate becomes the largest prefill that fits: 2,048 tokens** (M = 2,048: the LM prefill's
  projections on tcgen05 over many 256-row tiles). Racecheck's hazards are shared-memory races within a block, which
  the number of tiles does not change. memcheck, synccheck and initcheck keep their 4,096-token results (clean).
- **The 1,024-token result stays information only.**
- Run on the host with the same guard (SIGTERM, recorded NOT COMPLETED, if MemAvailable drops under 8 GB), after
  quiet window 7 finishes, by `tools/thor/tgemm_host_race2048.sh` (a waiter: window 7's QUIET-V7-FINISHED, then no
  compute-sanitizer running); results in `build/tgemm-r3-host2048/`.

## Rule 4b pair 2 results (window 7, 2026-11-17, ON first then OFF; `build/quiet-tcgen05-4b-pair2/`): rule 4b FAIL, only because the OFF baseline failed crowd identity
Verdict by `tools/tcgen05_4b_pair_verdict.py` under amendments 2 and 2a (`verdict.json`):
| | ON (`tgemm-on1`, server `0193b11b`) | OFF (`tgemm-off1`, server `54ee76ac`) |
|---|---|---|
| non-memory gates | all true | **crowd identity false** (all others true) |
| crowd after, unstreamed / streamed | 56/56 / 56/56 | **52/56** / 56/56 |
| replies; in-soak differ | 3,398; 0 | 3,406; 0 |
| server RSS idle growth (gate <= 64) | +4.0 MiB, pass | -38.1 MiB, pass |
| cgroup anon at checkpoints 1-4 (MiB) | 418.0 / 554.7 / 641.4 / 584.1 | 447.5 / 563.6 / 671.2 / 587.5 |
| cgroup anon growth, checkpoints 2 to 4 (gate <= 64) | +29.4, pass | +23.9, pass |
| run verdict | **pass** | **fail** (crowd) |
| reported only: `host_used_less_client` growth | +244.2 MiB (the harness's own memory verdict false) | -6.2 MiB |
| reported only: 429 retries per reply | 16.06 | 20.22 |
- **Equivalence (2a, ON's growth <= max(OFF's growth, 0) + 32):** server RSS: ON +4.0 against the bound
  max(-38.1, 0) + 32 = 32: **pass** (ON minus OFF is +42.1; 2a gates ON's growth against the bound, not the
  difference, so the script's pass matches 2a as written). cgroup anon: ON +29.4 against the bound 23.9 + 32 = 55.9:
  **pass**.
- **Retries:** ON/OFF retries per reply 0.795 (ON lower): no investigation.
- **rule_4b_pass = false, only because the OFF run failed crowd identity.** ON passes every gate.
- **OFF's crowd failure is a recorded failure, cause not yet known.** The four unstreamed cases that differed:
  `budget-truncated-s1`, `numbers-normalized-s1`, `image-mirror`, `image-noise`; streamed 56/56; in-soak differ 0. The
  same OFF binary had the crowd 56/56 in window 5. The harness's crowd check counts a non-200 as a differ and saves
  neither the status nor the reply, so a 429 after its retries run out (3,000 x 10 ms) cannot be told apart from a
  content difference in these records. The baseline is main's path (no tcgen05); Tripp is investigating it on main.
  Nothing rerun.
- **Reported, not gated, and noted:** ON's `host_used_less_client` grew +244.2 MiB (amendment 2 moved it out of the
  gate as including driver memory that swings ~500 MiB at idle; window 5's OFF alone grew 57.3). Not analysed further
  here; its per-reading records are in `on/soak-4b.readings.jsonl`.
**Rule 4b status: FAIL as recorded** (window 4 FAIL; window 5 not decisive; pair 2 FAIL on the OFF baseline's crowd,
cause unknown, ON passing its gates). The build option stays OFF.

**Rule 3, the 2,048-token racecheck on the host (amendment 3; 2026-11-17, after window 7; `build/tgemm-r3-host2048/`):
clean.** Binary `tgemm-on1` lfm3-speech (`7e3da591...`), pinned runtime libraries. Plain host run 11:14:16-11:14:20,
exit 0. Racecheck 11:14:20-11:24:20, exit 0, 600 s: **0 hazards (0 errors, 0 warnings); all 5 outputs identical** to
the plain run. MemAvailable 118.6 GiB at the start, low 53.3 GiB (11:18:20), 118.0 GiB after; the 8 GB guard never
came near.
**Rule 3 under amendment 3: PASS.** memcheck (no leaks), synccheck and initcheck clean on the 4,096-token prefill and
on the vision cases; racecheck clean on the vision cases (the 8,192-patch case included) and on the 2,048-token prefill;
every output identical to its plain run; both positive controls detected. Recorded beside it: the 4,096-token racecheck
NOT COMPLETED (infeasible on Thor), the 1,024-token racecheck information only.

## Standing of the round (2026-11-17)
| rule | verdict |
|---|---|
| 1 GEMM exactness at the entry point | **PASS** (353/353, 0 bytes differing) |
| 2 every byte-identity regression | **PASS** (all identical; the batch-server and stream checks passed on their quiet rerun) |
| 3 sanitizers (amendment 3) | **PASS** |
| 4a synthetic tensor-memory soak (amendment 1) | **PASS** (685,356 calls, 0 differ, worst 5.72x the soak median) |
| 4b server soak | **FAIL as recorded**: window 4 FAIL, window 5 not decisive, pair 2 (amendments 2, 2a) FAIL **only because the OFF baseline failed crowd identity** (4 unstreamed cases, cause unknown; ON passes every gate). The baseline is main's path; under investigation on main (crowd-identity-v1, a probe queued in window 9 after window 8). |
| 5 speed | **PASS** (window 3) |
| 6 coherency | **PASS** |
**`LIQUID_LFM3_TCGEN05` stays OFF** until rule 4b passes.
- Pair 2's OFF crowd failure, not reproduced (main `dff4bf1`, `validation/crowd-identity-v1/README.md`, window 9): on the same OFF binaries (server `54ee76ac`, worker `b211b296`), a 600 s soak (584 replies, 0 differ) then 20 crowd rounds, every one 56/56 identical both ways (2,240 replies). Under that probe's reading set beforehand nothing is concluded: window 7's four differences stay a recorded failure of unknown cause, and the state after an hour of load was not tested. Rule 4b stays FAIL as recorded; the option stays OFF.
