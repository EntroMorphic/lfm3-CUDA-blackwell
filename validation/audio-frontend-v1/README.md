# The audio front end made fast, off the engine thread, its bytes unchanged

Written 2026-11-16, before any code. Nothing below has been run except where marked as indicative. The model is
`validation/image-queue-v1/` (where the work runs) and `validation/images-speed-v1/` (the rules' form).

## What it will be
`lfm::audio_features::log_mel` (`lfm/audio_features.hpp`, audio-in-v1) turns a clip into its [128][T] log-mel
features. It is a direct DFT in double on one host thread, run by the speech engine's admission (`lfm3_speech.cuh`)
on the thread that steps every row. audio-speed-v1 found it to be about 12 ms per second of audio: 75–88% of an
audio request's prefill (0.13 s at 10.4 s of audio, 1.09 s at 90 s). While it runs, every other row's decode waits.
This round:
1. makes `log_mel` fast, **its output byte-identical** to today's;
2. moves it, in `liquid-lfm3-server`, onto the request's own handler thread before the engine sees the request, under
   image-queue-v1's waiting-slot pattern (`Service::decode_images`). The engine thread then only encodes and prefills.

Out of scope: the GPU encoder (audio-speed-v1), the WAV and base64 parsers, and the features' definition.

## Today's exactness standard (from the records and the binaries)
- **It was validated against FP64, not against HF's bytes.** audio-in-v1 rule 1 (rounds 2 and 3): on every clip, the
  native features' maximum absolute difference from the FP64 features (`tools/lfm3_audio_in_hf.py` `mel_fp64`: torch
  STFT in float64, a float64 window and filters) is at most HF's. Measured: native at most 1.1e-4, HF at most 4.4e-4
  (HF: `Lfm3OmniFeatureExtractor`, float32 FFT and matmul) on the 40 dumped clips. On the 9 edge cases native was
  equal or closer (equal, at 0, on the 1-sample clip).
- **What remains of native's difference is HF's own arithmetic choices, kept deliberately.** Native forms the window
  product in float32 (`float(x * window)`) and uses HF's float32 mel filters; then the DFT, power and mel sums run in
  double, and the result is rounded to float.
- **Since then, it has been checked byte for byte against its own recorded runs.** audio-assurance-v1 rule 8 (the
  360 + 93 stage files, mel included) and rule 2 (all 2,620 token sequences).
- **Found here: today's bytes depend on the compiler's fusion.** lfm3-speech is built by nvcc, whose host pass
  (GCC 13, `-O3`) fuses the double arithmetic. In `vimg-a5/lfm3-speech`'s `log_mel` (disassembly):
  - the DFT sums are `fmadd`/`fmsub` (`re = fma(f, c, re)`, `im = im - f·s` fused);
  - the power is one `fmul` and one `fmadd`;
  - the mel sum is `fmla` (vector FMA);
  - the Hann window is `0.5 - 0.5·cos`, fused, then rounded to float.
  So "the current path" means these fused operations in this order. A host build that fused differently
  (`audio-input-tool` is built by g++ `-O2`; not checked) could differ in the last bit. The new code makes every
  fusion explicit (`std::fma`, or the NEON `vfmaq_f64`/`vfmsq_f64` intrinsics), reproducing the shipped binary's
  operations, so the bytes no longer depend on the compiler.
- `audio-encoder-bench`'s `log_mel` (built the engine's way) reproduced the recorded audioin-v2 dump of
  4446-2275-0004 byte for byte (26,880 floats; FNV-1a `86fd42e9778863eb`).

## Options for the arithmetic (only (a1) is the design; the rest is recorded)
- **(a1) The same DFT, vectorized across bins and threaded across frames: the design.** Each bin's sum keeps its
  order over n and its fused operations. Vector lanes hold different bins (NEON, 2 doubles per lane); threads hold
  different frames. That changes no result: bins and frames are independent. The basis `cos[(n·k) mod 400]`,
  `sin[...]` becomes a precomputed [400][201] table of the same doubles (1.3 MB, built once). Today's loop is
  latency-bound (one dependent FMA chain per bin, about 1.3 GFMA/s). Independent lanes remove that bound.
  Estimate, not a measurement: several times faster per core, with a few threads well under 1 ms per second of audio.
- **(a2) The same DFT on the GPU in FP64: not chosen.** It could match bit for bit (`__fma_rn` is the same operation),
  except `log10`: CUDA's double `log10` is not glibc's, so that step would stay on the host. Thor's FP64 rate is low:
  `audio-encoder-bench --fp64` measured 65 GFLOP/s, 1/120 of FP32 (indicative: one run on an idle but shared GPU;
  the quiet figure comes from audio-speed-v1 rule 0's run). The DFT's 32 MFLOP per second of audio would take about
  0.5 ms per second of audio there. That is fast, but it is GPU time taken from the same decode it is meant to
  unblock, while (a1) uses idle CPU cores.
- **(b) An FFT in double: not chosen, the fallback.** It would be far fewer operations (400 = 2^4·5^2, mixed radix),
  but its sums run in a different order, so it is not byte-identical. It would be used only if (a1) fails rule 2,
  and only after an amendment, under the bound pre-registered in rule 1b.

## Where it runs
- **`liquid-lfm3-server` (in process):** on the request's handler thread, after the body is parsed and before
  `frontend_call`. It runs under a counting semaphore of 2 clips at a time that a request **waits** for, within its
  deadline (504 `frontend_deadline` past it), as image-queue-v1's `decode_images`. Each clip runs on at most
  `kMelThreads` = 4 threads (fixed before rule 2; changing it needs an amendment). The features (at most
  128 × 9,000 floats, 4.6 MB, for the 4 MiB body's ~90 s) travel with the job. `WorkerCore::parse_speech` and the
  speech engine's admission then use the attached features instead of calling `log_mel`.
- **The out-of-process worker and `lfm3-speech`:** `log_mel` stays where it is, now fast. All three call the same
  function.
- This needs image-queue-v1's pattern on main; the code waits for that commit.

## Rules, set before any code
0. **Measure first** (Tripp, quiet Thor):
   - audio-speed-v1 rule 0's run (`tools/thor/audio_speed_rule0.sh`) gives `log_mel`'s time and share of time to first
     token at 2.1, ~5.8, 10.4, 35 and 90 s.
   - Also, the stall it causes. The current server with 8 and with 32 concurrent rows (`--speech-rows`; if 32 speech
     rows cannot run, 8 speech rows and text /generate streams up to 32, recorded) streaming interleaved replies.
     One ASR request is admitted for each of the five clips. Record the longest gap between consecutive stream
     events on the other rows, against the same run with no clip.
   If the 90-second clip's stall is under 50 ms, part 2 of the design is dropped and this README says so before code.
1. **Exactness:**
   - a. `log_mel`'s output **byte-identical** to the current build's on all 2,620 LibriSpeech test-clean clips, the
     40 dumped clips' recorded mel files, and the 9 edge cases (`build/server-stream-v1-check/edge`). The reference
     bytes are the current source compiled the engine's way (`audio-encoder-bench --mel-only`, a tool mode added
     with this round). Also with 1, 2, 4 and 12 threads, and twice in a row.
   - b. *Only if (b), the FFT, is taken by amendment:*
     - per clip, the maximum absolute difference from a long-double reference of native's own arithmetic (the same
       float32 window and filters, every sum in long double) is at most the current path's;
     - every element is within 1 float ulp of the current path's;
     - all 2,620 transcripts are identical to audioin-v2's.
2. **Speed** (Tripp, quiet; labeled):
   - `log_mel` alone (front end only), one request, is at least **8× faster** than today at 10.4, 35 and 90 s, and not
     slower at 2.1 s;
   - time to first token (end to end), lfm3-speech and the server, is not slower than today at any length and at most
     HF's at 10.4, 35 and 90 s (HF's measured by audio-speed-v1 rule 0's method; the 2.1 s figure reported).
3. **Stall** (server, rule 0's method): at 8 and at 32 rows, for each clip, the longest gap on the other rows is at
   most the no-clip run's longest gap + the clip's GPU encode + its language-model prefill (rule 0's figures) +
   **10 ms**. In other words, the front end itself costs the other rows at most 10 ms.
4. **Every audio-in reply identical:**
   - lfm3-speech's 2,620 token sequences equal audioin-v2's (audio-assurance-v1 rule 2), and in a batch of 8 they
     equal the run alone (rule 3);
   - the 360 + 93 stage files are identical;
   - the server's audio-in check (`tools/check_lfm3_audio_in_server.py`): every reply byte-identical to lfm3-speech's,
     including with 8 and 32 audio requests sent at once;
   - the worker's replies are identical.
5. **Sanitizers:**
   - the host suites under ASan+UBSan and TSan (the threaded `log_mel`, the semaphore, cancellation and a client that
     leaves while waiting), with a test that runs `log_mel` from 8 threads at once and checks the bytes;
   - compute-sanitizer memcheck (leaks), racecheck, synccheck and initcheck on lfm3-speech with the 1-sample edge clip,
     a typical clip, the 90-second recording and a two-clip request. 0 errors, outputs identical to the plain run.
6. **Nothing else changes:**
   - lfm3-speech's regression (36), e2e and split (90 + 90);
   - images (images-speed-v1 rule 6's checks);
   - the text gate (control `7ea05068...`) and text's 429 behavior;
   - the server checks (batch, stream, TTS, images);
   - the HTTP host tests (plain, ASan+UBSan, TSan).
   - **Bounded:** peak RSS with 16 simultaneous 90-second requests is at most idle RSS + 1 GB.
7. **Red-team:** one pass by a fresh reader with the code only: deadlines, cancellation while waiting, a client that
   leaves, a body at the size limit, a thread that throws.

Failures are recorded here, not discarded. Every speed figure is labeled front end only, encode-only, decode loop or
end to end.

## The code (2026-11-16, after Tripp's go; before rule 0's run)
- `lfm::audio_features::log_mel(samples, threads = kMelThreads)` (`lfm/audio_features.hpp`):
  - the DFT's basis is a [400][204] table of the original's doubles;
  - 4 frames share each pass over it;
  - each (frame, bin) sum runs over n in ascending order, with `std::fma` where the shipped binary fused. That is the
    window, the DFT sums, the power and the mel sums. `linspace`'s fused form is written too; at start 0 it changes
    nothing;
  - frames are split across threads, at least 64 frames each, and a thread that cannot be started runs on the calling
    thread instead.
  The shipped binary's operations were read from `vimg-a5/lfm3-speech`'s disassembly of `log_mel` (all inlined,
  721 instructions). The filters, tables and `mel_to_hz` are unfused; `hz_to_mel` is folded at compile time.
- lfm3-speech and the worker call it where they did, now threaded (4 by default).
- The handler-thread move (part 2), after image-queue-v1 part 1 landed on main (`a350659`), rebased onto it:
  - **Where the features live:** `Service::compute_audio_features` puts each clip's features in a process-wide
    `FeatureCache`, keyed by the clip's base64 text exactly as the engine request carries it. The request holds a
    `Lease` on its entries until it ends.
  - **Engine side:** `WorkerCore::parse_speech` finds them and hands them on in `Options::audio_features`. The speech
    engine uses them instead of calling `log_mel`. A miss means the engine computes the features itself, the same
    bytes. Misses: the out-of-process worker, lfm3-speech, a skipped clip.
  - **Waiting:** decode_images' pattern, with `kAudioFeatureSlots` = 2.
  - **Refusals unchanged:** nothing in the new step refuses a request. A malformed message or WAV is skipped and
    refused later by encode_speech, as before.
  - **Not in the design, needed by rule 4 (8 and 32 audio requests at once, every reply as lfm3-speech's):** audio
    requests now wait for `frontend_call` within their deadline, as image requests do, instead of a 429.
    Text keeps its 429.
  - `Policy::audio_features_ahead` is on in liquid-lfm3-server only. The HTTP harness defaults it on (as the server)
    and its fake engine fails any request whose clip has no features waiting, or features unequal to `log_mel`'s
    (compared in full up to 20 s, by length beyond). `--audio-features engine` turns the hand-off off.
- Tools: `native/tests/audio_mel_check.cpp` (`tools/thor/build_mel_check.sh`, compiled as the engine compiles
  `log_mel`) and `tools/audio_mel_lists.py`.

## Rule 1 results (2026-11-16): **PASS**
The reference is `audio-mel-ref`: audio-mel-check built from the original header (commit `80bc435`), nvcc's host pass
at -O3 as the engine. Its `log_mel` compiles to the same 74 floating-point instructions as `vimg-a5/lfm3-speech`'s
(only the filter setup's scheduling differs). It reproduced the engine's recorded features byte for byte on the 40
dumped clips (`audioin-v2/native-dump`) and the 12 clips of the 9 edge cases, then wrote all 2,620 clips' features.
The new code is `audio-mel-fe-a1` (sha256 `b76c3716...`), in `build/audio-fe-r1/` on Thor:

| | clips | threads | identical |
|---|---|---|---|
| 40 dumped clips, against the engine's recorded runs | 40 | 1, 2, 4, 12 (each run twice) | 40 at every count |
| 9 edge cases (12 clips), against the recorded runs | 12 | 1, 2, 4, 12 (each run twice) | 12 at every count |
| all of LibriSpeech test-clean, against the reference | 2,620 | 1, 2; 4 and 12 run twice | 2,620 at every count |

**On x86 too** (workstation, g++ 11.4, `-O3 -march=native`, glibc's libm): the new code gives Thor's bytes on all
2,620, the 40 and the 12, at 1, 4 and 12 threads. Built without hardware FMA (`-O3`, so `std::fma` runs in software):
also 2,620 of 2,620 and 12 of 12.

Control: the original code built on x86 without FMA differs from Thor's bytes on 9 of the 2,620 clips, 1 float each
(at most 6e-8), and on 1 of the 40 dumped clips. Its bytes depended on the compiler's fusion; the explicit `std::fma`
removes that.

Host suites: `tests/test_audio_input_host.py`, 58 passed in its plain, ASan+UBSan and TSan builds:
- a TSan variant is new (clang++, ASLR off, as the HTTP suites);
- a new test runs `log_mel` at 1 to 12 threads and from 8 callers at once, on 1, 400, 16,000 and 160,000 samples, all
  byte-identical.

## Rule 5, the lfm3-speech part (before the server change)
`tools/thor/audio_frontend_sanitizers.sh audio-fe-speech-a1 audio-fe-san`. lfm3-speech built from this code (rule 1's
`log_mel`, before the rebase), on cut-1, the two-clip request, 1089-134686-0000 and the 90-second recording, every
stage dumped, 16 tokens:
- the plain run's stage files equal the recorded runs' (cut-1: 4; two-clip: 8; 1089-134686-0000: 4);
- memcheck: 0 errors, 0 bytes leaked, outputs identical (157 s);
- initcheck: 0 errors, outputs identical (112 s);
- synccheck: 0 errors, outputs identical (98 s);
- racecheck: **stopped, not passed.** It ran 12:30–13:31 and was stopped by SIGTERM (Tripp's quiet window began
  early, at 13:25) after 22 output files, with no result. It reruns from the start on the final build after the
  window. A defunct `TreeLauncherSub` (the sanitizer's launcher, exited) was left in lfm3-profile-v1's process table.

## Rule 4 and 6, the lfm3-speech part (rebased build `audio-fe-speech-a2`, sha256 `c51c9e6a...`; `audio-fe-r4/`)
`tools/thor/audio_frontend_rule4.sh`, exactness only (shared GPU):
- the 40 dumped clips: 360 of 360 output files identical to audioin-v2's (stages and tokens);
- the 9 edge cases: 93 of 93 identical;
- speech-e2e-v1: 90 of 90 identical; the split long-form runs: 90 of 90 identical;
- the corpus alone (2,620): **stopped, not run to the end.** Started 13:25, stopped by SIGTERM at 13:31 for the quiet
  window, with no comparison made. The 8-row corpus run never started. Both rerun after the window.

## Host suites (workstation)
With the server change: `tests/test_lfm3_http_audio_in.py` 75 passed (plain, ASan+UBSan, TSan), including 4 new
tests:
- 12 audio requests at once, each with its prompt as alone, a two-clip request and the same clip in two requests;
- deadlines while waiting give 504, never 429;
- text keeps its 429;
- the engine computing the features itself (`--audio-features engine`).

The whole host suite, run from the worktree: 2,158 passed, 235 skipped. The skips are suites that look for the
checkpoint beside the repository, which a worktree lacks; a rerun from a copy that has it is under way.

## Amendment 1 (2026-11-16, Tripp, after audio-speed-v1 rule 0; before any FFT code): a double-precision FFT first
**Why.** Rule 0 (audio-speed-v1's quiet run, `build/quiet-v1-audio/summary.json`) puts native time to first token at
3–9× HF's:

| clip | native | HF | native `log_mel` alone |
|---|---|---|---|
| 10.4 s | 0.173 s | 0.051 s | 0.104 s |
| 35 s | 0.485 s | 0.072 s | 0.349 s |
| 90 s | 1.20 s | 0.141 s | 0.898 s |

HF's feature extractor takes 3–5 ms. The exact direct DFT, even spread over 12 threads, stays far above that.

**Design, changed.** The DFT becomes an FFT in double:
- a 400-point real transform by mixed radix (400 = 2^4 · 5^2), its twiddles the same doubles as the DFT's table
  (`cos`/`sin` of 2π i/400);
- about 20× fewer operations per frame than the direct sums.

Unchanged: the windowed float32 samples, the power (`fma(re, re, im·im)`), the mel sums (same order, fused), `log10`,
and the float rounding. The FFT's power differs from the DFT's by about 1e-16 relative. A float output can differ
only where that moves a value across a float rounding point. The usual place is near a rounding boundary; the other is
near 0, where `(log10(mel) + 4) / 4` cancels and the float spacing is tiny. The direct DFT (option a1, which passed
rule 1) stays the fallback, and ships if the FFT fails amended rule 1.

**Threads: kept only if they still help.** Rule 2's quiet run times the FFT at 1 thread and at 4. `kMelThreads` stays
4 only if 4 threads cut the 90 s clip's `log_mel` time (front end only) by at least 20% against 1 thread; otherwise it
becomes 1.

**Rule 1, amended (exactness).** The FFT's features against the current path's (the reference bytes of rule 1) on all
2,620 clips, the 40 dumped clips and the 9 edge cases, at 1 and 4 threads:
- byte-identical, **or**, for every float that differs, an audit:
  - a reference value from a quad-precision recomputation of that output: the same float32 windowed samples and
    float32 filters, the DFT sums, power, mel sums and `log10` all in IEEE quad (aarch64 `long double`), then the same
    `(max(·, -12) + 4) / 4`;
  - the difference passes if the FFT's float is **at least as close** to the reference as the current path's
    (|FFT − ref| ≤ |current − ref|);
  - a difference where the FFT's float is farther from the reference **fails** rule 1.
- Every differing float is listed: clip, mel bin, frame, both floats, the reference, and how far it is from the
  rounding boundary or from 0.

**Rule 4, amended for clips whose features differ (only if any do).**
- Every clip whose features are byte-identical must still give byte-identical replies, all of rule 4 as written.
- For the clips whose features differ: their LibriSpeech transcripts against the current path's.
  - Every changed transcript is read (coherency).
  - Over the 2,620, the WER change against the current path has a paired-bootstrap 95% upper bound of at most
    **+0.05** points.
  - The server's replies still equal the new lfm3-speech's byte for byte.

**Rule 2, amended (speed; quiet Thor; labeled).** Time to first token (end to end, rule 0's method) at least as fast
as HF's at the 10.4, 35 and 90 s clips. The front end alone (front end only) is reported per clip at 1 and 4 threads.

**Expected, recorded before the run.** From rule 0's figures, the encoder and the language model's prefill alone take
about 0.07 s at 10.4 s and 0.30 s at 90 s, against HF's whole 0.051 and 0.141 s. Amended rule 2 can be met only if
those shrink too. The encoder may therefore need its own round with the same bar (time to first token at least as fast
as HF's): audio-speed-v1's design, under new rules. That is noted here and not started.

Rules 3 (stall), 5 (sanitizers), 6 (nothing else changes) and 7 (red-team) stand as written. The handler-thread move
stands.

## Amended rule 1 results (2026-11-16, workstation x86): **FAIL**; the direct DFT (option a1) ships
The FFT (`detail::fft`, mixed radix 4·4·5·5 in double, explicit `std::fma`, the DFT's table twiddles) behind
`Transform::kFft`. The default stays `kDft`. Refactoring the shared power/mel/log tail left the DFT path unchanged:
2,620 of 2,620 byte-identical to the reference again.
- **Byte comparison**, `audio-mel-check --transform fft` against the current path's bytes (rule 1's reference):
  - the 9 edge cases: 12 of 12 clips identical;
  - the 40 dumped clips: 38 of 40;
  - the 2,620: 2,584 identical; 37 floats differ in 36 clips (largest difference 6e-8, one float step);
  - every other float of the 253 million is identical.
- **The audit**, `native/tests/audio_mel_audit.cpp`: x86 `__float128` with libquadmath; exact quad twiddles; the same
  float32 windowed samples and filters. Over the 37 distinct floats that differ (`amendment1/audit.jsonl`, 40 lines:
  three of them are in clips that are both among the 40 and among the 2,620):
  - the FFT's float is at least as close to the quad reference on **16**, and **farther on 21**: amended rule 1 fails;
  - most are rounding ties, where neither path is truer: the median reference lies 1.2e-5 of a float step from the
    midpoint between the two floats;
  - a few are near-zero outputs, where cancellation (in weak bins, and in `log10(mel) + 4`) makes both paths miss by
    up to 1.3 float steps;
  - the FFT's 1e-16 does not move outputs toward the truth; it re-rolls ties.
- **Outcome, by the amendment:** the direct DFT, which passed rule 1, ships. The FFT stays in the tree as the audited
  candidate, not the default, so this record can be reproduced.
- **Speed seen while checking** (workstation, not quiet, indicative only): the whole 2,620-clip check at 12 threads
  took 13.6 s with the FFT against 32 s with the DFT.

**Amended rule 2 with the DFT, expected before the quiet run.** Rule 0's front end alone was 0.104, 0.349 and 0.898 s
on one thread. Even a perfect 4-thread split leaves about 0.026, 0.087 and 0.22 s, on top of the encoder and prefill
(about 0.07 and 0.30 s at 10.4 and 90 s), against HF's 0.051, 0.072 and 0.141 s. Amended rule 2 cannot pass with the
DFT. Options for Tripp, none started:
1. Features correctly rounded, as truly as the quad reference: a double-double (or compensated) transform, mel sum and
   `log10`. It would differ from the current bytes wherever they are off, but always toward the truth, which the
   amended audit counts as a pass. It is slower than a plain double FFT and likely still far faster than the DFT.
2. The same DFT on the GPU in FP64 (`__fma_rn`), `log10` on the host. Byte-identical by construction, about 0.5 ms
   per second of audio at the 65 GFLOP/s measured (indicative). It takes GPU time from the decode it is meant to
   unblock.
3. A rule change on ties (Tripp's call), e.g. accept a difference whose reference lies within a stated fraction of a
   float step of the midpoint. It would not cover the near-zero outputs where both paths miss by over half a step.

## Amendment 2 (2026-11-16, Tripp's decision: option 1; before its code): features correctly rounded, made fast
**"Exact", defined once.** Today's inputs, taken as exact numbers:
- the windowed float32 samples `float(x[n] · w[n])` (today's float32 window);
- today's float32 mel filters.

The pipeline's mathematics on them in exact arithmetic: the DFT with exact twiddles (cos and sin of 2π n k / 400),
`|X|²`, the mel sums, `log10`, the clamp at −12, then `(· + 4) / 4`, rounded once to float32 (to nearest, ties to
even). The new front end's output must be that float, for every output.

**How each output is computed.**
1. **Fast path.** A transform in double precision or better, then the power and the mel sums in double, then `log10`.
   Each output carries a rigorous bound `E` on |computed − exact| from the error analysis below. If the interval
   [v − E, v + E] contains no float32 rounding boundary, the computed value is rounded and is correct.
2. **Otherwise**, that one output is recomputed in quad precision (aarch64 `long double`, IEEE quad; `__float128`
   elsewhere):
   - the direct DFT of the bins its mel filter spans, with quad twiddles;
   - `log10` from quad series implemented here (no libquadmath), with a stated error bound;
   - rounded under the same test at quad's bound.
   If even the quad interval contains a boundary (expected never; about 1e-25 per output), the quad value is rounded
   and the case is counted and reported.

**The error bound (fast path), written down.** u = 2^-53.
- **Twiddles:** the table's `cos`/`sin` of `2.0 * M_PI * i / 400` are within δw = 24u of the exact values. That is the
  argument's three roundings (2π · 3.01u ≈ 19u, cos and sin being 1-Lipschitz) plus libm's documented ≤ 1 ulp for
  cos and sin.
- **Per stage, max-norm**, with A the largest |component| entering the stage and e its error bound:
  - twiddle multiply (two fused operations): e' = 1.415 e + A (2δw + 2.5u), A' = 1.415 A;
  - radix-4 butterfly (additions; ±i exact): e' = 4e + 8.1u A, A' = 4A;
  - radix-5 butterfly (four fused complex multiply-adds per output): e' = 7e + A (8δw + 80u), A' = 9A;
  - growth factors (1 + 4u) on each A' are carried.
- **In double**, over the stages n = 5, 25, 100, 400 from A0 = max|x[n]|: e ≤ 576,153 u A0 = 6.4e-11 A0 per
  component of every bin.
- **Power:** |P̂ − P| ≤ e (2(|re| + |im|) + 2e) + 2.02u P̂.
- **Mel sum** (filters ≥ 0, exact): Δ = Σ_k f_k |P̂_k − P_k| + (K + 1) 1.01u m̂, K the filter's nonzero taps.
- **log10:** E_L = Δ / ((m̂ − Δ) ln 10) + 8u |L̂|. The 8u covers twice glibc's documented ≤ 2 ulp for `log10`; a
  documented bound, not a proof, and rule 1's check of every output against the quad reference would expose a
  violation. If m̂ ≤ Δ, the quad path; if m̂ = Δ = 0 (silence), the output is exactly −2.
- **Clamp** (1-Lipschitz), then the addition: E = (E_L + 1.01u |L̂c + 4|) / 4, inflated by (1 + 1e-9) for the bound's
  own arithmetic.

**Why the fast path has two candidates.** With the plain double FFT, Tripp's first proposal, the bound above is
6.4e-11 A0 per bin. A mel band whose bins are 1% of A0 then has a v error near 1.4e-9: about a 5% chance per output
of falling back to quad. At 0.01% of A0, every output falls back. Weak high bands at that level are ordinary in speech,
and each quad recompute costs on the order of a millisecond. So both are coded:
- **F1:** the double FFT under the bound above;
- **F2:** the same FFT compensated (double-double: TwoSum and fma-based TwoProduct, error-free). Its bound is the same
  recurrence with u_dd = 2^-104 · 4 in place of u (and δw for the double-double twiddle table, ≤ 4 u_dd · 24). The
  quad path then falls to near never; its cost is about 5–15× the double FFT's arithmetic.

Their outputs are the same floats by construction; only speed differs. The candidate with the lower front-end time at
90 s (quiet Thor, rule 2's run) ships. Each run reports its quad-fallback count.

**Fallback to today's bytes.** The environment variable `LFM3_AUDIO_FEATURES=dft` makes every binary (lfm3-speech, the
worker, the server) use today's direct DFT, read once at startup. The default is the correctly rounded front end once
rule 1 passes; until then, today's DFT.

**Rules, amended (2026-11-16).**
1. **Exactness.** On all 2,620 clips, the 40 dumped clips and the 9 edge cases, on Thor (aarch64) and the workstation
   (x86-64), at 1 and 4 threads, for F1 and F2: every output float equals the quad reference's correctly rounded value.
   - The reference is a separate implementation in `audio_mel_audit`: libquadmath on x86, `long double` on aarch64,
     quad FFT with quad twiddles, its own interval test.
   - Every float that differs from today's bytes is listed with the reference; each must be strictly closer to it.
     That follows from correct rounding; the list makes it visible.
   - Any output the reference cannot decide at quad is reported.
2. **Speed** (as amended in amendment 1): time to first token at least as fast as HF's at 10.4, 35 and 90 s; the front
   end alone reported per clip for F1 and F2 at 1 and 4 threads; threads kept only if 4 cut the 90 s front-end time by
   at least 20%. **Expected to fail** for the reasons recorded under amendment 1: the encoder and the language model's
   prefill alone exceed HF's whole time to first token. The encoder's own round would follow under that bar.
3. **Transcripts.** lfm3-speech over all 2,620 LibriSpeech utterances with the new front end:
   - expected identical to today's token sequences except possibly on the clips whose features changed;
   - every changed transcript is read and listed with both texts;
   - corpus WER against today's: paired-bootstrap 95% upper bound at most +0.05 points;
   - against HF: upper bound at most +0.25 (audio-in-v1's rule 3).
4. **Audio-in replies.** The 360 + 93 stage files and the server's audio-in check: identical except where features
   changed, each change listed and its replies read. Every reply from a clip whose features did not change is
   byte-identical. The server's replies equal the new lfm3-speech's byte for byte.
5. **The fallback flag:** with `LFM3_AUDIO_FEATURES=dft`, rule 1's original reference bytes on all 2,620 + 40 + 12
   clips, and the 360 + 93 files, byte-identical.

Rules 3 (stall), 5 (sanitizers), 6 (nothing else changes) and 7 (red-team) of the original stand.

## Amendment 2, rule 1 on Thor (aarch64, 2026-11-16): **PASS**; rule 1 passes on both platforms
`audio-mel-a2` (audio-mel-check and audio-mel-audit, both compiled as the engine compiles `log_mel`; `long double` is
IEEE quad on aarch64), in lfm3-profile-v1 for the engine's libm, 8 workers. Binaries in `amendment2/thor/binaries.sha256`,
records in `amendment2/thor/`:
- F2 and F1 on the 40 dumped clips, the 12 edge clips and the 2,620: every float equals the reference's rounding,
  none undecided;
- quad recomputes: the same counts as x86 (F2 2 / 0 / 154; F1 43,477 / 1,734 / 2,325,008);
- the changed floats are x86's exactly: the same 29 (clip, mel, frame, old float, new float) in the 2,620, and the
  same 2 in the 40;
- F2 at 4 threads and F1 at 4 threads equal F2 at 1 thread on all 2,620.

## Amendment 2, rule 1 on x86 (2026-11-16, workstation): **PASS**
The code is `Transform::kExactF1` / `kExactF2` in `lfm/audio_features.hpp`. The default is still today's DFT, until
rule 1 passes on Thor too.

The reference is `audio-mel-audit exact`, written separately from the code under test:
- its own any-radix quad FFT (radices 2, 2, 2, 2, 5, 5) with libquadmath's `cosq`, `sinq` and `log10q`;
- its own interval test, a generous 1e6 quad ulps of A0 on every bin.

Records are in `amendment2/`:

| set | path | floats | equal to the reference's rounding | reference undecided | changed from today (all toward the reference) | outputs recomputed in quad |
|---|---|---|---|---|---|---|
| 40 dumped clips | F2 | 4,391,424 | all | 0 | 2 | 2 |
| 40 dumped clips | F1 | 4,391,424 | all | 0 | 2 | 43,477 |
| 9 edge cases (12 clips) | F2 | 259,072 | all | 0 | 0 | 0 |
| 9 edge cases (12 clips) | F1 | 259,072 | all | 0 | 0 | 1,734 |
| 2,620 LibriSpeech | F2 | 248,578,432 | all | 0 | 29 | 154 |
| 2,620 LibriSpeech | F1 | 248,578,432 | all | 0 | 29 | 2,325,008 |

- **Threads:** F2 at 4 threads and F1 at 4 threads equal F2 at 1 thread on all 2,620 clips. The audit runs each clip
  at 1 thread.
- **The changed floats.** 29 floats in 28 clips (7021-85628-0009 has 2):
  - each moves by one float step (at most 6e-8) onto the exact value's rounding;
  - the 2 in the dumped set (8455-210777-0002, 8555-292519-0007) are among them;
  - each one's quad reference is listed in `amendment2/exact-f2-all2620-ref.jsonl`.
  Rule 3's transcripts concern these 28 clips.
- **F1 against F2, as derived before the code.** F1's double-precision bound cannot decide about 0.94% of outputs (the
  weak bands), and each goes to quad. F2's compensated bound sends 154 in 249 million. The same floats either way;
  rule 2's quiet run decides between them. F2 is expected to win by a wide margin.
- Host tests (`test_audio_input_host.py`; plain, ASan+UBSan, TSan): F1 and F2 equal at 1–12 threads and from 8 callers
  at once, and `__float128` builds and runs under clang's TSan.
- **The whole host suite** (from a copy with the checkpoint beside it): 2,394 passed, 2 skipped, 22 failed, run while
  the 12-thread audits loaded the workstation (load average 14 on 12 cores).
  - 21 of the failures are wall-clock bounds in `test_speech_normalize.py`; the file passes in full on an idle
    machine.
  - The 22nd, `test_speech_text.py::test_every_validation_prompt_matches_hf_ids`, is **pre-existing**: it fails on the
    main checkout too. It is a known workstation limit (transformers 4.57.6 cannot load the checkpoint's remote
    code), confirmed by the coordinator, and unrelated to this round.

### Amendment 2 GPU checks on Thor (2026-11-16; shared GPU, exactness only)
`lfm3-speech` `audio-fe-speech-a3` (sha256 `214a576c...`), server and worker `audio-fe-srv-a2` (`9dbf1add...`,
`4af26a8d...`).
- With `LFM3_AUDIO_FEATURES=exact-f2`:
  - **the 9 edge cases:** 93 of 93 files identical to the recorded runs;
  - **the 40 dumped clips:** 355 of 360 identical. The 5 that differ are the two clips whose features changed:
    - 8455-210777-0002: only its mel file; the change does not reach the downsampler's output;
    - 8555-292519-0007: mel, downsampler, encoder and feature files;
    - both clips' tokens and replies are identical.
  - **speech-e2e-v1:** 90 of 90 identical; **the split runs:** 90 of 90 identical.
- **The driver was trimmed while running (SIGTERM to the two scripts only).** Its rule 5 pass would have repeated
  e2e, split and both corpus runs under `dft`. Rule 5 needs only today's bytes for the features and the 360 + 93
  files, and the repeat would have taken about 2.5 hours of a shared GPU. The 2,620-clip corpus pass (exact-f2)
  continued. `tools/thor/audio_frontend_amend2_rest.sh` waited for it and was in turn stopped (SIGTERM) before its
  sanitizer step, which would have run into quiet window 3. `audio_frontend_amend2_rest2.sh` ran the rest without
  the sanitizers, and they ran overnight after the windows.
- **Rule 3, transcripts (exact-f2):** **PASS.** The 2,620 utterances alone: every token sequence identical to
  audioin-v2's (`compare_speech_outputs.py --slots-only`, 2,620 of 2,620), including the 28 clips whose features
  changed. No transcript changed, so the WER is exactly today's (2.59% against HF's 2.61%) and there was none to read.
  The corpus in 8 staggered rows: 13,260 of 13,260 output files identical to the run alone.
- **Rule 4, audio-in replies (exact-f2):** **PASS.**
  - Stage files: as above (355 + 93; only the two changed clips' stage files differ, their replies identical).
  - The server's audio-in check (`check_lfm3_audio_in_server.py`): PASS in both modes, `dft` and `exact-f2`. Every
    reply equals that mode's lfm3-speech byte for byte, a repeated request is identical, the text control is
    identical, and the server exits 0.
  - 8 and 32 audio requests at once (`check_lfm3_audio_frontend_server.py`, exact-f2, `--queue-size 40 --handlers
    48`): **PASS** on the second run. All 8 and all 32 were served (none refused) and every reply was byte-identical
    to lfm3-speech's on the same conversations; the text control was identical and the server exited 0.
    `amendment2/server/`.
  - **First run failed, from a tool bug:** 8/8 and 32/32 served, 0 identical. The tool compared the server's text
    with rule 3's corpus outputs, which were made from `prompt_ids` and carry no `"text"` field, so every comparison
    was against `None`. Fixed: the tool now makes its own references, the same conversations through lfm3-speech
    `--batch` as messages, as `check_lfm3_audio_in_server.py` does, and refuses a missing reference text.
  - Coherency: I read the 32 replies (`amendment2/server/at-once-32-v2.json`): coherent transcriptions from 1.3 s to
    35 s, with the model's known slips ("Ham, to be or not to be"), all byte-identical to today's.
- **Rule 5, the fallback flag (`LFM3_AUDIO_FEATURES=dft`):** **PASS.**
  - The features of all 2,620 + 40 + 12 clips are byte-identical to rule 1's reference (audio-mel-check, 4 threads).
  - The 360 + 93 stage files are identical to the recorded runs.
- **Sanitizers (rule 5 of the original), exact-f2**, `audio_frontend_sanitizers.sh audio-fe-speech-a3 audio-fe-a2-san`
  (overnight, sharing the GPU with another agent's sanitizers):
  - plain run: the stage files of cut-1, the two-clip request and 1089-134686-0000 equal the recorded runs';
  - memcheck: 0 errors, 0 bytes leaked; initcheck: 0 errors; synccheck: 0 errors; outputs identical to the plain run;
  - racecheck: running.

### Rule 2 results (amended; quiet window 3, 2026-11-16, run by Tripp): **FAIL**, as recorded in advance; **F2 ships**
`tools/thor/quiet_audio_frontend.sh` (MAXN, GPU clocks locked, the power state unchanged start to end), `quiet-w3/`.
Bench `audio-bench-a2` (`3b92e75a...`), lfm3-speech `audio-fe-speech-a3`. Each figure is the median of 5:
- `log_mel` alone at 1 and 4 threads: front end only;
- the encoder alone: encode-only;
- time to first token: end to end (native: prefill + one mean decode step; HF: from reading the file to the first
  new token).

| clip | HF time to first token | today's DFT* | F1 | F2 |
|---|---|---|---|---|
| 10.4 s | 0.050 s | 0.080 s | 0.118 s | 0.090 s |
| 35 s | 0.072 s | 0.174 s | 0.326 s | 0.204 s |
| 90 s | 0.141 s | 0.400 s | 0.699 s | 0.477 s |
| 90 s, `log_mel` alone, 1 / 4 threads | 3–5 ms | 0.373 / 0.095 s | 1.455 / 0.395 s | 0.680 / 0.171 s |

\* "Today's DFT" here is rule 1's code: the same bytes as rule 0's front end, threaded and vectorized across bins.
- **Amended rule 2 fails in every mode**, as recorded before the run. With F2, time to first token is 1.8–3.4× HF's.
  The rest is the encoder (0.166 s at 90 s, encode-only, more than HF's whole 0.141 s) and the language model's
  prefill, beyond the front end.
- **Against rule 0** (the shipped engine, quiet-v1): F2's time to first token is 0.173 → 0.090 s at 10.4 s,
  0.485 → 0.204 s at 35 s, and 1.20 → 0.477 s at 90 s, with every feature correctly rounded.
- **F2 ships:** its 90 s front end at 4 threads is 0.171 s against F1's 0.395 s.
- **4 threads are kept:** 0.171 s against 0.680 s at 1 thread, a 75% cut (the bar was 20%).
- **The default is now F2** (`lfm::audio_features::default_transform()`), for lfm3-speech, the worker and the server.
  `LFM3_AUDIO_FEATURES=dft` restores today's bytes (rule 5). Host suite `test_audio_input_host.py` with F2 as the
  default: 70 passed (plain, ASan+UBSan, TSan).
- **The next bar is the encoder:** the encoder round under amendment 1's bar (time to first token at least as fast
  as HF's), audio-speed-v1's design under new rules, as noted; not started.
