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
