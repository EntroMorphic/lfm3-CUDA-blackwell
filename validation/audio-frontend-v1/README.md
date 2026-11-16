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
