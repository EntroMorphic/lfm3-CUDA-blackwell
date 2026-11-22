# mel-tiered-v1: the correctly rounded log-mel front end, made fast by tiered precision

**Goal.** Cut log_mel, the largest part of audio time to first token, without giving up the guarantee that every feature is correctly rounded (validation/audio-frontend-v1 amendment 2).
- At 90 s, time to first token is 0.425 s (audio-ttft-v1 amendment 6) and log_mel is 0.171 s of it: F2 on 4 threads. HF's front end takes 0.015 s.
- Tripp's go: 2026-11-21, after audio-ttft-v1 was merged and deployed.

## Where F2's time goes (from audio-frontend-v1's record, before any code here)
- **F2** computes every frame by a double-double FFT (error-free TwoSum and TwoProduct), about 5–15× the arithmetic of a double FFT. Its bound decides nearly every output, so quad almost never runs: 0.680 s at 1 thread, 0.171 s at 4, for the 90 s clip.
- **F1** computes every frame by a double FFT. Its bound leaves a few outputs undecided, about 0.02% on the 2,620 utterances (43,477 quad recomputes). Each is recomputed in quad, a direct 400-term DFT per tap at about a millisecond. So F1 was slower than F2 (1.455 / 0.395 s), almost all of it quad.
- Both produce the same floats: every decided output is the correctly rounded value of the same exact mathematics, and that value is unique.

## The change: F3, tiered precision
Each frame:
1. F1's double FFT, power, mel sums and bound, as today's F1.
2. An output F1's bound cannot decide takes F2's path for its frame: the double-double FFT of that frame, computed at most once per frame, and that mel's sum and bound exactly as F2 computes them.
3. An output F2's bound cannot decide takes quad, as today. Undecided even in quad: counted and reported (expected never).

Every output is decided by a bound already validated (F1's or F2's, audio-frontend-v1 amendment 2 rule 1 on Thor and x86). So **F3's floats must equal F2's, byte for byte.**
- Expected cost: about the double path's on ordinary speech, and at most about F2's plus the double path's on clips where most frames fall back (near silence).
- `LFM3_AUDIO_FEATURES=exact-f3` selects F3. If rules 1–4 pass, F3 becomes the default. `exact-f2`, `exact-f1` and `dft` still select the others.
- The thread count stays `kMelThreads` = 4.

## Rules (set before any code)
1. **Byte-identical to F2 (gated).** `audio-mel-check --transform f3 --compare` against F2's features, on Thor (aarch64) and on the x86 workstation, at 1 and 4 threads:
   - every one of the 2,620 utterances, the 40 dumped clips and the 12 edge clips identical;
   - F3's fast, double-double and quad counts reported, and undecided 0;
   - `--repeat 2` (every run equal to the first).
2. **Speed (gated; quiet Thor, the usual window).** `audio-encoder-bench` `log_mel` alone, F2 and F3 in one window, median of 5, at 4 threads, on audio-ttft-v1's five clips:
   - F3 at 90 s at most **0.060 s** (F2: 0.171 s);
   - F3 at most F2 at every one of the five clips.

   Reported: F3 at 1 thread; the edge clip with the most fallbacks, F2 against F3.
3. **Time to first token (gated; the same window).** The new server against the deployed one (`4025f693822c`'s build), ABAB on the five clips (`measure_lfm3_audio_ttft.py`, 1 untimed + 5 timed):
   - the new server's median at most the deployed one's at every clip;
   - texts identical.

   Reported: the saving at each clip.
4. **Nothing else changes (gated).** The features are the same bytes, so every downstream output must be too:
   - `lfm3-speech`'s TTS regression 36 SAME;
   - rule-6 audio-in files (360 + 93), e2e and split identical to amendment 6's x2 run (`a6-speech-rule6`);
   - the server's audio in, 8 and 32 at once, the text gate;
   - the host suites (with `test_audio_input_host.py`'s F3 cases at 1–12 threads and from 8 callers, plain, ASan+UBSan and TSan);
   - `LFM3_AUDIO_FEATURES=exact-f2` and `dft` still give their bytes.
