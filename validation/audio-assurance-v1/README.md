# Audio in and out: assurance before images

Tripp (2026-11-11): "I'd rather do some additional testing and red-teaming before moving on to images. Images and
video may take us a while so I want to be 110% certain about our audio in/out."

The engine under test is the one in the tree: liquid-CUDA 8f87d69's sources (lfm3-speech speed-a4, server speed-s2;
round 2 of the speed work was not merged). Thor, MAXN, clocks locked, power recorded at start and end.

## What has never been checked (found before writing these rules)
- **No compute-sanitizer run on the batched speech engine.** The last ones (depth-v1, codec-v1) predate batching,
  the per-row codec streams, parallel host sampling, the deferred language-model head, the staged FP32 kernel and
  the audio encoder.
- **No long run of the server with mixed audio traffic.** Every server check is a bounded crowd.
- **The full LibriSpeech run (2,620 utterances) was measured on audioin-v2.** The audio encoder has since moved to
  the staged FP32 kernel (checked bit for bit on its shapes, not end to end on the corpus). Audio input has not been
  run on the corpus in a batch.
- **No hostile-input run against the live server.** The parsers were fuzzed host-side (audio-in-v1's red-team).

Three independent reviewers, with the code and the records only, are looking for more: one on the batched speech
engine, one on the server and worker, and one auditing every audio claim against the build it was measured on. Their
findings, and any test added for them, go in a dated section below before that test runs.

## Rules (set before any run)
1. **Sanitizers clean.** compute-sanitizer memcheck (leak check on), racecheck, synccheck and initcheck on lfm3-speech
   in a 4-row batch with staggered arrivals: short TTS (both voices), an interleaved reply, a LibriSpeech
   transcription and a spoken question answered interleaved. Also run on the same list one request at a time. Pass:
   0 errors and 0 hazards in every tool, and every output byte-identical to the plain build's run.
2. **The corpus, unchanged.** lfm3-speech over all 2,620 LibriSpeech test-clean utterances, one at a time. Pass:
   every emitted token sequence equals audioin-v2's native run (`build/audioin-v2/native-full`), and so the WER
   against HF is audio-in-v1's (2.59 against 2.61). A difference fails this rule. It is then reported and scored
   (WER against HF, paired-bootstrap 95% upper bound at most +0.25, audio-in-v1's rule 3).
3. **The corpus in a batch.** The same 2,620 utterances through `--concurrent 8`, staggered. Pass: every output file
   byte-identical to rule 2's run alone.
4. **A long mixed run.** The server (8 speech rows) for 2 hours under continuous mixed traffic from 12 clients:
   - TTS, unstreamed and streamed, both voices;
   - interleaved replies;
   - LibriSpeech transcriptions and spoken questions;
   - text /generate;
   - about 1 in 10 streamed requests dropped by the client mid-stream.

   Every completed reply is compared with the same request's reply alone on a fresh server (all fields but timings;
   streamed, every event).

   Pass:
   - every completed reply is identical;
   - no 5xx;
   - a 429 is retried and counted;
   - the server and worker stay up;
   - each process's RSS at the end is at most its maximum over the first 15 minutes plus 64 MiB, and the same holds
     for the GPU memory in use;
   - after the run, a fresh crowd of the 42 cases is still identical;
   - the server exits 0 on SIGTERM.
5. **Hostile input, live.** At least 3,000 requests mutated from valid /speech and audio-input requests:
   - JSON field mutations, wrong types, huge and negative values, unicode;
   - base64 corruption;
   - WAV header and size mutations;
   - truncated bodies, slow and dropped connections.

   These are interleaved with valid requests whose replies must stay exact. Pass:
   - every reply is a well-formed 2xx (exact, for a valid request) or 4xx; a 429 or 503 counts only when documented;
   - nothing crashes or hangs (each request finishes within 120 s);
   - afterwards the 42-case crowd is identical, RSS stays within rule 4's bound, and the server exits 0.
6. **Coherency.** I read the soak's and the fuzz run's valid replies (sampled: 30 of each kind) and listen-check
   nothing new unless a byte changed. Any changed byte is a failure under rules 2–5 first.

Speed is not measured here.

## Added 2026-11-11, from the evidence audit (set before these runs)
The auditor (records only) found that every exactness check since speech-batch-v1 compares a build with itself or
with the build just before it, not with the builds on which fidelity was measured. It also found that no check since
depth-v1 reads the depth decoder's logits: forced replays force the codes, and a sampled code rarely moves at
temperature 0.4. Rules 7–9 tie the engine in the tree back to the measured builds. Rule 10 is a defect the audit
found.

7. **The depth decoder at its measured precision.** `tools/thor/depth_round.sh` at this tree (liquid-depth on the
   current depth.cuh and staged FP32 kernel) against depth-anchor-v1 and tts-trace-v2. Pass:
   - A1 and B1 pass;
   - the tripwire holds (median relative L2 from FP64 ≤ 1e-6, max ≤ 1e-5, KL ≤ 1e-9).
8. **The audio encoder's stages, unchanged.** lfm3-speech speed-a4 on audio-in-v1's 40 dumped clips
   (`build/audioin-v2/list-dump.txt`) and on its 9 edge cases (`build/server-stream-v1-check/edge`). Pass: every
   output file (features, mel, downsampler, encoder and projected outputs, tokens) equals the recorded native run's
   bytes, timings aside. Those runs were tied to HF: the stage errors 0.15–0.19 times HF BF16's, and the edge cases
   token-identical to HF.
9. **The sampled speech, unchanged.** speed-a4 on the 30 requests of speech-e2e-v1 and the 30 split long-form runs
   (speech-chunk-native-v1), on which C1 (WER against HF), C2 and the listening verdicts were measured. Pass: every
   `raw.i32`, `frames.i32` and `f32` byte-identical to the recorded outputs.
10. **Normalization only where it applies.** Today `POST /speech` in audio mode normalizes by default whatever
    language the system prompt names (`lfm3_http.hpp`), and the normalizer is American English only. A host test is
    written first and must fail on today's code. Its cases:
    - TTS requests in Spanish, German, French and Japanese with digits;
    - American English and British English;
    - no system prompt;
    - explicit `normalize: true` with a non-English language.

    The pass condition, after the fix: default normalization only when the TTS system prompt names American or
    British English (or there is no system prompt); otherwise off. An explicit `normalize: true` with another
    language is refused (400). The English defaults keep every validated reply byte-identical (the server checks of
    rules 4–5).

    The sentence splitter is English-only as well: it splits at `. ! ?` followed by a space. Whether multilingual
    TTS is in scope is Tripp's call. Until then, the splitter's default is left as is and reported.
11. **The staged FP32 kernel under the sanitizers.** fp32-linear-check (speed-a4) under memcheck, racecheck,
    synccheck and initcheck. Pass: 0 errors, 0 hazards, and the check passes.
