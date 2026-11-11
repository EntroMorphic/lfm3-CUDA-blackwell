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
