# Text no longer waits for speech to drain

Tripp (2026-11-11): "fix the text wait next round"; (2026-11-13) "Let's proceed with the 1..4 items you listed"
(item 4: the text-wait round, design B, with the timeout defaults, closed by its own soak).

The defect (audio-assurance-v1, server red-team finding 5; soak rule 4's open part): one engine-owner thread makes
every engine call, and while any speech row runs it makes only speech calls. A text request waits for every row
running when it arrived to finish, and a text client waits twice (create, then generate). In the assure-s4 soak, 3 of
215 text requests were never admitted. The analysis, the designs considered and an on-paper red-team are in
`audit/TEXT_WAIT_DESIGN.md`. This round builds design B: text generation sliced into `text_begin` and `text_step`,
interleaved one text token per speech step while speech rows run, and short session operations run between speech
steps.

## Builds
- Baseline: assure-s4 (`liquid-lfm3-server` c2fe0ac7…, liquid-CUDA 23c840f) and assure-a3 (`lfm3-speech` 7f68161a…).
- Under test: twa-N (server), built from this round's sources. `lfm3-speech` does not compile the worker or the
  server, so it should be unchanged; if its sources change, it is rebuilt and rule 1's engine checks rerun on it.

Thor must be quiet for every timing rule (rules 2–5, 7's soak), with nothing else on the GPU. Speed is labeled
decode loop or end to end. Every failure is recorded where it happens.

## Rules (set before any data, including the baseline)
1. **Exact.**
   - Text: the soak's 3 text bodies, the text gate's prompts and a sampled body (temperature 0.528, seed 42) give
     `generated_ids`, text and `last_logits` (`return_logits`) bit-identical:
     - alone on assure-s4;
     - alone on twa;
     - under 8 streaming speech rows on twa.
   - Speech: all 42 cases, unstreamed and streamed, with every event identical to alone, while a text client
     generates continuously.
   - The same check with `CUDA_LAUNCH_BLOCKING=1`.
   - `lfm3-speech` regression 36/36, engine exactness 41/41, the text gate and the depth tripwire are unchanged.
   - Pass: 0 differences.
2. **Text no longer waits for a drain (end to end).**
   - Setup: 8 rows of long-form TTS streaming, each with at least 60 s left. A new client then creates a session and
     generates 128 tokens of the reflective prompt (stops disabled).
   - Pass: the median of 10 arrivals is at most 3.5 × its median alone and at most 10 s, and no arrival exceeds 15 s.
   - Today's code must fail this rule. It is run on assure-s4 first, and its failure is recorded.
3. **Speech stays real time under text load.**
   - The stream check's real-time rule, at 8 concurrent streams (TTS and interleaved), while one text client
     generates back to back (128-token requests).
   - Pass: no player runs dry.
   - First audio of a TTS request arriving under that load: median at most 0.6 s over 10 arrivals (0.445 s without
     text load, validation/speech-batch-v1/README.md).
4. **Speech is not delayed by a long text generation.**
   - Setup: one text generation of 2,000 tokens is running; a TTS request arrives.
   - Pass: its first audio arrives within 1.0 s. Today it waits for the whole generation, so this fails on assure-s4.
5. **Nothing slower alone.** Paired medians, twa against assure-s4, interleaved runs, 5 pairs each:
   - text alone, end to end (one 128-token generation through the server): ratio at least 0.99;
   - speech through the server alone, 1 request and a crowd of 8 (audio seconds per wall second): ratio at least
     0.99.
   - `lfm3-speech` (engine, decode loop) at 1 and 8 rows: identical binary, or ratio at least 0.99.

   Speed round 2 failed on a 2.8% text slowdown, so 1% is the line.
6. **Deadlines and cancellation.**
   - Server started with `--request-timeout 20`; a text request arrives behind 8 rows that each have 60 s left.
     Pass: 200. Today: 504 `queue_deadline`.
   - A text client that resets mid-generation: its session's cache is reset, and the next generate on the session is
     identical to a fresh one.
   - A text request that passes its deadline mid-generation: cancelled at the next slice; its reply is 504 and the
     cache is reset.
   - SIGTERM with a text generation active and 8 rows running: exit 0, and every reply is either complete or an error.
7. **Hostile input and soak.** The audio-assurance-v1 fuzz (rule 5) and a 2 h soak (rule 4), unchanged except that the
   soak's text share rises from 1 in 10 to 1 in 4. Same pass rules, the text requests included: every text request is
   admitted and answered.
8. **Sanitizers.** memcheck, racecheck, synccheck and initcheck on an engine test that interleaves `text_step` and
   `speech_step` (4 speech rows and one text session). Pass: 0 errors and outputs identical, with the accepted
   initcheck exception for unloaded rows only.
9. **Timeout defaults** (the audio-assurance-v1 finding):
   - After this change, the engine timeout watches slices: one speech step, one text slice or one prefill. The
     defaults stay 300 s and 330 s; the round records the longest single call measured.
   - The session idle timer must not count time a session spends queued or generating. A test idles a session for
     longer than the idle timeout behind work and expects 404; one queued for that long expects its reply.

**Tests that fail on today's code, written first:**
- `tests/test_lfm3_server_text_wait.py`: rules 2, 4 and 6 (first case) as fast black-box checks on the harness with
  short speech (`--speech-rows 2`).
- A scheduler test on `lfm3_http_harness` (its fake backend records the call order): with rows running and a queued
  `generate`, the recorded calls interleave `speech_step` and `text_step`.
- `native/tests`: `text_begin` + k × `text_step` equals op `generate`, bitwise (ids and logits), for greedy, sampled,
  full prefill, reuse and cancelled requests, with k = 1, 3 and 256.

**Behavior changes this round makes (documented in the server's header):**
- A text request that passes its deadline while generating is cancelled at the next slice (504, cache reset). Before,
  it ran to its end and its cache stayed valid.
- An active text generation leaves the queue, as an admitted speech row does, so one more request is in flight than
  before at the same `--queue-size`.
- `generate_batch` stays one engine call and keeps the drain rule.
