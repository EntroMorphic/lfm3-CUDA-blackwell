# Streaming speech out of POST /speech

Tripp (2026-11-10): "Yes, do streaming next." Today `/speech` answers only when the whole reply is generated: a
paragraph of TTS waits for every sentence, a spoken answer for its last word. Streaming sends the audio as it is made.

## Design (written before the code)
**Where audio becomes final.** The engine hands emitted frames to the codec 12 at a time (lfm3_speech.cuh); each such
call makes 12 x 1920 samples final (the stream's first call trims its start, and only the finalizing call releases the
last 480). So audio exists every 12 frames, about 0.96 s of speech, inside a sentence as well as at its end. Streaming
at that granularity serves split TTS and unsplit replies alike, including interleaved spoken answers to spoken questions.

**Engine.** `lfm3_speech::Options` gains observers that the generation loop calls: `on_audio(samples, n)` after each
codec call with the samples it made final, and `on_slot(slot)` for each emitted slot (the worker forwards all but the
speech-output slot `<|audio_out|>`, which is not text; named `on_text` here at first, corrected after the red-team). `run_chunks` calls `on_chunk(k, info)` as each chunk ends. Observers only read: the samples, slots and
frames are the ones the non-streamed reply concatenates, in the same order.

**Worker op.** `"stream": true` on op `"speech"` makes `WorkerCore` write progress lines through an in-process hook
before its reply: `{"audio": base64 of 16-bit PCM}`, `{"text": [ids]}`, `{"chunk": {"frames", "slots",
"ended_by_eos"}}`. The 16-bit conversion is `wav_bytes`'s own, factored out, so the PCM is the WAV's data section. The
final reply is as before without `wav_base64`. A stream request without the hook (the standalone worker on pipes) is
refused.

**HTTP.** `/speech` takes `"stream": bool` (default false; any mode). With it, the reply is `application/x-ndjson`
(HTTP/1.0, no Content-Length, the body ends when the connection closes), one JSON object per line:
- `{"event": "audio", "chunk": k, "pcm_base64": ...}`: 24 kHz mono 16-bit little-endian PCM; all of them in order are
  the non-streamed WAV's samples;
- `{"event": "text", "chunk": k, "text": ...}`: the next piece of the decoded text (interleaved and text modes),
  held back while it ends inside a UTF-8 character;
- `{"event": "chunk", "chunk": k, "text" (split only), "frames", "ended_by_eos"}`: chunk k has ended;
- last, `{"event": "done", ...}`: every field of the non-streamed reply but `audio_wav_base64`; or, if the engine fails
  after the stream began, `{"event": "error", "error": code}`.

The status line is sent with the first event, so every refusal and every failure before the engine's first output is
an ordinary JSON error with its status, as today. If the client goes away mid-stream (a send fails or times out), the
request is cancelled at the engine's next step and the engine is free for the next request. Events queue between the
engine thread and the connection's handler thread; the engine never waits on a client.

## Rules (set before any streamed run)
Run on Thor (MAXN, clocks locked, power recorded), one warm server, each request sent streamed and non-streamed.
Cases: every request of `build/speech-e2e-v1/requests` (TTS fox, museum, numbers and long, interleaved fridge,
reflective and study; seeds 1-3; 30 requests, miscounted as 27 when this was written), the two long-form split cases of check_lfm3_speech_server.py
(lighthouse, bread), the normalized numbers paragraph of check_lfm3_audio_in_server.py, the five spoken questions of
the audio-input listening set answered interleaved (`q0`-`q4`) and three LibriSpeech clips in text mode (ASR).
1. **Exact.** For every case: the streamed PCM equals the non-streamed WAV's data byte for byte; the done event equals
   the non-streamed reply field for field except `audio_wav_base64`, `prefill_seconds` and `generate_seconds`; the
   text events joined equal the done event's text.
2. **First audio soon** (end to end at the client, loopback, from the request's first byte to the first audio event):
   median at most 1.0 s and maximum at most 1.5 s over the TTS cases; median at most 1.5 s over the interleaved cases
   (their replies start with text).
3. **No gaps.** A player starting at the first audio event and playing at 24 kHz never runs dry before the done
   event, in any case (no added buffering).
4. **Nothing else moves.** Non-streamed replies stay byte-identical to the previous build's (check_lfm3_speech_server.py,
   check_lfm3_audio_in_server.py), lfm3-speech's regression check stays 36/36 bit-identical, and the text gate passes.
5. **Cancellation.** A client that closes the connection after its first audio event: the generation stops early
   (the server's next request starts within 1 s) and the next request's reply equals its reference.
6. **Cost.** On the lighthouse case, streamed `generate_seconds` within 3% of non-streamed (median of 5 each).

Host tests (fake engine, plain, ASan+UBSan and TSan) cover the protocol: framing, ordering, errors before and after the
start, disconnect, shutdown mid-stream, and the refusals.

## Rule 6 re-measurement (written before running it)
On server-stream-v2 (after the red-team's fixes), rule 6 failed: streamed median 11.24 s against 10.77 s, ratio 1.044,
while the unstreamed runs alone spread from 10.43 to 12.09 s (on server-stream-v1 they spread 10.41-10.43 s and the
ratio was 1.005). Clocks were locked at MAXN before and after. The failure stands as recorded. To tell noise from cost,
one more measurement, set now: the same lighthouse request on server-stream-v2, 15 pairs, each pair one unstreamed and
one streamed run with the order alternating pair by pair; the cost is the median over pairs of streamed / unstreamed
`generate_seconds`. Streaming costs at most 3% if that median is at most 1.03.
**Result:** passes. The median over 15 pairs of streamed / unstreamed is 1.004: medians of 10.60 s streamed and 10.51 s
unstreamed (`build-v2/cost-remeasure.json`, clocks locked at MAXN before and after). Pairs 5 to 8 ran about 12 s
whether streamed or not, the same kind of slowdown the 5-run median caught.

## Results
On Thor (MAXN, clocks locked, power recorded): lfm3-speech stream-v1, server-stream-v1 (first build), then
server-stream-v2 (after the red-team's fixes; sources identical to this commit, `build-v2/sources-server.sha256`;
lfm3-speech's sources unchanged). Binary sha256 in `binaries.sha256`. 41 cases: 30 e2e requests, the lighthouse,
the bread, the normalized numbers, 5 spoken questions and 3 ASR clips.

| rule | server-stream-v1 | server-stream-v2 |
|---|---|---|
| 1. exact (PCM, done event, text, chunks) | 41 of 41 | 41 of 41 |
| 2. first audio, TTS median / max | 0.266 s / 0.315 s | 0.267 s / 0.338 s |
| 2. first audio, interleaved median | 0.417 s | 0.409 s |
| 3. no gaps (a player starting at the first audio never runs dry) | 0 stalls; least slack 0.51 s | 0 stalls; least slack 0.51 s |
| 4. nothing else moves | pass (below) | pass (below) |
| 5. a client that leaves: next request's first audio, delay | +0.53 s, reply exact | +0.06 s, reply exact |
| 6. cost, lighthouse generate_seconds | 1.005 | 1.044, **fail**; re-measured 1.004 (above) |

- **Rule 4, nothing else moves:**
  - lfm3-speech's regression check: 36 of 36 outputs bit-identical.
  - The server's TTS WAVs equal the validated ones; the audio-input server cases pass; the full text gate passes. All on
    both server builds.
  - The audio-input edge cases are token-identical to HF (9 of 9), and the 22 refusal and rendering checks pass, on
    stream-v1.
- **Rule 5, before and after the fixes:** on v1 a client that left was noticed only at the next send. v2 also watches
  the socket for the peer closing, which takes the delay from 0.53 s to 0.06 s.
- **Speed:** the first audio arrives about 0.27 s after a TTS request and about 0.41 s after a spoken or written
  question answered in interleaved speech, whether the reply is one sentence or 39 s of speech (both end to end at the
  client). Generation runs ahead of playback, about 3.6 times faster for TTS (lighthouse: 38.2 s of audio in 10.4 s)
  and about 2.5 times for interleaved (fridge: 25.4 s in 10.2 s), so playback never waits.

**Host tests:** 727 passed and 2 skipped: the earlier 661, plus the streaming suite's 22 tests on each of the plain,
ASan+UBSan and TSan builds. The fake engine's tests cover framing, order, errors before and after the start, deadlines,
shutdown, a client leaving before and after its first event, concurrent streams and every refusal. The one failure is
the known `test_speech_text.py::test_every_validation_prompt_matches_hf_ids`, which needs a newer transformers than
this workstation has. Across all of `tests/`, `test_text_batch_cancellation.py`'s 46 UBSan cases fail to compile on
this workstation (an unused `base64` in the worker, `-Werror`). They fail the same way at fba8e30, before this change.

## Red-team (one pass, an independent reviewer with the code only)
No serious defect. Its findings and what was done:
1. A text piece held back for an unfinished character was flushed only after the call, so it arrived after its chunk's end event.
   **Fixed:** it is flushed before the end event.
2. A client that left was noticed only at a later send: while queued or in a long prefill, it was not noticed before
   its first audio was made. **Fixed:** the handler also polls the socket for the peer closing, from before the request
   is queued (POLLRDHUP: an HTTP/1.0 client that half-closes its sending side counts as gone). A client that stops
   reading but keeps the socket open still runs to the end, bounded by the step budget, as unstreamed requests do.
3. An unused variable. **Removed.**
4. The concurrency test could not tell the streams' audio apart. **Fixed:** the fake engine's audio now depends on the
   seed. Also **added** a test for a client that leaves before its first event.
5. The GPU check could crash rather than fail on an error event in the cost loop, and did not assert the e2e request
   count. **Fixed** (the assertion then showed that the README had miscounted the requests as 27).
6. The README named the observer `on_text`; the code calls it `on_slot`. **Corrected.**

## Using it
`tools/lfm3_speech_stream_client.py PORT OUT.wav --text "..."` (or `--wav QUESTION_16K.wav` for a spoken question)
prints the text as it comes, writes the WAV, and with `--play` pipes the audio to `aplay` as it arrives.

## Second red-team (a fresh reviewer with the code only), and server-stream-v3
No serious defect. Its findings and what was done:
1. **Half-closing clients:** a client that half-closes its sending side after the request (`nc -N`, socat) got an empty
   reply: the POLLRDHUP check took it for gone. **Fixed:** the socket now reports three states (open, the peer sent
   its end, broken). When the peer's end arrives before anything was sent, the server sends the status line at once. A
   half-closed client reads on. A client that is really gone answers that write with a reset, which cancels the
   request. Test added.
   - The cost: a client that closes mid-stream is now noticed at the next send (0.28 s on the GPU check, rule 5),
     not on the 50 ms poll (0.06 s on v2).
2. **Playback-speed readers:** a client reading at playback speed holds its handler for the length of the audio, so a
   few long listeners could occupy every handler. **Fixed:**
   - At most max(1, handlers / 2) streams run at once; others get 429 `stream_limit`. Test added.
   - The bundled client now reads the network as fast as the server sends and feeds the player from a thread.
3. **Streams cut without a done line:** a stream can end without a done or error line, when a reader stalls for the
   socket timeout or at shutdown after the drain. **Documented** in the API comment: clients must treat that as a
   failure. Test added: a stalled reader is cut off and its generation cancelled.
4. **Error after all audio:** a request that completes just past its deadline gets an error event even though all
   its audio was sent. That is the unstreamed path's rule (the outcome past the deadline is unknown). **Kept.**
5. **Test gaps, all fixed:**
   - A test that one client leaving leaves another stream alone.
   - The GPU check now requires every audio and text event's `chunk` to be the running chunk.
   - A GPU case long enough for the step budget to cut it (`budget-truncated-s1`: 371 s of audio, truncated, exact,
     no stalls).

Also fixed: `tests/test_text_batch_cancellation.py`'s 46 UBSan cases had failed to compile since 933e419 (the worker's
own `base64` helper was unused outside speech). The worker now uses `lfm::base64::encode`, tested equal to Python's
on lengths 0 to 300 and 1 MiB.

server-stream-v3 (sources identical to this commit, `build-v3/`):
- **Stream check:** 42 of 42 exact (the 41 cases and the budget case); first audio TTS median 0.266 s / max 0.270 s,
  interleaved median 0.417 s; no stalls; rule 5 +0.28 s, reply exact; cost 1.004.
- **Unchanged output:** the server's TTS WAVs equal the validated ones (the WAV now base64-encoded by the shared
  encoder); the audio-input server cases and the full text gate pass.
- **Host tests:** 786 passed, 2 skipped: the earlier 727, the 12 new stream tests, the encoder test and the 46 cases
  that compile again. The one failure is the known transformers-version test.

## Third red-team (a fresh reviewer, the second round's fixes only)
No serious defect. The reviewer mutation-tested the new tests: making half-close count as gone fails the half-close
test, and a 30 s socket timeout fails the stalled-reader test. Its findings and what was done:
1. **No test reached the reset branch:** with resets mapped to the half-close state, every test still passed.
   **Added** a test: a client half-closes (the reply is committed), then resets during a 2 s gap with no events. It
   must be cancelled at once, not at the next send. With the same mutation the new test fails; without it, it passes.
2. **Cap before validation:** the stream cap was checked before validation, so at the cap a malformed request got a
   retryable 429. **Fixed:** it is checked after validation (test added).
3. **Player failure:** the bundled client reported success when its player died. **Fixed:** it exits nonzero.

These changes are host-side only (the HTTP layer's order of checks, tests, the client): 181 stream and speech HTTP
tests pass on the plain, ASan+UBSan and TSan builds. The next server build (concurrent speech) reruns every GPU check.
