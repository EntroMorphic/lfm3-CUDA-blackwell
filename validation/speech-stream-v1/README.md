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
