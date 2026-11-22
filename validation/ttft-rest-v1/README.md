# ttft-rest-v1: the time to first token that is neither log_mel, the encoder nor the prefill

**Goal.** Find and cut the part of audio time to first token that no stage accounts for.
- After mel-tiered-v1, at 90 s: from the request's first byte to its first text event, 0.2784 s (`build/mel3-run/` B1).
- Of that: the engine's `prefill_seconds` (encoder and language model) 0.185 s, and log_mel 0.0365 s. That leaves **0.057 s**.
- The same remainder is 0.026 s at 35 s and 0.002 s at 2 s, so it grows with the clip's bytes: about 15 ns per byte of the 3.84 MB base64 body.
- Separately, the client's own base64 and JSON encoding is 0.010 s at 90 s (from the file against from the first byte).

Tripp's go: 2026-11-21 ("Proceed").

## What the handler does with a clip (read from the code before any measurement)
For a `/speech` request with one audio part, on the request's handler thread, before the engine:
1. `strict_json` (`lfm3_http.hpp`): `valid_utf8` over the body, a nesting-depth scan, a copy, then `Json::parse`.
2. `parse_messages`: the clip's string copied into the message part.
3. `encode_speech`: **`clip_from_base64` decodes the whole clip (base64, WAV, samples to float) only to count its samples** for the token count. The samples are then dropped.
4. `compute_audio_features`: a copy of the clip, **`clip_from_base64` again**, then `log_mel`.
5. The feature cache's `put`, keyed by the clip's base64 text.

## Stage 0: measure (a measurement, no rule; set before the data)
- **`audio-handler-bench`** (`native/tests/audio_handler_bench.cpp`): each step above timed on its own, with the server's own functions, on the body `measure_lfm3_audio_ttft.py` sends. One thread; median of 20; the five clips.
  - Built as the engine builds the header: nvcc's host pass, `-O3`.
- **In the same quiet window:** one pass of time to first token of the deployed server (`merge3-srv`, 1 untimed + 5 timed).
- **The accounting:** remainder = from the first byte − `prefill_seconds` − the handler's sequence (steps 1–5, log_mel included). The remainder is the body's receipt, the hand-off to the engine and the first text event's emission.
