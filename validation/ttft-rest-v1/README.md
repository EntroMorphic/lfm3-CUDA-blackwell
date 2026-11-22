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

## Stage 0 results (2026-11-21, Thor, quiet 22:40–22:41; `build/rest-s0/`)
- `audio-handler-bench` `ffc519bf…`, one thread, median of 20.
- Time to first token from `merge3-srv` (`96e8510c…`, deployed), 1 untimed + 5 timed.
- The service was stopped and power unchanged.

Seconds:

| clip | valid_utf8 | strict_json (all) | clip_from_base64 (one) | of which base64::decode | WAV parse | log_mel | handler sequence | from the first byte | prefill_seconds | remainder |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.1 s | 0.0001 | 0.0002 | 0.0004 | 0.0004 | 0.0000 | 0.0012 | 0.0022 | 0.0401 | 0.0370 | 0.0009 |
| 5.8 s | 0.0002 | 0.0005 | 0.0013 | 0.0013 | 0.0000 | 0.0028 | 0.0058 | 0.0482 | 0.0408 | 0.0016 |
| 10.4 s | 0.0003 | 0.0008 | 0.0023 | 0.0023 | 0.0000 | 0.0045 | 0.0100 | 0.0613 | 0.0487 | 0.0025 |
| 35 s | 0.0011 | 0.0027 | 0.0083 | 0.0082 | 0.0001 | 0.0163 | 0.0357 | 0.1278 | 0.0850 | 0.0072 |
| 90 s | 0.0030 | 0.0070 | 0.0193 | 0.0190 | 0.0004 | 0.0367 | 0.0829 | 0.2791 | 0.1856 | 0.0106 |

`parse_messages`, the clip's copy and the cache's put are each 0.0002 s or less at 90 s.

**At 90 s, the 0.057 s beyond log_mel, the encoder and the prefill:**
- **0.0386 s: the clip decoded twice.** Nearly all of it is `base64::decode`, at about 5 ns per character; the WAV parse is 0.4 ms.
- **0.0070 s: `strict_json`.** Of that, 3.0 ms is `valid_utf8`; the rest is the depth scan, a copy and the parse.
- **0.0106 s remains:** receiving the body, the hand-off to the engine, and emitting the first event. It is not measured apart here.

## Stage 1 (2026-11-21, before any code): the clip decoded once, and decoded fast
**Changes.** Each produces the same bytes and the same refusals as today.
1. **Decode once.** `encode_speech` keeps each clip's samples beside its span. `compute_audio_features` takes them instead of copying and decoding the clip again. When the features are not computed ahead (`audio_features_ahead` off), the samples are dropped as today.
2. **`base64::decode` by table.** A 256-entry table (invalid characters marked), four characters to three bytes, into a buffer sized once. It is the same function of its input: the same bytes, and a refusal for exactly the same strings (length not a multiple of 4, any invalid character, padding only at the end and only in its two legal forms, nonzero bits under the padding).
3. **`valid_utf8` with an ASCII fast path:** 8 bytes at a time while all are under 0x80, then today's code from the first byte that is not. The same result for every input.

The depth scan and `Json::parse` stay as they are (together about 4 ms at 90 s): a smaller gain for more risk, left for later.

**Rules (set before any code):**
1. **Equivalence (gated), host tests, plain, ASan+UBSan and TSan builds:**
   - the new `base64::decode` against today's (kept in the test as the reference): the same output or the same refusal on at least 1,000,000 strings. These include every padding form, every invalid character at every position mod 4, nonzero padding bits, empty and short strings, and real encoded WAVs;
   - the new `valid_utf8` against today's on at least 1,000,000 strings, plus the existing UTF-8 tests: ASCII runs of every length mod 8 around multibyte, overlong, surrogate, truncated and out-of-range sequences.
2. **Nothing else changes (gated), on Thor, the new server:**
   - the server's audio-in check, and 8 and 32 at once, identical to `mel3-speech` (unchanged: `lfm3-speech` does not take this path);
   - images through the server and the image cache (images use `base64::decode` too);
   - the TTS check; the text gate;
   - the full host suites.
3. **Speed (gated; quiet Thor, one window):**
   - `audio-handler-bench`: the handler's sequence without log_mel at 90 s at most **0.015 s** (stage 0: 0.0462 s);
   - time to first token, the new server against the deployed one ABAB on the five clips: at most the deployed at every clip, and at 90 s at least **0.025 s** saved.

## Stage 1 results (2026-11-21, Thor; `build/rest-s1/`, `tools/thor/ttft_rest_stage1.sh`)
Binaries from the stage's tree (`build/rest-src1`):
- server `rest-srv` `71a6e1d7…`, worker `6f54d6fd…`; `audio-handler-bench` `107dcf38…`;
- deployed, for comparison: `merge3-srv` `96e8510c…`.

**Rule 3: PASS** (quiet 23:03–23:05; the service stopped; MONITOR CLEAN, 89 samples, foreign processes in 0; power checked before every run).

The handler, one thread, median of 20, in seconds. The new sequence is `strict_json` + `parse_messages` + one decode + put:

| clip | valid_utf8 | strict_json | clip_from_base64 | of which base64 | the sequence without log_mel (stage 0: two decodes) |
|---|---|---|---|---|---|
| 2.1 s | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0002 (0.0010) |
| 5.8 s | 0.0000 | 0.0003 | 0.0001 | 0.0001 | 0.0004 (0.0031) |
| 10.4 s | 0.0000 | 0.0006 | 0.0002 | 0.0002 | 0.0008 (0.0055) |
| 35 s | 0.0001 | 0.0023 | 0.0007 | 0.0005 | 0.0030 (0.0194) |
| 90 s | 0.0002 | 0.0050 | 0.0017 | 0.0013 | **0.0070** (0.0462; bound 0.015) |

`base64::decode` is 15× faster (19.0 → 1.3 ms at 90 s); `valid_utf8` 3.0 → 0.2 ms. `strict_json`'s other 4.8 ms (the depth scan, a copy, `Json::parse`) stays, as planned.

Time to first token from the file, the deployed server against the new one, ABAB, in seconds:

| clip | deployed A1 | new B1 | deployed A2 | new B2 | saved | change |
|---|---|---|---|---|---|---|
| 2.1 s | 0.0400 | 0.0390 | 0.0406 | 0.0394 | 0.0011 | −2.7% |
| 5.8 s | 0.0487 | 0.0459 | 0.0493 | 0.0461 | 0.0030 | −6.1% |
| 10.4 s | 0.0627 | 0.0577 | 0.0620 | 0.0578 | 0.0046 | −7.4% |
| 35 s | 0.1316 | 0.1158 | 0.1304 | 0.1160 | 0.0151 | −11.5% |
| 90 s | 0.2873 | 0.2512 | 0.2884 | 0.2518 | **0.0364** (bound 0.025) | −12.6% |

Texts are identical in every run.

**Rule 2, the server (23:05–23:18, not timed):**
- audio in against `mel3-speech`: PASS;
- 8 and 32 at once: 8 of 8 and 32 of 32 identical to `lfm3-speech`, PASS. The 32-at-once check took 5.5 s against 4.0–4.4 s in earlier runs. It is an exactness check with no timing criterion and ran outside a quiet window, so this is noted, not judged;
- images through the server: PASS; the image cache 24 of 24 identical;
- TTS exit 0; text gate exit 0.

**Host suites** (`tests` and `audit/test_host_contracts.py`, workstation): **2,590 passed, 3 skipped, 0 failed**. That is the 2,584 before, plus this stage's 6: the two equivalences in three builds.

**Rule 1: PASS** (3d51799). In the plain, ASan+UBSan and TSan builds:
- the table `base64::decode` gives the same bytes or the same refusal as the original on 1,000,000 strings;
- the fast `valid_utf8` gives the same verdict as the original on 1,000,000 strings;
- each run had over 1,000 strings accepted and over 1,000 refused.

## Conclusion (2026-11-21)
**Every rule passes.**
- At 90 s, time to first token is 36 ms lower: 0.288 → 0.252 s in one window against the deployed server. At 35 s it is 15 ms lower.
- Every output is unchanged.

**What changed:**
- the clip is decoded once, not twice;
- `base64::decode` runs by table (it serves images as well);
- `valid_utf8` skips ASCII 8 bytes at a time.

**Where the 90 s time to first token goes now** (each part measured on its own): about 0.252 s in all.

| part | time |
|---|---|
| language model prefill | about 0.139 s |
| encoder | 0.046 s |
| log_mel | 0.037 s |
| the body's receipt, the hand-off to the engine and the first event (not split) | about 0.011 s |
| strict_json | 0.005 s |
| one decode | 0.002 s |

HF's whole time to first token is 0.151 s (measured in an earlier window).
