# ttft-rest-v2: the rest of audio time to first token at 90 s, timed point by point

**Goal.** Split, and then cut, the part of 90 s audio time to first token that no stage accounts for, plus `strict_json`'s remainder.
- ttft-rest-v1 left, at 90 s:
  - about **0.011 s** that was not split ("the body's receipt, the hand-off to the engine and the first event");
  - **0.005 s** in `strict_json` after its fast `valid_utf8`: the depth scan, a copy of the body, `Json::parse`.
- Since then, tcgen05-v2 cut the language model's prefill.
  - The deployed server (`merge5-srv`, `f19f4ee4…`) reaches the first text at about 0.186 s from the file at 90 s (pinned-uploads-v1's quiet window, A1/A2).
  - HF BF16 takes 0.151 s, from an earlier window.
  - These two parts are now about 9% of the deployed time to first token.

Tripp's go: 2026-11-23 ("Merge alloc then proceed with item 4").

## What the code does between the first byte and the first text event (read before any measurement)
The in-process server with the speech batch (`--speech-rows 8`), on a streamed `/speech` request with one clip:
1. **Handler thread, `Connection`:**
   - `read_head` reads 4,096 bytes at a time;
   - `post` reads the body 64 KiB at a time (`poll`, `recv` into a stack block, appended to `buffer_`, then appended to `body`, which is not reserved);
   - `strict_json`.
2. **`Service::speech_stream`:**
   - `validate_speech`;
   - `encode_speech` (one decode of the clip);
   - `compute_audio_features` (`log_mel`);
   - `enqueue_ticket`.
   - Then the thread waits on the request's `Lines`.
3. **Engine-owner thread, `speech_cycle`:**
   - pops the ticket;
   - `speech_admit` (in process): `Batch::admit`, whose `t0`…`t1` is `prefill_seconds`. That covers the encoder, a synchronize, the language model's prefill and a synchronize.
   - Then `speech_step`. The first text token's progress line is parsed (`strict_json`) and passed to `StreamState::event`, which pushes the text event.
4. **Handler thread:** wakes on `Lines`, `send_all`s the event line (the status line and headers go with it).
5. **Client** (`measure_lfm3_audio_ttft.py`): times "from the first byte" from just before `sendall`, and stops at the first `recv` holding `"event":"text"`.

## Stage 0: measure (a measurement, no rule; set 2026-11-23, before the data)
**A trace build.**
- `lfm/ttft_trace.hpp`: `LFM3_TTFT_MARK(name)` appends `name CLOCK_MONOTONIC-ns` to the file named by `LFM3_TTFT_TRACE`. It compiles to nothing unless `-DLFM3_TTFT_TRACE`.
- The marks:
  - handler: `h_handle` (the handler thread starts, just after `accept`), `h_head` (`post` entered), `h_body` (the body is whole), `h_parsed` (`strict_json` returned);
  - `speech_stream`: `s_validated`, `s_encoded`, `s_features`, `s_enqueued`;
  - engine owner: `o_popped` (the ticket is taken), `c_t0`, `c_encoded` (the encoder synchronized), `c_forward` (the prefill's `forward` returned, before the synchronize), `c_t1`, `o_admitted` (the admit call returned), `c_step` (each `Batch::step` begins), `o_text` (each text event, just before it is pushed);
  - handler: `h_sent` (each event line's `send_all` returned).
- **The client:** `tools/trace_lfm3_audio_ttft.py`, the same request as `measure_lfm3_audio_ttft.py`, with `time.monotonic_ns()` (the same clock) taken at:
  - `t_send` (before `sendall`);
  - `t_sent` (after it);
  - `t_first` (the first `recv` that holds a text event).
- **The intervals reported, medians of the 5 timed passes per clip:**
  - send → sent;
  - send → `h_head`; `h_head` → `h_body`;
  - `strict_json`;
  - `validate_speech`;
  - `encode_speech`;
  - `log_mel` (`s_features`);
  - enqueue;
  - the queue wait (`s_enqueued` → `o_popped`);
  - `o_popped` → `c_t0`;
  - the encoder;
  - `c_encoded` → `c_forward`;
  - `c_forward` → `c_t1`;
  - `c_t1` → `o_admitted`;
  - `o_admitted` → the first `o_text`, with the number of `c_step`s between them;
  - the first `o_text` → the first `h_sent` after it;
  - that `h_sent` → `t_first`.

  Their sum is `t_first − t_send`, the measure script's "from the first byte".
- **Cost of the trace:** each mark is one `fprintf` and `fflush` (a `write`). The trace build's "from the first byte" is reported beside the deployed server's in the same window. A difference over 1 ms at 90 s is noted against the split.

**`strict_json` split** (`audio-handler-bench`, extended; one thread, median of 20, the five clips):
- `valid_utf8`;
- the depth scan (a copy of `strict_json`'s loop, kept identical);
- the copy into `std::string`;
- `Json::parse`;
- a copy of the parsed tree: `parse` returns a `const` local, which cannot be moved.

**The window** (quiet Thor; the service stopped with an EXIT-trap restart; the 1 s sampler; `power_state.sh` before each step):
1. the bench on the five clips;
2. the trace server, then the deployed server, then the trace server, then the deployed server: 1 untimed + 5 timed passes on the five clips each (T1 A1 T2 A2).

The trace server's texts must equal the deployed server's in every run, or the split is void.

**What stage 1 will be.** Chosen from the split, as its own rules before code.
- Candidates already visible in the code:
  - the body read straight into a reserved `body` (one copy instead of three, no regrowth);
  - `strict_json`: the depth scan and `quoted`'s run scan by 8 bytes at a time, as `valid_utf8` now is; no copy of the body; `parse` returning by move.
- A cut is proposed only for an interval of at least 1 ms at 90 s.

## Stage 0 results (2026-11-23, Thor, quiet 16:47–16:49; `build/rest2-s0/`, `tools/thor/ttft_rest2_stage0.sh`)
**Builds:**
- trace server `rest2-trace` `8f47fa8a…` (a334096, `-DLFM3_TTFT_TRACE`, tcgen05 ON);
- `audio-handler-bench` `1492a7b5…`;
- deployed `merge5-srv` `f19f4ee4…`.

The marks check failed twice on the script before the run, with the build correct both times (`run-1-nomarks.log`, `run-2-pipefail.log`):
- `strings` without `-a`;
- then `grep -q` under `pipefail`: grep exits at its first match, `strings` gets SIGPIPE, and the pipeline reads as failed.

**The window:**
- service stopped 16:47:22, started again 16:49:25;
- power unchanged;
- sampler 85 samples, foreign in 0. **MONITOR: CLEAN.**
- Texts identical across T1, A1, T2 and A2 at every clip.

**The trace costs nothing measurable.** From the first byte, in seconds:

| clip | T1 | A1 (deployed) | T2 | A2 (deployed) |
|---|---|---|---|---|
| 2.1 s | 0.0378 | 0.0383 | 0.0378 | 0.0384 |
| 5.8 s | 0.0446 | 0.0448 | 0.0444 | 0.0449 |
| 10.4 s | 0.0559 | 0.0559 | 0.0558 | 0.0558 |
| 35 s | 0.0862 | 0.0864 | 0.0850 | 0.0857 |
| 90 s | 0.1770 | 0.1770 | 0.1749 | 0.1774 |

**The split at 90 s**, T1 / T2 medians of 5 passes, in ms. The intervals sum to within 0.6 ms of the total (0.1765 / 0.1749 s against 0.1770 / 0.1749 s).

| interval | 90 s | 35 s | 2.1 s |
|---|---|---|---|
| send → `h_head` (connect, request head) | 0.65 / 0.53 | 0.11 / 0.11 | 0.10 / 0.11 |
| body receipt | 2.76 / 2.39 | 1.54 / 1.26 | 0.06 / 0.06 |
| `strict_json` | 6.53 / 5.90 | 2.86 / 2.66 | 0.14 / 0.14 |
| `validate_speech` | 1.41 / 0.80 | 0.64 / 0.52 | 0.03 / 0.04 |
| `encode_speech` | 7.08 / 6.95 | 4.10 / 3.75 | 0.18 / 0.18 |
| `log_mel` | 38.85 / 38.28 | 17.78 / 17.55 | 1.30 / 1.28 |
| enqueue + queue wait + to `t0` | 0.24 / 0.24 | 0.11 / 0.12 | 0.05 / 0.05 |
| encoder (allocations included) | 41.00 / 41.62 | 19.20 / 19.18 | 7.54 / 7.21 |
| language model `forward` (call) | 76.57 / 76.78 | 38.23 / 38.39 | 27.33 / 27.39 |
| `forward` → `t1` (synchronize) | 0.98 / 0.97 | 0.98 / 0.99 | 0.94 / 0.94 |
| `t1` → the first text event (admit return, one step) | 0.36 / 0.36 | 0.24 / 0.25 | 0.21 / 0.26 |
| event → sent → client | 0.06 / 0.07 | 0.05 / 0.08 | 0.05 / 0.05 |

`steps_before_text` is 1 for every request: the first text token is sampled in the first step after the admit, and its event is sent within 0.1 ms.

**`strict_json` split** (bench, one thread, warm, median of 20, ms at 90 s):

| part | ms |
|---|---|
| `valid_utf8` | 0.23 |
| depth scan | 2.01 |
| copy of the body | 0.15 |
| `Json::parse` | 2.53 |
| copy of the tree | 0.15 |
| `strict_json` | 4.90 |

**Reading.**
- **The hand-off and the first event are not the remainder.** The queue, the admit's return, the step and the send come to about 0.6 ms at 90 s.
- **The remainder is the handler copying the clip.**
  - `validate_speech` and `encode_speech` each take 1.4 and 7.0 ms in the server. In the warm bench, `parse_messages` was 0.2 ms and one decode 1.7 ms.
  - `strict_json` is 6.2 ms against the bench's 4.9.
  - The body receipt grows with the clip: 2.6 ms at 90 s.
- **Why the server pays more than the bench.** The server fixes glibc's mmap threshold at 128 KiB (audio-assurance-v1, the soak's memory finding), so every buffer of 128 KiB or more is a fresh mapping. Each copy of the 3.84 MB clip is a memcpy plus about 940 page faults.
- **The clip's copies before `log_mel`, by the code:**
  1. the body, grown by doubling from the first 64 KiB (not reserved);
  2. `strict_json`'s `std::string(raw)`;
  3. the parsed tree's string, which is needed;
  4. `Json::parse`'s `const Json result`, copied on return;
  5. `validate_speech`'s `parse_messages`, which copies every part's `data`;
  6. `encode_speech`'s `parse_messages`, again;
  7. `audio_json += part.data`: one reallocation to the clip's size, then the closing `"` regrows it to about 7.7 MB with a second copy.
- Beyond the handler: `log_mel` 38.6 ms, the encoder 41.3 ms and the language model 77.7 ms are 89% of the 90 s total.
  - The encoder's ~10 ms of per-clip allocation is its own later round: keeping the scratch is safe only with pinned staging, pinned-uploads-v1.

## Stage 1 (2026-11-23, before any code): the clip copied once
**Changes.** Each gives the same bytes and the same refusals.
1. **The body read into a string reserved at its length** (`post`). The bytes already read and each `fill` are appended as before, with the same deadlines and errors. It no longer regrows.
2. **`strict_json` and `Json::parse`:**
   - `Json::parse` takes a `std::string_view`, so `strict_json` parses the body in place (no copy);
   - `parse` returns a non-`const` local (moved, not copied);
   - the depth scan and `quoted`'s plain run look 8 bytes at a time for a byte that ends the run. In the depth scan that is `"` or `\`; in `quoted`, `"`, `\` or one under 0x20. From that byte, today's code runs unchanged.
3. **`parse_messages(..., copy_media)`:** with `copy_media` false, an audio or image part's `data` stays empty and `media` points at the request's own string. `ContentPart::source()` gives either one.
   - `validate_speech`, `decode_images` and `encode_speech` use it. The request's `Json` outlives them: `encode_speech` drops it last, as now.
   - Every other caller copies as before.
4. **`audio_json` reserved** at its final size before the clips are appended.

**Rules (set before any code):**
1. **Equivalence (gated), host tests in the plain, ASan+UBSan and TSan builds:**
   - `strict_json` against today's (kept verbatim in the test as the reference) on at least 1,000,000 strings. The outcome must be the same tree (every value, key order and `integral` flag) or the same refusal: exception type and message.
     - The strings: random JSON documents with strings around every 8-byte alignment, with every escape, `\u` escapes (surrogate pairs valid and not), raw control characters, multibyte and invalid UTF-8, nesting to 70, and numbers at their edges.
     - Each is then changed by one byte replaced, inserted or deleted, or truncated.
     - At least 1,000 accepted and 1,000 refused.
   - `Json::parse` alone against today's, on the same strings.
   - `parse_messages` with `copy_media` false against true, on at least 100,000 random message lists with text, audio and image parts, valid and not. The same messages (`source()` equal to the copy's `data`) or the same refusal.
2. **Nothing else changes (gated), on Thor, the new server:**
   - the server's audio-in check, and 8 and 32 at once, identical to `mel3-speech`;
   - images through the server and the image cache;
   - the TTS check; the text gate;
   - the full host suites.
3. **Speed (gated; quiet Thor, one window):** time to first token, the new server (B) against the deployed one (A), ABAB on the five clips:
   - mean(B) ≤ mean(A) at every clip;
   - at 90 s, mean(A) − mean(B) ≥ **0.006 s**. The estimate is about 0.010 s; the bound is set below it for the run-to-run spread.
   - A trace build of the new tree, one pass of 1 + 5, shows where the time went (reported, not gated).
