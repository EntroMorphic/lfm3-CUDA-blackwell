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
