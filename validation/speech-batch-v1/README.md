# Concurrent speech: many /speech requests at once, batched on the GPU

Tripp (2026-11-10): "...then, do concurrency next." Today the server runs one speech request at a time: a second
waits in the queue for the first to finish. Speed here means aggregate throughput at batch (as for text,
validation/batch-decode-v1).

## Where the time goes today (one request; tts-long-male-s1, speech-e2e-v1)
Per frame step about 23 ms: the language model's decode step 16.3 ms, the depth decoder 5.9 ms (3 passes of a
3-layer transformer over the frame's 12 codebook slots), host sampling 0.5 ms, codec 0.46 ms. That is about 43 frames
a second, 3.4 times real time. The batched text decoder (lfm3_batch::BatchDecoder, which speech already uses with one
row) steps 8 rows in about the time of 1 (16 ms at 1 row, 18.5 ms at 32: validation/batch-decode-v1), so batching
the language model is nearly free; the depth decoder must be batched too, or it dominates.

## Design (written before the code)
**Engine.** One speech batch of up to B rows, each row one request:
- **Language model:** the BatchDecoder with B rows: text rows and frame rows step together (it already embeds a
  frame row's codes in place of a token). Text picks are greedy per row with each row's own suppression set.
- **Depth decoder:** batched over the rows that sample a frame this step: each of the 3 reveal passes runs every such
  row's 12 slots in one call (attention and rotary positions within each row's 12). The FP32 operators are
  row-independent (fp32_ops.cuh), so a row's logits do not depend on its batch.
- **Per row:** each row keeps its own mode machine, frame sampler (seeded as today), codec stream, chunk loop (long-form
  TTS by sentences, the step budget) and observers (streaming).
- **One implementation:** the single-request Session becomes a batch of one, so lfm3-speech, the server and the batch
  share one code path.
- **Admission:** a new request joins at a step boundary: it is prefilled by the Runner (as today) and loaded into a
  free row; a row that finishes frees its slot.

**Server.** Speech requests go to a speech queue. The engine-owner thread runs a speech batch while any speech request
is running or queued. Between steps it admits queued requests into free rows, delivers each row's events (streaming)
and finished replies, and applies cancellations. While a text job is waiting it admits no new speech, so the batch
drains and the text job runs. The engine timeout watches each step, not the whole batch.

**Exactness is the bar.** Rows are independent by construction, so a request's output in a batch must equal its
output alone, byte for byte, whatever shares its batch and whenever it joins.

## Rules (set before any batched run)
Run on Thor (MAXN, clocks locked, power recorded). The single-request engine of 2ef3a89 (lfm3-speech stream-v1,
server-stream-v3) is the reference.
1. **Exact.** Every request's reply (WAV bytes, text, chunks, truncation) in a batch equals its reply alone, for these
   mixes:
   - the 30 speech-e2e-v1 requests, the 2 long-form cases, the normalized numbers paragraph, the 5 spoken questions
     and 3 ASR clips (41) sent at once, then in random arrival order with random gaps (0-2 s);
   - each twice: unstreamed and streamed;
   - one request cut by the step budget among them.
2. **Throughput** (decode loop, aggregate seconds of audio generated per wall second, the 15 TTS requests of the e2e
   set looped to keep N rows busy, after a warm-up): measured at N = 1, 2, 4, 8, 16. Pass: N = 8 at least 4 times
   N = 1's.
3. **Real time per stream:** at N = 8 concurrent streamed TTS requests, no stream's player runs dry (rule 3 of
   speech-stream-v1, per stream). The largest N with no gap is reported.
4. **First audio under load:** a TTS request arriving while 7 others stream: its first audio within 1.0 s (median of
   10 arrivals).
5. **Text is served:** a /generate arriving during continuous speech load starts within the time the running rows take
   to finish (measured and reported), and its output equals its reference.
6. **Nothing else moves:** lfm3-speech's regression 36/36 bit-identical through the batch-of-one path; the server's
   TTS, audio-input and streaming checks; the text gate; host tests.

Host tests (fake engine; plain, ASan+UBSan, TSan) cover the scheduler: admission, finishing, cancellation of one row,
deadlines, shutdown with rows running, text jobs draining the batch, streaming per row.

## Corrections to the rules' wording (made before the measured runs, recorded here)
- **Rule 2's TTS count:** "the 15 TTS requests of the e2e set" is a miscount; the set holds 21. The measurement loops all
  21, max(42, 4N) requests per run.
- **Rule 1's reference:** "alone" means the single-request server of 2ef3a89 (server-stream-v3), as the rule says.
  The first version of the server check took it from the new binary and was corrected after the red-team (below).

## Engine (lfm3-speech batch-a4; Thor, MAXN, clocks locked)
- **Batch of one:** lfm3-speech's regression check, 36 of 36 outputs bit-identical. The single-request Session is now
  a batch of one.
- **Exact under concurrency (rule 1, engine level):** each of the 41 requests in a batch writes the same WAV, FP32
  waveform, frames, raw frames and JSON (timings aside) as alone, in 41 of 41 cases, at 4 rows, 8 rows, 8 rows
  staggered (a random order, each joining 0 to 60 steps after the last) and 16 rows staggered
  (`build-a4/engine-exact.json`).
- **A row at its capacity (red-team finding 1, below):** a row runs to exactly its capacity and finishes while another
  row decodes beside it.
  - Fixed decoder: the other row's logits are the same, bit for bit, at all 62 steps as with the capped row never
    loaded.
  - Pre-fix decoder, the same check: they differ from step 56, the step the capped row finished.
  - `build-a4/capacity-check*.json`.
- **Throughput (rule 2): FAIL.** Aggregate audio seconds per wall second (two runs each, median):

  | rows | runs | median |
  |---:|---|---:|
  | 1 | 3.65, 3.65 | 3.65 |
  | 2 | 6.54, 6.52 | 6.53 |
  | 4 | 10.67, 10.64 | 10.66 |
  | 8 | 13.42, 12.88 | **13.15** |
  | 16 | 17.57, 17.49 | 17.53 |

  - The ratio at 8 rows is **3.60**, against at least 4: the rule fails, and that stands. At 16 rows it is 4.8.
  - Where the step goes at 8 rows (nsys, lighthouse): the language model's step does not grow, 16.2 ms against 15.7
    ms alone. The depth decoder grows to about 17 ms, against 5.3 ms alone: its FP32 matrix multiplies, kept FP32
    and in their validated summation order so that every frame is bit-identical, are about 40% of the GPU's time.
    Each row's codec, about 0.4 ms per step, and host sampling grow with the rows too.
  - Tried and rejected (`native/tests/fp32_linear_bench.cu`): seven other block shapes for the FP32 kernel, all
    bit-identical to it. None was faster on the trunk's shapes; a 32-row block was 40% slower.
  - Kept: per-codebook heads (each embedding read once per pass), and batched rotary, attention and head norm.
    Rotary had cost 23 µs a call in FP64; it now uses a table computed on the device by the same expressions.

## Server (liquid-lfm3-server batch-s3: 8 speech rows by default; sources identical to this commit)
- **Exact in a crowd (rule 1): PASS.** The server check sends the 42 cases (check_lfm3_speech_stream.py's 41 and the
  budget-truncated one) four ways, against each case's replies alone on the single-request server
  (server-stream-v3), unstreamed and streamed:
  - unstreamed, all at once: 42 of 42 identical;
  - unstreamed, random order and 0-2 s gaps: 42 of 42;
  - streamed, all at once: 42 of 42 (every event and the done event);
  - streamed, random arrival: 42 of 42.

  The crowd server ran with 96 handlers and a queue of 64, so it could take them all; a 429 is retried, as a client
  does (`build-s3/batch-server.json`).
- **Real time (rule 3): PASS.** No player ran dry at 8, 12 or 16 concurrent streams; first audio then was 0.52-0.57 s,
  0.63-0.75 s and 0.73-0.92 s.
- **First audio under load (rule 4): PASS.** A TTS request arriving while 7 others stream gets its first audio after
  a median of 0.445 s over 10 arrivals (0.43-0.53 s).
- **Text under load (rule 5):** its ids are identical to its reference. Its latency was 34.0 s, while the rows running
  when it arrived finished 15.0 s later. A text client makes two requests (create a session, then generate), and each
  waits for the speech admitted before it to finish: the batch drains for a text job but does not pause for it.
  Speech load delays text by up to one drain per text request; this is open (below).
- **Nothing else moves (rule 6):**
  - streamed /speech: 42 of 42 exact, first audio as before, no gaps;
  - the server's TTS WAVs equal the validated ones;
  - the audio-input cases pass;
  - the full text gate passes;
  - lfm3-speech's regression check is 36 of 36 bit-identical.
- **Host tests:** 885 passed, 2 skipped: the HTTP, framing, chat, context, batch and cancellation gates, speech,
  streaming, the new scheduler suite (`tests/test_lfm3_http_speech_batch.py`), audio input, the normalizer and the
  batch-cancellation suite, on the plain, ASan+UBSan and TSan builds. The one failure is the known transformers-version
  test.

## Red-team (one pass, an independent reviewer with the code only)
1. **A finished row at its capacity corrupted its neighbour (high).** The batched decoder wrote every row's K/V at its
   cursor, finished rows included. A request that ran exactly to its capacity left its cursor there, so later steps
   wrote one slot past its cache: into its next head, or the next row's cache. That is a silent change to another
   request's output, and the text engine's batched decoder could do the same.
   - **Fixed:** inactive rows, and any cursor at the capacity, are not written.
   - **Verified:** the capacity check above, which fails on the pre-fix decoder at the finishing step. None of the
     rule-1 cases had run a request to exactly its capacity.
2. **Constant text traffic starved speech (medium; reproduced: a 504).** Speech was admitted only while no text job
   waited. **Fixed:** the two queues are served in arrival order. Test added (it fails with the old rule).
3. **Smaller fixes:**
   - Queued tickets whose client left, or past their deadline, now give up their places before admission.
   - One limit counts text and speech waiting together, as the single queue did.
   - Every row's suppression mask is zeroed at the start.
   - A test was added for a client that leaves while queued.
4. **Check gaps, fixed:**
   - The server check's reference is the single-request server.
   - The crowds are sent all at once and at random, both unstreamed and streamed.
   - Text latency is reported against the drain time.
   - Several stream tests had proved cancellation only because a follow-up request queued behind the cancelled one.
     They now run with one row, where it must.
5. **Kept, as a choice:** any engine-side refusal of speech_admit still fails the service closed, as every engine
   refusal does. Requests are fully validated before they reach the engine.

Found while verifying: the server's accept queue was http.server's 5. Under a crowd arriving at once it overflowed, and
about one connection in 360 was reset (reproduced on the host with the fake engine). It is now max(16, handlers), and
a 120-client test checks that every client gets a reply.

## Open
- **Throughput at 8 rows (rule 2 failed):** the batched FP32 depth decoder, the codec per row and host sampling are
  what grow with the rows. Making them faster without changing a bit (an FP32 kernel in the same summation order;
  codecs batched across rows; parallel sampling) is the next speed work.
- **Text under speech load** waits for a drain per request. Interleaving text jobs between speech steps would end
  that, at the cost of pausing every running stream for the text job's length.
- **Crowds hit 429 frontend_busy:** the server prepares one request at a time and refuses a second meanwhile, as the
  reference server does; clients retry. A short wait instead of the refusal would suit crowds better.

## Second red-team (a fresh reviewer with the code only), and server batch-s4
No cross-row coupling: the reviewer traced every buffer, RNG and kernel of the batch path and found none. The text
engine is unchanged by the masked scatter: every row it steps is active and below its capacity. Findings and what was
done:
1. **Long requests timed out under load (medium, design).** In a batch every row runs slower than alone. A request near
   the step budget (about 227 s alone, sized under the 330 s request deadline) would time out at 8 rows or more. It
   would get a 504 where alone it gets its reply. **Changed:** the request deadline bounds only the wait in the queue.
   - Once admitted, a request runs to its end, bounded by the step budget; a client that leaves still cancels it.
   - A request still queued at its deadline gets 504 `queue_deadline` and never runs. Streamed, if its reply was
     already committed, it gets the error event instead.
   - Tests added for both.
   - The single-request path (`--speech-rows 0`, the fake engine only) keeps its old deadline.
2. **The accept-queue test did not catch the bug.** With a backlog of 5 the 120-client test still passed: here an
   overflowing queue drops connection attempts and the clients retry. **Replaced:** the test reads the listening
   socket's backlog (`ss`, Send-Q), and fails with a backlog of 5 (checked). A crowd larger than the handlers is still
   closed at accept, by design (LocalHTTPServer); the crowd checks ran with 96 handlers.
3. **First-audio timing:** the GPU check timed first audio from a request's last attempt, so time spent retrying a 429
   was not counted. **Fixed:** it times from the first attempt.
4. **Admission stalls:** an admission's audio encoding and prefill run inside a step and stall the other rows. **Now
   checked:** 7 TTS streams while the 5 spoken questions join one by one: no stalls.
5. **Unvalidated row counts:** `--speech-rows` allowed 64, but exactness and memory were validated only to 16 (each
   row holds its own codec, 147 MB). **Capped at 16.**
6. **Event-line limit:** streamed event lines in the batch were allowed 64 MiB. They are bounded at 4 MiB again; only
   a finished reply, with its WAV, may be larger.
7. **Kept, documented:** the fake engine ends a cancelled row before its step's work. The real batch lets a row whose
   generation just ended finish normally first, as the single-request loop did. A split request may then start its
   next chunk and be cancelled one step later.

**batch-s4** (the second red-team's fixes; sources identical to this commit, `build-s4/`):
- **Crowds (rule 1):** 42 of 42 identical in all four.
- **Real time (rule 3):** no stalls at 8, 12 and 16 streams, nor with 7 streams and the spoken questions joining.
- **First audio under load (rule 4):** median 0.445 s.
- **Text under load (rule 5):** identical ids; 34.1 s, against a 15.0 s drain (as above).
- **Nothing else moves (rule 6):** streaming, audio-input, TTS and text gate pass.
- **Host tests:** 888 passed, 2 skipped; the one failure is the known transformers-version test.
