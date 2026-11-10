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
