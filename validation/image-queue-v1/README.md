# Image requests that wait their turn, and an engine that never decodes

## What it will be (Tripp, 2026-11-16: "Proceed as you see fit on the rest")
Two findings of images-v1, both about where image decoding runs in `liquid-lfm3-server`:
1. **Simultaneous image requests are refused.** The Service decodes and preprocesses each image inside
   `frontend_call`, which takes one encode at a time and answers a second with `429 frontend_busy` (by design: the
   text front end's admission was validated that way). Image decoding holds it long enough (up to ~0.4 s at 4K) that
   8 of 11 simultaneous image requests were refused first (`validation/images-v1/rule6/vimg-srv-check/`).
2. **The engine-owner thread decodes images.** `WorkerCore::parse_speech` decodes the same bytes again, on the thread
   that runs every speech row's steps: a 4K image's ~0.36 s of CPU work stalls every other request's audio for that
   long. Not yet measured; this round measures it first.

Out of scope: the decoders themselves (bit-identical to HF; images-v1 rule 1), and moving them to the GPU (later).

## Design (to be confirmed by the measurement in rule 0)
- Images are decoded and preprocessed on the request's own handler thread, **before** `frontend_call`, under a
  counting semaphore of 2 decodes at a time (bounding memory: 2 x ~200 MB at the 64-megapixel cap) that a request
  **waits** for, within its deadline (504 `frontend_deadline` past it), instead of being refused. `frontend_call` then
  only renders and tokenizes, as for text.
- The decoded result travels to the engine with the request (in process: the pixel values themselves, not the bytes),
  so the engine-owner thread never decodes; the out-of-process worker keeps decoding bytes, as now.
- Text requests are unchanged: their `frontend_call` admission, 429s included.

## Rules, set before any code
0. **Measure first:** the stall of rule 2's condition on the current server (a 4K image admitted while 7 rows stream
   speech: the longest gap between audio events on the other rows, against the same run without the image). If there
   is no stall above 50 ms, part 2 of the design is dropped and this README says so before code.
1. **No refusals:** 11 image requests sent at once (the stage-0 set) all succeed with no 429, each reply byte-identical
   to `lfm3-speech`'s; 32 at once with `--queue-size` exhausted are refused 503 `queue_full` exactly as text is.
2. **No stall:** with a 4K image admitted while 7 rows stream speech, no other row's gap between audio events grows by
   more than 50 ms over the same run without it.
3. **Bounded:** peak RSS with 16 simultaneous 4K image requests at most the server's idle RSS + 1 GB.
4. **Nothing else changes:** images-speed-v1 rule 6's checks, and every HTTP host test (plain, ASan+UBSan, TSan);
   text's 429 behavior byte-identical in `test_lfm3_http_native.py`.
5. **Sanitizers and red-team** as for every server change: the HTTP host suite under TSan covers the new semaphore;
   one red-team pass for deadlines, cancellation while waiting, and a client that leaves.

## Part 1 implemented (2026-11-16), before rule 0's run (part 2 waits on it)
- `Service::decode_images` (`lfm3_http.hpp`): each image part decoded on the request's handler thread before
  `frontend_call`, under `kImageDecodeSlots` = 2 waiting slots; the counts travel in `Speech::image_tokens` to
  `encode_speech`. While waiting: 503 `worker_unavailable` when the server stops, 504 `frontend_deadline` past the
  deadline, 499 `client_gone` when the client has gone (unstreamed: the connection's `gone`; streamed: a reset peer).
- Not in the design, needed by rule 1: after decoding, an image request still meets `frontend_call`'s one-at-a-time
  render and tokenize; two decodes finishing together would have the second refused 429. An image request therefore
  waits for it too (`frontend_call(..., wait = true)`, a timed lock within its deadline); text keeps `try_lock` and
  its 429, unchanged.
- The front end counts an image's tokens with smart_resize alone (`lfm::image::tokens`, preprocess's own count), no
  longer running the full resize the engine runs again: refusals still need the full decode, the resize did not.
- Host tests (`tests/test_lfm3_http_image_in.py`, plain, ASan+UBSan, TSan): 11 simultaneous image requests all 200
  with their prompts' lengths as when sent alone, a two-image request's order kept under concurrency, deadlines while
  waiting 504 never 429, text's 429 kept while images decode. All HTTP host suites: 356 passed, 2 skipped (72 of the
  image file's own). Rules 1-5 on Thor follow; rule 0 runs on the server built before this change (`vimg-srv-a3`).

## Rule 1 results (2026-11-16): **FAIL as written** (the rule was mis-specified); its intent met (`rule1/`)
`check_lfm3_image_server.py` on `vimg-srv-q1` (quiet session 1, step 6): every reply identical to `lfm3-speech`'s,
streamed, unstreamed and with the 11 sent at once; but 4 of the 11 simultaneous requests met 429s first (retried by
the tool). Their code (`q429` probe, the same 11 at once, no retries): **`queue_full`**, not `frontend_busy`: at the
server's default `--queue-size 4` the engine admits one request at a time and an image prefill takes about a second,
so 11 arrivals overflow 4 waiting places, as 11 text requests would. My rule asked for no 429 at all without fixing
the queue size, and named the queue's refusal 503 where it is 429 `queue_full`: both errors mine, recorded.
What the rule meant (no refusal by the front end): the same probe at `--queue-size 16`: **11 of 11 served, no 429**
on the new server; **6 of 11** on the server before the change (`vimg-srv-a3`), five refused `frontend_busy`.

## Rule 0 results (2026-11-16): stall **1.06 s** > 50 ms: part 2 proceeds (`rule0/`)
`tools/measure_lfm3_image_stall.py` on the server before part 1 (`vimg-srv-a3`), quiet session 2 (the first attempt,
session 1, failed in the tool: its 7 streams met text's 429 `frontend_busy` and did not retry; fixed and recorded,
commit "measure_lfm3_image_stall.py: the streaming rows retry 429"). Three paired runs: the longest gap between audio events on the 7
streaming rows, without the image 0.336 / 0.340 / 0.339 s, with a 4K photograph admitted 1.429 / 1.397 / 1.393 s:
**+1.06 s** (median), every row but one (that row, by then, between chunks). The engine-owner thread's share: decoding
and preprocessing the image again (~0.4 s), the encoder (~0.36 s) and the image prompt's prefill (~0.25 s). Part 2
removes the first; the encode and prefill would still hold every row for ~0.6 s (interleaving prefill with decode
steps is a separate design, not this round's).
Found on the way, not yet explained: the image request itself returned its 32-token reply after **20.0 s** with the 7
streams running (alone: ~1.3 s).
