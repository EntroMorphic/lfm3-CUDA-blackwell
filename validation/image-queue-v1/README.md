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
