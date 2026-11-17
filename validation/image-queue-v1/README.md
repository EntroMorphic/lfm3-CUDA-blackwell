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
Found on the way, then **retracted**: "the image request itself returned after 20.0 s" was the tool's error (it timed
the request after waiting for the 7 streams to end; fixed). Measured directly (shared GPU, indicative): unstreamed with
7 streams, 7.9 s on this server and 5.8 s with part 2 and the cache (a hit); the first image request after a server
starts takes ~16 s before its first token (one-time initialization, to examine separately).

## Part 2 implemented (2026-11-16), with image-cache-v1
In process, the handler thread decodes and preprocesses each image (`Service::decode_images`, under the 2 slots) and
leaves the pixel values in `lfm::image::shared_pixel_handoff()` (`lfm/image_handoff.hpp`, a `LeasedMap` keyed by
the image part's full base64 text, held by the request until it ends); `WorkerCore::parse_speech` takes them from
there instead of decoding (`Policy::image_handoff`, set by `liquid-lfm3-server`). A miss (the out-of-process worker; a
front end that had the image's count cached and so decoded nothing) decodes on the engine thread, as before. The
harness's fake engine checks every handed-over image against its own decode, byte for byte
(`--image-handoff on --image-log`; tests in `tests/test_lfm3_http_image_in.py`). Found by the full host suite and fixed
before commit: the refactor first left smart_resize's aspect-ratio refusal outside the refusal mapping (3 failures,
one per build variant; `decode_image` now runs the count or preprocessing under it).
Exactness through the server: image-cache-v1 rule 1's check runs the in-process server with the handoff on (32 of 32).
Rule 2 (the stall with part 2) and rule 3 (memory) follow in a quiet window.

## Rule 2 results (2026-11-16, quiet window 3): **FAIL** (stall 0.66 s > 50 ms; was 1.06 s) (`rule2/`)
`tools/measure_lfm3_image_stall.py` on `vimg-srv-c2` (part 2, image cache off: `--image-cache-mb 0`), cherry (4K), 3
repeats, MAXN throughout (`power-*.json`), tegrastats recorded. Worst gap between audio events on the 7 streaming rows:
without the image 0.823 / 0.339 / 0.341 s, with it 1.017 / 0.993 / 1.000 s; medians **0.341 -> 1.000 s, stall 0.66 s**
(rule 0 on the old server: 1.06 s). Part 2 took the decode off the engine thread (~0.4 s), as designed; what remains
is the encode and the 4K prefill, which run on the engine thread between decode steps: part 2 was never going to remove
them, and the 50 ms bar cannot be met without splitting the prefill into chunks interleaved with decoding (a separate
round, not started; recorded as a candidate). Every row still completes, every reply 200.
The cache-on run crashed in the tool (`stall-new-cache.log`): the warm-up run on a cold server got no audio inside its
window and the gap computation raised. Fixed (`f2106db`: the warm-up measures nothing; a measured run with no gaps
says so); rerun queued for the next quiet window. With a hit, the encode is gone too, so that run will show the
prefill alone.

## Rule 3 results (2026-11-16, quiet window 3): **PASS** (`rule3/`)
`tools/measure_lfm3_image_rss.py`, 16 simultaneous cherry (4K) requests on `vimg-srv-c2`: idle 5,506 MB, peak
**6,031 MB (+525 MB, bar +1 GB)**, after 5,662 MB. Statuses: 200s and 429 `queue_full` for those past the queue (the
default queue size; the rule is about memory, and the refused ones still arrived and were parsed).

## Rule 2 with the cache (2026-11-16, quiet window 4): **FAIL**, stall **0.28 s** (`rule2-w4/`)
The same measurement with the image cache on (window 3's run crashed in the tool; fixed `f2106db`). Worst gaps without
the image 0.384 / 0.339 / 0.337 s, with it 1.016 / 0.622 / 0.619 s (the first repeat the image's miss, the next two
hits: image request 2.18 s, then 1.36 and 1.35 s); medians **0.339 -> 0.622 s, stall 0.28 s**. Old server 1.06 s,
part 2 alone 0.66 s, with a cache hit 0.28 s: what remains is the 4K prefill itself on the engine thread. A chunked
prefill interleaved with decode steps is the candidate for the 50 ms bar (not started).

## Red-team (rule 5) (2026-11-17, x86 host only; commit `8104cd5`)
I read `Service::decode_images`, `frontend_call`, `speech`/`speech_stream`, `lfm/leased_map.hpp`, `image_handoff.hpp`
and `WorkerCore::parse_speech` and looked for faults, then wrote a test for each finding before fixing it. The
results, by area:
- **A lease copied its key (fixed; low to medium, memory).** `LeasedMap::Lease` kept its own copy of the key, which is
  the image part's whole base64 text (up to 16 MiB). Every in-flight request therefore held its images' text once
  more, for its whole life, on top of the map's copy and the body. Test: `native/tests/lru_bytes_check.cpp`, the
  LeasedMap block (counts the bytes `operator new` hands out: the first lease of a 1 MiB key took 2 MiB, and a second
  lease of the same key took another 1 MiB). Fix: a lease holds its map iterator. It is valid because an entry is
  erased only at its last lease's release.
- **Lease lifetime on every exit path: no defect.** The leases live in `Speech::image_leases`, so each ends when its
  request's `Speech` does. That holds for a refusal after decoding (a later image refused, or the prompt over
  capacity in `encode_speech`), a 504, a 499, 429 `queue_full`, a ticket cancelled or abandoned, and a request that
  completes. If an exception is thrown while a lease is being pushed, the temporary lease releases itself. The decode
  slot is released by the RAII `Slot`. New guard test: `test_every_way_an_image_request_ends_releases_its_leases`.
  The harness's fake engine logs the handoff's size (`--handoff-log`). After refused requests and clients that reset
  while waiting, decoding, queued or running, the next request's engine call sees only its own entry. A mutation (a
  lease never released) makes the test fail.
- **Stop while waiting: no defect.** `close()` sets `stopping_`, and waiters see it within one 50 ms poll and answer
  503 `worker_unavailable`. New test: `test_image_requests_waiting_to_decode_when_the_server_stops_get_503`. It sends
  12 large images and stops the server during the decodes. Every reply is 200 or 503 `worker_unavailable`, at least
  one is 503, and the process exits cleanly (no sanitizer report).
- **Deadline while waiting: no defect** (existing test: 504 `frontend_deadline`, never 429). A decode that has
  started is not interrupted; it is bounded by the decoder's limits (~0.4 s at 4K), and `frontend_call` then refuses
  504.
- **A client that leaves: no defect in this change; a limitation inherited from the earlier design.** Only a reset is
  seen as `gone` (POLLHUP/POLLERR). A client that closes with a FIN looks the same as one that half-closes and still
  reads (audio-assurance-v1's rule), so its request keeps its place. Unstreamed, that request runs to its end; a
  streamed one ends at its first write. `gone` is not polled while a request waits for `frontend_call`'s timed mutex.
  That mutex is held only to render and tokenize (milliseconds), and the batched wait polls `gone` right after.
- **A failed service (`failed_`, engine fault) is not checked while waiting to decode: no change.** An image request
  still decodes before meeting the 503, because image refusals are validation and come before 503, as text's 400 comes
  before its lease's 503. The cost is CPU in a terminal state that needs a restart anyway.
- **Thread safety: no defect.** The slots are a counter under `image_slots_lock_` with a condition variable,
  `stopping_` is atomic, and the handoff is a mutex-guarded map of `shared_ptr<const PixelValues>`. A lease can be
  released on the engine-owner thread (the last `Speech` reference is dropped there) under the same mutex. The HTTP
  suites pass under TSan.
- **Engine-side cost of the handoff (GPU-side, described rather than changed; low to medium, memory and engine-thread
  time).** `parse_speech` copies the handed-over `PixelValues` (`images.push_back(*pixels)`). A full-size image is
  8,192 patches x 1,536 floats = **50 MB**. `ChunkRun::next` copies the Options again, and `Batch::admit` copies them
  into the row (`w.o = o`). That makes ~150 MB of memcpy per full-size image on the engine thread. The row keeps its
  copy until the row is next admitted, which is after the request has ended. See image-cache-v1's red-team (C4) for
  the precise fix.
Host suites after the fixes: see image-cache-v1's red-team section (one run covers both).
