# An image seen once is not decoded or encoded again

## What it will be (Tripp, 2026-11-16: "Let's do all of these in the order you see fit")
A conversation re-sends its images on every turn (the API is stateless, as HF's and every chat API is), and every turn
pays for them again: today a 4K photograph is decoded and preprocessed twice per request (once by the server's front
end to count its tokens, `lfm3_http.hpp` `image_tokens`; once by the engine, `WorkerCore::parse_speech`), ~0.32-0.42 s
each, then encoded, ~0.36 s (images-speed-v1). HF re-encodes too. This round keeps what an image's bytes produce.

## Design (to be confirmed by rule 0)
- **Key: the image's bytes themselves**, not a hash of them: a fast hash picks the bucket, then the full bytes are
  compared. Two different images can never share an entry, whatever their hash (no cryptographic assumption, no
  poisoning by a crafted collision). The bytes are the base64-decoded JPEG/PNG; the front end's settings are fixed
  (one checkpoint, one processor configuration), so the bytes alone determine the pixels, and the pixels the features
  (the encoder is deterministic: images-speed-v1's runs byte-identical across builds and sanitizers).
- **Engine side** (in process: the engine-owner thread; out of process: the worker): bytes -> {grid, tokens, the
  encoder's output rows on the GPU (BF16, `tokens x 2048`: 8 MB for a 4K photograph)}. A hit skips decode,
  preprocessing and the encoder; the rows are copied device to device into the prompt's span.
- **Front-end side** (the server's handler threads, under a mutex): bytes -> tokens. A hit skips the front end's decode.
- **Bounded**: least-recently-used eviction under one budget counting the kept bytes and the GPU rows,
  `--image-cache-mb` (default 512; 0 disables the cache entirely). Only a fully encoded image is inserted: a refused,
  failed or cancelled one never is.
- **Disclosure**: a shared cache is a timing side channel (a client can tell whether an image was sent recently by
  anyone). The server is loopback-only by default; SECURITY.md and docs/server-api.md say so, and 0 turns it off.
- Order: after image-queue-v1 (its handler-thread decode is where the front-end lookup goes).

## Rules, set before any code
0. **Measure first** (quiet Thor): a three-turn conversation re-sending one 4K photograph (and the stage-0 "two"
   request's pair), per turn: end to end, time to first token, and its front-end, encode and prefill parts. If turn 2's
   front end plus encode is under 100 ms in total, the round is dropped and this README says so before code.
1. **Exact**: with the cache on, every reply byte-identical to the same request with `--image-cache-mb 0`: the 11
   stage-0 requests each sent twice in a row, the multi-turn set of rule 0, and the listening set
   (`outputs/images-listening-v1/`); streamed and unstreamed; in process and through the worker; the encoder rows of a
   hit byte-identical to the miss's (`record_image`).
2. **Faster**: on a hit, the time from the request to the start of prefill at most 30 ms at 4K (the miss: ~1 s), and
   turn 2's end to end shorter than turn 1's by at least 80% of turn 1's front end plus encode.
3. **Bounded**: the cache's bytes (reported by `stats`) never above the budget through a run of 200 distinct images
   at random sizes; eviction least-recently-used (a unit test); `--image-cache-mb 0`: every rule-6 output
   byte-identical to the current build's and no cache memory allocated.
4. **Never the wrong image**: a test build whose bucket hash is a constant (every image collides) passes rule 1 and
   the stage-0 set mixed and repeated; one byte changed in an image is a miss; a refused image (every refusal code of
   images-v1) is refused again identically on its second sending, never served from the cache.
5. **Nothing else changes**: images-speed-v1 rule 6's checks; the host suites (plain, ASan+UBSan, TSan: the front
   end's mutex); text and audio requests untouched.
6. **Sanitizers and red-team**: compute-sanitizer memcheck on the engine with hits, misses and evictions; one red-team
   pass (eviction while a request is using an entry, cancellation mid-encode, the worker restarting).
