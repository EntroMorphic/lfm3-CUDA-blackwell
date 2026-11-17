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

## Rule 0 results (2026-11-16): **proceed** (`rule0/`)
`tools/measure_lfm3_image_turns.py`, quiet session 1, server `vimg-srv-a3`, 4 conversations (the first warm-up), medians:
| request | turn | end to end | first text | engine prefill (decode + preprocess + encode + prefill) |
|---|---|---|---|---|
| cherry (4K) | 1 | 5.93 | 4.82 | 3.87 (bimodal: 0.62, 3.87, 0.59, 4.05: the prefill stall, below) |
| cherry | 2 | 2.53 | 1.60 | 0.65 |
| cherry | 3 | 2.72 | 1.61 | 0.66 |
| two (two small images) | 1-3 | 1.32-1.34 | 0.23-0.25 | 0.14-0.15 |
At 4K each turn spends ~0.95 s between the request and the engine's prefill (the server's decode and the engine's
second decode and preprocessing) plus ~0.36 s of encode inside the prefill: far above the 100 ms bar. Texts were
identical across conversations. The turn-1 times expose an intermittent 3-4 s prefill stall, investigated separately
(`validation/prefill-stall-v1/`).

## Implemented (2026-11-16), with image-queue-v1 part 2
- `lfm/lru_bytes.hpp` (`LruBytes`, `LockedLruBytes`): least recently used under a byte budget, keys compared in full
  by `std::map` (no hash anywhere, so rule 4's constant-hash build is moot: there is no hash to make constant);
  `lfm/leased_map.hpp` (`LeasedMap`, the pattern of the audio round's `FeatureCache`) for part 2's handoff.
- The front end (`Service::decode_images`): an image part's base64 text -> its token count; a hit skips the decode.
- The engine (`WorkerCore::image_cache_`, `lfm3_speech::ImageCache`, engine thread only): the text -> the encoder's
  rows on the GPU (`ImageRows`); `parse_speech` finds them (a hit: no decode, no encode; the request holds its own
  reference, so an eviction cannot pull rows from a request in flight); the prefill copies them into the prompt's
  span, or encodes and keeps them (only after a whole encode). `lfm3-speech` sets no cache: unchanged.
- `--image-cache-mb` (default 512, 0 none) is each side's budget (the design said one budget for both; two caches in
  two places, each bounded, is what was built: recorded). The engine's figures in the worker's `stats`
  (`image_cache_entries`, `image_cache_bytes`, `image_cache_budget_bytes`).
- Host tests: `tests/test_lru_bytes.py` (200 keys at random sizes never over budget; LRU order; same key keeps its
  value; over-budget entries not kept; keys differing in their last byte distinct; evicted values live on for holders;
  budget 0; ASan+UBSan and TSan); `tests/test_lfm3_http_image_in.py` (a re-sent image skips the front end's decode; one
  byte different is decoded; a refused image is refused again; budget 0 keeps nothing).

## Rule 1 results (2026-11-16): **PASS** (`rule1/`)
`tools/check_lfm3_image_cache.py` on `vimg-srv-c1` (server `841aba10`, worker `e8258fef`; shared Thor, exactness):
**32 of 32 identical**: each of the 11 stage-0 requests sent twice unstreamed then twice streamed (every send after
the first a cache hit) equal to `lfm3-speech`'s text, cache on and off; the 11 at once, twice, on both; the
three-turn conversations on cherry and two, twice each, the same on and off; the 8 listening-set requests
(interleaved, 1,024 tokens) text and WAV identical on and off (the record keeps each WAV's sha256).
**Correction (2026-11-17, from the rule 6 red-team):** the rule asks for "in process and through the worker". The
check ran the in-process server only; the out-of-process worker has no image cache and no handoff, so through the
worker there is nothing to compare (cache on and off are the same code there). Rule 1 holds for the in-process
server, where the cache exists; "through the worker" is vacuous, not passed.

## Rule 2 results (2026-11-16, quiet window 3): **part (a) FAIL as written, part (b) PASS in substance** (`rule2/`)
`tools/measure_lfm3_image_turns.py`, 4 conversations (the first a warm-up), medians; new `vimg-srv-c2` (cache on, the
default) against old `vimg-srv-a3`, same window, MAXN:
| request | turn | new end to end | old end to end | new first text | old first text | new engine prefill | old |
|---|---|---|---|---|---|---|---|
| cherry (4K) | 1 | 1.466 | 2.643 | 0.354 | 1.534 | 0.227 | 0.592 |
| cherry | 2 | **1.345** | 2.529 | **0.415** | 1.599 | 0.288 | 0.653 |
| cherry | 3 | 1.530 | 2.700 | 0.419 | 1.589 | 0.291 | 0.656 |
| two | 1-3 | 1.17-1.19 | 1.31-1.33 | 0.085-0.100 | 0.229-0.242 | 0.068-0.082 | 0.136-0.150 |
Texts identical across conversations on both servers.
- **Rule as written could not be applied to turn 1 vs turn 2**: the tool's warm-up conversation sends the same image,
  so on the new server every measured turn 1 is already a hit (the warm-up's own turn 1, the one miss, was the
  server's first request: 3.38 s, cold). The comparison that measures the same thing is new against old (old: every
  turn a miss). **(b)** Turn 2 is shorter by **1.18 s** against the bar of 80% of a miss's front end plus encode (~0.95
  + ~0.36 s from rule 0: 1.05 s): met.
- **(a) "request to prefill start at most 30 ms on a hit"**: not isolated by this tool; its upper bound, first text
  minus engine prefill (which also holds the first generated token), is **0.127 s** for cherry and 0.018 s for two.
  The difference scales with the payload: cherry is a 9.8 MB PNG, a 13 MB base64 body, which every request must
  still send, receive and parse as JSON (in process: parsed by the front end and again by `parse_speech`) and
  compare as a key. The cache cannot remove the payload; the bar was written without it. **FAIL as written**; a
  breakdown (receive, parse, lookup) is queued, and parsing once instead of twice is a candidate fix.
- The old server's runs show two 4.4 s prefills (conversations 3 and 4): the prefill stall (`validation/prefill-stall-v1/`).

## Red-team (rule 6) (2026-11-17, x86 host only; commit `8104cd5`)
I read `lfm/lru_bytes.hpp`, the front end's token cache (`Service::decode_images`), `WorkerCore::parse_speech`,
`set_image_cache_mb`, `stats`, and `Batch::admit`'s keep and copy in `src/lfm3_speech.cuh`. Host findings have a
failing test first, then the fix. GPU-side findings are described rather than changed (no GPU on this host, and no
Thor time for this pass).
- **C1, keys held twice (fixed; medium, memory: rule 3's bound).** `LruBytes` kept each key in its map and a second
  copy in its order list, but charged it once. The front end's token cache, whose entries are almost entirely key,
  could therefore hold ~2x its budget (about 1 GiB at the default 512). The engine's host-side keys were doubled the
  same way. Found with it: `put` pushed the key into the order list before inserting the map entry. An allocation
  failure in the insert left a list entry with no map entry, and `trim` would then dereference `end()`. Test:
  `lru_bytes_check.cpp` (putting a 1 MiB key took 2 MiB). Fix: the order list holds pointers to the map's keys (a
  `std::map` node's key never moves), and the map entry is inserted first and removed again if the list's
  allocation fails.
- **C2, cost overflow (fixed; low).** `key.size() + value_bytes` could wrap to a small cost and be kept under any
  budget. Today's callers pass bounded sizes. Test: a `SIZE_MAX` value was kept under a budget of 1,000. Fix: the
  comparison is made without forming the sum.
- **C3, a front-end hit without the engine's rows decodes on the engine thread (GPU-side, described; medium, the
  stall image-queue part 2 removed comes back; replies stay correct).** When the front end's token cache hits, the
  handler skips the decode, so it hands nothing over. If the engine's cache then lacks the rows, `parse_speech`
  decodes and preprocesses on the engine-owner thread (~0.4 s at 4K, every row stalled). Ways that happens:
  (a) The two budgets fill at very different rates. A 4K image costs the engine ~13 MB of key plus 8 MB of rows, but
  costs the front end ~13 MB of key alone. The engine therefore evicts first, after ~24 distinct 4K images at the
  default budgets.
  (b) The first request with an image never reached the encoder (504 `queue_deadline`, cancelled while queued, 429
  `queue_full` after decoding).
  (c) Two requests sent together with the same new image, where the first leaves the queue before the second is
  admitted.
  The fake engine has no GPU cache, so the host suite shows this only as the expected `decode` line in
  `test_a_resent_image_skips_the_front_ends_decode_and_changes_nothing`. Proposed fix: the engine publishes the keys
  it holds. For example, `lfm3_speech::ImageCache` takes an insert/evict hook that keeps a mutex-guarded set beside
  `shared_pixel_handoff()`. On a token-count hit, `decode_images` (in-process, `policy_.image_handoff`) skips the
  decode only when the engine holds that key. Otherwise it decodes and leases as on a miss; the cached count stays
  valid. An eviction between that check and `parse_speech` would still decode on the engine thread, but rarely.
  **Thor check needed:** rule 2's stall with 30 distinct 4K images sent first, then a re-send.
- **C4, rows and media outlive their request (GPU-side, described; medium, memory outside every budget).**
  `Batch::admit` copies the whole `Options` into the row (`w.o = o`), and `ChunkRun` keeps its own `o_`. Both hold the
  images' `PixelValues` (50 MB for a full-size image), the keys (up to 16 MiB each) and the `shared_ptr<ImageRows>` of
  cache hits. `ChunkRun` goes away when the request ends, but `Row::o` lives until that row is next admitted. With 8
  to 16 rows, evicted GPU rows (8 MB each), pixel values and keys stay allocated after their requests end, which the
  budget does not count. The copies (`images.push_back(*pixels)` in `parse_speech`, `o = o_` in `ChunkRun::next`,
  `w.o = o`) are also ~150 MB of memcpy per full-size image on the engine thread. Proposed fix (bookkeeping only, no
  numerics):
  (1) In `admit`, after the prefill, clear `w.o.images`, `w.o.image_rows`, `w.o.image_keys` and `w.o.audio`. After
  `admit`, the row reads only `forced_*`, `codec_finalize`, `on_audio`, `on_slot`, `cancelled` and `record_hidden`.
  (2) In `ChunkRun::next`, move the media out of `o_` instead of copying them. Image and audio requests are never
  chunked, so they have exactly one chunk.
  (3) Let `Options` hold `shared_ptr<const PixelValues>` so the handoff's pixels are shared, not copied.
  **Thor check needed:** compute-sanitizer memcheck, and rule 1's exactness after the change.
- **C5, keeping rows can fail the server closed (GPU-side, described; low to medium).** The keep in `Batch::admit`
  (`std::make_shared<ImageRows>` -> `cudaMalloc`, then a D2D copy) throws on an allocation failure. A throw from
  `speech_admit` is a rejected native operation, which the Service treats as fatal, so every row and queued request
  gets 503 and the server stays down until restarted. That happens over an optional cache, after an encode that
  succeeded. Proposed fix: wrap the keep in `try { ... } catch (const std::exception&) { cudaGetLastError(); }`
  (`cudaErrorMemoryAllocation` is not sticky) and continue uncached. Any sticky error still surfaces at the
  `cudaDeviceSynchronize` that follows.
- **Eviction while a request uses an entry: no defect.** The request holds `shared_ptr<const ImageRows>` from
  `parse_speech` until it is done with it (longer than needed: C4). The vision encoder, the keep's copy and the hit's
  copy all run on the legacy default stream, so a buffer freed after eviction is never read by work still in flight.
  `cudaFree` also synchronizes.
- **Cancellation mid-encode: no defect.** The encode runs synchronously inside `admit`, and cancellation is observed
  only between steps. A row is kept only after `vision_->encode` returned exactly `im.tokens` rows. If the request is
  cancelled later, the kept rows are still the image's exact encoder output.
- **The worker restarting: no defect, but the design does not match the README.** In process, the engine never
  restarts: a fault fails the server closed, and a process restart starts both caches empty. The front end's cache
  holds only counts, which are a function of the bytes, so they cannot go stale. The out-of-process
  `liquid-lfm3-worker` has **no image cache at all** (no flag, so its budget stays 0, and no handoff). The design's
  "out of process: the worker" was not built, so rule 1's "through the worker" compared the cache off with the cache
  off. Recorded; no change.
- **Never the wrong image (rule 4): no defect.** Keys are the full base64 text, compared in full by `std::map` on
  every side. An image part is exactly `{"type": "image", "image": ...}` (any other key is refused), so there is no
  MIME type or detail setting to differ under the same text. The format is sniffed from the bytes, and the
  preprocessing (`PreprocessConfig`, min/max pixels) is fixed with no per-request option. The same text always gives
  the same pixels, and therefore the same rows. The same bytes in different text (padding, for instance) make separate
  entries, which is harmless. An empty key is refused by `parse_messages`, and the engine uses `""` only as its "not
  cached" marker. A huge key is at most 16 MiB (the body limit), is charged in full, and a key over the budget is not
  kept. A hit is checked against the image's grid and token count in `admit`.
- **A refusal is never cached: no defect.** The front end stores a count only after a decode and count that succeeded.
  The engine keeps rows only after a whole encode. The existing test sends a corrupt image twice and gets 400 twice.
- **Budget 0 or a change at runtime: no defect.** At 0, `put` keeps nothing and the engine passes no keys
  (`caching = budget() > 0`); an existing test covers it. The budget is set once, before the Service's threads
  start. `set_budget` trims, and `LockedLruBytes` locks it.
- **Thread safety: no defect.** The engine's `image_cache_` is touched only on the engine-owner thread:
  `parse_speech`, `Batch::admit` and `stats` all run in `WorkerCore::handle`, whose only in-process caller is the
  Service's `rpc` on that thread. `set_image_cache_mb` runs in `main` before the Service starts its threads. The
  front end's cache is a `LockedLruBytes`.
- **`stats` (low, recorded).** `image_cache_entries`, `image_cache_bytes` (keys plus GPU rows) and
  `image_cache_budget_bytes` are reported by the worker's op `stats`. But `liquid-lfm3-server`, the one binary with a
  cache, never sends `stats` and has no endpoint for it. The out-of-process worker always reports a budget of 0. The
  front end's cache has no figures at all. Rule 3's "the cache's bytes (reported by `stats`)" therefore cannot be read
  from the server as built; it needs an API decision (for example, fields on `/health`).
- **Disclosure (fixed, docs).** The design promised that SECURITY.md and docs/server-api.md would describe the
  timing side channel and `--image-cache-mb`; neither did. Both now do.
- **Lookup cost (low, accepted).** Without a hash, a crafted set of images whose base64 shares a long prefix makes
  each `std::map` comparison read up to the whole key. That costs ~log2(entries) x 16 MiB of memcmp per request at
  worst (~10 ms), partly on the engine thread. It is bounded by the body limit.

Host suites after the fixes (plain, ASan+UBSan, TSan; every `tests/test_lfm3_http_*.py` and `tests/test_lru_bytes.py`):
**619 passed, 2 skipped** (36 min; 3 warnings, unhandled exceptions in client threads of existing tests whose connection closed without a reply). `test_lru_bytes.py` now also runs a plain build. Its byte-counting checks run in the plain and ASan
builds; TSan's runtime owns `operator new`, so that build checks the threads only.

## Red-team fixes C3-C5: rules, set before the code (2026-11-17)
1. **Exact:** image-cache rule 1's check (`tools/check_lfm3_image_cache.py`) 32 of 32 on the new build; the 11
   stage-0 requests through `lfm3-speech` byte-identical to `vimg-e2e-a4`; the host suites (plain, ASan+UBSan, TSan)
   all passing.
2. **C3:** after 30 distinct 4K images have been sent (more than the engine's budget holds), re-sending the first
   image decodes on the handler thread, not the engine's (`decode` vs `handoff` in the engine's log, or the harness's
   `--image-log` on the host), and image-queue rule 2's stall measured in that state is within 0.1 s of the cache-hit
   stall (window 4: 0.28 s), not the cache-off one (0.66 s).
3. **C4:** after 16 distinct 4K image requests have finished, the engine process's device memory beyond its cache
   budget and its host RSS are no higher than after the first one has finished, plus 64 MiB; compute-sanitizer
   memcheck on `lfm3-speech` with image hits, misses and evictions: 0 errors, no leaks.
4. **C5:** a fault-injection build (keep allocation forced to fail) serves the request, replies identically to the
   cache-off reply, keeps the server up, and caches nothing.
