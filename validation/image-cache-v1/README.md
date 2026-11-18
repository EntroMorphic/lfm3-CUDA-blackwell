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

## C3-C5 Thor procedure (set before the run) (2026-11-17)
Set before any of its data exists; the four rules above are not changed. Runner: `tools/thor/quiet_session_v10.sh`
(results in `build/quiet-w10-c3c5/`). It starts after window 9's `QUIET-V9-FINISHED` (or `NOT QUIET`) line, once no
engine, sanitizer, probe, build or HF process has run for three checks 30 s apart. Builds and exactness come first;
the stall comes last, after a second quiet check. Every server command runs in `lfm3-profile-v1` through
`docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia`, as in window 9. Processes are stopped
with SIGTERM only.

**Builds** (at the start of the window, from the tree as synced, in parallel; sha256s in `binaries.sha256`):
- `vimg-srv-c3` (server and worker): `tools/thor/build_native.sh` as is, in the pinned compiler image, `--network none`.
  No script in `tools/thor` records how `vimg-srv-c2` was configured. The image options are not CMake options: the
  CMake build compiles the vendored libjpeg-turbo itself. So the production configuration is used unchanged.
- `vimg-srv-c3f`: the same with `-DCMAKE_CUDA_FLAGS=-DLFM3_FAULT_IMAGE_KEEP`. Checked on x86 with CMake: the define
  reaches the server's CUDA compile line. Checked by the runner: the fault binary contains "injected: image rows not
  kept" and the normal one does not; neither links cuBLAS.
- `vimg-c3` (`lfm3-speech`): `tools/thor/build_speech.sh`. The brief's nvcc line named `tests/lfm3_speech.cu`, but the
  source is now `native/apps/lfm3_speech.cu`, linked with the image front end and libjpeg-turbo, which is what
  `build_speech.sh` does (same image and flags).

**Images** (`tools/make_lfm3_c3_images.py`, standard library only; sha256s in `images.sha256` and each manifest):
`images/prime/c3-01..30.png` and `images/fresh/c3-31.png`. Each is cherry (3840 x 2160) with 16 bytes of one scanline
changed (row `71 x index mod 2160`), recompressed at zlib level 6. Each file is about 9.89 MB. Its base64 key is about
13,181,400 bytes, and it is 1,980 tokens, so 8,110,080 bytes of rows. Cost per image: about 21.29 MB in the engine
(key plus rows) and 13.18 MB in the front end (key). Cherry's own key is 13,111,720 bytes.
With the default budgets (512 MiB = 536,870,912 bytes each):
- The engine gets cherry plus 30 primes, about 660 MB, so cherry is evicted.
- The front end gets small, cherry and 30 keys, about 408.6 MB, so cherry's count is kept.

The check tool recomputes both figures from the images and refuses to run if either does not hold.

**Rule 1 (exact).**
- (a) `$X lfm3-speech MODEL --batch` on `vimg-e2e-a4`'s 11 requests (`build/vimg-e2e-a4/req`, written by
  `tools/thor/vision_e2e_round.sh`: text mode, 128 tokens, `record_image`). Then `tools/compare_speech_outputs.py
  build/vimg-e2e-a4/out e2e/out`: every file identical, the JSON's timing fields excepted, as in earlier rounds.
- (b) `tools/check_lfm3_image_cache.py vimg-srv-c3 MODEL vimg-e2e-a4 image-cache.json LISTEN`: **32 of 32**. LISTEN
  is the listening set's `req/`. The runner finds it as the `build/*/req` that holds `01-china.json` and
  `08-mm05.json`, or takes `LISTEN_DIR`.
- (c) The host suites, on x86 at `e2f8b7f` (which contains 0d743af): every `tests/test_lfm3_http_*.py` and
  `tests/test_lru_bytes.py`, plain, ASan+UBSan and TSan, **622 passed, 2 skipped** (28 min; 3 warnings, the known
  client-thread exceptions). The new tools' own tests (`tests/test_lfm3_image_cache_fixes_tools.py`, against
  `tests/fake_lfm3_image_server.py`): 18 passed.

**Rule 2 (C3).**
- **Where the image is decoded: cannot be measured as written.** The rule names "`decode` vs `handoff` in the
  engine's log". The real server has no such log; those lines exist only in the host harness (`--image-log`).
  **Stand-in:** each thread's CPU time (`/proc/PID/task/TID/stat`, utime + stime) across each request, read by
  `tools/check_lfm3_image_cache_fixes.py c3`. A 4K decode is ~0.3-0.4 s of one thread's CPU. The engine-owner thread
  is the one that spends the most on a 64-token text request.
- **c3 sequence:** text and small (warm-up); text (finds the engine thread); cherry (the seed); cherry (the hit);
  c3-01..30, each once with 1 token; cherry (**the probe**); cherry (a hit again); c3-31 (a never-seen image).
  Cherry requests are the stage-0 request: 128 tokens.
- **Reading:** decoded on the handler when the probe's largest non-engine thread used **>= 0.20 s** and the engine
  thread used at most the fresh image's engine CPU **+ 0.20 s**. Decoded on the engine thread when the other threads
  used <= 0.10 s and the engine used >= the fresh image's + 0.20 s.
- **Void unless the controls hold:**
  - the hit's largest non-engine thread <= 0.10 s;
  - the fresh image's >= 0.20 s;
  - the probe's `prefill_seconds` at least the midpoint of the hit's and the fresh image's (the engine had to encode
    again, so the rows had been evicted).
- **Also required:** every cherry reply's text equal to `lfm3-speech`'s, and every status 200.
- **The stall:** `tools/measure_lfm3_image_stall.py` (cherry, 3 repeats), three runs on `vimg-srv-c3` in the same
  window:
  - cache on, as in window 4;
  - **`--prime-images images/prime`** (new option): after the warm-up, cherry is sent once alone. Then before
    **every** measured run, with and without the image, c3-01..30 are sent one by one. So every "with" run re-sends
    cherry in the C3 state. Priming only once would leave repeats 2 and 3 as plain hits.
  - `--image-cache-mb 0`.

  Without the option the tool is unchanged (host test: arguments and output keys).
  - **Bar as written:** the primed stall within 0.10 s of 0.28 s.
  - **Prediction, recorded now:** about **0.66 s**, so the bar as written is expected to fail. In the C3 state the
    engine no longer holds the rows. With the fix, the re-send is a miss whose decode ran on a handler: the engine
    thread still encodes (~0.36 s) and prefills. That is window 3's cache-off condition (0.66 s), not a hit (0.28 s).
    Without the fix the engine thread would also decode (~0.4 s), as before image-queue part 2 (1.06 s). So the
    rule's two reference points are one step off.
  - **Also reported:** whether the primed stall is <= this window's cache-off stall + 0.10 s, meaning the engine
    thread did not decode.

  Both lines are printed. The rule's verdict stays the literal one.

**Rule 3 (C4).**
- **Memory: "device memory" cannot be measured per process on Thor.** The GPU allocates from system memory, and no
  `/proc` field shows it (`tools/thor/host_memory_sampler.py`). "Beyond its cache budget" is read as "beyond what its
  caches hold".
- The server reports no cache figures (see `stats` in the red-team), so the caches' growth is computed from the
  images:
  - host: each image's key twice (the front end's copy and the engine's): 15 x 2 x ~13,181,400 = ~395.4 MB;
  - device: each image's rows: 15 x 8,110,080 = 121.65 MB.
- **Stand-ins:**
  - host: the server's VmRSS;
  - device: Thor's in-use memory (MemTotal - MemAvailable), less the server's RssAnon + RssShmem, less the client's.
- `check_lfm3_image_cache_fixes.py c4`, server with `--queue-size 16` (default 8 rows):
  1. Text and small (warm-up).
  2. c3-01 alone, then **after_1**.
  3. c3-02..16 sent **at once**, so that they spread over the rows (sent one by one, they would all reuse one row and
     hide C4). Then **after_16**.
  4. A reading 10 s later is recorded, not gated.

  Each reading is the median of 5 samples, 0.5 s apart, after 3 s idle. All 16 requests must return 200.
- **Gate:** VmRSS(after_16) - VmRSS(after_1) - 395.4 MB **<= 64 MiB**, and the device stand-in's growth minus
  121.65 MB **<= 64 MiB**. Allocation rounding of the kept rows falls inside the margin.
- **Control, not gated:** the same procedure on `vimg-srv-c2` (before C1 and C4), to show the measure moves.
- **Memcheck: cannot be measured as written on `lfm3-speech`.** It has no image cache ("`lfm3-speech` sets no cache"
  above), so it has no hits and no evictions. Two runs instead:
  - **(i) `lfm3-speech`** under `compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99`,
    `--concurrent 4`, on cherry, two, small, small, flower (16 tokens; `speech-list`). This covers C4's code: shared
    pixels, `admit` by value, the one-chunk move, the row without media, rows reused, and a repeated image. Required:
    exit 0, **ERROR SUMMARY 0 errors, 0 bytes leaked**, outputs identical to the same run without the sanitizer.
  - **(ii) The server**, the only binary with the cache, at `--image-cache-mb 1` under memcheck. There is **no leak
    check**: the server ends with `std::_Exit` by design and frees nothing, so every live allocation would be
    reported. The sequence is small, small, wide, tall, small, tall, china, two, wide (8 tokens each), then small,
    tall and wide at once, then small. Expected engine states, simulated from the stage-0 sizes: miss, hit, miss,
    miss (evicts 1), miss (evicts 1), hit, over budget (not kept), over and over, miss (evicts 1). Required: exit 0,
    ERROR SUMMARY 0 errors, every status 200, texts equal to the plain run of the same sequence.

**Rule 4 (C5).** `check_lfm3_image_cache_fixes.py c5` runs three servers in turn:
- `vimg-srv-c3 --image-cache-mb 0` (off);
- `vimg-srv-c3` (on, the calibration);
- `vimg-srv-c3f` (fault).

Each gets: cherry 3 times unstreamed and once streamed; two, twice; then `GET /health`, a text request, and SIGTERM.
- "Replies identically to the cache-off reply": every text equal to the off server's, and to `lfm3-speech`'s.
- "Keeps the server up": `/health` 200, the text request after it 200, the process alive, exit 0 on SIGTERM.
- **"Caches nothing" cannot be read directly** (no stats endpoint). **Stand-in:** each re-send of cherry to the fault
  server is decoded on a handler again (largest non-engine thread >= 0.20 s: the engine published no held rows). Its
  `prefill_seconds` must also be at least the midpoint between the cache-on server's miss and hit (it was encoded
  again).
- Calibration required: the cache-on server's re-sends are hits (prefill at least 0.1 s below its miss, no handler
  decode). `on` must also equal `off`.

**Expected runtime:** builds ~25 min, exactness ~30 min, C3/C4/C5 ~15 min, memchecks ~1-3 h, stall ~20 min.

## C3-C5 Thor results (quiet window 10, 2026-11-17 15:01-15:31; `quiet-w10-c3c5/`)
Builds: `vimg-srv-c3` (server `f653f8d0`, worker `66aeeed4`), the fault build `vimg-srv-c3f` (`aee511fb`; the
injected string in it only), `vimg-c3/lfm3-speech` (`b281be64`); no cuBLAS.

| rule | verdict | figures |
|---|---|---|
| 1 exact | **PASS** | `lfm3-speech`: 67 of 67 files identical to `vimg-e2e-a4`; `check_lfm3_image_cache.py`: **32 of 32**; host suites passed before the run |
| 2 C3, where the evicted image is decoded | **NOT MEASURED** (instrument failure) | its own control failed: the never-seen image (must decode on a handler) showed 0.00 s on every non-engine thread. The server runs each connection on a detached thread that exits with the request (`lfm3_http.hpp:2534`), so its CPU is gone from `/proc/PID/task` before the second reading |
| 2 C3, the stall | **FAIL as written**, as predicted before the run | C3 state **0.625 s** against the bar 0.28 +/- 0.10; this window's hit 0.256, cache off 0.616 (window 3: 0.66; before image-queue part 2, the engine decoding: 1.06) |
| 2 C3, second check (set before the run) | yes | C3 state 0.625 <= cache off 0.616 + 0.10: the engine thread is not decoding (that would add ~0.4 s) |
| 3 C4 memory | **PASS** | after 1 vs 16 distinct 4K requests: RSS +380.2 MiB, the caches' keys +377.1, **excess +3.1 MiB** (bar 64); device stand-in +2.5 MiB against rows +116.0. Control, the pre-fix `vimg-srv-c2`: excess **+797.3 MiB** |
| 3 C4 memcheck, `lfm3-speech` | **PASS** | 0 errors, 0 bytes leaked; 25 of 25 outputs identical to the plain run (the runner's comparison could not write into the root-owned directory; rerun on the host the same evening, written beside it) |
| 3 C4 memcheck, the server | **FAIL as written** | 0 errors (`ERROR SUMMARY: 0 errors`), all 200, texts equal to plain; but exit **15**, not 0 (plain: exit 0). Suspected, not verified: the server blocks SIGTERM in `main` and `sigwait`s; threads the sanitizer starts first do not block it, so SIGTERM kills the process by default action |
| 4 C5 | **FAIL as written** (instrument) | the fault build equals cache off and `lfm3-speech`, stays up; re-sends' prefill 0.592/0.592 s (cache-on miss 0.614, hit 0.228): **nothing was served from the cache**. The gate's other half (decoded on a handler again) used the same failed thread-CPU reading |

Next (a later window): the thread-CPU reading from `/proc/PID/stat` (the process total, which keeps exited threads'
time) minus the engine thread; C3's location and C5 rerun with it, their gates unchanged; the server under memcheck
with every thread's blocked-signal mask (`SigBlk`) recorded before the SIGTERM, to test the suspected cause.

### Amendment to the C3-C5 procedure (2026-11-17, after window 10, before any rerun)
- **The thread-CPU reading** (C3's location, C5's "decoded on a handler again"): "the largest non-engine thread" cannot
  be read for this server, whose handler threads end with their request. It becomes **every non-engine thread's CPU,
  ended threads included**: the process's total from `/proc/PID/stat` minus the engine thread's. The thresholds are
  unchanged (decoded >= 0.20 s, not decoded <= 0.10 s, engine margin 0.20 s), and so are the controls that say whether
  the reading works (the hit <= 0.10 s; the never-seen image >= 0.20 s). If a control fails again, the location stays
  not measured. The process total also holds the handlers' other work on a 13 MB body (receive, two JSON parses); the
  hit control bounds it. The fake server in the host tests now runs a thread per connection, as the server does (the old
  reading fails against it; the new passes, 18 of 18).
- **The server under memcheck**: before the SIGTERM, every thread's `SigBlk` is recorded. If some thread doesn't block
  SIGTERM, the suspected cause of window 10's exit 15 holds, and it's the server's signal design, not the image cache.
  The gate stays exit 0: a fix would be the server's (a handler that only wakes the waiting thread, so it doesn't matter
  which thread takes the signal), made and checked as a change of its own.

## C3-C5 rerun (quiet window 13, 2026-11-17 17:41-17:45; `quiet-w13-c3c5/`), with the amended reading
Window 10's builds and images, unchanged.
| Check | Result | Reading |
|---|---|---|
| C3 location | **PASS** | other threads 0.45 s on the evicted re-send (hit control 0.05, never-seen control 0.47), engine 0.16 s (its miss 0.15); texts equal; decoded on a handler |
| C5 (fault build) | **PASS** | all 6 checks pass: fault re-sends prefill 0.592 s (on: miss 0.612, hit 0.228), other-thread CPU 0.45-0.46 s (decoded on a handler every time), replies equal to the cache off and to lfm3-speech |
| Server memcheck, `--image-cache-mb 1` | **FAIL as written (exit 15); 0 errors** | plain: exit 0, statuses 200, expected cache counts. memcheck: 0 errors, texts equal to plain, same cache counts, exit 15 |

**Why memcheck exits 15.** Before the SIGTERM:
- The plain server has 14 threads, and 1 does not block SIGTERM. That is the signal thread itself: while a thread waits in `sigwait`, Linux takes the waited signals out of its blocked mask.
- Under memcheck there are 19 threads, and 2 do not block SIGTERM. The extra one is a thread the server didn't start with the mask. A process-directed SIGTERM can be delivered to that thread. Its action is the default one, so the process dies of signal 15 instead of shutting down in order.

So the cause suspected after window 10 holds. It's the server's signal design, not the image cache. The fix is its own change: a SIGTERM/SIGINT handler that only forwards the signal to the waiting thread, so it doesn't matter which thread receives it. It is checked by the host suites and a memcheck rerun.

**C3-C5 standing:**
- C3 location: PASS.
- C3 stall: FAIL as written (window 10), but the engine thread does not decode.
- C4 memory: PASS. lfm3-speech memcheck: PASS. Server memcheck: 0 errors, exit gate pending the signal fix.
- C5: PASS.
