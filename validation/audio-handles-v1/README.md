# Audio clips handed to the in-process engine by handle, not by their bytes

Written 2026-11-17, before any code. Nothing below has been run.

## What it will be
validation/audio-frontend-v1 moved each clip's features off the engine thread: the handler computes them (F2) and the
in-process engine takes them from a cache. Rule 3 of that round (the stall a clip's admission causes the other rows)
still failed as written, by 1–54 ms in 6 of 10 cells.

Quiet window 16's nsys trace (audio-frontend-v1, "What the trace shows") accounts for the 90 s clip's excess at 8 rows:
- **about 35 ms:** the engine thread parses the 3.8 MB request line and decodes the clip's base64 and WAV
  (`parse_speech`), only to count its samples, before it finds the features;
- **about 15–20 ms:** contention from the handler's `log_mel` while it runs;
- **about 5–10 ms:** the clip's own row in the batch.

This round removes the first. The second and third are out of scope (below). The round depends on audio-frontend-v1's
F2 and is branched from it (`fa00262`).

## The change (as proposed in audio-frontend-v1, 2026-11-17)
- When every clip of a request has its features in the cache, the in-process server's engine request carries a
  handle for each clip (`"audio_ref"`) instead of its base64.
- The cache entry holds the features and the clip's sample count, the only thing the engine still needs from the
  samples (the token count and the features' length check).
- `WorkerCore::parse_speech` resolves each handle under its lease. It neither parses a multi-megabyte JSON string nor
  decodes base64 or WAV.
- A handle is assigned per cache entry and stays valid while the request's lease lives, so it cannot name another
  clip.
- An unknown handle is a refusal with its own message, never a silent recomputation; it is unreachable by
  construction.
- A request whose clips are not all in the cache, the out-of-process worker, and lfm3-speech all send and take the
  base64 exactly as today.

Expected: about 35 ms less of the 90 s clip's stall (about 12 ms at 35 s). That should bring rule 3's 8-row cells near
or under the bar, except where `log_mel`'s contention lands in the interval.

## Rules, set before any code (audio-frontend-v1's proposal, rules 1–5, as written)
1. **Exactness:**
   - the server's audio-in check (both front ends) and 8 and 32 requests at once: every reply byte-identical to
     lfm3-speech's;
   - the 360 + 93 stage files and the 2,620 corpus through lfm3-speech: unchanged (lfm3-speech does not use handles);
   - a request mixing cached and uncached clips sends base64, and gives the same replies.
2. **Rule 3 again, as written** (audio-frontend-v1 amendment 3's method, the same bar): at 8 and 16 rows, all five
   clips. A remaining failure is recorded, and the contention part is then the next diagnostic.
3. **The engine thread's admission work, reported** (nsys, as window 16): the CPU-only time before the encoder's first
   kernel for the 90 s clip. Expected under 3 ms against 35.4.
4. **Safety:**
   - host suites in full (plain, ASan+UBSan, TSan), with tests for a handle outliving its request (refused), a request
     cancelled after its handles were sent, and the same clip in requests at once;
   - compute-sanitizer memcheck on the server path;
   - a red-team pass on the handle lifetime.
5. **Nothing else changes:** audio-frontend-v1 amendment 3's rule 6 list again; the out-of-process worker's protocol
   and replies unchanged.

## Out of scope, open
The remaining 15–20 ms is `log_mel`'s contention with the decode steps. Three options, none started, each needing a
measurement first:
1. lower the `log_mel` threads' scheduling priority;
2. run them on cores the engine thread does not use;
3. the cheaper direct DFT, which is not correctly rounded. That would reverse audio-frontend-v1 amendment 2's decision
   and is **Tripp's call**.

## The code (2026-11-17, after the rules)
- `lfm::audio_features::FeatureCache` (`audio_features.hpp`):
  - each entry holds a handle (a 64-bit counter, assigned when the entry is created and never reused) and the clip's
    sample count beside its features;
  - `resolve(handle)` returns the features and sample count only while some lease holds the entry;
  - `clip_of(handle)` is for tests.
- `Service::compute_audio_features` (`lfm3_http.hpp`): when every clip of a request has a lease, the request gets
  `audio_refs_json`, and `audio_field` puts `"audio_ref": [handles]` into the engine request in place of `"audio"`.
  Otherwise the base64 goes as before, for every clip. `LFM3_AUDIO_AHEAD_SKIP=k` (diagnostic, an integer 0..63)
  leaves a request's k-th clip out, for rule 1's mixed requests.
- `WorkerCore::parse_speech` (`lfm3_worker.cu`):
  - takes `"audio_ref"`, resolving each handle;
  - an unknown or expired handle is refused ("unknown audio handle N"), as is a request carrying both `"audio"` and
    `"audio_ref"`;
  - it fills `Options::audio_lengths` (sample counts) and `audio_features`, and no samples.
- The speech engine (`lfm3_speech.cuh`): the sample counts come from `audio_clip_length()` (samples or
  `audio_lengths`). A request by handle must have every clip's features and no samples, or it is refused.
- The out-of-process worker and lfm3-speech never see `"audio_ref"`: `Policy::audio_features_ahead` is on in
  liquid-lfm3-server only.
- Tools:
  - `tools/nsys_audio_admission.py` reads rule 3's reported figure from a trace. On window 16's trace it gives 35.3 ms,
    as found by hand;
  - `tools/thor/quiet_audio_rest.sh` takes `QAR_OUT` and `QAR_TAG`, so window 19 runs it from this tree without
    touching window 12's builds or results;
  - `tools/thor/quiet_session_v19.sh` (not launched).

## Rule 4's red-team of the handle lifetime (2026-11-17): the cases checked, and how
| case | how checked | finding |
|---|---|---|
| A handle sent while its request's lease is gone | Code. The engine request is framed and passed to the in-process engine in one synchronous call (`backend_.call`) from code that holds the request: `speech()`'s `request`, `speech_stream()`'s, and the batch scheduler's ticket (`t->request`) for `speech_admit`. `parse_speech` resolves every handle inside that call. | Sound: a handle is only ever resolved while its lease lives |
| The engine's use after the request ends | Code. `resolve` copies the features' `shared_ptr` into `Options`. The encoder reads them at admission, inside the same call. | Sound: no use of the entry after resolution |
| A handle outliving its request | Unit test (`audio-input-tool cache-handles`, all three builds). After the last lease is released, `resolve` and `clip_of` return nothing. | Refused, as required |
| An unknown handle | Unit test (handles 0 and 2^64−1). In `parse_speech`, any id ≤ 0 or unresolved is refused with its number. | Refused, never recomputed |
| A handle naming another clip | Unit test: handles are never reused (a clip put again after release gets a new one). HTTP harness: the fake engine checks every handle's features against `log_mel` of the clip the handle names (`clip_of`), including two clips differing only in their last sample, sent at once 14 times. | Sound |
| The same clip in requests at once, or twice in one | Unit test: one entry, one handle, two leases, freed after the last. HTTP tests: the same clip in requests at once, and two clips in one request. | Sound |
| A request cancelled after its handles were sent | HTTP test `test_features_are_dropped_when_a_request_ends_however_it_ends`: a refused request, a client gone while queued, a 504 deadline, a stream dropped mid-reply; then a request alone finds exactly its own entry, by handle ("entries 1 refs"). | Sound: entries released |
| Concurrency | Unit test under TSan: 8 threads, put, resolve and `clip_of` over shared clips, 2,000 times each. `Lease::handle()` reads the handle without the lock; it is written once, under the lock, before any lease on the entry exists, so the lock orders the write before every read. | No race reported |
| Exceptions | Code. `put` erases a half-made entry if indexing its handle throws. The `Lease` constructor and `handle()` cannot throw. | Sound |
| A malformed request to the engine | Code. `"audio"` and `"audio_ref"` together are refused, as is a request by handle with samples or with a clip lacking features (`lfm3_speech.cuh`). | Refused |
| The out-of-process worker sent a handle | Code. It never is: the policy is off outside liquid-lfm3-server. If one arrived, its empty cache makes it an unknown handle. | Refused |
| Mixed requests | HTTP test: with a clip left out, the whole request goes as base64 with the same prompt, and the fake engine still checks the held clip's features. | Sound |

**Not changed:**
- a request by handle still keeps its clips' base64 in `Speech::audio_json` until it ends (memory as audio-frontend-v1
  measured it: growth 309 MiB for 16 × 90 s);
- this pass was mine, not an independent reviewer's. Rule 4 asks for "a red-team pass on the handle lifetime"; an
  independent one can follow if Tripp wants it.
