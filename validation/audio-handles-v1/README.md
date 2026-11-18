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
