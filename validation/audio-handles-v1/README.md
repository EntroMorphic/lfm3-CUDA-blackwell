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

## Rule 4, host suites (2026-11-17, workstation): PASS
`tests` and `audit/test_host_contracts.py`, in full, plain, ASan+UBSan and TSan, from a copy of this branch with the
checkpoint beside it: **2,442 passed, 2 skipped, 1 failed** (53 minutes).
- The failure is the known pre-existing `test_speech_text.py::test_every_validation_prompt_matches_hf_ids`
  (transformers 4.57.6 cannot load the checkpoint's remote code; it fails on main too).
- The 2 skips are `test_lfm3_http_native.py`'s memory tests, which run only on the uninstrumented build.
- New in this round:
  - `test_audio_input_host.py::test_feature_cache_handles` (the handle lifetime unit test, all three builds);
  - `test_lfm3_http_audio_in.py::test_a_request_with_a_clip_left_out_sends_base64_with_the_same_prompt`;
  - the eviction test now checks that the probe's own clip came by handle.
  - The audio-in HTTP suite alone: 84 of 84.

## What runs on Thor, not yet run
`tools/thor/quiet_session_v19.sh` (waits for `QUIET-V18-FINISHED`; window 15's `busy()` and `wait_quiet`):
- rule 3 as written;
- rule 5's list (through `quiet_audio_rest.sh` with its own names);
- rule 1's exactness: audio-in with each front end, mixed requests, 8 and 32 at once;
- rule 4's memcheck on the server path;
- rule 3's reported figure from an nsys trace.

It needs this branch synced to `build/audio-src-h` and the base sources in `build/audio-src-base` (already there).

## Quiet window 19 (2026-11-18, Thor, `f1dac0a`): NOT RUNNABLE, a fault in this branch
Results in Thor's `build/quiet-w19-handles/` and `build/quiet-w19-handles-rest/`, as the coordinator reported them:
- **The new server (`audio-h-w-srv`) answered every request containing an audio clip with 503 `worker_unavailable`,
  and every request after it, text included.** It failed closed within 0.01 s of the first audio request, with
  nothing on stderr. No rule could be measured on it.
- The memcheck step (3) ran for 6 hours before it was stopped. Not yet explained (below).
- The base server's stall step at 8 rows stopped with "no row streamed audio across the window": NOT RUNNABLE. The
  baseline worst gaps at 8 rows are about 1.15 s in earlier windows.

### The cause (found on the workstation, 2026-11-19)
`WorkerCore` checks every request's fields against a fixed list per op (`schema()` in `lfm3_worker.cu`) before it reads
any of them. `"audio_ref"` was never added to the list for ops `speech` and `speech_admit`. Every request by handle
was therefore refused with "unsupported request field: audio_ref". The Service treats any refused engine operation as
fatal, so it failed closed, and answered 503 to everything after.

**Why the host suites passed:**
- the host harness's fake engine reimplements `parse_speech`'s checks, and it accepted `"audio_ref"`;
- it had no copy of `schema()`;
- the real `WorkerCore` runs only on Thor.

**Why stderr was empty:**
- the Service turned the worker's error into `WorkerFailure("native operation rejected")`, dropping the reason;
- `fatal()` logged nothing.

### The fix
- The field lists moved to `native/include/lfm/lfm3_worker_schema.hpp`, with `"audio_ref"` added for ops `speech` and
  `speech_admit`.
  - `WorkerCore` calls it as before.
  - The fake engine now runs every frame the Service sends through the same function, so a field the Service starts
    sending is refused on the host exactly as on Thor.
- **Reproduced on the host:** the same header with `"audio_ref"` taken back out fails 27 of the 28 audio-in HTTP tests
  (plain build) with 503 `worker_unavailable`. The one that passes is the engine-computed half of a test whose other
  half fails. With the fix, all 28 pass.
- **Logging:**
  - the refused operation now carries the worker's own error ("native operation rejected: unsupported request field:
    audio_ref");
  - `fatal(where)` writes the first failure's reason to stderr once:
    `liquid-lfm3-server: worker failed closed (WHERE): REASON` (at most 1,000 characters; clients still see only 503);
  - the watchdog writes `liquid-lfm3-server: worker failed closed (watchdog): an engine call ran past engine_timeout`;
  - shutdown's own `fatal()` call logs nothing.
- **`tools/measure_lfm3_audio_stall.py`:** a run it cannot measure now says why, without changing the measurement:
  with or without the clip, and per row the tries, the first attempt, the stream's start and end, and the first and
  last audio, each in seconds from the clip request's start.
- **Host tests:**
  - the tests that fail the engine on purpose (`--fail-op`, `--speech-stream-fault`, the engine timeout: 13 in each
    build) now require exactly the one failed-closed line, with its reason;
  - every other test still requires an empty stderr;
  - `test_text_batch_cancellation.py` builds `WorkerCore`'s source on the host, and now includes the new header.
- **Host suites in full again** (`tests` and `audit/test_host_contracts.py`; plain, ASan+UBSan, TSan; 2026-11-19,
  workstation, 51 minutes): **2,440 passed, 4 skipped, 1 failed**.
  - The failure is the known pre-existing `test_every_validation_prompt_matches_hf_ids`.
  - The skips are the 2 memory tests, as before, and `test_speech_normalize.py`'s 2 corpus tests: the corpus data was
    not in this copy.
- **The server's translation unit on x86 (nvcc 13.5, sm_90):** only the 3 known errors from the quad type
  (`__float128` there, `long double` on Thor). Thor's build is the real compile check.

### The memcheck step's 6 hours: explained from run 1's files, and fixed
**From run 1's files** (`quiet-w19-handles-run1/memcheck/`, the coordinator's account of the processes at 05:12):
- the check sent its 9 cases by 23:14, every one a 503, then stopped the server with `proc.terminate()`;
- `proc` was `srv-memcheck.sh`, which execs compute-sanitizer. SIGTERM killed compute-sanitizer itself (the log has
  "exit: -15"), and compute-sanitizer did not pass the signal on;
- the server lived on under TreeLauncherSubreaper, idle, holding the check's stderr pipe;
- `proc.wait()` returned at once; `proc.stderr.read()` then waited for an end of file that never came;
- `audio-in-server.json` was written at 05:13, just after the coordinator sent SIGTERM to the server inside the
  container;
- `memcheck-1407.log` has only its header: compute-sanitizer died before any summary.

So it was not a full pipe, but a read to the end of a pipe that a surviving process held open. The pipes were not
drained either, which a talkative server could also have blocked on.

**The fix** (`tools/check_lfm3_audio_in_server.py`):
- SIGTERM goes to the server itself: the process at or below `proc` whose argv[0] is `liquid-lfm3-server`, found in
  `/proc`. Under compute-sanitizer, that is the target. compute-sanitizer then sees it exit, writes its summary and
  exits with the target's code;
- run directly, `proc` is the server and gets the signal as before;
- stdout and stderr are drained in threads from the start, so nothing blocks on a full pipe or on a pipe's end;
- the wait for the exit is bounded (900 s, for memcheck's leak check), never followed by SIGKILL. A server still alive
  is recorded in the JSON (`still_running_after_sigterm`) and printed;
- the server is stopped even when a case raises.

**Checked on the workstation** with a stand-in launcher that, like compute-sanitizer, dies of SIGTERM and waits for a
child server holding its pipes:
- the server got the signal;
- the launcher wrote its summary and exited 0;
- the stderr line was kept;
- no process was left.

**In window 19's rerun (run 2, launched 06:43 on `16bbe85`):**
- at 06:47:11, during step 1 and before any run of the check, the coordinator swapped `f8e2299`'s
  `tools/check_lfm3_audio_in_server.py` (sha256 `01388a82…`) into Thor's `build/audio-src-h`;
- that tree's `SOURCE_COMMIT` reads "16bbe85 + tools/check_lfm3_audio_in_server.py from f8e2299";
- steps 1 (rule 6's list), 2 and 3 all use this check.

Run 1's outputs are kept as `build/quiet-w19-handles-run1/`, `build/quiet-w19-handles-rest-run1/` and
`quiet-w19-run1.log`.

### The base server's "no row streamed audio across the window" (run 1)
`stall-audio-fe-w-base-srv-8.log`, 8 rows:

| clip | worst gap without | with | stall |
|---|---|---|---|
| short | 1.1540 s | 1.2934 s | 0.1393 s |
| median | 1.1591 s | 1.3957 s | 0.2367 s |
| typical | 1.1579 s | 1.4613 s | 0.3034 s |
| long | 0.4580 s | 1.4325 s | 0.9746 s |
| 90 s | (stopped: no row had audio inside the window) | | |

- The long clip's "without" median, 0.458 s, is far below the other clips' 1.15 s, unexplained.
- **Hypothesis, not checked:** the 90 s clip's admission held every row silent for the whole 4 s window. Rule 0's
  3.3 s stall at 90 s was close to that. The tool's definition then has no gap to measure.
- The tool's message now says whether the run had the clip, and when each row's audio was, so the rerun will show it.

### To rerun on Thor (the fix's commit synced to `build/audio-src-h`; nothing else changes)
`tools/thor/quiet_session_v19.sh` in full, as window 19 ran it:
- the handles server rebuilds, because its tree hash changes;
- the base server's build is reused.

## Quiet window 19, run 2 (2026-11-19, Thor, 06:43–08:41; QUIET-V19-FINISHED)
- **Source:** `16bbe85`, plus `f8e2299`'s audio-in check, swapped in before its first use (above).
- **Logs on Thor:** `build/quiet-w19.log`, `build/quiet-w19-handles/`, `build/quiet-w19-handles-rest/`.
- Figures as the coordinator reported them.

**Overall: rule 3 FAILS as written (the long clip, 2 of 10 cells). Rules 1 and 5 pass. Rule 3's reported figure is
1.5 ms, under the expected 3 ms. Rule 4's memcheck is clean (0 errors, 0 leaks), but its check has one unexplained
FALSE, open below.**

### Rule 1, exactness: PASS
- **The audio-in check, three ways:** with each front end (`dft`, `exact-f2`) and with mixed requests (`skip1`). Each
  time the repeat was identical, the text control identical, and the exit 0.
- **At once:** 8 at once gave 8 of 8 identical to lfm3-speech (2.5 s); 32 at once gave 32 of 32 (4.3 s).
- **The 360 + 93 stage files:** unchanged (in rule 5's list below). F2's 5 differences are the expected ones.

### Rule 3 again, as written: FAIL
The stall in seconds, both servers in the same window, with each cell's bar.

| rows | clip | base (rule 0's server) | new (handles) | bar | |
|---|---|---|---|---|---|
| 8 | short | 0.0740 | 0.0338 | 0.0468 | PASS |
| 8 | median | 0.1046 | 0.0427 | 0.0547 | PASS |
| 8 | typical | 0.1701 | 0.0587 | 0.0702 | PASS |
| 8 | long | 0.4922 | 0.1489 | 0.1413 | **FAIL** (+5.4%) |
| 8 | joined 90 s | 1.2291 | 0.3137 | 0.3177 | PASS |
| 16 | short | 0.0493 | 0.0282 | 0.0468 | PASS |
| 16 | median | 0.0986 | 0.0376 | 0.0547 | PASS |
| 16 | typical | 0.1644 | 0.0574 | 0.0702 | PASS |
| 16 | long | 0.4915 | 0.1491 | 0.1413 | **FAIL** (+5.5%) |
| 16 | joined 90 s | 1.2222 | 0.3161 | 0.3177 | PASS |

- **The new server:** under the bar in 8 of 10 cells. The long clip is over by about 8 ms, at both row counts.
- **"Base" is rule 0's base server** (`audio-fe-w-base-srv`, before the handler-thread front end), as in window 12's
  table. It is not audio-frontend-v1. Its figures agree with window 12's to within a few ms (90 s: 1.229 and 1.222
  against 1.234 and 1.230).
- **Against audio-frontend-v1's own figures (window 12, a different window; its bars were 0.3020 and 0.1327 there):**

  | rows | clip | audio-frontend-v1 (w12) | handles (w19) | change |
  |---|---|---|---|---|
  | 8 | short | 0.0376 | 0.0338 | −3.8 ms |
  | 8 | median | 0.0501 | 0.0427 | −7.4 ms |
  | 8 | typical | 0.0693 | 0.0587 | −10.6 ms |
  | 8 | long | 0.1440 | 0.1489 | +4.9 ms |
  | 8 | 90 s | 0.3555 | 0.3137 | −41.8 ms |
  | 16 | short | 0.0323 | 0.0282 | −4.1 ms |
  | 16 | median | 0.0437 | 0.0376 | −6.1 ms |
  | 16 | typical | 0.0592 | 0.0574 | −1.8 ms |
  | 16 | long | 0.1384 | 0.1491 | +10.7 ms |
  | 16 | 90 s | 0.3533 | 0.3161 | −37.2 ms |

  - At 90 s the handles took off 37–42 ms, as predicted (about 35 ms from window 16's trace).
  - The long clip came out 5–11 ms worse, at both row counts.
  - **Not explained:** the parse saved there should have been about 12 ms. The same base server moved by up to
    14.5 ms between windows 12 and 19 (8 rows, short: 0.0595 against 0.0740), so a cross-window change of this size
    is not yet evidence. A same-window comparison of audio-frontend-v1 and the handles server would settle it.
  - The bars themselves moved between windows: the long clip's from 0.1327 to 0.1413, the 90 s clip's from 0.3020 to
    0.3177.
- **Recorded as written:** a failure. Closing the remaining 8 ms is the `log_mel` contention question (out of scope,
  above).
- **Run 1's base-server stop at the 90 s clip did not recur:** this time the base server measured all five clips, the
  90 s clip with a 1.23 s stall. That does not support run 1's hypothesis (a stall longer than the 4 s window). Run 1's
  stop stays unexplained, with no diagnostics from that run.

### Rule 3's reported figure (nsys, the 90 s clip at 8 rows, new server)
| figure | run 2 | window 16 (base) |
|---|---|---|
| the engine thread's CPU-only time before the encoder | **1.5 ms** (expected under 3) | 35.3 ms |
| CPU-only time ending before the launch | 96.4 ms | |
| encoder, GPU | 157.1 ms | |
| decode step before | 25.6 ms | |

### Rule 4, safety
- **Host suites:** PASS (above).
- **Red-team:** done (above).
- **compute-sanitizer memcheck on the server path:**
  - ERROR SUMMARY 0 errors; LEAK SUMMARY 0 bytes in 0 allocations;
  - the hang is fixed: the server got SIGTERM, and compute-sanitizer wrote its summary.
- **The server exited 15** ("Target application returned an error"). Known: this branch has no
  `lfm/stop_signals.hpp`, since it predates main's stop-signal forwarding (`064c1bb`, windows 13 and 17's same exit
  15). Merging main brings it in.
- **The check under memcheck:**
  - the 90 s case missed its wall bound (242.4 s against 240), expected under the sanitizer;
  - every other case returned 200 and was identical;
  - **`text_control_identical` was FALSE, not explained.** The check kept neither the second `/generate`'s status nor
    the ids. Stderr was empty, so the service did not fail closed.
- **The leading hypothesis, not checked:** the control session was reaped.
  - It is created and reset before the cases, then sits idle until the second `/generate`.
  - The server's default `session_idle_timeout` is 300 s.
  - Under memcheck the cases take far longer: the 90 s case alone took 242 s. The session would then be reaped, and
    the second `/generate` answered 404 `unknown_session`, which the check counted as FALSE.
  - Without the sanitizer the cases take a minute or two, which fits all five unsanitized runs passing the control.
- **Other causes the record must rule out:**
  - a client timeout: the second `/generate` has a 600 s bound, likely enough under memcheck for 128 tokens, but not
    certain;
  - real nondeterminism after audio requests that appears only under memcheck. Memcheck serializes kernels and slows
    the host, so any race between streams or host threads would surface there first.
- **The check now records the control** (`text_control` in its JSON; host-checked against a fake server, with the
  session kept, reaped, and differing ids):
  - both statuses, the reset's status and how long the session sat idle;
  - both replies' `generated_ids` and the first differing index;
  - the error of a non-200 reply;
  - when the session was reaped, the control also run on a fresh session, which must give the same ids, with
    `session_reaped` recorded;
  - a request that times out or fails is recorded as a client error instead of ending the check without a report.
- **Next:** rerun the memcheck step alone, about 15 minutes. It is exactness, not timing, so it can overlap the
  always-on service.

### Rule 5, nothing else changes: PASS
- **The memory bound:** idle 5,405 MiB, peak 5,627, growth 222 MiB (bound 1,024); 16 of 16 served.
- **Rule 6's list, in full:**
  - lfm3-speech regression (36);
  - the audio-in files (360 + 93; F2's 5 expected);
  - e2e, split, the concurrent engine;
  - the batch server, the stream, the audio-in server, the TTS server;
  - the text gate, images through the server.
