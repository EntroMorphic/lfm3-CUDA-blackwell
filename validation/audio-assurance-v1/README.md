# Audio in and out: assurance before images

Tripp (2026-11-11): "I'd rather do some additional testing and red-teaming before moving on to images. Images and
video may take us a while so I want to be 110% certain about our audio in/out."

The engine under test is the one in the tree: liquid-CUDA 8f87d69's sources (lfm3-speech speed-a4, server speed-s2;
round 2 of the speed work was not merged). Thor, MAXN, clocks locked, power recorded at start and end.

## What has never been checked (found before writing these rules)
- **No compute-sanitizer run on the batched speech engine.** The last ones (depth-v1, codec-v1) predate batching,
  the per-row codec streams, parallel host sampling, the deferred language-model head, the staged FP32 kernel and
  the audio encoder.
- **No long run of the server with mixed audio traffic.** Every server check is a bounded crowd.
- **The full LibriSpeech run (2,620 utterances) was measured on audioin-v2.** The audio encoder has since moved to
  the staged FP32 kernel (checked bit for bit on its shapes, not end to end on the corpus). Audio input has not been
  run on the corpus in a batch.
- **No hostile-input run against the live server.** The parsers were fuzzed host-side (audio-in-v1's red-team).

Three independent reviewers, with the code and the records only, are looking for more: one on the batched speech
engine, one on the server and worker, and one auditing every audio claim against the build it was measured on. Their
findings, and any test added for them, go in a dated section below before that test runs.

## Rules (set before any run)
1. **Sanitizers clean.** compute-sanitizer memcheck (leak check on), racecheck, synccheck and initcheck on lfm3-speech
   in a 4-row batch with staggered arrivals: short TTS (both voices), an interleaved reply, a LibriSpeech
   transcription and a spoken question answered interleaved. Also run on the same list one request at a time. Pass:
   0 errors and 0 hazards in every tool, and every output byte-identical to the plain build's run.
2. **The corpus, unchanged.** lfm3-speech over all 2,620 LibriSpeech test-clean utterances, one at a time. Pass:
   every emitted token sequence equals audioin-v2's native run (`build/audioin-v2/native-full`), and so the WER
   against HF is audio-in-v1's (2.59 against 2.61). A difference fails this rule. It is then reported and scored
   (WER against HF, paired-bootstrap 95% upper bound at most +0.25, audio-in-v1's rule 3).
3. **The corpus in a batch.** The same 2,620 utterances through `--concurrent 8`, staggered. Pass: every output file
   byte-identical to rule 2's run alone.
4. **A long mixed run.** The server (8 speech rows) for 2 hours under continuous mixed traffic from 12 clients:
   - TTS, unstreamed and streamed, both voices;
   - interleaved replies;
   - LibriSpeech transcriptions and spoken questions;
   - text /generate;
   - about 1 in 10 streamed requests dropped by the client mid-stream.

   Every completed reply is compared with the same request's reply alone on a fresh server (all fields but timings;
   streamed, every event).

   Pass:
   - every completed reply is identical;
   - no 5xx;
   - a 429 is retried and counted;
   - the server and worker stay up;
   - each process's RSS at the end is at most its maximum over the first 15 minutes plus 64 MiB, and the same holds
     for the GPU memory in use;
   - after the run, a fresh crowd of the 42 cases is still identical;
   - the server exits 0 on SIGTERM.
5. **Hostile input, live.** At least 3,000 requests mutated from valid /speech and audio-input requests:
   - JSON field mutations, wrong types, huge and negative values, unicode;
   - base64 corruption;
   - WAV header and size mutations;
   - truncated bodies, slow and dropped connections.

   These are interleaved with valid requests whose replies must stay exact. Pass:
   - every reply is a well-formed 2xx (exact, for a valid request) or 4xx; a 429 or 503 counts only when documented;
   - nothing crashes or hangs (each request finishes within 120 s);
   - afterwards the 42-case crowd is identical, RSS stays within rule 4's bound, and the server exits 0.
6. **Coherency.** I read the soak's and the fuzz run's valid replies (sampled: 30 of each kind) and listen-check
   nothing new unless a byte changed. Any changed byte is a failure under rules 2–5 first.

Speed is not measured here.

## Added 2026-11-11, from the evidence audit (set before these runs)
The auditor (records only) found that every exactness check since speech-batch-v1 compares a build with itself or
with the build just before it, not with the builds on which fidelity was measured. It also found that no check since
depth-v1 reads the depth decoder's logits: forced replays force the codes, and a sampled code rarely moves at
temperature 0.4. Rules 7–9 tie the engine in the tree back to the measured builds. Rule 10 is a defect the audit
found.

7. **The depth decoder at its measured precision.** `tools/thor/depth_round.sh` at this tree (liquid-depth on the
   current depth.cuh and staged FP32 kernel) against depth-anchor-v1 and tts-trace-v2. Pass:
   - A1 and B1 pass;
   - the tripwire holds (median relative L2 from FP64 ≤ 1e-6, max ≤ 1e-5, KL ≤ 1e-9).
8. **The audio encoder's stages, unchanged.** lfm3-speech speed-a4 on audio-in-v1's 40 dumped clips
   (`build/audioin-v2/list-dump.txt`) and on its 9 edge cases (`build/server-stream-v1-check/edge`). Pass: every
   output file (features, mel, downsampler, encoder and projected outputs, tokens) equals the recorded native run's
   bytes, timings aside. Those runs were tied to HF: the stage errors 0.15–0.19 times HF BF16's, and the edge cases
   token-identical to HF.
9. **The sampled speech, unchanged.** speed-a4 on the 30 requests of speech-e2e-v1 and the 30 split long-form runs
   (speech-chunk-native-v1), on which C1 (WER against HF), C2 and the listening verdicts were measured. Pass: every
   `raw.i32`, `frames.i32` and `f32` byte-identical to the recorded outputs.
10. **Normalization only where it applies.** Today `POST /speech` in audio mode normalizes by default whatever
    language the system prompt names (`lfm3_http.hpp`), and the normalizer is American English only. A host test is
    written first and must fail on today's code. Its cases:
    - TTS requests in Spanish, German, French and Japanese with digits;
    - American English and British English;
    - no system prompt;
    - explicit `normalize: true` with a non-English language.

    The pass condition, after the fix: default normalization only when the TTS system prompt names American or
    British English (or there is no system prompt); otherwise off. An explicit `normalize: true` with another
    language is refused (400). The English defaults keep every validated reply byte-identical (the server checks of
    rules 4–5).

    The sentence splitter is English-only as well: it splits at `. ! ?` followed by a space. Whether multilingual
    TTS is in scope is Tripp's call. Until then, the splitter's default is left as is and reported.
11. **The staged FP32 kernel under the sanitizers.** fp32-linear-check (speed-a4) under memcheck, racecheck,
    synccheck and initcheck. Pass: 0 errors, 0 hazards, and the check passes.

## Added before the soak runs
- **Rule 4's GPU memory.** Thor's GPU allocates from system memory and reports no per-process figure (nvidia-smi
  shows 0 MiB per process, N/A in total). The soak samples the host's memory in use (MemTotal minus MemAvailable,
  with nothing else running) in its place, under the same bound. The tool is `tools/check_lfm3_audio_soak.py`.
- **Rule 12, from the engine red-team.** compute-sanitizer's racecheck sees only shared memory, so it cannot see a
  race between the main stream and a row's codec stream. The engine check's 8-row staggered run is repeated with
  `CUDA_LAUNCH_BLOCKING=1`, which serializes every launch. Pass: every output byte-identical to the normal run.
- **One process.** Rules 4 and 5 say "the server and worker". The worker is compiled into the server
  (`lfm3_server.cu` includes `lfm3_worker.cu`) and runs in the server's process, so the server's RSS covers both. Found
  in the harness's dry runs (which also caught its RSS bound being taken after the references instead of rule 4's
  first-15-minutes maximum); no measured run had been made.

## Decisions (Tripp, 2026-11-11)
- The bearer token stays: it keeps web pages and other local processes off the loopback server.
- **Next round:** a text request can wait behind long speech rows past its deadline (server red-team finding 5);
  this is the open item "text waits for a speech drain".
- **Accepted:** an unstreamed client that only half-closes is not cancelled. A reset client is cancelled (fixed
  here).
- **The token without setup (Tripp: "make the token easier").** With no `--token-file` and no `LFM3_SERVER_TOKEN`,
  the server reads `~/.config/liquid-lfm3/token` (or `$XDG_CONFIG_HOME/liquid-lfm3/token`). It creates the file on
  its first start: a random token, mode 0600, in a mode-0700 directory, opened with O_EXCL and no symlinks. Later
  starts reuse it. The server logs the file's path, never the token.

  Clients read the same file through `tools/lfm3_token.py`; the stream client now needs no token setting. The
  explicit forms keep their precedence. A token file that is not an owned regular mode-0600 file is refused, and so
  is a directory that is a symlink or that others can write. Tests: `tests/test_lfm3_server_token.py` (every build
  variant).

  The change is to startup only. The server builds measured before it (assure-s1) are replaced by assure-s2, and the
  server checks, the fuzz run and the soak count on assure-s2.

## Incidents and deviations (recorded as they happened)
- **A run on a shared Thor (2026-11-11, 11:53–13:50).** I started the server checks and the fuzz run on assure-s2
  while racecheck (rule 1) was still running on the same GPU. It was my scheduling error. Decoding ran about 3× slower,
  and three checks failed on timing alone:
  - the batch server check: while the check built its references, `budget-truncated-s1` took about 300 s on the
    reference server, the old single-request build server-stream-v3, not assure-s2. That build runs a /speech
    request as one engine call, so the default `--engine-timeout` (300 s) times the whole request, and it answered
    503 `worker_unavailable`, as designed. assure-s2 applies the timeout to each step;
  - the stream check: every reply bit-identical, cancel and cost passed, but the stall rule failed for interleaved
    replies (3–16 stalls each);
  - the TTS server check: every case byte-identical and the repeat identical, but the worst-case budget request took
    424 s (bar 240 s), and the text control failed. The control session sat idle through it longer than the 300 s
    session idle timeout, so the second call probably met an expired session. The check did not record the statuses;
    it does now.

  These runs are kept (`build-final/shared-gpu/`) and do not count. The checks are repeated on a quiet Thor. The rule
  for every later round: timing checks (the soak, stalls, timeouts, speed) run on a quiet Thor; exactness checks may
  share it.
- **The fuzz harness's own bug.** In that run, 5 of 3,000 requests never left the client: Python's `struct.pack`
  refused hostile channel and bit counts whose block size overflows the WAV header's fields. The harness now writes
  them wrapped, as a hostile file would carry them. It also records which mutation preceded each connection the
  server reset (20 in that run, all JSON mutations) so each can be accounted for. Otherwise that run showed no bad
  status, no slow request, 375/375 valid replies exact, the crowd identical and exit 0.
- **The sanitizer jobs side by side.** Racecheck on the 4 staggered rows ran more than 3 hours. To use the idle CPU
  cores, I stopped only the script that queued the remaining jobs (racecheck kept running) and started them alongside
  it: racecheck one at a time, then fp32-linear-check under memcheck, initcheck, synccheck and racecheck. Every job
  the rule names runs on the same inputs and against the same bars; only the order changed. Sanitizer verdicts and
  output bytes do not depend on timing.

  **It failed, and it cost the 3-hour run.** Timing was not the issue; memory was. lfm3-profile-v1 is limited to 48
  GiB. Racecheck with 4 rows held ~30 GB and one at a time ~20 GB, so the kernel's cgroup OOM killer ended both
  lfm3-speech processes at 15:00:33. Both reported 0 hazards up to that point, and neither finished (their
  outputs are incomplete). These records are kept (`build/assure/oom-1500/`) and do not count. The two racecheck
  runs were restarted one after the other at 15:10 (`FP32=0`). Only the small fp32-linear-check runs share the
  container with them. My error: I checked the CPU headroom and the host's free memory, but not the container's limit.
- **initcheck with 4 staggered rows.** It reports 479,232 reads of uninitialized memory, all in `convolution7` on
  rows not yet admitted (accepted in the engine red-team). The outputs are identical to the run without a sanitizer,
  and initcheck one at a time reports 0. Rule 1's "0 errors" holds for memcheck and synccheck in both modes, and for
  initcheck one at a time; for initcheck with staggered rows, this accepted exception is the verdict.
- **Timeout defaults (finding, for the next round).** The defaults (engine 300 s, session idle 300 s) assume an
  unshared, fast GPU. On the old single-request server, a near-maximum TTS request at a third of normal speed
  reaches the engine timeout. On assure-s2 the engine timeout applies per step, but a session left idle while other
  work runs still expires after 300 s. They are not
  changed here; this goes with the text-wait round.
- **The normalizer audit, and a third server build.** Tripp asked for the normalizer audit while Thor ran the
  sanitizers ("Do all four"). Its rules were written before any scoring (`normalizer-audit.md`). Three independent
  red-team rounds found regressions in the first fixes. The fixes now pass a real-text gate
  (`tools/normalizer_gate.py`, 485,135 sentences): 39,888 read differently from HEAD, every one reviewed; 39,744
  are improvements, 144 neutral and 0 regressions. I reran the gate myself and it passed. No stored reference text
  normalizes differently. The change is in the server's code, so the server checks, the fuzz run and the soak count
  on **assure-s3** (server a628ddb5…, 0 warnings), which replaces assure-s2.
- **Racecheck with 4 staggered rows took 15.5 hours** (55,662 s): 0 hazards, outputs identical to the run without a
  sanitizer. Racecheck one at a time then started (06:30, 2026-11-12). Racecheck needs no quiet machine, but the
  server checks do, and it ran first. Tripp chose to pause it ("Go with b"): SIGTERM at 07:05 after about 35
  minutes (the partial run is kept in `build/assure/paused-0705/` and does not count). Then the server checks, the
  fuzz run and the soak run on a quiet Thor on assure-s3, and racecheck one at a time runs again from the start
  afterwards (`build/assure/day2.sh`).
- **The soak on assure-s3 failed rule 4 (2026-11-12, 07:58–10:17; `build-final/s3/soak.json`).** The server checks
  and the fuzz run before it passed. In the soak, every completed speech reply was identical (596 unstreamed, 677
  streamed, 89 dropped), with no 5xx, no transport error and the server alive throughout. The crowd afterwards was
  42/42 identical both ways, and the server exited 0. Two parts failed:
  1. **Memory.** Server RSS ended at 5,963 MiB against a bound of 5,659 MiB (the first-15-minute maximum, 5,595,
     plus 64). It climbed about 530 MiB over the 2 hours, steadily, with no plateau. The host's memory in use failed
     the same way (14,943 against 14,246).
  2. **Text starvation.** 7 of 171 text `/generate` requests were never admitted: the client retried the 429 for
     about 10 minutes (30,000 retries) and gave up. The soak's clients made 1.29 million 429 retries in all. This is
     the open item "text waits for a speech drain" (Decisions; `audit/TEXT_WAIT_DESIGN.md`), scheduled for the next
     round. The soak did not change any reply.
- **The memory diagnosis (`tools/diag_lfm3_server_memory.py`, `build-final/memdiag/`).** One fresh server ran each
  kind of traffic alone while the server's smaps was summed by mapping. Racecheck shared the GPU (memory only: no
  timing is judged here).

  | Phase (assure-s3) | Requests | RSS change |
  |---|---|---|
  | idle 60 s | none | 0 |
  | session create and close | 20,000 (14,781 were 429) | +1.1 MiB |
  | refused 400 | 20,000 | −0.2 MiB |
  | storm: long speech while text hammers 429 | 283 speech, 3.07 M 429s, 15 min | +112.8 MiB, all anonymous |
  | unstreamed speech | 200 | +100.6 MiB, anonymous |
  | streamed speech | 200 | −0.1 MiB |
  | dropped streams | 200 | +7.2 MiB |

  Only replies that carry a whole WAV grow, and the growth is in anonymous memory. The worker and the server keep
  nothing per ticket: each reply's WAV, base64 text and JSON are local strings, freed after the reply. An A/B test
  of 3 × 200 unstreamed requests, run side by side on two fresh servers:
  - as built: +162, +389, +47 MiB;
  - with `MALLOC_MMAP_THRESHOLD_=131072`: +3.7, +0.4, +0.6 MiB.

  **No leak; allocator retention.** glibc raises its mmap threshold to as much as 32 MiB after the first large
  free. From then on each reply's MB-sized buffers come from a malloc arena (allocated on the engine-owner thread,
  freed on a handler thread). The arena fragments and keeps the pages, irregularly and with no bound one can rely
  on.
- **The fix, assure-s4 (server c2fe0ac7…, 0 warnings).** The server's `main` fixes the mmap threshold at 128 KiB
  (`mallopt(M_MMAP_THRESHOLD)`, unless `MALLOC_MMAP_THRESHOLD_` is set). Every buffer of 128 KiB or more is then its
  own mapping, returned to the system when freed. The change affects allocation only, so no output byte can change.
  Its cost is a few page faults per MB-sized reply, against seconds of generation. It is in the server's code, so
  the server checks, the fuzz run and the soak all repeat on assure-s4, on a quiet Thor after racecheck one at a
  time (`build/assure/day3.sh`). assure-s3's records are kept (`build-final/s3/`).

  The memory diagnosis on assure-s4, with no environment variable (`build-final/memdiag/s4/`), ran 600 unstreamed
  replies (+2.9, +6.1, +0.1 MiB), 200 dropped streams (+5.2), and a 10-minute storm (202 long speech replies and
  2.14 M 429s; +5.9). Churn, refusals and idle stayed flat. In all: +21 MiB, against +213 MiB on assure-s3 for
  fewer requests (200 unstreamed and a 15-minute storm). Every request answered 200, 400 or 429. The server exited 0.
- **How the assure-s4 soak is judged (set before it runs).** The tool and bars are unchanged. Rule 4 passes only if
  every part passes, the text requests included. A text request still unadmitted after the client's 30,000 retries
  fails rule 4. Until the text-wait round lands, that part is expected to fail again. The report then gives:
  - the memory verdict;
  - the exactness verdict;
  - the starved text requests, counted.

  Rule 4 stays open, not passed, until the text-wait fix is soaked.
- **The container's memory limit, raised (2026-11-12, 15:50).** Racecheck one at a time grows its shadow state with
  the audio's length. With three of its five cases done, its peak reached 48.4 GB, the container's 48 GiB limit, and
  the cgroup reclaimed page cache 180,838 times to keep it alive (no new OOM kill). The two longest cases were still
  ahead. With Tripp's approval, `lfm3-profile-v1`'s limit was raised live to 96 GiB (`docker update --memory 96g
  --memory-swap 96g`; no swap, as before). The run was not interrupted. A sanitizer's verdict and output bytes do not
  depend on its memory limit.
- **Racecheck one at a time, again (2026-11-12 10:17 to 11-13 05:03).** It took 67,566 s (18.8 h): 0 hazards, and
  every output identical to the run without a sanitizer. The four cases finished at 11:38, 13:13, 14:25 and 19:39,
  and il-reflective (about 1,500 frames, the longest) ran until 05:03.
- **The checks on assure-s4 (2026-11-13, 05:03–08:07, `build-final/s4/`).** Racecheck had finished and Thor was
  quiet (`build/assure/day3.sh` waited for both). Every server check passed, the fuzz run passed, and the soak's
  memory and exactness passed. 3 text requests starved (Results, rule 4).
- **The engine binary at this tree (assure-a3).** The server builds compile the engine in but do not build
  `lfm3-speech`. The sanitizer, corpus and fidelity runs (rules 1–3, 7–9, 11, 12) used assure-a2. Its sources differ
  from this tree only in `speech_normalize.hpp` (the normalizer audit), which `lfm3-speech` compiles in for
  `normalize: true` requests. So `lfm3-speech` was rebuilt from this tree as **assure-a3** (7f68161a…;
  `build-final/engine-a3/sources.sha256` differs from a2's in that one file):
  - the speech regression: 36/36 identical, refusals clean;
  - the engine exactness check: 41/41 identical to speed-final2's run alone, and every concurrent mode (4, 8, 8
    staggered, 16 staggered) identical.

  The normalizer is host code. Its tests pass under ASan and UBSan. It runs no GPU kernel the sanitizers judged.
- **The fuzz run's 20 resets, accounted for.** In both fuzz runs (s3 and s4), all 20 resets followed
  `json:many-messages`. That mutation repeats the messages ×30 or ×300; with LibriSpeech audio, the body is then
  several MiB over the 4 MiB limit. The server refuses at the headers (`413 body_limit`) and closes without reading
  the body. A client still uploading fails its write before it reads the 413. Checked on the HTTP harness (the same
  HTTP layer): for 5, 9 and 40 MiB bodies, the socket holds `HTTP/1.0 413 Request Entity Too Large`, and the
  client's send fails with a broken pipe after 0.45–1.8 MB. So each reset is a correct 413 the client never read.
  Draining the body before the 413 would let every client read it, but it would also make the server read up to
  the client's claimed length. That is left as is.

## Results (2026-11-13)
Builds:
- engine: assure-a2 (`lfm3-speech` b8eac364…) for rules 1–3, 7–9, 11 and 12; assure-a3 (7f68161a…, this tree) for
  the regression and exactness check above;
- server: assure-s4 (`liquid-lfm3-server` c2fe0ac7…, this tree) for rules 4–6 and the server checks.

Hashes: `binaries.sha256`, `sources.sha256`. Thor: MAXN, clocks locked, power recorded.

| Rule | Verdict | Evidence (`build-final/`) |
|---|---|---|
| 1. Sanitizers clean | **PASS**, with initcheck's accepted exception | memcheck: 0 errors and 0 bytes leaked, both modes. synccheck: 0, both modes. initcheck: 0 one at a time; the staggered rows' 479,232 reads are all in `convolution7` on rows not yet admitted (Incidents). racecheck: 0 hazards with 4 staggered rows (55,662 s) and one at a time (67,566 s). Every output byte-identical to the plain run. `sanitizers/` |
| 2. The corpus, unchanged | **PASS** | 2,620/2,620 token sequences equal audioin-v2's native run, so WER is audio-in-v1's (2.59 against HF's 2.61). `engine/rule2-slots.json` |
| 3. The corpus in a batch | **PASS** | 8 rows, staggered: 13,260/13,260 files byte-identical to rule 2's run. `engine/rule3-files.json` |
| 4. A long mixed run | **OPEN**: memory and exactness pass, 3 text requests starved. **Closed by validation/text-wait-v1 (2026-11-14):** twa-2's soak at a text share of 1 in 4, 1,418 text replies, every one admitted and identical (rule 7, run 4). That round also found this soak's host-memory test could not resolve its figure (it includes the client, and Thor's host cycles about 500 MiB on its own): s4's pass on it was luck; its server RSS was flat | assure-s4, 2 h, 12 clients: 781 unstreamed, 866 streamed and 112 dropped speech replies and 212 text replies, every completed one identical. No 5xx, no transport error, alive throughout. Server RSS 5,436 MiB (first 15 minutes' maximum), 5,412 at the end; host memory 14,150 and 14,150. The crowd after: 42/42 identical both ways; exit 0. 3 of 215 text requests were never admitted (30,000 retries each); that is the text-wait item. `s4/soak.json` (assure-s3's failed run: `s3/`) |
| 5. Hostile input, live | **PASS** | assure-s4, 3,000 requests: 375/375 valid replies exact; no bad status, no slow request, no error. 200: 492, 400: 1,882, 401: 46, 404: 30, 413: 10, 429: 9, 501: 31, client-closed: 125, and 20 unread 413s (above). Crowd identical, RSS within bound, exit 0. `s4/fuzz.json` |
| 6. Coherency | **PASS** | Every valid soak and fuzz reply was byte-identical to its case's reference, so the 42 cases' replies and the 3 text replies are every distinct reply they produced. I read all 45 (`rule6/replies.json`): fluent, on topic and coherent. No audio byte changed from the validated builds (the batch-server check against server-stream-v3, rule 9 against the measured runs), so nothing new to listen to. The content errors are the model's, the same in HF's replies: the refrigerator answers (all 3 seeds) get the cycle's physics wrong ("low-pressure gas" from the compressor, the condenser "inside the fridge"); one spoken answer suggests 200 g of pasta for four; and the ASR keeps its known slips ("flour fat and sauce", the sonnet's archaic words). |
| 7. The depth decoder at its precision | **PASS** | A1: native median relative L2 from FP64 1.57e-7 (bar 4.91e-3). B1 PASS. Tripwire: median 1.57e-7 ≤ 1e-6, max 6.72e-7 ≤ 1e-5, KL 1.55e-12 ≤ 1e-9. `engine/assure-rerun.log`, `engine/a1b1.json` |
| 8. The audio encoder's stages | **PASS** | 360/360 files of the 40 dumped clips and 93/93 of the 9 edge cases byte-identical. `engine/assure-rerun.log` |
| 9. The sampled speech | **PASS** | speech-e2e-v1 90/90 and the split long-form runs 90/90 byte-identical. `engine/assure-rerun.log` |
| 10. Normalization only where it applies | **PASS** | The host test failed on the old code and passes now (`tests/test_lfm3_http_speech.py::test_normalization_follows_the_language`). Every validated reply is unchanged. The sentence splitter stays English-only; multilingual speech is shelved by Tripp (`validation/multilingual-speech-v1/`). |
| 11. The staged FP32 kernel under the sanitizers | **PASS** | fp32-linear-check under memcheck, initcheck, synccheck and racecheck (23,012 s): 0 errors, 0 hazards, the check passes. `sanitizers/` |
| 12. The codec streams serialized | **PASS** | `CUDA_LAUNCH_BLOCKING=1`: 205/205 output files byte-identical to the normal run. The normal run equals speed-final2's run alone (205/205; the input list is not an output). `engine/assure-blocking.log` |

Server checks on assure-s4 (`s4/`), all PASS:
- the batch server;
- the stream check: first audio, median 0.237 s for TTS and 0.329 s interleaved; 0 stalls;
- audio in;
- the TTS server: worst-case budget 124.4 s, large-cap 40.3 s and truncated, text control [200, 200];
- the text gate (exit 0).

The same checks passed on assure-s3 (`s3/`).

**What this round found and fixed:**
- the token without setup;
- normalization by language (rule 10);
- the normalizer audit (39,744 improvements, 0 regressions);
- a reset client cancelled;
- the fuzz harness's own WAV packing bug;
- the server's allocator retention (assure-s4).

**Open, for the next round:** text waits for a speech drain (rule 4's last part; `audit/TEXT_WAIT_DESIGN.md`, design
B), with the timeout defaults. Rule 4 closes when that fix passes this soak.
