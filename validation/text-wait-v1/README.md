# Text no longer waits for speech to drain

Tripp (2026-11-11): "fix the text wait next round"; (2026-11-13) "Let's proceed with the 1..4 items you listed"
(item 4: the text-wait round, design B, with the timeout defaults, closed by its own soak).

The defect (audio-assurance-v1, server red-team finding 5; soak rule 4's open part): one engine-owner thread makes
every engine call, and while any speech row runs it makes only speech calls. A text request waits for every row
running when it arrived to finish, and a text client waits twice (create, then generate). In the assure-s4 soak, 3 of
215 text requests were never admitted. The analysis, the designs considered and an on-paper red-team are in
`audit/TEXT_WAIT_DESIGN.md`. This round builds design B: text generation sliced into `text_begin` and `text_step`,
interleaved one text token per speech step while speech rows run, and short session operations run between speech
steps.

## Builds
- Baseline: assure-s4 (`liquid-lfm3-server` c2fe0ac7…, liquid-CUDA 23c840f) and assure-a3 (`lfm3-speech` 7f68161a…).
- Under test: twa-N (server), built from this round's sources. `lfm3-speech` does not compile the worker or the
  server, so it should be unchanged; if its sources change, it is rebuilt and rule 1's engine checks rerun on it.

Thor must be quiet for every timing rule (rules 2–5, 7's soak), with nothing else on the GPU. Speed is labeled
decode loop or end to end. Every failure is recorded where it happens.

## Rules (set before any data, including the baseline)
1. **Exact.**
   - Text: the soak's 3 text bodies, the text gate's prompts and a sampled body (temperature 0.528, seed 42) give
     `generated_ids`, text and `last_logits` (`return_logits`) bit-identical:
     - alone on assure-s4;
     - alone on twa;
     - under 8 streaming speech rows on twa.
   - Speech: all 42 cases, unstreamed and streamed, with every event identical to alone, while a text client
     generates continuously.
   - The same check with `CUDA_LAUNCH_BLOCKING=1`.
   - `lfm3-speech` regression 36/36, engine exactness 41/41, the text gate and the depth tripwire are unchanged.
   - Pass: 0 differences.
2. **Text no longer waits for a drain (end to end).**
   - Setup: 8 rows of long-form TTS streaming, each with at least 60 s left. A new client then creates a session and
     generates 128 tokens of the reflective prompt (stops disabled).
   - Pass: the median of 10 arrivals is at most 3.5 × its median alone and at most 10 s, and no arrival exceeds 15 s.
   - Today's code must fail this rule. It is run on assure-s4 first, and its failure is recorded.
3. **Speech stays real time under text load.**
   - The stream check's real-time rule, at 8 concurrent streams (TTS and interleaved), while one text client
     generates back to back (128-token requests).
   - Pass: no player runs dry.
   - First audio of a TTS request arriving under that load: median at most 0.6 s over 10 arrivals (0.445 s without
     text load, validation/speech-batch-v1/README.md).
4. **Speech is not delayed by a long text generation.**
   - Setup: one text generation of 2,000 tokens is running; a TTS request arrives.
   - Pass: its first audio arrives within 1.0 s. Today it waits for the whole generation, so this fails on assure-s4.
5. **Nothing slower alone.** Paired medians, twa against assure-s4, interleaved runs, 5 pairs each:
   - text alone, end to end (one 128-token generation through the server): ratio at least 0.99;
   - speech through the server alone, 1 request and a crowd of 8 (audio seconds per wall second): ratio at least
     0.99.
   - `lfm3-speech` (engine, decode loop) at 1 and 8 rows: identical binary, or ratio at least 0.99.

   Speed round 2 failed on a 2.8% text slowdown, so 1% is the line.
6. **Deadlines and cancellation.**
   - Server started with `--request-timeout 20`; a text request arrives behind 8 rows that each have 60 s left.
     Pass: 200. Today: 504 `queue_deadline`.
   - A text client that resets mid-generation: its session's cache is reset, and the next generate on the session is
     identical to a fresh one.
   - A text request that passes its deadline mid-generation: cancelled at the next slice; its reply is 504 and the
     cache is reset.
   - SIGTERM with a text generation active and 8 rows running: exit 0, and every reply is either complete or an error.
7. **Hostile input and soak.** The audio-assurance-v1 fuzz (rule 5) and a 2 h soak (rule 4), unchanged except that the
   soak's text share rises from 1 in 10 to 1 in 4. Same pass rules, the text requests included: every text request is
   admitted and answered.
8. **Sanitizers.** memcheck, racecheck, synccheck and initcheck on an engine test that interleaves `text_step` and
   `speech_step` (4 speech rows and one text session). Pass: 0 errors and outputs identical, with the accepted
   initcheck exception for unloaded rows only.
9. **Timeout defaults** (the audio-assurance-v1 finding):
   - After this change, the engine timeout watches slices: one speech step, one text slice or one prefill. The
     defaults stay 300 s and 330 s; the round records the longest single call measured.
   - The session idle timer must not count time a session spends queued or generating. A test idles a session for
     longer than the idle timeout behind work and expects 404; one queued for that long expects its reply.

**Tests that fail on today's code, written first:**
- `tests/test_lfm3_server_text_wait.py`: rules 2, 4 and 6 (first case) as fast black-box checks on the harness with
  short speech (`--speech-rows 2`).
- A scheduler test on `lfm3_http_harness` (its fake backend records the call order): with rows running and a queued
  `generate`, the recorded calls interleave `speech_step` and `text_step`.
- `native/tests`: `text_begin` + k × `text_step` equals op `generate`, bitwise (ids and logits), for greedy, sampled,
  full prefill, reuse and cancelled requests, with k = 1, 3 and 256.

**Behavior changes this round makes (documented in the server's header):**
- A text request that passes its deadline while generating is cancelled at the next slice (504, cache reset). Before,
  it ran to its end and its cache stayed valid.
- An active text generation leaves the queue, as an admitted speech row does, so one more request is in flight than
  before at the same `--queue-size`.
- `generate_batch` stays one engine call and keeps the drain rule.

## Corrections to the rules' wording (made before any data, recorded here)
- **Logits and stops are not HTTP fields.** `/generate` accepts `session_id`, `messages`, `max_new_tokens` and
  `sampling` only. So:
  - rule 1's `last_logits` comparison is made where logits exist: in the engine test (`text_begin` + k × `text_step`
    equals op `generate`, ids and `last_logits`, bitwise). Over HTTP, rule 1 compares `generated_ids` and text;
  - rules 2 and 4 cannot disable stops. They use prompts that ask for long answers, and record each reply's token
    count. Alone and loaded, a request generates the same tokens (rule 1), so the timing ratio compares equal work.
    Rule 2's prompt: "Hypothetically, might reflective recursion be a function of cognition? Answer in at least 300
    words." (128 tokens). Rule 4's: "Write a long, detailed story about a lighthouse keeper, at least 3,000 words."
    (2,000 tokens).
- **Rule 2's 8 rows** are 8 streams of `budget-truncated-s1` (the 6,144-step worst case, several minutes at 8 rows).
  Each text arrival starts once every stream has played 5 s of audio; the streams are then read to their end. Each of
  the 10 arrivals gets fresh rows.
- **Rule 3's arrivals** need a free row, so the load is 7 streams (4 TTS long-form, 3 interleaved) and the text client,
  and each arrival is the 8th row (as speech-batch-v1 measured its 0.445 s).
- **The tool:** `tools/check_lfm3_text_wait.py` (black box), rules 2, 3, 4 and 6's first case.

## Results (2026-11-14)
Builds: twa-1 (server 414c1a9c…), and twa-2 (server 610c8c1d…, the reset fix in the HTTP layer only; worker
fb95efcf… identical to twa-1's). `lfm3-speech` is source-identical to assure-a3: none of this round's changed files
(`lfm3_http.hpp`, `lfm3_worker.cu`, `CMakeLists.txt`, the harness) is among the 32 it compiles, so its regression,
engine exactness and decode-loop speed carry over. Records: `build-final/{baseline-s4,twa-1,twa-2,twa-2-memdiag*}/`.

| Rule | assure-s4 (baseline) | twa-1 | twa-2 (final) |
|---|---|---|---|
| 1. Exact | (reference) | PASS: text alone, beside 8 streams, the crowd 42/42 both ways, 26 and 27 text sets, 0 differ, normal and `CUDA_LAUNCH_BLOCKING=1`; slice check (`text_begin` + k x `text_step` = `generate`, ids and logits) 0 failures in both modes; text gate passed (control `7ea05068…`) | PASS: the same over HTTP, 0 differ in both modes; text gate passed, same control; worker identical, so the slice check carries over |
| 2. No drain wait | FAIL: median 161.8 s (68.6x alone), max 178.1 | PASS: median 5.98 s (2.72x; alone 2.19), max 5.99 | PASS: median 5.93 s (2.74x; alone 2.17), max 5.95 |
| 3. Real time under text | first audio median 16.6 s | attempt 1 FAIL (9 arrivals of 10); attempt 2 PASS: 10 arrivals, median 0.574 s, 0 stalls | PASS: 10 arrivals, median 0.573 s, 0 stalls |
| 4. Speech beside a long text | FAIL | PASS: max 0.641 s (2,000 tokens in about 35 s) | PASS: max 0.598 s |
| 5. Nothing slower | (reference) | PASS: text 1.0003, one 1.0017, crowd 1.0012 | PASS: 0.9987, 0.9996, 1.0004; `lfm3-speech` identical source |
| 6. Deadlines, cancellation | deadline: 504 at 20.0 s | deadline PASS (200 in 5.95 s); reset mid-generation FAIL (found 2026-11-14) | deadline PASS (200 in 5.94 s); reset PASS; deadline mid-generation and SIGTERM with text and rows: host tests pass (plain, ASan+UBSan, TSan: 492 passed) |
| 7. Fuzz and soak | (audio-assurance-v1) | fuzz PASS; soak passed on the endpoint test (luck, see Incidents) | fuzz PASS (375 valid, 0 differ); soak: run 1 FAIL (host figure), run 2 FAIL (load medians), run 3 gate passed (not counted), **run 4 PASS** (idle checkpoints with a fixed sequence; server RSS +0.5 MiB; 1,418 text replies, all admitted and identical) |
| 8. Sanitizers | | memcheck 0 errors 0 leaks; synccheck 0; racecheck 0 hazards (30/30); initcheck: only the accepted unloaded-rows reads in `convolution7` | carries over (worker identical; the slice check differs only in an nvcc temporary's name) |
| 9. Timeout defaults | | idle-timer host tests pass | same; **gap:** the longest single engine call was not measured, so the 300 s and 330 s defaults are kept, not re-derived |

**Verdict:** every rule passes on twa-2 except one recorded gap (rule 9's measurement). Rule 7's soak needed four
runs; the measurement changed twice after failures, each change set before the run it judged (Incidents). Rule 7
also closes audio-assurance-v1 rule 4's open part (the text wait).

## Incidents and deviations (recorded as they happened)
- **Baseline, attempt 1 (assure-s4).** The tool's streams did not retry `429 frontend_busy` (the frontend prepares one
  request at a time, so 8 streams sent at once collide), and its first rule-2 setup failed. Every other check
  retries a 429 as a client does; the tool now does too (streams, and text creates and generates). The log is kept
  (`baseline-s4/attempt1-no-429-retry.log`); no measurement was taken.
- **Tests written first, run on the code before the change** (`host/before-3bda17f.txt`, plain build): 7 of 10
  failed, each for the defect (text waited 10.8 s for the rows against a 6.5 s bar; 504 instead of 200 at the
  deadline; the cancel landed on a generation that had not started) or for the new ops and options it needs; the 3
  that passed check what does not change. That first run also caught three bugs in the tests themselves (replies of
  two sessions compared with their session ids; `/cancel` answers 202, not 200; two requests sent at the same instant
  collide on the frontend lock), fixed before the code was run.
- **Found by the tests on the new code, fixed before any Thor run:**
  - a text generation cancelled for its deadline failed the server closed: the cancelled reply was "unsolicited"
    (`generation_response` accepts a cancelled reply only when the client's event is set). The owner now sets the
    event for its own cancellations, as shutdown already did.
  - the idle text slice (256 tokens, about 4 s on Thor) would have held new speech back past rule 4's 1.0 s; it is
    16 tokens (`Settings::text_slice`). Changed while the tests were written, before any measured run.
  - a session operation queued behind a text generation waiting for the active one would have waited for it too
    (head-of-line). The owner takes the first runnable job in arrival order; an exclusive job still holds back
    everything behind it.
- **Red-team (an independent reviewer with the code only):** no defect against bit-identity, no deadlock, no double
  settle, no leaked lease (TSan clean on the new host tests). Found and fixed:
  1. at shutdown, an active text generation was answered 503, not its cancelled reply: `close()` sets the service
     failed together with `cancel_all_`, and the failure check ran first. Now the cancelling `text_step` is sent
     while the engine is open. The shutdown test now requires the cancelled reply.
  2. an exclusive job could pass speech queued before it while the owner was busy in a short call; the scan now
     stops at an exclusive job that arrived after the oldest queued speech. (Checked by reading: a test needs three
     sessions, and `--max-sessions` allows 2.)
  3. a cancelled slice left one unread frame in the cancellation pipe (harmless, bounded); the worker reads the pipe
     before the flag.
  4. a throw while writing the cancelled reply would have left `pending` set; the worker takes ownership of it
     first.
  5. no test of a speech failure while text is active: added.
- **The engine check's own bugs (lfm3-text-slice-check, before its first full run):** an "unbounded" k of 2^40 that
  `Json::integer()` refuses (now `INT_MAX`, as op `generate` uses); a range-for over a temporary's member (dangling in
  C++20); and the speech comparison included the replies' wall-clock fields (`prefill_seconds`,
  `generate_seconds`). Now excluded, and the check also requires two runs alone to be identical, so the comparison is
  shown to be exact rather than vacuous.
- **Rule 1 over HTTP, attempts 1 and 2: the tool's error.** The tool started the servers with the default 16 handlers,
  and its crowd of 42 at once (plus a text client) had surplus connections refused or reset, the server's documented
  saturation behavior. The server was alive throughout and exited 0. The soak and the batch-server check give
  their crowd servers 96 handlers and a queue of 64; so does this tool now. Every comparison made before the crowd
  was identical in both attempts. Records: `twa-1/rule1-attempt1-server-died.log` (named before the cause was known),
  `twa-1/rule1-attempt2/`.
- **Rule 3, attempt 1 (twa-1): FAIL on the count.** All 7 load streams ended after 9 arrivals, and the tool stops
  arriving when the load ends; the rule asks for 10. The 9 measured: first audio median 0.567 s (bar 0.6 s), 0
  stalls in every load stream and arrival, every text 200. The tool now restarts each load row's stream when it ends,
  so the 7 rows stay busy to the 10th arrival; rule 3 runs again alone after the round (`twa-1/rule3.sh`).
- **Rule 6, second case: FAIL on twa-1, found while writing the results (2026-11-14).** No test covered a text client
  that resets its connection mid-generation; `/cancel` was tested, but a reset takes another path. The test written
  then (`test_a_client_that_resets_mid_generation_leaves_its_session_as_a_fresh_one`) failed on twa-1's code: the
  server never noticed the reset, the generation ran to its end for no one, and its cache stayed valid (the
  continuation reused 35 tokens where a fresh session reuses 0). Nothing wrong was ever returned, but the rule says
  the cache is reset. The test's first two versions were vacuous (a repeated prompt reuses nothing on the fake engine,
  reset or not); its control now shows that a client that stays for its reply keeps its cache.
  - **The fix (server HTTP layer only):** `/generate` now waits as unstreamed `/speech` does, polling its connection
    every 50 ms; a reset sets the session's cancel event, as `/cancel` does, and the generation stops at its next
    slice with its cache reset. A half-close (FIN after the request) is not a reset: a new test shows that client
    still gets its full reply. The poll is shared with `/speech` (`reset_by_peer`).
  - **Build twa-2** (server 610c8c1d…, 0 warnings). Its worker is byte-identical to twa-1's (fb95efcf…), so the
    engine is unchanged; `lfm3-text-slice-check` differs from twa-1's in 3 bytes, the name of an nvcc temporary file
    in its symbol table (`tmpxft_00000048` against `tmpxft_0000017a`). The engine check and the sanitizers (rule 8)
    therefore carry over from twa-1; everything that runs the server is run again on twa-2
    (`tools/thor/textwait_rerun.sh`): rule 1 over HTTP in both modes, rules 2, 3, 4 and the deadline, rule 5, the fuzz,
    the soak and the text gate.
- **Rule 7, the soak on twa-2: FAIL on the host's memory (2026-11-14, 09:48-11:55).** Everything else passed: 6,305
  replies (2,041 unstreamed, 2,477 streamed, 277 dropped, 1,510 text), every completed one identical, all 200, no
  transport error, alive throughout, the crowd 42/42 both ways, exit 0, server RSS flat (5,478.9 MiB early maximum,
  5,468.6 at the end). The host's memory in use ended at 12,528.6 MiB against a bound of 12,068.1 (the first 15
  minutes' maximum, 12,004.1, plus 64). After the server exited, Thor was back to 4.3 GB in use.
  - **The earlier passes were luck.** The same drift is in every soak's records: by the median of the first 15
    minutes against the last 45, about +200 MiB (assure-s4), +420 (twa-1) and +500 (twa-2). twa-1 and s4 passed only
    because a single sample swings about 200 MiB with load: their first 15 minutes held a high sample (a loose bound),
    or their last sample was a low one. The test (one sample against the first 15 minutes' maximum) could not
    tell drift from noise.
  - **Red-team of the harness** (`tools/check_lfm3_audio_soak.py` and the `Server` helpers it uses), five findings:
    (1) the host figure includes the soak client itself, on the same host; (2) the one-sample test above; (3) the
    server's stdout and stderr were pipes never read while it ran: the server writes there only at start and on a
    fatal error, so it never mattered, but a full pipe would block it and look like a server hang; (4) a transport
    error on a text request was recorded under a speech case's name; (5) `stream()` buffers by concatenation,
    quadratic in client CPU on long streams (conservative: it can only add to measured times; recorded, not
    changed, as the timing tools share it).
  - **Diagnosis** (`tools/thor/host_memory_sampler.py` on the Thor host beside the soak, every 30 s: the host's
    memory in use split into the server's anonymous memory, the client's, every other process's, the kernel's, the
    file cache MemAvailable leaves out, and a residual, on Thor mostly GPU memory). The first diagnostic run (old
    harness, stopped by SIGTERM at 30 minutes once the new harness was ready; `twa-2-memdiag-stopped-old-harness/`,
    1,588 requests, 0 differ) showed the client climbing (56 to 425 MiB) and the server and residual flat.
  - **Second soak (harness fixes 1, 3 and 4; and the flat test: the median of the last 15 minutes against minutes 5
    to 20, plus 64 MiB, on server RSS and on the host's memory less the client's RSS), 12:50-15:04: FAIL.** 6,343
    replies, 0 differ, 0 errors, all 200, crowd 42/42, exit 0. Server RSS +11.8 MiB: pass. The host's memory less
    the client: +75.7 MiB: fail. But the breakdown sampler, measuring the same quantity over the same windows at
    other instants, found -78.1 MiB; the residual (GPU) has no trend (+23 MiB/h, sample deviation 106 MiB); the
    server's anonymous memory +12.8, the kernel's -2.0. Under load a sample of the GPU's share swings about 100 MiB,
    so a 30-sample median moves by more than 64 MiB between two samplers: that test could not judge either. The
    client grew 163.7 MiB, 152.1 of it still there after glibc's `malloc_trim` (live Python memory, not
    fragmentation; the client is not the system under test, and its figure is now subtracted).
    `twa-2-memdiag2/`.
  - **Third soak, the idle checkpoint test (set before it ran):** at 5 minutes, every 20 minutes after and at the
    end, the clients pause, the server drains (no request in flight), 10 s pass, and the median of 10 readings 1 s
    apart is the checkpoint. Pass: the last checkpoint at most the first plus 64 MiB, on server RSS and on the host's
    memory less the client's, at least 4 checkpoints. Checked offline first on a fake server (every reading taken
    with 0 requests in flight; flat passes; a side process growing 8 MiB/s fails at +366 MiB).
    **Result (15:10-17:27): the gate passed, and I do not count it.** 6,036 replies (1,938 unstreamed, 2,386
    streamed, 261 dropped, 1,451 text), 0 differ, 0 errors, all 200, crowd 42/42, exit 0. Server RSS at the 7
    checkpoints: 5,400.0, 5,400.7, 5,409.3, 5,409.1, 5,408.8, 5,408.8, 5,408.6 MiB (+8.6). The host's memory less the
    client: 12,092.1, 12,190.9, 11,942.4, 12,218.4, 11,770.6, 12,230.0, 11,803.5 MiB, a 460 MiB range, and three
    checkpoints spread about 510 MiB across their own 10 readings. It "passed" at -288.6 because the last checkpoint
    landed low. The breakdown sampler shows why: the GPU residual sits on one of two flat levels (8,433-8,467 or
    8,633-8,665 MiB; neither drifts in 2 h), with brief excursions near 8,950, because the server frees and
    reallocates some device buffers sized to the request just served (`MathAttentionDynamic`'s scratch, the
    `DeviceArray` temporaries in `fp32_ops.cuh`, among others). An idle reading depends on the last request, and a
    dropped stream can still be finishing on the server when the client counts itself idle. Predicted while the run
    was going, before its last checkpoint, and reported to Tripp then. `twa-2-memdiag3/`.
  - **Fourth soak (Tripp: "Go with the extra run"), the fixed-request checkpoint (set before it ran):** as the third,
    but after the drain and 10 s, the same four requests run alone and in order before the readings: text
    `/generate` (the first text), `spoken-q0` unstreamed, the first TTS case streamed, `asr-1089-134686-0000`
    unstreamed, each compared with its reference (a miss is a differ); then 10 s and the readings. Every checkpoint
    then follows the same last requests, so the request-sized buffers are the same and a difference between
    checkpoints is growth. Same pass rule. Checked offline first on fakes (flat passes; outside growth of 8 MiB/s
    fails at +367; a wrong fixed reply is recorded as a differ at each checkpoint and fails the run).
    **Result (17:27-19:43): PASS.** 5,928 replies (1,904 unstreamed, 2,324 streamed, 254 dropped, 1,418 text), 0
    differ, 0 errors, all 200, alive throughout, crowd 42/42 both ways, exit 0; the fixed sequence identical to its
    references at all 7 checkpoints. Server RSS at the checkpoints: 5,408.8, 5,408.8, 5,409.4, 5,408.7, 5,408.8,
    5,409.0, 5,409.3 MiB (+0.5). The host's memory less the client: 12,163.6, 11,805.6, 11,809.5, 12,284.2, 11,804.4,
    11,989.1, 12,008.7 (-154.9: passes, and still scattered across 480 MiB with the same last requests).
  - **What the host figure can resolve.** The scatter is not the server's: with no server running, Thor's memory in
    use cycles between about 4,370 and 4,880 MiB every 30 s (`host-idle-no-server.txt`; the desktop session and
    other users' processes run on Thor). So the host-wide figure resolves about 500 MiB, whatever the server does. The
    evidence of no growth is: the server's RSS (+0.5 MiB in run 4, +8.6 in run 3, exact), the gate passing as set
    before run 4, the low checkpoints at the same level to within 6 MiB across an hour (11,804.4-11,809.5; an
    observation, not a gated test), and the breakdown sampler's GPU residual sitting on flat levels in runs 2, 3 and
    4 with no trend. The client's growth (about 200 MiB in 2 h, not released by `malloc_trim`) is in the Python
    harness, a slow leak there; it is not the system under test and is left as recorded.
  - The rule's intent is unchanged (no memory growth in the server or the GPU over a 2 h run); its measurement
    changed twice, each time after a failure, each change set before the run it judged. Both failures stay recorded.
- **A pre-existing test warning, not from this round.** `test_listener_closes_and_second_signal_exits` SIGTERMs the
  server during a client's call; the client thread fails to parse the empty reply (a pytest thread warning). The
  same on the code before this round.
