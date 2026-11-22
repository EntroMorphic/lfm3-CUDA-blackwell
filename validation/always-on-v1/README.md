# always-on-v1: liquid-lfm3-server as an always-on, pre-warmed service on Thor

**Status: rules set 2026-11-17, before any code or measurement.** The last item of the P0 list (Tripp, 2026-11-16: "I
want to have our inference engine running as a service for our models so they are available in real-time").

## What it will be
1. **Warm-up at startup** (`--warmup`): before the ready line and before `/health` answers 200, the server runs one
   request of each kind through its own front end and engine (text, TTS, a 4K image, audio in) and discards the
   replies. Opt-in, not the default: every existing tool and test starts the server and measures it cold, and the
   service's unit file is the one place that needs it. (The 2026-11-16 plan said `--no-warmup` for tests; opt-in
   does the same with nothing else changed.)
2. **A systemd user service** on Thor: starts with the machine (needs `loginctl enable-linger`, Tripp's to run),
   restarts on failure, stops by SIGTERM, loopback only, the existing mode-0600 token file.

Found earlier (image-queue-v1, shared GPU, indicative): the first image request after a server starts waited ~16 s
for its first token. Where those seconds go has not been measured.

## Rules
0. **Measure first** (quiet Thor): on a fresh server without warm-up, for each kind (text, TTS streamed, 4K image,
   audio in, in that order and then each kind first on its own fresh server), the time to first token of the 1st, 2nd
   and 3rd request. Proceed with warm-up only if some kind's 1st request is slower than its 3rd by more than 1 s;
   record where the difference goes as far as the server's own timings show it.
1. **Warm**: with `--warmup`, each kind's 1st request after the ready line has its time to first token within 10% or
   50 ms (the larger) of that kind's 3rd request on the same server; median of 3 fresh servers; quiet Thor.
2. **Exact**: with `--warmup`, every reply byte-identical (timings aside) to the same request on a server without
   it, each request the first of its kind after startup: the 11 stage-0 image requests, the speech stream check's
   cases, the three soak text bodies, the audio-in cases. And `check_lfm3_image_cache.py` 32 of 32 with `--warmup`.
3. **Leaves nothing behind**: after warm-up, no session exists, the queue is empty, and the image caches hold
   nothing a client can hit (the warm-up image is generated, not one of any test set). Idle RSS after warm-up at most
   the idle RSS of a server without warm-up after the same four requests, plus 64 MiB.
4. **Ready means warm**: `/health` answers 503 until warm-up ends, then 200; the ready line is written after it. If
   warm-up fails, the server writes why and exits non-zero (never serves cold without saying so). Startup time
   (process start to ready) is reported.
5. **The service** (on Thor, after linger): starts at boot and is warm within rule 4's startup time + 10 s; after the
   engine is killed (SIGTERM to the process, as a crash stand-in: never SIGKILL), it is back and warm within the same
   bound; `systemctl --user stop` exits 0 within 30 s with a request in flight (the request cancelled cleanly); it
   listens on 127.0.0.1 only. Host check: `systemd-analyze --user verify` on the unit.
6. **Nothing else changes**: without `--warmup`, the server is byte-identical in behaviour (every existing host suite
   passes: plain, ASan+UBSan, TSan); the stall, crowd and image checks need no change.

## Notes on measuring the rules as written (2026-11-17, with the code; the rules above are unchanged)
- **Rules 0-1, text's "time to first token".** `/generate` does not stream, so a client cannot see its first token.
  `tools/measure_lfm3_first_request.py` measures it as the whole reply of the same request at `max_new_tokens` 1 (the
  prefill and the first token; the session's create and close are outside the timing). The `/speech` kinds are
  streamed and timed to their first audio (TTS), text (image) or either (audio in) event.
- **Rule 5, "after the engine is killed (SIGTERM ...), it is back".** The server exits 0 on SIGTERM, and systemd's
  `Restart=on-failure` does not restart a unit whose process exited 0. As written the rule needs the unit to restart on
  exit 0 too: `deploy/liquid-lfm3.service` adds `RestartForceExitStatus=0` (`systemctl --user stop` still never
  restarts it). Without that line the rule cannot be met.
- **Rule 3, RSS "after the same four requests".** The warm-up's requests are generated inside the server; the tool
  rebuilds them (the same bodies, the same clip byte for byte, the same image pixels in its own PNG encoding; checked
  against the server's generator on the host, tests/test_lfm3_http_warmup.py) and sends them to a cold server.
- **Rule 2 and rule 5 on Thor.** The exactness checks start the server by path with fixed arguments; with `--warmup`
  they are run with SERVER pointing at a two-line wrapper that execs the binary with `--warmup` added. Rule 5 needs the
  binary to run on the host (the unit cannot reach a process started by `docker exec` with SIGTERM); see docs/service.md.

## Thor results (quiet window 15, 2026-11-17 18:37-19:21; `build/quiet-w15-always-on/`)
- **Build:** main d0a5e80 (the merge 7ce8546), in the container `lfm3-profile-v1`.
- **Container caveat:** since the 2026-11-14 boot this container cannot open `/dev/nvidia-uvm` (prefill-stall-v1, window 14), and any request in it can carry a prefill stall. The repeats below agree to a few ms, so no stall shows in these numbers.
- **Measurement:** time to first token in seconds. "Seq" means the kinds in order on one cold server; "alone" means each kind first on its own fresh server; "warm" is the median of 3 fresh `--warmup` servers.

| Kind | Cold, seq: 1st / 3rd | Cold, alone: 1st / 3rd | Warm: 1st / 3rd (median of 3) |
|---|---|---|---|
| text (`max_new_tokens` 1) | 2.731 / 0.035 | 0.843 / 0.035 | 0.0296 / 0.0293 |
| TTS, first audio | 0.709 / 0.240 | 1.482 / 0.240 | 0.243 / 0.241 |
| 4K image, first text | 1.140 / 0.329 | 2.305 / 0.296 | **1.073 / 0.300** |
| audio in, first event | 0.086 / 0.080 | 1.274 / 0.080 | 0.080 / 0.080 |

- **Rule 0: proceed.** Every kind's first request on a fresh server is slower than its third by more than 1 s in at least one ordering:
  - text +2.70 s when it is the server's very first request, +0.81 s alone;
  - TTS +1.24 s alone;
  - image +2.01 s alone;
  - audio in +1.19 s alone.

  The server's own timings show where the time goes. The engine's `prefill_seconds` for the first and third request differ by at most 0.07 s, so the cold cost is outside the engine's timed prefill and generate: first-use initialisation, not the model's work.
- **Rule 1: FAIL as written.** Text, TTS and audio in pass: their warm first request is within 50 ms of the third. The image does not: 1.073 against 0.300.
  - The tool sends the same photograph three times, so on any server the 2nd and 3rd are hits in the image caches, while the 1st must decode and encode it.
  - Engine prefill confirms this: 0.595 s on the 1st against 0.228 s on the 2nd and 3rd, the same miss and hit as image-cache-v1's C3 (0.60 and 0.228).
  - The warm-up does remove the image's cold cost. The first image on a cold server alone was 2.305 s; on a warm server it is 1.073 s, which matches the third image on the same server when it is a miss.
  - As written, the rule compares a miss with a hit. The amendment below changes the comparison, not the bar.
- **Rule 2 (`--warmup`): PASS for what was run**, every check on a fresh server whose first request of each kind follows the warm-up:
  - the 11 stage-0 image requests (IMAGE SERVER PASS);
  - `check_lfm3_image_cache.py` 32 of 32 identical;
  - the speech stream check PASS;
  - the audio-in server check PASS.

  **Not run:** the three soak text bodies.
- **Rule 3: PASS.** Idle RSS, 10 s after the four requests:

  | Server | Idle RSS |
  |---|---|
  | warm | 5,696.4 MiB |
  | cold, after the warm-up's four requests (replica) | 5,697.1 MiB |

  The bound was the cold figure plus 64 MiB. Nothing is left behind: host-tested, and the warm-up checks it itself before ready.
- **Rule 4: startup reported.** Process start to ready, warm-up included: 4.175, 4.155 and 4.196 s (median 4.18 s). `/health` 503 until warm, the ready line after the warm-up, and failures exiting 1 are covered by the host tests (`tests/test_lfm3_http_warmup.py`).
- **Rule 6: PASS for the same checks without `--warmup`** on the same build (all four checks above). The host suites passed before the merge.
- **Rule 5:** not yet. It needs the host and Tripp's `enable-linger`.

### Amendment to rule 1 (2026-11-17, after window 15, before any rerun)
- **The change:** each kind's 1st request after ready is compared with a later request of that kind that does the same work.
  - For the image, the 3rd request carries a different 4K image of the same size, never sent before: a miss, as the 1st is.
  - Text, TTS and audio in have no input cache, so they are unchanged.
- **The bar is unchanged:** 10% or 50 ms, whichever is larger; median of 3 fresh servers.
- **What I expect:** the warm 1st image within the bar of the 3rd miss. Window 15's sequence already suggests it: cold "seq" image 1st 1.140 s after the other kinds had warmed the server, against the warm 1st 1.073 s.
- **Rule 0's reading is not affected:** cold alone, the first image was 2.305 s against the 1.07 s a miss costs on a warm server.

## Rule 1 as amended (quiet window 18, 2026-11-17 19:43-19:45; `build/quiet-w18-always-on/`): **PASS**
- **Build:** window 15's, unchanged.
- **Tool:** main 07c4d17, with `--image-third` set to `quiet-w10-c3c5/images/fresh/c3-31.png` (3840 x 2160, never sent to these servers).
- **Servers:** 3 fresh `--warmup` servers.
- **Measurement:** time to first token in seconds, median over the 3 servers.

| Kind | 1st | 3rd | Bound | Verdict |
|---|---|---|---|---|
| text | 0.0293 | 0.0286 | 0.050 | PASS |
| TTS | 0.2424 | 0.2405 | 0.050 | PASS |
| image (3rd a miss) | 1.0857 | 1.0749 | 0.1075 | PASS |
| audio in | 0.0808 | 0.0802 | 0.050 | PASS |

The image's 2nd request, a hit on the 1st's photograph, is 0.297-0.302 s. Engine prefill is 0.595-0.624 s on the 1st and the 3rd (misses) and 0.228 s on the 2nd (a hit).

**always-on-v1 standing:**
- Rule 0: proceed.
- Rule 1: PASS as amended.
- Rule 2: PASS for the checks run. The three soak text bodies are still to run.
- Rule 3: PASS.
- Rule 4: startup 4.18 s, with the behaviour host-tested.
- Rule 6: PASS.
- Rule 5: needs the host run and Tripp's `enable-linger`.

## Rule 2, the text part (2026-11-17 19:47-19:49; `build/always-on-warm-text/`): **PASS**
**Check:** `tools/check_lfm3_warm_text.py` on window 15's build. Each of the soak's three text bodies is the first request on a fresh server with `--warmup` and on a fresh server without it, six servers in all. The `/generate` replies are compared field by field: text, generated_ids, finish_reason, consumed and reused tokens, seed, do_sample, cancelled, cache_valid, gpu_batching.
- **Run 1 (kept as `always-on-warm-text-run1-session-id/`): DIFFER, a tool fault.** The only field that differed was `session_id`, a random id per session, which is not output. The fake server in the host test did not return it, so the test missed it.
- **Fix:** the tool now ignores `session_id` along with timings, and the fake returns it.
- **Run 2:** all 3 identical, every server exits 0.

**always-on-v1 standing:**
- Rule 0: proceed.
- Rules 1, 2, 3 and 6: PASS (rule 1 as amended).
- Rule 4: startup 4.18 s.
- Rule 5: to do. It needs the host run and Tripp's `enable-linger`.

## Rule 5, run 1 (2026-11-17 22:10, host; `build/always-on-rule5/`): **FAIL, a server bug found; the run then broke on a script bug**
**Install:** `deploy/install-user-service.sh`.
- Binary: `build/sig-srv`, main 064c1bb, copied to `~/.local/lib/liquid-lfm3/157506acd07b/`.
- Libraries: the pinned CUDA libraries on `--library-path`.
- Install exit 0, enabled, `systemd-analyze --user verify` clean.

**Start:** `systemctl --user start` returned once warm, after **4.94 s**, and `/health` answered 200. The server's line: startup 4.937 s, warm-up 4.084 s (text 1.403, TTS 1.114, image 0.976, audio 0.592). It listens on **127.0.0.1:8080 only**.

**Crash 1 (SIGTERM to the main process, 22:10:15):**
- The server stopped cleanly (exit 0), and systemd scheduled the restart 5 s later, as the unit says.
- **Every restart failed:** `cannot listen on 127.0.0.1: Address already in use`, five times 5 s apart, until systemd's start limit stopped it (22:10:44, "Start request repeated too quickly").
- **Cause:** the listener did not set `SO_REUSEADDR`. The server closes its connections first, so they sit in TIME_WAIT on port 8080 for 60 s, and a new server cannot bind until they clear. Here, the `/health` checks were enough. Any service with recent clients would have failed to come back after a crash.
- **Fix:** `SO_REUSEADDR` on the listener (`lfm/lfm3_http.hpp`). It does not let a second listener share a port (tested).
- **New host test:** a harness restarted at once on the same port after serving requests. It fails without the fix with exactly this error, and passes with it.

**The script then broke (my bug).** With no server running, `MainPID` read 0, and the next crash step ran `kill -TERM 0`, which signals the script's own process group. It ended there, silently. Nothing else was signalled: the group held only the script and its children.
- The script now stops the crash checks unless the PID is a live `liquid-lfm3` process.

**Not measured in run 1:** crashes 2 and 3, the stop with a request in flight, the final start. Run 2, on a build with the fix, does all of rule 5 again, boot aside.

## Rule 5, run 2 (2026-11-17 23:08-23:09, host, the fixed build; `build/always-on-rule5-run2/`)
- **Build:** `build/sig-srv2`, main 67c0617 (`SO_REUSEADDR`), installed at `~/.local/lib/liquid-lfm3/000aa16aa5cc/`. Install and verify clean.
- **Start:** warm and active after 8.12 s; listening on 127.0.0.1:8080 only.
- **Three SIGTERM crash stand-ins:** every one restarted and came back warm, so the run-1 bug is fixed. As written, **FAIL by 0.14-0.20 s**: warm again after 14.32, 14.35 and 14.38 s, against rule 4's startup + 10 s = 14.18 s.
  - **Where the time goes** (journal): the exit about 0.45 s after the signal; the unit's `RestartSec=5` (5.1 s); then the startup, 8.75-8.79 s.
  - **Unexplained:** startup was 8.1-8.8 s here against 4.94 s on run 1's start an hour earlier and 4.18 s in window 15. The warm-up's **text** request took 4.4-5.0 s here against 1.4 s on run 1. The other three kinds took about the same. The power state was identical: MAXN, clocks locked. Not yet investigated: a diagnostic comparing the old and new binary was stopped for the night.
- **Stop with a streamed TTS request in flight: PASS.**
  - `systemctl --user stop` exited 0 after 0.45 s.
  - The unit's result: success, exit status 0.
  - The client got 200 and its first audio, then a `done` event with `"cancelled":true`.
  - No server process was left.
- **Left running:** started again, enabled and active.

**Open, for the morning:**
1. Why the text warm-up went from 1.4 s to 5 s.
2. Whether to cut `RestartSec` from 5 s to 1 s. That is a change to the unit, not to the bar; `StartLimitBurst` still guards crash loops. If so, record it as a change, then run the crash checks again.
3. The boot check: Tripp reboots.

## The slow text warm-up, and the change to the unit (2026-11-18, before rule 5 run 3)
**The slow text warm-up was not reproduced; its cause is unknown** (`build/warmup-diag-v1/`).
- Last night's five starts, by the server's own line: text 1.40 s (22:10), then 4.40, 4.97, 5.04 and 4.99 s (23:08:52-23:09:35), then 1.41 s (23:09:41).
- **The build is not the cause:** `sig-srv` and `sig-srv2`, 4 starts each, alternating: startup 4.98-5.14 s, text 1.28-1.36 s.
- **A build just before is not the cause:** the same server build, then 8 starts at once: startup 4.55-5.05 s, text 0.79-1.29 s.
- Power state was the same last night. Whether it recurs is watched in run 3's starts.

**Change to the unit:** `RestartSec=5` becomes `RestartSec=1`. The bar is unchanged (14.18 s). `StartLimitBurst=5` in 10 minutes still ends a crash loop.
**Expected in run 3:** warm again about 6.5-7 s after each SIGTERM (exit about 0.5 s, restart delay 1 s, startup about 5 s).

## Rule 5, run 3 (2026-11-18 17:34-17:35, host; `build/always-on-rule5-run3/`, unit `RestartSec=1`)
- **Crash recovery: PASS.** Warm again 6.80, 6.79 and 6.59 s after each SIGTERM (bound 14.18 s; expected 6.5-7 s).
- **Stop with a streamed TTS request in flight: PASS.** Exit 0 after 0.44 s, unit result success, the client got `done` with `"cancelled":true`, no process left.
- **Listening on 127.0.0.1:8080 only: PASS.** Install and `systemd-analyze --user verify` clean. Left running, enabled.
- **Still to do for rule 5:** starts at boot (Tripp's reboot).

**The slow text warm-up came back on this run's first start** (startup 8.76 s, text 5.00 s); the four starts after it were normal (text 1.40 s). A test of the GPU's idle time before a start (`build/warmup-diag-v1/idle-*`, service stopped, 16 starts):

| Idle before the start | Starts | Text warm-up |
|---|---|---|
| 1-8 s | 11 | all normal (0.90-1.39 s) |
| 10-60 s | 5 | 3 slow (3.31-3.47 s), 2 normal (1.39 s) |

**Reading: associated with idle, not explained by it.** Last night's slow restarts followed only about 5.5 s of idle, and 5-8 s here was normal. The cause is still unknown. It costs a start at most about 4 s and no client request: the running service holds the GPU open.

## The boot check, set before the reboot (2026-11-18)
Tripp reboots Thor; nobody logs in before the check's data is read (ssh afterwards does not change the timestamps it reads).
- **Started by itself:** the unit is active after boot, with linger on, and its start came from the user manager, not from a login or from a person.
- **Warm in time:** from the unit's start (`InactiveExitTimestamp`) to active (`ActiveEnterTimestamp`, which `Type=notify` sets only once warm) within rule 4's startup + 10 s = **14.18 s**.
- **Serving:** `/health` answers 200, and one real request of each of text and TTS succeeds.
- **Recorded as well:** the server's own warm-up line (the page cache is cold after a boot, so the checkpoint's load is part of this start), and the time from kernel boot to active.
- **The UVM question:** the major number of `/dev/nvidia-uvm` (488 before the reboot), and whether `lfm3-profile-v1-pre-uvm`, started briefly, can open it. If the major is still 488 and it still cannot, the "stale allowlist" explanation is wrong.

## The boot check: results (2026-11-18, 22:29 boot)
Data in Thor's `build/always-on-boot/` (the unit's `systemctl show`, this boot's journal, the replies). Times are seconds since kernel boot (monotonic).

| Check | Result | Verdict |
|---|---|---|
| Started by itself | linger on; `user@1000.service` started by systemd (pid 1) at 13.22 s, active at 13.56 s; the unit started at 13.57 s, `NRestarts=0`. The first login of the user was at 31.0 s (Tripp's desktop password login at the console), after the unit was already active. | **PASS** |
| Warm in time | `InactiveExitTimestamp` 13.566 s to `ActiveEnterTimestamp` 19.544 s: **5.98 s** (bound 14.18 s) | **PASS** |
| Serving | `/health` 200. Text (`/generate`, the lighthouse prompt, 96 new tokens): 200, `stop`, a coherent paragraph. TTS ("Thor is back up, and the service started on its own."): 200, 4.16 s of audio, which the service itself transcribes back word for word (ASR at 16 kHz). The clip is `outputs/always-on-boot/tts-after-boot.wav`. | **PASS** |

**Recorded as well:** the warm-up line: startup 5.98 s, warm-up 5.15 s (text 2.02 s, TTS 1.31 s, image 1.10 s, audio 0.72 s), with a cold page cache. From kernel boot to active: 19.54 s.

**Started at boot: PASS. With it, every rule of always-on-v1 has a verdict.**

**The UVM question.** The major of `/dev/nvidia-uvm` is still 488. `lfm3-profile-v1-pre-uvm`, started briefly, cannot open it: it now has no `/dev/nvidia-uvm` node at all, as uid 0 or 1000 (before the reboot it had a node that gave EPERM). So, by the rule set above, **the "stale allowlist" explanation is wrong.** After the reboot the old container behaves as a freshly created nvidia-runtime container does on this Thor: no UVM node. Why it had a node before the reboot is not known. One unconfirmed guess is that something inside it once created the node, which the device cgroup then refused. The fix holds either way: `lfm3-profile-v1` (with `--device /dev/nvidia-uvm`) opens it as uid 0 and 1000 after the reboot. Both containers have restart policy "no", so `lfm3-profile-v1` needs a `docker start` after each reboot.

## The merged server deployed (2026-11-20)
The always-on service moved from main 67c0617's server (`000aa16aa5cc`) to the merge of audio-handles-v1 (ee2a8c1). The new server is `build/merge-srv`, sha256 `a681eab4c040…`, installed in `~/.local/lib/liquid-lfm3/a681eab4c040/`. The unit is cc7c30b's, with `OOMScoreAdjust=100`. The only line that changed is `ExecStart`, and `systemd-analyze --user verify` is clean.

**Before the deploy, on Thor, the merged tree checked out as follows:**
- Image server: PASS.
- Image cache: 32 of 32 identical.
- Audio-in server: PASS.
- 32 at once: 32 of 32 identical to `lfm3-speech`.
- Speech stream check: PASS, in a quiet window (06:21-06:32, the service stopped, MAXN, clocks locked at the start and the end). All cases exact. TTS first audio: median 0.239 s, max 0.241 s (n 25). Interleaved first audio: median 0.359 s, max 0.416 s (n 14). After a cancel, the next request's first audio was 0.042 s later and the request was exact. Streaming cost 1.004x plain.

An earlier run of the stream check overlapped other GPU work (14:23-14:39 on 2026-11-19), so its timing is void. A second attempt waited for quiet and was ended by a reboot of Thor at 19:42 on 2026-11-19 (Tripp's, by hand; the journal stops at 19:38 with no shutdown entry). The service came back at 19:42 on its own, which is a second, unplanned pass of "started at boot".

**After the deploy:**
- Restart: 4.57 s. Warm-up: startup 4.35 s, warm-up 3.51 s (text 0.68 s, TTS 1.15 s, image 1.06 s, audio 0.61 s).
- `/health`: 200. `oom_score_adj`: 100.
- One text request (`/speech`, text mode: "why is the sky blue"): 200, one coherent sentence.
- One TTS request: 200, 4.0 s of audio, ended by EOS.

**Correction (2026-11-20):** the "restart policy 'no'" above was true when written. With Tripp's approval it was later set to `unless-stopped` on both containers (verified with `docker inspect` on 2026-11-20: `lfm3-profile-v1` and `lfm3-omni-dev`, both `unless-stopped`). Both came back by themselves after the 2026-11-19 reboot, so no `docker start` is needed after a reboot.

## The audio-ttft-v1 merge deployed (2026-11-21)
The always-on service moved from `a681eab4c040` (the audio-handles-v1 merge) to main 717cca6, the merge of audio-ttft-v1: the audio encoder on the tensor cores.
- The new server is `build/merge2-srv`, sha256 `4025f693822c…`, installed in `~/.local/lib/liquid-lfm3/4025f693822c/`. The previous binary stays in place for a rollback.
- The only line of the unit that changed is `ExecStart` (the binary's path). The unit before is kept as `build/merge2-check/liquid-lfm3.service.before`. `systemd-analyze --user verify` is clean.
- `native/` is identical to the source of `a6-srv`, which passed audio-ttft-v1 amendment 6. The binaries differ only by their build path, so the checks were run again on this one.

**Before the deploy, on Thor (`build/merge2-check/`, `tools/thor/deploy_check_merge2.sh`):**
- **Quiet, 13:38–13:49:**
  - speech stream check: PASS;
  - time to first token on the five clips: 0.0372, 0.0512, 0.0733, 0.1762 and 0.4086 s, texts identical to amendment 6's;
  - MONITOR CLEAN (621 samples); power unchanged.

  These times are 4–17% under amendment 6's same-binary-source runs from another window (0.0448–0.4247 s). That is not explained. This pass followed the stream check's heavy GPU work, so it may be the open allocation effect of audio-ttft-v1. It is recorded, not claimed as a gain.
- **Exact:** images PASS; image cache 24 of 24 identical, PASS; audio in PASS; 8 and 32 at once identical to `lfm3-speech`; TTS exit 0; text gate exit 0.
  - The image cache count is 24 here against 32 at the last deploy. That run also passed a listening directory, which adds cases; this one did not.

**After the deploy (14:02):**
- Restart: 5.0 s. Warm-up: startup 4.77 s, warm-up 3.97 s (text 1.30, TTS 1.10, image 0.97, audio 0.60 s).
- `/health`: 200. `oom_score_adj`: 100. The process runs `~/.local/lib/liquid-lfm3/4025f693822c/liquid-lfm3-server`.
- Text ("why is the sky blue"): 200, a coherent answer.
- TTS: 200, 3.68 s of audio, ended by EOS.
- Audio in, the 90 s clip (ASR): 200, the right transcript. The first request after the restart reached its first text 0.401 s after the send; the second, 0.399 s. This is the live service, not a quiet window.
  - So the warm-up's audio request already covers the encoder's first-use costs.

## The mel-tiered-v1 merge deployed (2026-11-21)
The always-on service moved from `4025f693822c` (the audio-ttft-v1 merge) to main d90b115, the merge of mel-tiered-v1: the correctly rounded front end by tiered precision (F3, the same bytes as F2).
- The new server is `build/merge3-srv`, sha256 `96e8510cdd3e…`, installed in `~/.local/lib/liquid-lfm3/96e8510cdd3e/`. The previous two binaries stay in place for a rollback.
- The only line of the unit that changed is `ExecStart`. The unit before is kept as `build/merge3-check/liquid-lfm3.service.before`. `systemd-analyze --user verify` is clean.
- `native/` is identical to the source of `mel3-srv`, which passed mel-tiered-v1.

**Before the deploy, on Thor (`build/merge3-check/`, `tools/thor/deploy_check_merge3.sh`; the server on its own defaults):**
- **Quiet, 20:54–21:05:**
  - speech stream check: PASS;
  - time to first token on the five clips: 0.0325, 0.0428, 0.0577, 0.1250 and 0.2748 s, texts identical to mel-tiered-v1's runs;
  - MONITOR CLEAN (625 samples); power unchanged.

  Again, as at the last deploy, this pass ran right after the stream check and came in under the round's own runs of the same source (0.0400–0.2890 s), by 5–19%. Two deploys now show it. It fits the open finding of audio-ttft-v1: something about the GPU's recent work changes the language model's prefill speed. Recorded, not claimed.
- **Exact:** images PASS; image cache 24 of 24 identical; audio in PASS; 8 and 32 at once identical to `lfm3-speech`; TTS exit 0; text gate exit 0.

**After the deploy (21:18):**
- Restart: 5.11 s. Warm-up: startup 4.86 s, warm-up 3.97 s.
- `/health`: 200. `oom_score_adj`: 100. The process runs `~/.local/lib/liquid-lfm3/96e8510cdd3e/liquid-lfm3-server`.
- Text: 200, coherent. TTS: 200, 3.52 s of audio, ended by EOS.
- Audio in, the 90 s clip: 200, the right transcript. The first text came 0.2645 and 0.2646 s after the send on the first two requests after the restart (the live service, not a quiet window; 0.401 s on the previous server).

## The ttft-rest-v1 merge deployed (2026-11-22)
The always-on service moved from `96e8510cdd3e` (the mel-tiered-v1 merge) to main 6943e84, the merge of ttft-rest-v1: the audio clip decoded once, base64 by table, valid_utf8's ASCII fast path.
- The new server is `build/merge4-srv`, sha256 `71a6e1d793a4…`, installed in `~/.local/lib/liquid-lfm3/71a6e1d793a4/`. The previous binaries stay in place for a rollback.
- It is byte-identical to `rest-srv`, the binary that passed ttft-rest-v1.
- The only line of the unit that changed is `ExecStart`. The unit before is kept as `build/merge4-check/liquid-lfm3.service.before`. `systemd-analyze --user verify` is clean.

**Before the deploy (`build/merge4-check/`, `tools/thor/deploy_check_merge4.sh`):**
- **Quiet, 09:09–09:20:**
  - speech stream check: PASS;
  - time to first token on the five clips: 0.0316, 0.0405, 0.0534, 0.1086 and 0.2380 s, texts identical to the round's;
  - MONITOR CLEAN (631 samples); power unchanged.

  This is the third deploy whose timing pass, run right after the stream check, came in under the round's own runs (here by 5–19%). Recorded, not claimed.
- **Exact:** images PASS; image cache 24 of 24 identical; audio in PASS; 8 and 32 at once identical to `lfm3-speech`; TTS exit 0; text gate exit 0.
  - 32 at once took 4.0 s, back in its usual range. The 5.5 s in the round's run came while the service's warm-up shared the GPU.

**After the deploy (09:33):**
- Restart: 5.0 s. Warm-up: startup 4.76 s, warm-up 3.97 s.
- `/health`: 200. `oom_score_adj`: 100. The process runs `~/.local/lib/liquid-lfm3/71a6e1d793a4/liquid-lfm3-server`.
- Text: 200, coherent. TTS: 200, 2.72 s of audio, ended by EOS.
- Audio in, the 90 s clip: 200, the right transcript. The first text came 0.2279 and 0.2273 s after the send on the first two requests after the restart. That is the live service, not a quiet window; it was 0.265 s on the previous server.
