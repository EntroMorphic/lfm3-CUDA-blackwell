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
