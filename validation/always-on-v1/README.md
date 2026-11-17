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
