# Running as a service

`liquid-lfm3-server` can run as an always-on, pre-warmed **systemd user service**: it starts with the machine,
warms up before it reports ready, restarts when it fails, and stops on `SIGTERM` (validation/always-on-v1).

## Warm-up: `--warmup`

Off by default; without it the server behaves exactly as before. With it, after the engine is loaded and before
the ready line, the server sends itself one request of each kind through its own front end and engine, and
discards the replies:

| Kind | Request |
|---|---|
| text | `/generate` in a fresh session (capacity 1024, 16 new tokens), the session closed after |
| TTS | `/speech`, audio mode, streamed, one short sentence (48 frames) |
| image | `/speech`, text mode, a generated 3840 x 2160 PNG of coloured tiles (8 new tokens) |
| audio in | `/speech`, interleaved, a generated 2 s, 16 kHz tone glide (32 new tokens) |

The inputs are generated in code (`native/include/lfm/lfm3_warmup.hpp`), the same bytes every time; none comes
from a test set. A kind the engine cannot run (no speech, no vision or audio encoder for the checkpoint) is skipped
and reported as skipped.

While it warms up the server already listens: `GET /health` answers **503** `{"ready":false,...}`, and every
`POST` answers **503** `{"error":"warming_up"}`. Then:

1. one JSON line on **stderr**, the timings:
   `{"warmup":{"startup_seconds":S,"warmup_seconds":W,"requests":[{"kind":"text","seconds":...},{"kind":"tts","seconds":...,"prefill_seconds":...,"generate_seconds":...},...]}}`
   (`startup_seconds`: from process start to the end of the warm-up; `prefill_seconds` and `generate_seconds`: the
   engine's own, from the `/speech` replies);
2. the ready line on **stdout**, with one key added: `{"ready":true,"port":N,"warmup_seconds":W}` (only with
   `--warmup`; every tool in this repository reads the ready line as JSON and picks the keys it needs);
3. `GET /health` answers 200.

The warm-up leaves nothing behind: its session is closed, the queue is empty, and the warm-up image is in neither
image cache (the front end's token counts, the engine's encoder rows). Both caches keep nothing while the server
warms up, and get their `--image-cache-mb` budget once it is done: the generated image could be rebuilt by any
client, so it is never cached, not even briefly. (Skipping rather than clearing afterwards: nothing to undo and no
new engine operation. The cost: the cache's own first insertion, one GPU allocation for the first image a client
sends, is not warmed.) The warm-up checks all this itself before it reports ready.

If any warm-up request fails, the server writes `<name>: warm-up failed: warm-up <kind> request: <reason>` to stderr
and exits **1**: it never serves cold without saying so. A `SIGTERM` during the warm-up is a stop: `stopped during
warm-up` on stderr, exit 0.

## The unit: `deploy/liquid-lfm3.service`

A template, filled in by `deploy/install-user-service.sh`:

- `Type=notify`: the unit is *active* only once the server is warm. The server sends sd_notify's `READY=1` itself
  to `$NOTIFY_SOCKET` right after its ready line (no libsystemd); without `NOTIFY_SOCKET` it sends nothing.
- `ExecStart=... --experimental MODEL --port PORT --warmup`. **Loopback only**: the server binds `127.0.0.1` and
  has no option to bind anything else. The bearer token is the default file `~/.config/liquid-lfm3/token` (mode
  0600, created on the first start), which `tools/lfm3_token.py` reads for clients.
- `Restart=on-failure` (a non-zero exit, a crash, a failed warm-up), `RestartSec=1`, at most 5 starts in 10 minutes;
  plus `RestartForceExitStatus=0`: the server exits 0 on `SIGTERM`, so a `SIGTERM` from outside systemd would
  otherwise leave it down. `systemctl --user stop` is never followed by a restart.
- `KillSignal=SIGTERM`, `TimeoutStopSec=30`, **`SendSIGKILL=no`**: this project never SIGKILLs a CUDA process. A server
  still running 30 s after `SIGTERM` leaves the unit failed and the process running, for a person to look at.
- `TimeoutStartSec=900` for the checkpoint load and the warm-up.
- `OOMScoreAdjust=100`: user units default to 200, so when memory runs out the kernel kills this service first
  (2026-11-19, three times in 16 minutes alongside another GPU workload; systemd restarted it each time). 100 is the
  lowest a user unit can set without privilege, still above system and docker processes (0); lower needs root, an
  `OOMScoreAdjust` drop-in for `user@UID.service`. The kernel's OOM kill is a SIGKILL nothing here can prevent.

## Install

```sh
deploy/install-user-service.sh --server /path/to/liquid-lfm3-server --model /path/to/models/LFM3-Omni-2B \
    [--port 8080] [--library-path DIRS] [--speech-rows N] [--image-cache-mb N] [--dry-run]
```

It writes `~/.config/systemd/user/liquid-lfm3.service` (`$XDG_CONFIG_HOME/systemd/user`), then runs
`systemctl --user daemon-reload` and `systemctl --user enable liquid-lfm3.service`. It does not start the service
and never uses `sudo`. `--library-path` sets `LD_LIBRARY_PATH` for the server (the CUDA libraries it was built
against, when they are not on the loader's default path); `--dry-run` prints the unit and changes nothing.

For the service to start at boot without anyone logged in, lingering must be on for the user. Run once, yourself:

```sh
sudo loginctl enable-linger $USER
```

Then:

```sh
systemctl --user start liquid-lfm3        # or reboot
systemctl --user status liquid-lfm3
journalctl --user -u liquid-lfm3 -f       # the warm-up line, failures
curl -s http://127.0.0.1:8080/health      # {"ready":true,...} once warm
systemctl --user stop liquid-lfm3         # SIGTERM; running requests are cancelled at their next token
```

The unit runs the binary directly, not through `docker exec`: `docker exec` does not forward the signals it receives
to the process it started, so neither `systemctl stop` nor a restart would reach the server. The binary must therefore run
on the host, with its libraries on `--library-path`.

Check a unit before installing it: `deploy/install-user-service.sh ... --dry-run > unit.service &&
systemd-analyze --user verify unit.service`.
