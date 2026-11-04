# Local text server (experimental)

## Additional signal-mask remediation

Independent verification of the startup fix found that blocking SIGTERM/SIGINT
around process creation also blocked them in the exec'd child (observed child
mask `0x4002`). `restore_signals=True` does not restore that mask. The transport
now uses a small Python exec-only launcher to restore the original mask before
executing the worker, without a thread-unsafe `preexec_fn`. A host regression
checks exact child/parent masks and requires ordinary SIGTERM exit within one
second, independently of forced-cleanup fallback. This closes the signal-mask
defect; native HTTP evidence is recorded separately below when available.

`tools/lfm3_server.py` adds an authenticated, IPv4-loopback-only HTTP wrapper
around `liquid-lfm3-worker`. This is **not GPU batching**, a public-facing server,
or evidence that the native GPU correctness gates passed. Every inference and
lifecycle RPC runs through one serialized owner thread. Native weights are shared;
each of at most two sessions has its own native KV/cache allocation. Aggregate
reserved capacity is at most 32768 tokens, not 32768 per concurrent session.

## Explicit startup (not run by the implementation/tests)

Supply `--worker /absolute/path/liquid-lfm3-worker --checkpoint /path/to/checkpoint`
to `python tools/lfm3_server.py`, plus `--token-file /path/to/token` (owned regular
file, exact mode 0600, no symlink). Alternatively configure `LFM3_SERVER_TOKEN`.
The token must contain 32–256 URL-safe characters; generate a cryptographically
random value, not a human password. Tokens, prompts and session IDs are not logged.
Worker stderr is discarded rather than captured in an unbounded pipe/log. The
process environment supplies CUDA library discovery; the server does not change
the checkpoint, model configuration, or library paths. Port defaults to 8080;
the only accepted bind address is the literal `127.0.0.1`.

The official local tokenizer frontend is used without a Transformers model,
processor, audio codec, remote download, or checkpoint-code execution. Its existing
tokenizer dependency must already be installed; HTTP/transport need only stdlib.

## API

All POSTs require `Authorization: Bearer TOKEN`, `Content-Type: application/json`
and exactly one decimal `Content-Length`. Transfer encoding is rejected. Objects
reject duplicate keys, nonfinite numbers and nesting deeper than 64. UTF-8 is
strict; the frontend also rejects escaped invalid Unicode and reserved token
placeholders. Unknown request/message/sampling fields are rejected. No tools are
executed; plain tool-response text is supported by the existing frontend, but
tool definitions/calls and multimodal inputs are not.

* `POST /sessions` with `{"capacity":32768}` returns a random 192-bit
  `session_id`, capacity, and `gpu_batching:false`. For two sessions choose smaller
  allocations whose sum fits 32768. Configuration may lower either quota.
* `POST /generate` with `{"session_id":"ID","messages":[{"role":"user",
  "content":"Hello"}],"max_new_tokens":32,"sampling":{"do_sample":false}}`.
  Sampling is optional and follows `lfm3_chat.sampling_options`, including explicit
  seed validation. Return fields: `session_id`, decoded `text`, `generated_ids`
  (including a final stop token if present), `reused_tokens`, `consumed_tokens`,
  `do_sample`, `seed`, `gpu_batching:false`.
* `POST /sessions/ID/reset` with `{}` clears that cache but retains its allocation.
* `POST /sessions/ID/close` with `{}` releases that session/allocation.
* `GET /health` is unauthenticated and returns only `ready` and
  `gpu_batching:false` (HTTP 503 after worker failure).

There is no batch-input endpoint, streaming, automatic session history, or silent
history truncation. Send the full conversation each time. The native worker checks
the cached prefix and re-prefills on mismatch. The full rendered prompt plus
**all requested** output tokens must fit the session capacity, capped at 32768.
An individual 32K allocation uses the entire aggregate budget. A session ID is
not a separate authorization principal: all users of the global bearer are
trusted local peers with access to all sessions if they know their IDs.
Sessions expire after 300 seconds idle by default (`--session-idle-timeout`,
positive finite seconds, at most 86400). Creation and completion of generation
or reset restart that interval. A queued/executing session lease never expires.
Expiration closes the native allocation and removes its session ID; subsequent
use returns 404. Clients must create a new session after expiration.

## Bounds, failure and operational limits

Defaults: eight simultaneous HTTP handlers, four pending owner-queue jobs (plus
one executing), 4 MiB request/response bodies, 4096 messages and 4 MiB combined
conversation text under the existing frontend. Admission is nonblocking: session
quota, queue saturation or tokenizer contention returns 429; an already leased
session returns 409. Tokenization and metadata/budget checks finish **before GPU
queue admission**. A session lease includes validation, queue wait and execution;
reset/close cannot race generation. No stdin writes from HTTP threads occur.
The listener backlog is the stdlib bounded socket backlog; connections exceeding
handler capacity are closed without spawning a thread, not promised a JSON 429.
Tokenizer input/work concurrency is bounded, but an in-process tokenizer call is
not preemptible; the queue/request deadline starts after frontend validation.

Header and body deadlines default to ten seconds each (header trickling has an
absolute deadline). Socket writes also time out. Each worker RPC uses one shared
absolute write/read deadline, default 300 seconds, and 4 MiB bounded framing.
Queue/request wait defaults to 330 seconds; expired queued jobs do not execute.
HTTP 504 for a running request is **outcome unknown**, not cancellation: its lease
remains held until execution completes/fails. A disconnected caller similarly
does not cancel execution. A create that exceeds its request deadline is special:
the owner closes its newly created allocation after completion, before releasing
the reserved quota, instead of retaining an unreachable ID. Completion publication
and abandonment are synchronized so a timeout racing completion cannot strand it.
A lost successful HTTP create response is recovered by idle expiration, not by
idempotent retry: retrying can return 429 until that allocation expires. The owner
checks expiration between every job and at least every 50 ms while idle. Native
close RPCs are serialized with inference, so expiry can be delayed by the bounded
current RPC and pending expiration closes (at most two sessions); it never runs
native work on a handler thread. A failed expiration close fails the entire service
closed. There is no request-rate limiter.

Any native error, pipe EOF/deadline, malformed response or inconsistent generation
metadata fails the service closed, clears **all** server session records, terminates
and reaps the process group, and makes subsequent admitted work return 503. There
is no automatic restart: stop/restart the server, then create fresh session IDs.
Native error text is not exposed to clients. Worker cleanup uses bounded TERM/KILL
waits and closes pipes. The bearer environment variable is removed from the
native child's environment. Service shutdown rejects new work, invalidates sessions,
drains rejected queued jobs, waits for the bounded running operation, then closes
the worker. The CLI installs SIGTERM handling before worker construction, retains
ownership throughout partial initialization, and handles SIGINT and SIGTERM through
this cleanup path, including stalled handshakes. Spawn briefly masks these signals
until the child handle is recorded; cleanup retains bounded KILL escalation even
when a child or descendant ignores/blocks TERM. A
supervisor should still terminate the whole service group if force-killing the
Python process, since SIGKILL cannot run cleanup.

This focused stdlib wrapper has no TLS, per-client ACLs, durable sessions,
multi-process serving, true GPU batching, native cooperative cancellation, public
deployment hardening, throughput SLA, or demonstrated multi-user load stability.
Do not expose it via a public proxy or reinterpret loopback plus a global bearer
as tenant isolation. Health readiness is process/service state, not a GPU probe.

## Host verification

`python -m pytest -q tests/test_lfm3_worker_client.py tests/test_lfm3_server.py`
uses fake subprocesses, fake workers/frontends and in-memory HTTP handlers. It
does not start any network listener, load a model, use SSH, or access a GPU.
Coverage includes strict JSON, frame limits, EOF/timeout/write-backpressure and
cleanup, handshake rejection, quota/budget validation, lifecycle isolation,
per-session leases, serialized ownership, queue rejection, timeout lease retention,
abandoned-create reclamation, idle allocation expiration, startup SIGTERM descendant
cleanup, fail-closed invalidation (including ambiguous nested sampling metadata),
token-file permissions and mocked-handler HTTP validation.
Real native GPU gates and end-to-end HTTP/load tests remain separate requirements.

Host contract/regression evidence is not GPU acceptance or serving load evidence.
Separate socketpair framing tests are maintained in `tests/test_lfm3_server_sockets.py`.

The complete stabilized host suite passes **552 tests** (including 15 actual
socketpair parser/framing/deadline/admission tests and seven HTTP gate
counterexamples): `validation/host-http-remediated-v1.log`. Socket tests also
verify that a non-reading peer cannot retain a response writer or provoke a
second reply after a partial timed-out response. Counterexamples reject wrong
control IDs, lost cache reuse, failed recall/reset/invalidation and binary changes.
`tools/check_lfm3_server.py` separately starts an explicit temporary authenticated
loopback listener for native acceptance; these host counts do not imply it passed.

## Native HTTP verification

The corrected `http-native-remediated-v2` run **passed** against worker
`3bc90ece...`. Evidence: `validation/http-native-remediated-v2/`; actual response:
`../tmp/http-native-remediated-v2/control-response.txt`. Local and remote server,
transport and gate source hashes match and are recorded in `verification.json`.

The finite gate checks exact control IDs, authenticated create/quota/rejections,
split-delimiter rejection, A/B history/cache reuse, sampled code-word recall,
reset parity, close, normal SIGTERM fault injection, health failure and all-session
invalidation. Three eight-client rounds yield six successful exact-output requests
and eighteen busy-session409 rejections, followed each round by exact output from
both sessions. No worker/test process remains after cleanup. This is evidence of
bounded admission and serialized execution, not GPU batching or a throughput SLA.

The failed first run and traceback remain in `validation/http-native-remediated-v1/`.
Its assertions were not weakened. Native cancellation, sustained soak/resource/
performance acceptance and the integrated sanitizer gaps remain open.

### Final-source confirmation

Two further cleanup regressions ensure final reap timeout still closes the
parent's pipes and cleanup failure cannot kill the owner thread before callers
receive503. The service remains failed with `cleanup_failed=true`; a reap timeout
is not a claim that an uninterruptible child has exited. The full suite now
passes **554 tests** (`validation/host-http-remediated-v2.log`).

The final-source native rerun **passed** in `validation/http-native-remediated-v3/`:
six exact-output successes, sixteen409 busy and two429 contention rejections over
three eight-client rounds; both sessions remain exact after every round. Ordinary
SIGTERM fault injection and all-session invalidation pass, and no worker/test
process remains. Remote/local source hashes match `verification.json`. Actual
control text: `../tmp/http-native-remediated-v3/control-response.txt`.
