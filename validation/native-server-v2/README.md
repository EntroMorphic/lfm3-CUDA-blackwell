# G7 red-team remediation: liquid-lfm3-server v2

This follows `validation/native-server-v1`, which has the design and the full G7 evidence. A red-team pass
over that server found no high-severity issues, two medium ones and several low ones. All of them are
fixed here and verified again on Thor.

**Build:** `build/cmake-server-v3`, from the pinned compiler image, with 0 warnings. Hashes are in
`sha256.txt`.
- `liquid-lfm3-worker` is byte-identical to v2 (`f4f22ae7…`), so the 10 existing gates in
  `native-server-v1/gates/` still apply unchanged.
- `liquid-lfm3`'s sha256 changed, from `2dcd128c…` to `322b9cfc…`, but its source did not. The two
  binaries differ in 2 bytes, inside nvcc's PID-numbered temporary filename in `.symtab`/`.strtab`
  (`tmpxft_000000b0` against `tmpxft_000000ae`). After `objcopy --strip-all` they are identical
  (`7447d154…`).

## What changed (`native/include/lfm/lfm3_http.hpp`)

| Finding | Severity | Fix |
|---|---|---|
| Memory amplification: eight concurrent 4 MiB bodies were parsed at once, and each parse tree was then deep-copied, which peaked at 2.38 GB | Medium | Bodies over 1 MiB are parsed one at a time under a mutex, and the raw buffer is freed before the parse tree is used. The tree is shared (`shared_ptr<const Json>`) instead of copied, and dropped once the prompt is encoded. The peak is now 718 MB |
| An engine call that never returns would hang the service | Medium | `Settings::engine_timeout` (default 300 s; `--engine-timeout`). A watchdog marks the service failed, as `WorkerFailure` would, and answers 503 `worker_unavailable` from then on. Live sessions are dropped, cancel frames are sent, and a late reply is refused |
| Header parsing differed from `http.client.parse_headers` / `email.feedparser` | Low | Rewritten to follow the feedparser rules: lines split on `\r\n`, `\r` and `\n`; a malformed line ends the headers; an empty name is dropped; continuations fold; `From ` lines are skipped; EOF ends the headers |
| Limits differed | Low | The line limits are 65,536 bytes including the terminator (414 for the request line, 431 for a header). The 100-line header limit counts the terminating line, as Python does. Request-line whitespace follows `str.split`, including 0x85 and 0xa0 |
| `Server` header | Low | `LFM3Local ` with Python's trailing space |
| The listener stayed open during shutdown | Low | `serve()` closes the listener when it returns. Waiting for handlers is now a separate `drain()` |
| A stuck shutdown could not be overridden | Low | A second SIGTERM or SIGINT exits at once with status 1. `close()` waits at most `engine_timeout + 5` s, then reports `cleanup_failed` |

Deliberate differences from the Python server are listed in the header comment of `lfm3_http.hpp`.

## Evidence

| Check | Result |
|---|---|
| Server gate `tools/check_lfm3_native_server.py` (`gate/`) on the GPU: serving, cancellation, batching and a 40-cycle soak, robustness, a 22-request differential against the Python server, shutdown | **PASS.** Startup 0.92 s. Warm RSS 4.95 GB, flat. Descriptors 47, constant. Differential byte-identical. SIGTERM exits 0 in 0.31 s |
| Request-validation differential `tools/lfm3_http_diff.py` (`http-diff/`): 12,478 requests | **0 mismatches** on aarch64. 9 expected differences, the same as v1. The case, reference and native files hash identically to v1 |
| HTTP framing parity `tests/test_lfm3_http_framing.py`: 79 raw requests against Python's real `LocalHTTPServer`/`Handler` (`http.server`, `http.client`, `email.feedparser`) | 79 passed (x86-64) |
| Host tests `tests/test_lfm3_http_native.py`: 26 scenarios × {plain, ASan+UBSan, clang TSan} | 78 passed (x86-64). On aarch64 (`host-http/`), 75 passed and 1 failed (below). The 81 skips are the 79 framing cases, because Transformers isn't installed on the Thor host, and the plain-only memory test under ASan and TSan |
| Full host suite | 1,566 passed, 2 skipped (x86-64) |

New scenarios are:
- `test_large_bodies_are_parsed_one_at_a_time`: eight concurrent 4 MiB bodies must add less than 1 GB of
  peak RSS.
- `test_engine_timeout_fails_closed`.
- `test_listener_closes_and_second_signal_exits`.

**The aarch64 failure** is `test_frontend_is_serialized[asan]`. Its 2.4M-character prompt does not
finish tokenizing within the test client's 60 s timeout. It is caused by ASan, not by the server:

| Build on Thor | Work | Time |
|---|---|---|
| Plain | 2.4M characters | 0.39 s |
| Plain | 300K characters | 0.05 s |
| ASan+UBSan | 300K characters | 22.9 s |
| ASan only | Loading `tokenizer.json` at startup | 107 s |
| ASan+UBSan | Loading `tokenizer.json` at startup | 216–221 s |

- ASan on Thor's aarch64 host is about 460× slower than the plain build on allocation-heavy code.
- `-fno-omit-frame-pointer` doesn't change it.
- The cost is linear in input size. On x86-64, plain -O2 handles 1.2M characters in 0.14 s.
- The same scenario passes plain and under TSan on aarch64, and in all three variants on x86-64.

The ASan per-test startup is also why the aarch64 run took 1 h 46 min.
