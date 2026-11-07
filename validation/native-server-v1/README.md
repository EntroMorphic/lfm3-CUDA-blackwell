# G7: one executable for LFM3 text serving

`liquid-lfm3-server` runs loopback HTTP, the native text frontend and the engine in one process. Python,
Transformers, Jinja and Rust `tokenizers` are no longer on the serving path.

```
LFM3_SERVER_TOKEN=... liquid-lfm3-server --experimental MODEL_DIR [--token-file F] [--port N] ...
→ {"ready":true,"port":N}
```

The API is the one `tools/lfm3_server.py` serves:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness |
| `POST /sessions` | Create a session |
| `POST /generate` | Generate on one session |
| `POST /generate_batch` | Two-lane decode group |
| `POST /sessions/ID/reset` | Discard a session's cache |
| `POST /sessions/ID/close` | Close a session |
| `POST /sessions/ID/cancel` | Cancel a running generation |

It keeps that server's statuses, error codes, admission, lease and deadline rules, and its response
bytes.

**Build:** `build/cmake-server-v2`, from the pinned compiler image, with 0 warnings. Binary and corpus
hashes are in `sha256.txt`; Thor ran exactly the committed sources.

## How it is put together

| Part | File | Origin |
|---|---|---|
| HTTP and service layer, host-only | `native/include/lfm/lfm3_http.hpp` | Port of `lfm3_server.py` (Service, LocalHTTPServer, Handler) and the request rules it takes from `lfm3_chat.py` and `lfm3_worker_client.py` |
| Engine | `WorkerCore` in `native/src/lfm3_worker.cu` | The worker's request loop, extracted unchanged |
| Server binary | `native/src/lfm3_server.cu` | Joins the two |

**The worker itself:** `liquid-lfm3-worker` is now a thin stdin/stdout loop over `WorkerCore`. All 10
existing gates pass on it.

**Inside the server:**
- One engine-owner thread calls the core with protocol-2 lines, so the engine code paths are the
  worker's, unchanged.
- Cancellation uses the same nonblocking pipe and `CancellationReader` as the worker. Frames are written
  in-process.
- Prompts are rendered and tokenized by `lfm3_text.hpp` on the request thread, holding the one-at-a-time
  frontend lock.
- Text is decoded by the engine (`return_text`).

## Evidence

| Check | Result |
|---|---|
| Server gate `tools/check_lfm3_native_server.py` (`gate/`), on the GPU, black box | PASS |
| Request-validation differential `tools/lfm3_http_diff.py` (`http-diff/`): 12,478 requests to the Python Service and to the native layer, both on a fake engine; pinned stack transformers 5.22.1, tokenizers 0.23.3, jinja2 3.1.6 | **0 mismatches** on aarch64. 9 expected differences (below) |
| Host tests `tests/test_lfm3_http_native.py`: 23 scenarios × {plain, ASan+UBSan, clang TSan} against a fake engine | 69 passed |
| Text differential (G1–G3, rerun because `lfm::Json` gained an integral flag) | 0 mismatches |
| Native text end-to-end gate | PASS |
| All 10 existing native gates (`gates/`) on the refactored worker: forward, context, sampling, worker, grouped worker, server, cancellation, HTTP cancellation, 32K, HTTP batch (10-minute soak) | PASS. Control request and output pinned by sha256 |
| Host suite | 1,480 passed |

### Server gate

The gate starts the binary as a deployment would (token file, `--port 0`) and drives it only over HTTP.

**Serving:**
- **Control output:** the pinned 128-token control is reproduced exactly.
- **Requests refused:** unauthorized requests get 401 and the session quota gives 429. `top_p=0`,
  duplicate JSON keys and a split `<|im_end|>` get 400.
- **Requests accepted:** a benign multipart message gives identical ids.
- **History:** turns reuse the cache, and a sampled recall returns "cobalt".
- **Contention:** three rounds of 8 concurrent clients, each giving 200 or 409 with exact ids, and both
  sessions exact afterwards.
- **Lifecycle:** reset forces a full prefill, and close makes the session return 404.

**Cancellation:**
- The partial generation is an exact prefix of the control, and the cache is discarded.
- A second cancel returns 409.
- The cancelled session recovers with a full prefill, and the peer session's cache is untouched.

**Batching:**
- Two lanes, one greedy and one sampled with `top_k=1`, give exact ids, with 31 paired decode calls each.
- Cancelling lane B leaves lane A exact and returns B's exact prefix.
- 40-cycle soak: warm RSS growth was 0 bytes (budget 32 MiB) and descriptor growth was 0 (budget 2).
- Median latency: 1.13 s for a group of two, against 1.85 s for the same two run serially.

**Robustness:** sampling with temperature 1e-300 or 1e6, repetition penalty 1e300, top_p 1e-300,
min_p 1 or top_k 128000 generates normally, and the server stays ready.

**Differential:** 22 scripted requests went to `tools/lfm3_server.py` (with the same build's worker) and
to `liquid-lfm3-server`. Status and response bytes were identical for all 22. The script covers controls,
rejections, history, two sampling configurations, both kinds of batch, reset, cancel, close, a 404 and a
capacity-fit rejection.

**Shutdown:** SIGTERM during a 2,048-token generation returned the cancelled reply and exited 0 in
0.26 s.

**Startup:** the server is ready 0.96 s after exec. The weights load lazily on the first generation, as
in the worker.

### Deliberate differences from `tools/lfm3_server.py`

Each is at a boundary no conforming client reaches, and each is listed in the `lfm3_http.hpp` header.
- **Numbers that underflow a double** (`1e-400`, `5e-324`, `1e-310`) are rejected with 400; Python reads
  them as 0 or a subnormal. These are the 9 expected differences in the request-validation differential.
- **Malformed HTTP** gets a JSON error instead of http.server's HTML page. This covers two-word request
  lines, folded headers, header lines without a colon and unsupported methods.
- **Shutdown** cancels running generations at their next token boundary instead of waiting for them to
  finish.
- **After a fatal engine error** the server keeps answering 503 as Python does. It keeps its GPU memory
  until it exits, because there is no separate worker process to kill. Restart it.
- **`--port 0`** is accepted and picks a free port. The ready line on stdout reports the port.

## Not yet done

- **One-shot CLI:** `tools/lfm3_chat.py` still uses the Transformers frontend. `liquid-lfm3` with
  `"messages"` is already the native one-shot path.
- **Streaming:** the server does not stream; replies are whole. `StreamDecoder` is ready for it.
- **cuBLAS:** still linked (G4).
