# Native text frontend wired into the engine binaries

`liquid-lfm3` and `liquid-lfm3-worker` now accept a `"messages"` field in place of `"input_ids"`.
The binary renders the chat template and tokenizes natively, then returns `"text"`, decoded natively.
An `input_ids` request can ask for the text with `"return_text": true`. Requests that use neither
field behave exactly as before, down to the output bytes. The contract is written in
`native/include/lfm/lfm3_text_request.hpp`.

**Build:** `build/cmake-text-v2`, from the pinned compiler image, with 0 warnings. The binary hashes
are in `hashes.json`. Thor ran exactly these sources (every hash checked on both machines).

## Results on the final build

| Check | Result |
|---|---|
| Differential suite (`diff/`): 1,167,834 encode, 16,200 decode, 12,006 chat cases against tokenizers 0.23.3, transformers 5.22.1 and jinja2 3.1.6 | 0 mismatches on aarch64 (Thor) and x86_64; the inputs are byte-identical to `g2-text-v1` |
| End-to-end gate `tools/check_lfm3_native_text.py` (`e2e/`) | PASS |
| All 10 existing native gates (`gates/`): forward, context, sampling, worker, grouped worker, server, cancellation, HTTP cancellation, 32K, HTTP batch | PASS. Control request and output pinned by sha256 |
| Host suite | 1,411 passed |

## What the end-to-end gate covers

**`liquid-lfm3`, 11 cases.** These include the frontend parity cases, Unicode and Python-only strip
whitespace, a multi-turn conversation with `</think>`, merged tool messages, a sampled run (seed 7)
and a 4,966-token prompt at capacity 8192.

Each case runs twice: once with the exact request `tools/lfm3_chat.make_request` builds, and once
with `messages` instead of `input_ids`. In every case:
- the native run echoes the Python `input_ids` exactly;
- both runs generate the same ids;
- the native `text` equals Transformers `frontend.decode`;
- the `input_ids` run carries no new fields.

Invalid messages exit with status 1 and write no output.

**`liquid-lfm3-worker`:**
- **Two-turn conversation:** one session takes `messages`, the other `input_ids`. Generations and
  prefix reuse are identical; turn 2 reuses 21 tokens.
- **`return_text` on an `input_ids` request:** returns the decoded text.
- **`generate_batch`:** one `messages` lane and one `input_ids` lane, GPU-batched, with identical
  generations.
- **Rejections:** 7 rejected requests give **the same reason `make_request` gives**, in the same
  check order. They include a reserved token, two reasoning fields, a late system message, a
  multimodal part, a zero count with bad messages, a count that fills the capacity, and a prompt
  over capacity. The worker stays usable afterwards.

## Red-team findings (independent review plus my own checks)

No input that both sides accept produces different output.

**Fixed:**
- **Two reasoning fields:** native validated each field before counting them, so the error reason
  differed. It now counts first.
- **Check order:** native checked the messages before the capacity. The binaries now follow
  `make_request`'s order: integer ranges, then messages, then the fit check. The tool follows
  `TextFrontend.encode_messages`.
- Regression tests for both were added.

**Kept as known and deliberate.** These are documented in `lfm3_text_request.hpp`; in each,
outputs agree whenever both sides accept the input:
- **Duplicate JSON keys** are rejected. The serving path's `read_json` rejects them too.
- **Whole-valued floats** such as `16.0` are accepted for integer fields. This is the existing
  contract of every native request field, because `lfm::Json` keeps numbers as doubles.
- **Lone surrogates, NaN and nesting deeper than 64** are rejected at parse time, with the parser's
  own wording.
- **Decode skips ids the tokenizer doesn't define**, as Rust `tokenizers` does. The engines never
  generate such ids (G6).

**Verified:**
- `python_space` equals Python 3.12's `str.strip` set, enumerated over every code point (29).
- Encoding 4 MiB pathological inputs takes 0.8–4.1 s with no quadratic case. Peak memory is
  370 MB, against 160 MB when only the tokenizer is loaded.

**Cost:**
- Loading the tokenizer takes about 0.65 s.
- `liquid-lfm3` loads it only for requests that use `messages` or `return_text`. The worker loads it
  once, on the first such request, and checks that it agrees with the verified token count (G6).

**Reference behaviour, reproduced as-is:** `<image>` (id 124907, a special token) is not caught by
the frontend's `<|…|>` reserved check, so user text that contains it encodes to that token. Both
sides do this identically. It is not one of the six suppressed modality ids.

## Not yet done

Superseded for serving: `liquid-lfm3-server` (see `validation/native-server-v1`) replaces the Python HTTP
server. At the time of this run, the Python CLI and HTTP server still called the Transformers frontend. Switching them to send
`messages` and use the returned `text` would drop Transformers from serving; the HTTP layer itself
stays Python until G7. The stream decoder is implemented and tested, but no binary streams yet.
