# G1–G3: native LFM3 text frontend, bit-exact against the reference

`native/include/lfm/lfm3_text.hpp` replaces Python, Transformers, Jinja and Rust `tokenizers` on the
text path. It has three parts:

- **G1:** chat-template render (text subset) and `encode_messages`.
- **G2:** byte-level BPE tokenizer.
- **G3:** detokenizer and streaming decoder.

**Reference:** tokenizers 0.23.3, transformers 5.22.1, jinja2 3.1.6 and Python 3.12.3, inside
`lfm3-omni-dev` on Thor, through `tools/lfm3_text_frontend.py`, the same calls text serving makes.

| Family | Cases | Mismatches (x86_64) | Mismatches (aarch64, Thor) |
|---|---:|---:|---:|
| encode | 1,167,834 | 0 | 0 |
| decode (full and streaming) | 16,200 | 0 | 0 |
| chat (render + ids; 2,732 rejected by both, with identical reasons) | 12,006 | 0 | 0 |

## What the cases cover

**Encode:**
- Every Unicode scalar value (1,112,064) in six regex contexts.
- Every edge of the measured letter and number class tables.
- WikiText-2 test, each line and the whole file.
- 30,200 random or adversarial strings: all whitespace kinds; CR/LF runs; contractions in every
  case, including U+017F and U+212A; digits from many scripts; combining marks; emoji and ZWJ;
  CJK; added tokens and fragments of them; long runs.

**Decode:**
- Random ids, including special ids and ids past the tokenizer (125033–127999).
- Raw-byte tokens, which stress lossy UTF-8.
- Byte runs split by special and non-special added tokens.
- decode(encode(x)), which returned x every time.

**Streaming:** the concatenated stream equals the full decode. At most 3 bytes are ever held back,
and `finish()` releases only replacement characters.

**Chat:**
- The frontend's parity cases plus random conversations.
- `compare` reports how often each template path was taken, and every path was hit. The paths
  include: reasoning kept and dropped, `</think>` split with Python-only strip whitespace,
  tool-call-id merging, `<tool_response>` passthrough, list content, empty system, and capacity
  rejection.

## Measured facts the implementation relies on

The character classes of the `Split` regex come from Oniguruma, not Python's Unicode tables.
- They were measured code point by code point in `validation/g2-classes-v1`.
- Oniguruma has 4,302 more letters and 80 more numbers than Python's unicodedata 15.1.
- The header is generated from those measurements.

A skipped special token does **not** end a byte run: bytes on either side of it join into one
character. 380 cases tell this rule apart from the alternative.

Whether a kept added token ends the run cannot be observed. Every added token is ASCII, and an ASCII
byte ends any partial UTF-8 sequence either way.

## How to reproduce

```
python tools/lfm3_text_diff.py cases --out build/g2-diff-v2           # inputs match cases.json sha256
docker exec lfm3-omni-dev python /workspace/liquid-CUDA/tools/lfm3_text_diff.py reference --dir ...
g++ -std=c++20 -O2 -Inative/include native/tests/lfm3_text_tool.cpp -o lfm3-text-tool
lfm3-text-tool TOKENIZER encode|chat X.jsonl X.native.jsonl; ... decode decode.in.jsonl decode.native.jsonl
python tools/lfm3_text_diff.py compare --dir build/g2-diff-v2
```

`tests/test_lfm3_text.py` (39 tests) pins what the differential suite cannot:
- The loader refuses 19 kinds of `tokenizer.json` deviation.
- Input limits are enforced, and invalid UTF-8 and lone surrogates are refused.
- The stream decoder releases each character as soon as it is complete.

## Not yet done

The engine binaries do not use the header yet. Text serving still renders and tokenizes through the
Python frontend. Wiring the header into `liquid-lfm3` and the worker is part of G7 (single
executable).
