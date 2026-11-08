# Tree-verify decode (single-request speed, step 2): exact, and up to 4.1x on one request

`native/src/lfm3_tree.cuh`, driven by `native/tests/lfm3_tree.cu` (`lfm3-tree`; `build/tree-v1`, hashes in
`sha256.txt`). Plan: `../../journal/lfm3-single-stream_synth.md`. Thor, LFM3-Omni-2B, no cuBLAS.

## What it does
Each pass carries, for every active request, its pending token plus a tree of guessed continuations. All rows are
verified in one weight read. Each request commits the longest path the model itself produces, plus the model's next
token. The guesses here are free: n-gram trees over the prompt and the output so far (prompt lookup).

**Exact by construction.** Each row reads its keys in plain decode's order, with the same 256-key splits: the
committed cache first, then its ancestors' keys from the pass scratch. Its conv window is the last six committed or
ancestor inputs, in `convolution7`'s arithmetic. So its logits are bitwise what plain decode computes at that position.
Tokens are picked per row exactly as plain decode picks them (greedy, or the same counter-based draw for that step).
So the output is token-identical to plain decoding, greedy or sampled, whatever the guesses. Accepted rows' K/V and
conv inputs are committed; the rest are discarded.

## Gates

| Gate | Result |
|---|---|
| Single request, 9 workloads × {greedy, T 0.528} × {no drafts, n-gram 32 rows, n-gram 64 rows}, vs plain batched decode | **54/54 token-identical** |
| Nine requests in one decoder, 64 rows (rows shared by the scheduler), greedy and sampled, vs plain | **9/9 identical in both** |
| compute-sanitizer memcheck, two requests, sampled, n-gram trees | 0 errors (`evidence/san-memcheck.log`) |
| Coherency (`coherency.md`, read in full) | coherent; no engine-attributable fault (below) |

**The coherency read.** The flaws found are the model's own:
- **Code edit:** the type-hint instruction is ignored. The text is identical to the accepted serial engine's.
- **Translation:** "tricot" (knitting) for weaving. Identical to the serial engine's.
- **Answer from a document:** the quoting is messy.
- **Math:** both paths write the garbled fraction "350 1/33" at the same point. The serial engine then answers
  10.6 min (wrong); this path, which diverges there by a rounding-level difference, answers 31.8 min (right).

## Speed, single request (decode loop, after prefill, one run each)

| workload | tokens | plain tok/s | n-gram, 32 rows: tok/pass, tok/s | gain |
|---|---|---|---|---|
| code edit | 530 | 61 | 4.90, **252** | **4.1x** |
| answer from a document | 248 | 61 | 2.55, 130 | 2.1x |
| code | 590 | 61 | 1.52, 83 | 1.4x |
| summary | 267 | 60 | 1.64, 83 | 1.4x |
| math | 400 | 61 | 1.53, 85 | 1.4x |
| JSON extraction | 73 | 62 | 1.47, 84 | 1.4x |
| explanation | 131 | 59 | 1.12, 65 | 1.1x |
| translation | 125 | 61 | 1.03, 60 | 1.0x |
| open chat | 871 | 49 | 1.21, 52 | 1.1x |

Sampled (T 0.528) gains are a little lower (code edit 199 tok/s, document answer 106). 64 rows gives slightly more
tokens per pass but a slower pass. These match the offline simulation (`tools/lfm3_acceptance.py`) within a few
percent.

## Found: speculation under load needs a better row policy
With 9 requests and 64 rows, tokens per pass rise from 5.42 to 6.43, but the pass grows from ~9 rows to 64 and slows.
Net: greedy 321 vs 324 tok/s, sampled 277 vs 311. The scheduler gives every request an equal share of rows. It should
give guess rows only to requests whose guesses are being accepted, and keep passes at ≤ 32 rows under load. Not done.

## Not done
The adaptive row policy; lookahead drafts (the verifier's own outputs); the trained draft head; lossless weights; the
server path.
