# Full text-serving acceptance target

Requested scope: text preprocessing/decoding, native generation with sampling,
multi-turn sessions, batching/concurrency, and the model-card context maximum.
Existing bounded experimental acceptance is preserved, not generalized silently.

## Context source of truth

`LiquidAI/LFM3-Omni-2B` model card, both current and pinned revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`, states **32K tokens, input + output**.
Source: https://huggingface.co/LiquidAI/LFM3-Omni-2B#model-details
The local `text_config.max_position_embeddings=128000` is not the advertised
context promise. No 256K context claim is established for this checkpoint. If
the target checkpoint/model-card changes, re-evaluate explicitly.

## Work gates

1. **Bounded-memory 32768-token runtime.** Tile attention query rows, preserve
   the previously validated <=4096 schedule, enforce a combined input/output
   serving budget, validate long-position RoPE and long-row softmax. Prefill,
   final-slot decode, overflow, reset, and long prompt logits require evidence.
   Bring-up requests above4096 require `experimental_long_context=true`; the
   public launcher is not yet raised to32K. This is not accepted32K serving yet.
2. **Tokenizer-only frontend.** Avoid importing the Omni processor/audio codec
   for text requests. Six official processor parity cases passed on Thor with
   exact UTF-8 rendering and token IDs. Integration into the entry point pending.
3. **Sampling.** Host-only sampler supports validated temperature/top-k/top-p/
   min-p/repetition penalty and seeded RNG. Host adversarial/statistical tests
   pass; native request/response and CLI integration, reference distribution
   comparisons and end-to-end reproducibility remain pending. No torch RNG parity
   claim is implied.
4. **Persistent multi-turn sessions.** Explicit cache ownership, isolation, reset,
   capacity accounting, cancellation, restart/error semantics and transcript
   parity. No reuse of unconsumed final predictions without forwarding them.
5. **Batching/concurrency and serving.** Independent sequence positions and
   convolution/KV states, bounded admissions/memory/queue, stop/cancellation,
   protocol limits and deterministic failure/recovery. A serialized subprocess
   wrapper must not be represented as GPU batching.
6. **Release acceptance.** Adversarial/unit/integration coverage, native/reference
   logits and greedy/distribution checks, request isolation, soak and resource
   measurements, hash-matched benchmark, and explicit sanitizer disposition.
   The reproduced cuBLAS initcheck tracking blocker is still open; no zero-fill,
   custom-kernel suppression or weakened numerical gate closes it.

## First implementation increment

`text_serving_limits.hpp` specifies32768 and128-query tiling. Attention score and
probability scratch at maximum capacity is256MiB combined, rather than64GiB for
two quadratic BF16 matrices. KV and other activation allocations are additional.
The new long-row softmax has32 scalar registers/thread to cover32768 columns;
its arithmetic above4096 is bring-up, not an exact ATen equivalence certificate.

`text_sampling.hpp` and `lfm3_text_frontend.py` are independently tested components;
neither is silently advertised as fully integrated native serving. Their detailed
contracts and remaining validation are in `SAMPLING.md` and `TEXT_FRONTEND.md`.
