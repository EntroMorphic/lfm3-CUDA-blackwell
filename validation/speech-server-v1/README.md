# #2: speech in liquid-lfm3-server (POST /speech)

The single server executable now speaks as well as writes. Tripp chose (2026-11-09) a native `/speech` endpoint in the
server's own style, one JSON reply with the WAV base64-encoded, and both validated modes: TTS (with the sentence
split) and interleaved.

## What it is
- **Engine** (`native/src/lfm3_worker.cu`, op `speech`):
  - **Runner and code:** the speech loop runs on its own runner (capacity 4096 per chunk; it shares the loaded weights
    and is built on first use, both parts or neither). It uses the same code as `lfm3-speech`: the chunk loop and the
    WAV writer live in the host header `native/include/lfm/speech_run.hpp` (moved out of the driver), and the CUDA
    session in `native/src/lfm3_speech.cuh`.
  - **Support:** the worker decides at startup whether speech can run. It needs a native-GEMM build, an Omni
    checkpoint with its codec, and the special ids that `lfm/audio_modes.hpp` fixes; the server refuses `/speech`
    otherwise.
  - **Cancellation:** shutdown and the engine timeout are checked before every step.
  - **Step budget:** each request may take at most 6144 decode steps across all its chunks. A chunk whose cap could use
    up what remains gets the remainder as its cap, finalizes the codec stream and is the last one run.
    - **`truncated`** means exactly that the budget stopped the text: chunks were skipped, or the chunk whose cap the
      budget lowered used all of it without ending by EOS. A chunk cut by its own `max_new_tokens` shows only in its
      `ended_by_eos`.
    - **In effect** a request is cut once its steps reach 6144 minus `max_new_tokens`. A per-chunk `max_new_tokens` sized
      to a sentence (512 is ample for 49 words) leaves almost all of the budget for text.
    - **Host test.** `tests/test_speech_run.py` checks the loop against a scripted fake session: seeds, codec continue
      and finalize, the step bound and the flag, in named cases and 20,000 random ones under ASan+UBSan. The random
      cases are also checked against an independently written schedule of which chunks run with which caps, and,
      far from the budget, against the same request with no budget. All seven bugs planted in the loop are caught.
- **HTTP** (`native/include/lfm/lfm3_http.hpp`, `Service::speech`):
  - Same authentication, queue, deadlines and error codes as `/generate`, and no session.
  - **Refusals (400):** bad fields, mode, `max_new_tokens` (1..4095, and prompt + it within 4096 per chunk), `seed`
    (0..2^31-1), `split` (audio only; default true), more than 512 chunks, more than 131,072 prompt tokens, or an engine
    request frame over 4 MiB (a backstop; the token cap keeps frames under about 0.9 MiB).
  - **Unsupported speech:** 501 `speech_unsupported`, so the server keeps serving text. This covers a model or build
    without speech, and an `--engine-timeout` under 300 s, which the step budget was not sized for.
  - The chat is rendered and split by the native frontend (`lfm/speech_text.hpp`).
  - **Interleaved text** is decoded from the emitted slots without `<|audio_out|>`, special tokens skipped, as
    `tools/lfm3_speech_cases.py` decoded the validated runs.
  - **Reply size:** speech replies may be 64 MiB (17.5 minutes of audio, far above what the step budget allows); text
    stays at 4 MiB.
  - Seconds are written as Python's `json.dumps` writes them (`py_float`, equal to `float.__repr__`).
- **Worst-case time.** The step budget bounds generation and the two caps bound prefill.
  - **Measured rates** (`budget/measurements.json`, locked clocks): 0.03 s prefill per short chunk, and 5,500 prefill
    tokens/s on 3,920-token prompts. Frames/s: 41 for short chunks; 38.4-38.5 for long ones, the slowest observed (a
    2,815-token prompt reaching position 3,085, and the 97-word paragraph unsplit, 1,182 frames).
  - **Bound:** at the slower rates assumed (34 frames/s, 4,700 tokens/s, 0.035 s per chunk), the worst case is about
    227 s. That is under 240 s, 80% of the default engine timeout, which would otherwise fail the service closed. With
    unlocked clocks (about 15% slower, ~32.6 frames/s) it would be about 234 s, still under.
  - **The worst case run** (`speech-check.json`, `worst_case_budget`): 512 chunks of ~235-token prompts at 12 steps
    each used all 6,144 steps and took 158 s in total (19 s of it prefill). It is not `truncated`: the budget ran out
    exactly at the last chunk, and every chunk was cut only by its own 12-step cap.
- **Reply:** `{"mode", "text", "audio_wav_base64", "sample_rate", "audio_seconds", "frames", "chunks": [{"text"
  (split only), "frames", "ended_by_eos"}], "seed", "split", "cancelled", "truncated", "prefill_seconds",
  "generate_seconds"}`. Fewer chunks than sentences only with `truncated`.
- **Limits, by design:**
  - A speech request is cancelled only by shutdown or the engine timeout. There is no per-request cancel, because
    speech has no session; a request past its deadline (504) runs to completion, as text generation does.
  - The speech runner's memory sits outside the session ledger. As with text, an allocation failure fails the server
    closed.
  - At the budget's limit (6,144 frames, 492 s of audio) the float waveform is about 47 MB and its base64 about
    31.5 MB. A reply passes through several such transient copies (about ten).

## Rules, set before the work, and results (final builds: server-speech-v4, speech-v11)
`sources.sha256` holds the source files both were built from (from Thor; identical to this commit's files);
`binaries.sha256` holds the binaries.
1. **The `lfm3-speech` driver is unchanged by the refactor** on every valid output. speech-v11's regression check
   gives 36 of 36 files identical, and all 9 refusals are clean. The 10 WAV files were checked byte for byte at
   speech-v8. Two deliberate changes:
   - a request file named as one of its own outputs is refused (the driver would otherwise delete it);
   - a NaN or infinite sample now fails the run instead of being written as full scale.

   **PASS.**
2. **The server's WAV equals `lfm3-speech`'s for the same request** (`tools/check_lfm3_speech_server.py`;
   `speech-check.json`, which records the server's sha256). It passes on all five validated runs, with every chunk
   ended by EOS:
   - lighthouse split, seed 1, 5 chunks, 38.2 s;
   - bread split, seed 1;
   - fox unsplit, seed 1;
   - il-study interleaved, seed 1 (its text also equals the validated native text);
   - the Phase B abbreviations under the default rule, seed 1.

   The same request twice gives the same bytes. **PASS.**
3. **Text is unaffected.**
   - A pinned greedy `/generate` gives the same ids before and after the speech requests.
   - The full text-server GPU gate passes on server-speech-v4 (`tools/check_lfm3_native_server.py`; `text-gate.json`,
     whose binary hashes match `binaries.sha256`): serving, cancellation, batching, extreme sampling, the byte-for-byte
     differential against `tools/lfm3_server.py`, and shutdown.
   - On the host, the HTTP, framing, speech and chunk-loop suites pass: 238 of 240 tests over the plain, ASan+UBSan
     and TSan builds, with 2 skipped as before.

   **PASS.**
4. **Refusals are clean and the reply shape is exact** (`tests/test_lfm3_http_speech.py`; a fake engine that refuses
   whatever WorkerCore refuses; all three builds; 82 tests). **PASS.** Covered:
   - the prompts sent are exactly the splitter's sentences, each rendered alone;
   - 17 malformed or oversized requests are refused with 400 and the server keeps serving;
   - the red-team's request that failed the service closed now gets a 400;
   - truncated replies;
   - unsupported speech, or an engine timeout under 300 s, gets a 501 while text is still served;
   - shutdown during speech; 401 without the token;
   - every reply byte-identical to Python's `json.dumps`;
   - `py_float` against `float.__repr__`.

   **Worst cases on Thor:**
   - the budget case above: 158 s;
   - 512 sentences each allowed 4,000 steps: cut early by the budget, `truncated`, 49 s.

   The server kept serving after both.
5. **Red-team, three passes.**
   - **The first pass** found:
     - two ways a request could fail the server closed: a model or build without speech, and an engine frame over 4 MiB;
     - an unmeasured timeout claim;
     - a fake engine looser than the real one;
     - a reply that could be half-written before an error;
     - a half-built runner;
     - a NaN sample written as full scale;
     - two wrong figures in the docs.

     All are fixed as described above.
   - **The second pass** found evidence gaps:
     - the "worst case" run did not use the budget;
     - the truncation logic had no host test;
     - `truncated` was inferred from `!ended_by_eos`;
     - the text-gate record mixed two runs;
     - several counts and figures were wrong;
     - three minor items.

     All are fixed. The true worst case is above, the loop has its host check, `truncated` is exact, and the records
     come from one run with hashes.
   - **The third pass** found no blocking bugs; it found six smaller items:
     - the frames/s rate was taken from the fastest case (long chunks are now measured: 38.4-38.5);
     - no source hashes in the repo (`sources.sha256` now);
     - a `truncated` quirk at exactly remaining = cap (the budget changed nothing there; fixed);
     - a partly circular property check (now with an independent schedule and a no-budget comparison);
     - an orphaned comment;
     - the "unchanged driver" wording (the two deliberate changes are now stated).

     All are fixed.

`binaries.sha256`: the server and worker (`tools/thor/build_native.sh` with `-DLIQUID_LFM3_NATIVE_GEMM=ON`; no
cuBLAS) and speech-v11's `lfm3-speech`. Thor's clocks have been locked (MAXN) since 2026-11-09. Speed is not part of
this round.
