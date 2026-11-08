# #2 step 3, part 2: the sampler (B2) and the mode machine (A4): PASS

Both are host code with no GPU, so the tests run on any machine with g++ (`tests/test_audio_sampling.py`,
`tests/test_audio_modes.py`). The fixtures come from Thor: real logits from the depth anchors, and HF's own control
flow on forced inputs. Quality bar: `journal/lfm3-tts-quality-bar.md`, B2 and A4.

**Record:** `report.json`, written by `tools/lfm3_audio_control_report.py`: every test's outcome, the figures below,
and the planted-bug runs. The figures are from fixed seeds. The step-3 red-team changed how the seeds are mixed, so
the χ² p-values differ from this README's first version; the exact differences do not.

## B2: the frame sampler (`native/include/lfm/audio_sampling.hpp`)

HF's `sample_audio_frame`:
- the codebooks are revealed in a random order, 4 per pass over 3 passes;
- each pass scores only the codebooks it reveals;
- each code is drawn from softmax(penalize(logits) / 0.4). The penalty is 1.2, applied once to every code that
  codebook emitted in the last 64 accepted frames: positive logits are divided, negative ones multiplied.

The draws reuse the text engine's `lfm::TextSampler` (one 64-bit Mersenne Twister draw per code; its own tests cover
its arithmetic). The order is a Fisher–Yates shuffle with unbiased bounded draws from a second engine. The two
engines are seeded from two splitmix64 mixes of the seed, so neither shares a stream with a text sampler given the
same seed. As HF's `_build_warpers`, temperature 0 is greedy and `top_k` keeps the k best penalized codes; HF's
shipped speech setting is 0.4 with no top-k. PyTorch's Philox stream cannot be reproduced, so the comparison with HF
is distributional.

**Oracle.** An independent float64 computation of HF's `RepetitionPenaltyLogitsProcessor` followed by
`TemperatureLogitsWarper`. It is run on six real depth-decoder logit vectors: the FP64 anchor for tts-museum-male,
across all three passes, with 0, 10 and 64 history codes (`tests/data_audio_sampling_fixture.npz`).

| Check | Result |
|---|---|
| Exact distribution vs analytic, 6 positions | max abs difference 1.4e-17 to 5.6e-17 |
| 60,000 draws per position, χ² against analytic | p = 0.65, 0.90, 0.35, 0.43, 0.92, 0.15 (34–174 bins) |
| The penalty matters at these positions | distribution moves by more than 1e-3 with and without history |
| Reveal order, 120,000 frames: position × codebook (144 cells) | χ² p = 0.93 |
| Each of the 66 pairs shares a pass with probability 3/11 | smallest Bonferroni-corrected p = 1.0 |
| Frame protocol, 20,000 frames | every pass picks 4 still-masked codebooks; masks change only by reveal; 3 passes per frame; every pass is given exactly the codes the earlier passes sampled |
| Per-codebook frame marginals, each codebook with its own logits and 64-frame history | χ² per codebook against its own analytic distribution, Bonferroni over 12, all pass |
| A history with repeated codes | one penalty per code (set), not one per occurrence |
| top_k 1, 5, 50 | exact distribution and draws match; nothing outside the top k is drawn |
| Temperature 0 | the argmax of the penalized logits, every draw, and `distribution` is that point mass (top_k ignored, as HF) |
| top_k above 2017 | keeps every code, as HF clamps it |
| Same seed as a text sampler | the draws agree only at the independent rate Σp² |
| History window | keeps the last 64 accepted frames; an EOS frame is refused |
| Seeds | the same seed reproduces; different seeds differ |

Two of the first tests were too weak. The red-team found that the frame test gave every codebook the same logits
and no history and ignored the codes passed to each pass. A bug writing codes to the wrong codebook passed it. It
now gives each codebook its own logits and history, and checks the codes each pass is given.
The first version of the pair test was vacuous. Every permutation puts exactly 18 pairs in a shared pass, so the
summed count could not fail (p = 1.000). It now tests each pair.

**Cost.** About 0.2 ms of host time per draw: `TextSampler` sorts all 2,017 candidates. That is about 2.4 ms per
frame for 12 draws, against an 80 ms frame.

## A4: the mode machine (`native/include/lfm/audio_modes.hpp`)

HF's `_generate` control flow, batch 1:
- **Markers:** the requested mode appends its marker to the prompt, unless the prompt already ends with one; with no
  mode, the prompt's last token chooses.
- **Audio:** a frame with any EOS code is rewritten to all-EOS, fed back but not emitted, and switches to text.
- **Interleaved:** 6 text slots and 12 frames alternate. `<|text_end|>` starts the audio tail; inside a block the
  markers are ordinary text.
- **Text:** `<|im_end|>` ends the turn, and so does `<|text_end|>` outside a block.
- **Suppression:** text steps never sample the placeholder tokens, and text-only requests never sample the markers.
- **Replayed speech:** with no mode requested, a prompt ending on a replayed `<|audio_out|>` frame that holds an
  EOS code resumes text (`begin` takes the last replayed frame and decides).
- **Last step:** it is not decoded, unless the caller supplied the cache (`caller_cache`); then it is.
- **History:** the caller's `FrameSampler` accepts every frame whose `Outcome::eos` is false, for the whole request.

**Reference** (`tools/lfm3_mode_reference.py` → `tests/data_mode_reference.json`): HF's real `_generate` on a
meta-device model, with only the neural calls stubbed (prefill and decode forwards, `lm_head`, `_sample`,
`sample_audio_frame`) to return forced tokens and frames. Everything else is HF's code.

There are 28 scenarios:
- 22 synthetic ones, one per branch:
  - text only;
  - text → audio, and text → interleaved;
  - an EOS frame in the middle of a block, then `<|audio_start|>`;
  - `<|im_end|>` inside the text of a block, and `<|text_end|>` outside one;
  - `max_new_tokens` cut in audio, interleaved and text;
  - the prompt already ending in a marker (`<|audio_start|>`, `<|mixed_start|>`);
  - `return_audio_eos`;
  - markers inside a block;
  - the suppress flags;
  - EOS on the first frame;
  - two interleaved blocks in one turn;
  - replayed speech in the prompt (`audio_out_codes`): a terminal EOS frame resuming text, a speech frame
    continuing audio, and a requested mode overriding the EOS frame;
  - the caller's cache (`past_key_values`) in text, audio and interleaved;
  - a repetition history across audio → text → audio and past the 64-frame window.
- The six traced cases. For these the harness reproduced the real run's emitted slots and frames exactly
  (asserted), so the reference is HF's behaviour.

**Native equals HF bit for bit in all 28.** That covers the marker, every step (what was sampled, what was fed back,
whether a decode ran, the suppression set, and the repetition history HF hands each speech step, as a length and a
hash), the emitted slots and frames, and the unconsumed inputs. Native's history is a `FrameSampler` fed by the
rule integration must follow: accept every frame whose `Outcome::eos` is false. HF refuses
`max_new_tokens = 0`, and so does native. HF also refuses an `<|audio_out|>` placeholder without its replayed
frame, and so does `begin`. (The first version of this reference had a scenario ending on `<|audio_out|>` with no
frame. Its stubbed prefill skipped HF's check, so the scenario showed behaviour HF never reaches. The red-team found
it, and the harness now asserts that such a prompt replays its frame.)

## The tests are not vacuous: 17 planted bugs, all caught (`report.json`)

Each bug is planted in a copy of the headers, and the test file is rerun against it. Each file has 32 tests, and
both pass 32 of 32 clean.

| Planted bug | Tests that fail |
|---|---|
| Sampler: codes written to the reveal position, scored from the first pick's logits | 1 (frame marginals) |
| Sampler: passes scored without the codes revealed so far | 1 (frame protocol) |
| Sampler: Sattolo's shuffle (cyclic permutations only) for the order | 1 (order) |
| Sampler: code engine on the seed a text sampler would use | 1 (streams) |
| Sampler: temperature 0 not greedy | 6 |
| Sampler: top_k ignored | 6 |
| Sampler: greedy `distribution` reports a temperature-1 softmax | 6 |
| Sampler: top_k above 2017 refused instead of clamped | 1 |
| Sampler: the window keeps 63 frames | 2 |
| Modes: EOS only when all 12 codebooks are EOS | 18 |
| Modes: cadence still counted after `<|text_end|>` | 3 |
| Modes: text-only requests may sample `<|audio_start|>` | 3 |
| Modes: a replayed EOS frame does not resume text | 2 |
| Modes: a replayed EOS frame resumes text even when the prompt ends elsewhere | 1 |
| Modes: `<|audio_out|>` accepted without its replayed frame | 1 |
| Modes: the caller's cache ignored | 3 |
| Modes: EOS frames reported as accepted speech | 18 |

The first plant is the step-3 red-team's: the first version of the frame test passed it.

**Settled along the way.** tts-trace-v1's cadence starts with `T7` because the trace counted from before the marker
`generate` appends. The 7 is the marker plus 6 text slots, read now from the code, not inferred.

## Open

- Step 4, end to end: the native LM's hidden states, then depth decoder, sampler, mode machine and codec in the
  single binary. Then C1–C4 against HF over a fixed prompt set and several seeds, with Tripp's listening last.
- Replaying past assistant speech (`audio_out_codes` in the prompt): its mode rule is covered here. Its embedding
  (A3) comes with step 4.
- HF keeps every tie at the k-th score under top-k; `TextSampler` keeps exactly k (lowest codes first). This only
  matters with top-k set, which HF's shipped speech setting does not do.
