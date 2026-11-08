# #2 step 3, part 2: the sampler (B2) and the mode machine (A4): PASS

Both are host code with no GPU, so the tests run on any machine with g++ (`tests/test_audio_sampling.py`,
`tests/test_audio_modes.py`). The fixtures come from Thor: real logits from the depth anchors, and HF's own control
flow on forced inputs. Quality bar: `journal/lfm3-tts-quality-bar.md`, B2 and A4.

## B2: the frame sampler (`native/include/lfm/audio_sampling.hpp`)

HF's `sample_audio_frame`:
- the codebooks are revealed in a random order, 4 per pass over 3 passes;
- each pass scores only the codebooks it reveals;
- each code is drawn from softmax(penalize(logits) / 0.4). The penalty is 1.2, applied once to every code that
  codebook emitted in the last 64 accepted frames: positive logits are divided, negative ones multiplied.

The draws reuse the text engine's `lfm::TextSampler` (one 64-bit Mersenne Twister draw per code; its own tests cover
its arithmetic). The order is a Fisher–Yates shuffle with unbiased bounded draws from a second engine, seeded from
the same seed. PyTorch's Philox stream cannot be reproduced, so the comparison with HF is distributional.

**Oracle.** An independent float64 computation of HF's `RepetitionPenaltyLogitsProcessor` followed by
`TemperatureLogitsWarper`. It is run on six real depth-decoder logit vectors: the FP64 anchor for tts-museum-male,
across all three passes, with 0, 10 and 64 history codes (`tests/data_audio_sampling_fixture.npz`).

| Check | Result |
|---|---|
| Exact distribution vs analytic, 6 positions | max abs difference 1.4e-17 to 5.6e-17 |
| 60,000 draws per position, χ² against analytic | p = 0.31, 0.20, 0.16, 0.71, 0.87, 0.26 (34–174 bins) |
| The penalty matters at these positions | distribution moves by more than 1e-3 with and without history |
| Reveal order, 120,000 frames: position × codebook (144 cells) | χ² p = 0.49 |
| Each of the 66 pairs shares a pass with probability 3/11 | smallest Bonferroni-corrected p = 0.31 |
| Frame protocol, 20,000 frames | every pass picks 4 still-masked codebooks; masks change only by reveal; 3 passes per frame |
| Per-codebook frame marginals | χ² per codebook, Bonferroni over 12, all pass |
| History window | keeps the last 64 accepted frames; an EOS frame is refused |
| Seeds | the same seed reproduces; different seeds differ |

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
- **Last step:** it is not decoded.

**Reference** (`tools/lfm3_mode_reference.py` → `tests/data_mode_reference.json`): HF's real `_generate` on a
meta-device model, with only the neural calls stubbed (prefill and decode forwards, `lm_head`, `_sample`,
`sample_audio_frame`) to return forced tokens and frames. Everything else is HF's code.

There are 22 scenarios:
- 16 synthetic ones, one per branch:
  - text only;
  - text → audio, and text → interleaved;
  - an EOS frame in the middle of a block, then `<|audio_start|>`;
  - `<|im_end|>` inside the text of a block, and `<|text_end|>` outside one;
  - `max_new_tokens` cut in audio, interleaved and text;
  - the prompt already ending in a marker (`<|audio_start|>`, `<|audio_out|>`, `<|mixed_start|>`);
  - `return_audio_eos`;
  - markers inside a block;
  - the suppress flags;
  - EOS on the first frame;
  - two interleaved blocks in one turn.
- The six traced cases. For these the harness reproduced the real run's emitted slots and frames exactly
  (asserted), so the reference is HF's behaviour.

**Native equals HF bit for bit in all 22.** That covers the marker, every step (what was sampled, what was fed back,
whether a decode ran, the suppression set), the emitted slots and frames, and the unconsumed inputs. HF refuses
`max_new_tokens = 0`, and so does native.

**The test is not vacuous.** Three planted bugs are each caught:

| Planted bug | Scenarios that catch it |
|---|---|
| EOS only when all 12 codebooks are EOS | 15 of 22 |
| Cadence still counted after `<|text_end|>` | 3 |
| Text-only requests may sample `<|audio_start|>` | 2 |

**Settled along the way.** tts-trace-v1's cadence starts with `T7` because the trace counted from before the marker
`generate` appends. The 7 is the marker plus 6 text slots, read now from the code, not inferred.

## Open

- Step 4, end to end: the native LM's hidden states, then depth decoder, sampler, mode machine and codec in the
  single binary. Then C1–C4 against HF over a fixed prompt set and several seeds, with Tripp's listening last.
- The replay of past assistant speech (`audio_out_codes` in the prompt) is outside these scenarios. It needs the
  codec encoder.
