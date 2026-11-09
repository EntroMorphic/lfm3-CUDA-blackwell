# #2 step 4: long-form TTS, the failure's mechanism and its fix

The C1 round (`../speech-e2e-v1`) found the native engine worse than HF on one case, the 97-word lighthouse paragraph
(`tts-long-male`): over 15 seeds, native WER 0.418 against HF BF16's 0.163 (p = 0.019), from runs that skip
clauses or loop on a phrase until the slot cap. Every per-step component had matched or beaten HF's fidelity
(`../speech-forced-v1`). The tables: `report.md` (`tools/lfm3_speech_long_form_report.py validation`).

## 1. The true model loops too (report section 1)
HF in FP32 (TF32 off), the same 15 seeds: WER 0.433, 6 runs above 0.3, up to 1,234 frames. Native against it: a
difference of -0.015 (CI -0.35 to +0.32, p = 0.95). Making either half of HF FP32 (the backbone, or the depth
decoder) is also worse than HF BF16. Only HF's full-BF16 rounding is better on this paragraph; native is faithful to
the model.

## 2. The mechanism: the sliding window loses the text (`alignment/`)
The backbone's 11 attention layers are 6 full-attention and 5 sliding-attention with a window of 256 keys (layers
4, 8, 12, 16, 20). In TTS mode the text is in the prompt and each speech frame is one more position, so the
window's trailing edge moves through the text about 4x faster than the speech does.
`tools/lfm3_speech_alignment.py` replays seven native runs (3 clean, 4 failing) through HF FP32 eager. On the clean
runs, the heads whose attention centroid follows the text most monotonically are full-attention layer 14 head 4
(Spearman 0.94) and layer 10's heads. `alignment/alignment.png`:
- every run reads the text in order until the window's trailing edge passes the word being spoken, about frame
  180 (14.5 s), word ~41;
- by frame ~250 the sliding layers' attention on the text is zero; only the full-attention layers keep the place;
- after the crossing the clean runs stay on a line; the failing runs jump ahead or back (loops, a sawtooth).
The first errors cluster at words 40-48 and 62-65 in every configuration (report, onset column).

## 3. Position, not content (report section 3)
Native, 15 seeds each: the sentence at the onset alone, perfect; the first three sentences (50 words: the edge never
reaches the reading position), perfect; the paragraph reordered, onsets again at word 41 on different words; an
unrelated 86-word paragraph (bread), onsets at words 40-43.

## 4. Widening the window does not fix it (report section 2)
FP32 model with the window overridden at inference: 512, WER 0.967; 4096, WER 0.855 (14 of 15 runs above 0.3
each). The sliding layers were trained with 256 keys; showing them more breaks them from the start.

## 5. The fix: one request per sentence (report section 4)
`tools/lfm3_speech_chunked_hf.py` splits the text at sentence ends (a sentence under 6 words joins the next) and
makes each sentence its own request, exactly as a single TTS request is built; native: `lfm3-speech` with
`"chunks"` (one prompt per sentence, chunk k seeded SEED * 1000 + k, one codec stream across the chunks).
- Native split, both paragraphs, 15 seeds each: **WER 0.000 on all 30 runs**, every chunk ended by its own EOS,
  3.42x real time end to end (prefills included). HF split, FP32 and BF16: 0.000 on all 60.
- Continuing one conversation instead (each sentence a new user turn after the previous speech) breaks the model:
  WER 0.835 on the first seed, chunks rushed (27 frames for 22 words); not pursued
  (`asr/hf-fp32-split-smoke-fresh-and-continue.json`).
- Unsplit requests are unchanged: speech-v2's lfm3-speech is bitwise identical to speech-v1's on three regression
  requests (raw frames, emitted frames, waveform).
- Seams: the pause at a sentence join is a median 0.36 s (p90 0.46 s) against 0.24 s (p90 0.32 s) at the model's
  own sentence ends unsplit.
- **Listening (C3): PASS, Tripp, 2026-11-08** (`outputs/listening-split-v1`, not in the repo). Seeds 1-3, blind A/B,
  native split against HF BF16 unsplit; bread paragraph seeds 1-3, native split alone. Verdict: native split good
  on every seed and both paragraphs, no seams heard; HF unsplit murmurs on "another" and a couple of other words
  in seeds 2 and 3. On a re-listen of seed 1 in the C3 set (2026-11-09) Tripp also heard HF murmur, where Whisper
  finds a loop in the last sentence ("visitors climb them tadvitas climb them..."). The ASR agrees: HF seed 2 "its own pattern from lutter" (word 41), seed 3 "tell when from
  whenever" (words 46-48), at the crossing; native split 0.000.
  Blinding caveat: the order was drawn with one fixed seed per single-case set, so A was native in all three
  pairs. Fixed in tools/lfm3_listening_set.py (the draw now depends on the seed); the verdict was per pair.

## Open
- A single sentence over ~40 words could still reach the crossing: the splitter should also break at commas past a
  length limit. Not yet done.
- ~~The split is built in Python~~ **Native since speech-v3 (2026-11-09):** `lfm3-speech` takes `"messages"` with
  `"split": true` and splits (`native/include/lfm/speech_text.hpp`) and renders each sentence natively.
  - The splitter equals the Python one on edge cases, 5,000 fuzzed texts, and whitespace on every code point.
  - Every validation prompt's per-sentence ids equal the HF processor's (`tests/test_speech_text.py`, run in
    lfm3-omni-dev).
  - Native-split lighthouse s1 and bread s1 are bit-identical to the validated Python-built runs (raw frames,
    emitted frames, waveform); so are unsplit tts-fox-female s1 and il-study s1 against speech-e2e-v1.
  - The Python prototype's short-last-sentence join had a bug (crash on two parts, wrong part overwritten on
    three or more), never reached by a validation prompt; fixed in both.
  - Still open: the server.
- HF BF16's better unsplit odds on this paragraph remain unexplained beyond "different rounding"; with the split,
  it no longer matters.

Records: `asr/` (lfm3_asr.py outputs per configuration), `alignment/` (alignment.json: head ranking and tracks),
`binaries.sha256`.
