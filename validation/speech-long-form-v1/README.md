# #2 step 4: long-form TTS, the failure's mechanism and its fix

The C1 round (`../speech-e2e-v1`) found the native engine worse than HF on one case, the 97-word lighthouse paragraph
(`tts-long-male`): over 15 seeds, native WER 0.418 against HF BF16's 0.163 (p = 0.019), from runs that skip clauses or
repeat phrases (up to 1,182 frames; every run still ended on the model's own EOS, none at the slot cap). Every
per-step component had matched or beaten HF's fidelity (`../speech-forced-v1`). The tables: `report.md`
(`tools/lfm3_speech_long_form_report.py validation`; it refuses a row without exactly the expected seeds).
Section 0 there holds the C1 escalations. On the fresh seeds 4-15 alone (seeds 1-3 raised the flag, so 1-15 leans
toward confirming it) tts-long-male is 0.301 against 0.164, p = 0.053: the same direction, weaker.

Corrected in the step-4 red-team (2026-11-09): several sentences below said more than the data does; each is now
worded to what the tables show, and two claims that had no committed data now do (`alignment/mass_by_layer_type.json`,
`tools/thor/speech_regression_check.sh`).

## 1. The FP32 model loops too (report section 1)
HF in FP32 (TF32 off), the same 15 seeds: WER 0.433, 6 runs above 0.3, up to 1,234 frames. Native against it
directly: -0.015 (95% CI -0.36 to +0.31, p = 0.95). The CI is wide (wider than native's gap to HF BF16), so this shows
no detectable difference from the FP32 model, not equality. Making half of HF FP32 moves the mean toward it (BF16 LM
with FP32 depth 0.353, FP32 LM with BF16 depth 0.336), but neither half alone is significantly worse than HF BF16
(p = 0.078 and 0.24). On this paragraph HF's full-BF16 rounding is the outlier, not native.

## 2. The mechanism: the sliding window loses the text (`alignment/`)
The backbone's 11 attention layers are 6 full-attention and 5 sliding-attention with a window of 256 keys (layers
4, 8, 12, 16, 20). In TTS mode the text is in the prompt and each speech frame is one more position, so the window's
trailing edge moves through the text about 4x faster than the speech does.
`tools/lfm3_speech_alignment.py` replays seven native runs through HF FP32 eager. Three are used to choose the
alignment heads: the three with the lowest WER (s9 0.031, s14 0.093, s2 0.134; low, not error-free: s14's first
error is at word 37); the other four are 0.258 to 1.96. On those three, the heads whose attention centroid follows the
text most monotonically are full-attention layer 14 head 4 (Spearman 0.94) and layer 10's heads.
- The window's trailing edge passes the text's first token at frame 133 and its last at frame 248 (every run).
- The sliding layers' attention mass on the text, averaged over their heads, is 0.16-0.19 for the first ~175
  frames, falls as the edge passes, and is exactly 0 from frame 250 in all seven runs
  (`alignment/mass_by_layer_type.json`). From there only the full-attention layers see the text, about 3% of their
  attention on average.
- In `alignment/alignment.png` every run tracks the text in order until the edge reaches the word being spoken,
  about frame 180 (14.5 s), word ~41; after that the lower-WER runs stay on a line and the others jump ahead or back.
The first errors of the window-256 configurations cluster at words 40-48 and 62-65 (report, onset column; HF BF16 also
at 72 and 80).

## 3. Position, not content (report section 3)
Native, 15 seeds each: the sentence at the onset alone, perfect; the first three sentences (50 words: the edge never
reaches the reading position), perfect; the paragraph reordered, onsets again at word 41 on different words; an
unrelated 86-word paragraph (bread), onsets at words 40-43 (and 75).

## 4. Widening the window does not fix it (report section 2)
FP32 model with the window overridden at inference: 512, WER 0.967; 4096, WER 0.855 (14 of 15 runs above 0.3 each).
Both are worse than the trained 256. Their onsets are identical seed for seed (word 62-67 in 13 of 15; none at 40-48),
so the two are one observation, not two: with a wider window the reading holds past word 41 and fails later, and
worse. The sliding layers were trained with 256 keys; more keys than that is out of distribution for them.

## 5. The fix: one request per sentence (report section 4)
`tools/lfm3_speech_chunked_hf.py` splits the text at sentence ends (a sentence under 6 words joins the next) and makes
each sentence its own request, exactly as a single TTS request is built; native: `lfm3-speech` with `"chunks"` (one
prompt per sentence, chunk k seeded SEED * 1000 + k, one codec stream across the chunks).
- Native split, both paragraphs, 15 seeds each: **WER 0.000 on all 30 runs**; all 150 chunks end as the HF tool
  defines it (last sampled frame EOS, last slot `<|im_end|>`; re-checked in the red-team, one EOS frame per chunk);
  3.42x real time end to end (prefills included). HF split, FP32 and BF16: 0.000 on all 60.
- Continuing one conversation instead (each sentence a new user turn after the previous speech) breaks the model:
  WER 0.835 on the first seed; not pursued. (`asr/hf-fp32-split-smoke-fresh-and-continue.json`: its two entries are
  not labelled; the continue run is the one at 0.835, the fresh one 0.000.)
- Unsplit requests are unchanged: bit-identical raw frames, emitted frames and waveform across speech-v1 to v4
  (`tools/thor/speech_regression_check.sh`).
- Seams: the pause at a sentence join is a median 0.36 s (p90 0.46 s) against 0.24 s (p90 0.32 s) at the model's own
  sentence ends unsplit (seeds 1-3, measured with a scratch script; not regenerated by a committed tool).
- **Listening (C3): PASS, Tripp, 2026-11-08** (`outputs/listening-split-v1`, not in the repo). Seeds 1-3, blind A/B,
  native split against HF BF16 unsplit; bread paragraph seeds 1-3, native split alone. Verdict: native split good
  on every seed and both paragraphs, no seams heard; HF unsplit murmurs on "another" and a couple of other words
  in seeds 2 and 3. On a re-listen of seed 1 in the C3 set (2026-11-09) Tripp also heard HF murmur, where Whisper
  finds a loop in the last sentence ("visitors climb them tadvitas climb them..."). The ASR agrees: HF seed 2 "its
  own pattern from lutter" (word 41), seed 3 "tell when from whenever" (words 46-48), at the crossing; native split
  0.000. Blinding: the A/B order of these sets came from seeded draws, A = native in all three split pairs; they
  were blind in practice (the order was not looked up). `tools/lfm3_listening_set.py` now draws from the operating
  system's entropy.

## 6. Native splitting (speech-v3, 2026-11-09)
`lfm3-speech` takes `"messages"` with `"split": true` and splits (`native/include/lfm/speech_text.hpp`) and renders
each sentence natively.
- The splitter equals the Python one on edge cases, 5,000 fuzzed texts (45,000 more in the red-team) and whitespace on
  every code point; the tests catch 31 of 31 non-equivalent planted bugs.
- Every validation prompt's per-sentence ids equal the HF processor's (`tests/test_speech_text.py`, in lfm3-omni-dev).
- Native-split lighthouse s1 and bread s1 are bit-identical to the validated Python-built runs (raw frames, emitted
  frames, waveform), with speech-v3 and speech-v4.
- The Python prototype's short-last-sentence join had a bug (crash on two parts, wrong part overwritten on three or
  more), never reached by a validation prompt; fixed in both.

## Open
- **The split has no length cap.** A chunk ends only at `. ! ?` followed by whitespace, so text without them stays one
  long request and can fail as before: paragraphs separated only by blank lines, CJK punctuation (`。！？`), dialogue
  with the period inside the quotes (`"Stop." Then`). Abbreviations (`p.m. on`) can also split mid-sentence (the
  6-word join hides most). Decision pending (with comma splits for long sentences).
- The server: speech is not yet in `liquid-lfm3`.
- HF BF16's better unsplit odds on this paragraph remain unexplained beyond "different rounding"; with the split, it
  no longer matters.

Records: `asr/` (lfm3_asr.py outputs per configuration, rescored with compound folding: no long-form value changed),
`alignment/` (alignment.json: head ranking and tracks; mass_by_layer_type.json), `binaries.sha256`.
