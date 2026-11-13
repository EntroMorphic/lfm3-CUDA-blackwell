# Speech in every locale: native against HF beyond American English

> **Shelved (Tripp, 2026-11-11):** "let's keep the other languages on the shelf for now. We'll come back to those."
> These rules are kept as written, before any data. Nothing has run and FLEURS is not downloaded. The open questions
> stay open until the round resumes. What still applies today: the language gate (audio-assurance-v1 rule 10)
> keeps non-English TTS from being normalized, and the splitter gap for unspaced scripts is recorded.

Tripp (2026-11-11): "We may do the images and other languages at the same time, splitting the Thor's resources to do
so." Every speech result so far is American English: the TTS and interleaved sets (speech-e2e-v1, speech-split-v1),
LibriSpeech for audio in (audio-in-v1), and every assurance run (audio-assurance-v1). The model card lists 19 speech
locales. This round checks the native engine against HF in all of them, with the same bars, before any run.

Written 2026-11-11 while Thor ran audio-assurance-v1's sanitizers. Nothing below has been run.

## The locales (model card, `models/LFM3-Omni-2B/README.md`)
- "Speech languages and locales": Arabic, Danish, German, Greek, British English, American English, Finnish, French,
  Hebrew, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Swedish, Turkish, Chinese (19 locales, 18
  languages).
- Voices: the TTS prompt is `Perform TTS in <language> using a <gender> voice.` with `female` or `male`; for English,
  `American English` or `British English`. There are no named voices, so 19 x 2 = 38 locale-voices.
- ASR: `Perform ASR in <language>.`, the language's English name (English for both English locales).
- Interleaved: `Respond with interleaved text and audio.` names no language; the reply's language follows the user.
- The model card scores multilingual ASR on FLEURS, with CER for Chinese, Japanese and Korean and WER for the rest.

## Found before writing these rules (from the code, not from runs)
- **The sentence splitter is built for English** (`lfm/speech_text.hpp`, equal to `tools/lfm3_speech_chunked_hf.py`
  `sentences()`). It cuts only after `.` `!` `?` followed by whitespace, and counts words by whitespace. A Chinese or
  Japanese paragraph is one chunk of one "word": never split, never capped. Arabic splits at `.` but not at its
  question mark (U+061F). Latin and Greek text splits by sentence (Greek's question mark is `;`, which the rule does not
  cut). Pinned in `tests/test_multilingual_tools.py`, so a fix shows up as a deliberate change. Long Chinese or
  Japanese text may therefore loop, as unsplit English did (speech-long-form-v1). Rule 3 measures it; fixing it is a
  separate round.
- **The number and abbreviation normalizer is American English** (`lfm/speech_normalize.hpp`). The server applies it
  by default only when the system prompt names American or British English, and refuses `normalize: true` otherwise
  (audio-assurance-v1 rule 10). British English gets American readings (no "and" in "one hundred and five", American
  date order). Rule 7 checks the gate; the British readings are for Tripp's ear (open question 4).
- **The ASR scorer is English only** (`tools/lfm3_asr.py`: Whisper with `language="en"` and the English normalizer).
  New: `tools/lfm3_asr_multilingual.py`, which leaves English scoring unchanged.
- **No multilingual speech data is on either machine.** FLEURS test (google/fleurs, CC-BY-4.0) is needed: 7.74 GB of
  test audio for the 18 languages (243-658 MB each), plus 8 MB of transcripts (test.tsv; sizes from the
  Hugging Face API, 2026-11-11). Downloading needs Tripp's OK (open question 1).

## Data and cases (rule 0, fixed before any run)
FLEURS test, 18 configs: ar_eg, da_dk, de_de, el_gr, en_us, fi_fi, fr_fr, he_il, it_it, ja_jp, ko_kr, nl_nl, pl_pl,
pt_br, es_419, sv_se, tr_tr, cmn_hans_cn. FLEURS sentences are FLoRes sentences, n-way parallel: the same sentence id
is the same sentence, professionally translated, in every language, so every locale reads the same content.
Fetched by `tools/fetch_fleurs.sh`, which records the sha256 of every test.tsv and archive before extracting.
Not yet checked: that the archives extract to `audio/test/*.wav` at 16 kHz. `tools/fleurs_cases.py` refuses a missing
or non-16 kHz clip, so a different layout fails loudly rather than being resampled. Then `tools/fleurs_cases.py`
(seed 2026):
- **The draw:** ids present in all 18 languages whose text has no digit in any language, with 8-30 English words,
  sorted; `random.Random(2026).sample(eligible, 13)`. The first 6 are the TTS sentences, the next 5 the long-form
  paragraph, and the last 2 the interleaved prompts. Digits are left out because they are the model's known weak spot
  (journal/lfm3-tts-quality-bar.md C3); numbers are covered by rule 7 and the English sets.
- **TTS** (`tts.json`): 38 locale-voices x 6 sentences = 228 cases, seeds 1-3, `max_new_tokens` 768, each locale
  reading its own language's text (both English locales read English).
- **Long form** (`long.json`): per locale, the 5 paragraph sentences joined by spaces, female voice, seeds 1-3.
- **Interleaved** (`interleaved.json`): per language, 2 sentences as the user turn, seeds 1-3 (36 cases).
- **ASR** (`asr-cases.json`): per language, the first 100 test utterances by (id, file name), 1,800 in all. Clips are
  checked as 16 kHz mono 16-bit PCM and refused otherwise, never resampled. The reference is FLEURS's normalized
  transcription.

Sampling as in every earlier speech round: HF BF16 SDPA, audio temperature 0.4, repetition penalty 1.2,
`torch.manual_seed(seed)`; native the same seeds. Text greedy.

## Rules (set before any run)
1. **Rendering and splitting equal HF's, exactly.** On the exact subset (seed 1 of every TTS locale-voice's first
   sentence, every long-form paragraph, every interleaved first sentence, and the first 2 ASR clips of every
   language: 111 requests), lfm3-speech run from the messages produces outputs byte-identical to lfm3-speech run on
   HF's prompt ids. That covers the native renderer and, for long form, the native splitter against HF's chunks.
   Pass: every file identical (`tools/compare_speech_outputs.py`).
2. **TTS intelligibility and endings, per case** (the C1 and C2 rules of speech-e2e-v1, unchanged):
   - **C1:** Whisper large-v3 in the case's language. CER for Chinese, Japanese and Korean, WER otherwise. A case is
     flagged when native's mean error over seeds 1-3 exceeds HF's worst seed. A flagged case runs seeds 4-15 in both
     engines. It **fails** if, over seeds 1-15, native's mean error is higher and the two-sided Mann-Whitney p is
     below 0.05. A failing case is then run in HF FP32 (`lfm3_speech_hf.py float32`) as referee before any cause is
     named. The failure stays recorded either way.
   - **C2:** per locale-voice (18 runs): native ends by its own EOS on no fewer runs than HF minus 1, and native's
     mean frames are within 0.85-1.15 of HF's. A miss is flagged, escalated like C1, and fails if it persists at 15
     seeds.
   - **Heard as its language:** Whisper's language detection on every clip is reported. A locale-voice where native is
     heard in its own language on 3 or more fewer runs than HF is flagged for rule 9.
3. **Long form per locale.** Native split against HF's same split (`lfm3_speech_chunked_hf.py fresh`), C1 as in
   rule 2, per case. Chunks per paragraph and chunks ended by EOS are reported for both engines. Prediction, made
   from the code: Chinese and Japanese are one chunk in both engines and may loop in both. That would be the
   splitter's gap, not native against HF, and is recorded as the open item for a splitter round. Those two cases
   still count under C1, since both engines read the same chunks.
4. **Recognition as good as HF's** (audio-in-v1 rule 3, per language). `tools/check_lfm3_multilingual_asr.py`:
   - Per language: the corpus error rate of each engine, and native minus HF with a paired bootstrap 95% interval.
   - Overall: the mean over languages of that difference, with a stratified bootstrap. Languages are averaged, not
     pooled, because characters and words are different units.
   - Pass: the overall upper bound is at most +0.25 points.
   - A language whose upper bound is over +1.0 point is flagged and runs its full FLEURS test split. It passes if the
     upper bound there is at most +0.5 points.
   - Token-identical utterances are reported (on LibriSpeech: 2,554 of 2,620).
5. **Interleaved: the speech says the text** (C4). Per case, in the case's language, the C1 rule of rule 2 with the
   model's own text as the intended words. The reply's language (Whisper detection) is reported for both engines;
   the prompt names no language, so an English reply is the model's choice, not a failure, unless one engine does it
   and the other does not (flagged, rule 10).
6. **The server equals the engine, exactly.** `tools/check_lfm3_multilingual_server.py`, on 92 cases: seed 1 of every
   locale-voice's first sentence with the server's defaults (split on; normalize on only for English), every
   language's first interleaved sentence, and the first 2 ASR clips of every language as base64. Each reply's text,
   WAV and chunk texts must equal lfm3-speech's from the same messages, byte for byte. The same request twice must
   give the same reply, and the server must exit 0.
7. **The language gate, live** (audio-assurance-v1 rule 10). For every non-English locale (17), `normalize: true`
   gets a 400 and no audio. A sentence with digits in that language (`GATE_TEXT`) is spoken with its digits
   untouched under the defaults. For both English locales the same sentence is normalized by default: no digit is
   left in the chunk texts, which must equal lfm3-speech's with normalize on.
8. **A batch equals alone, exactly.** The TTS and interleaved part of rule 1's subset (56 requests) through
   `--concurrent 8` staggered (seed 7): byte-identical to each request alone.
9. **Listening, the final gate** (Tripp; `outputs/multilingual/`, blind A/B, keys kept in the build directory):
   - seed 1 of the first sentence of every locale-voice (38 pairs);
   - the 19 long-form paragraphs.

   For each clip: is it clean speech in the asked voice, without loops, skips or garbling, and is one clip better?
   Tripp does not need to speak the language to judge this. A pair where native is heard worse is investigated
   before the round can pass.
10. **Coherency.** I read:
    - the transcripts of every flagged case;
    - every interleaved reply's text at seed 1;
    - 30 sampled ASR utterances where the engines differ.

Speed is reported, not gated: decode loop and end to end, as before, and only from runs on a quiet Thor. A phase that
shared Thor reports no speed.

## How it runs (`tools/thor/multilingual_round.sh TAG SPEECH_TAG SERVER_TAG PHASES`)
| phase | rules | GPU time (estimate) | may share Thor |
|---|---|---|---|
| prep (cases, every request; no model loaded) | 0 | none | yes |
| exact | 1, 8 | ~15 min | yes (exactness only) |
| server | 6, 7 | ~35 min (lfm3-speech loads once per case) | yes (exactness only) |
| tts | 2 | ~1 h 30 min (HF ~48 min, native ~28 min, Whisper ~10 min) | yes, with no speed reported |
| long | 3 | ~35 min | yes, with no speed reported |
| interleaved | 5 | ~25 min | yes, with no speed reported |
| asr | 4 | ~55 min (about 5 h of FLEURS audio; LibriSpeech's 5.4 h took 23 min native, 27 min HF) | yes, with no speed reported |
| escalate | 2 | ~15-25 min (10-20 flagged cases expected by chance at 228 cases) | yes, with no speed reported |
| listen | 9 | none (copies) | yes |

**Total: about 4.5-5 GPU hours.** That is more than the 2-3 hours I told Tripp before these rules existed; the ASR
corpus and the server's per-case engine runs are most of the difference. A smaller variant (TTS 4 sentences, ASR 50
utterances per language, rule 6 on female voices only) is about 2.5 hours, with wider intervals (open question 5).
The estimates come from the measured rates (native 3.2x real time and HF 1.9x for TTS, end to end; audio-in-v1's
corpus times), assuming about 8 s of speech per FLEURS sentence. They are not measured on this data.

**Sharing Thor with image work.** Every rule here is a comparison of outputs, never a timing, so every phase may run
beside other GPU work. Native outputs are bit-identical regardless of load (audio-assurance-v1 rules 3 and 12), and
HF is seeded. What sharing costs is speed figures (not reported for shared phases) and memory:
- `lfm3-profile-v1` has a 48 GiB cgroup limit (`docker inspect`; two racecheck runs side by side were OOM-killed on
  2026-11-11). lfm3-speech and the server run there, so check the container's headroom against whatever image work
  runs in it before starting. Never run a sanitizer beside these phases.
- HF and Whisper run in `lfm3-omni-dev` (no limit), about 10 GB for the BF16 model plus 4 GB for Whisper, from the
  same 122 GB.

Timing checks (the soak, stream stalls, timeouts, throughput) are not part of this round. They need a quiet Thor.

## Open questions for Tripp
1. Download FLEURS test for the 18 languages to Thor (`data/fleurs/`): 7.74 GB of audio and 8 MB of transcripts,
   from huggingface.co/datasets/google/fleurs, CC-BY-4.0. Without the audio, rules 0-3 and 5-9 can still run (they
   need only the transcripts), and rule 4 waits.
2. Rule 9 is 57 blind pairs (about 15-20 minutes of listening). Keep all of them, or one voice per locale (38)?
3. The splitter gap (rule 3): measure it here and fix it in its own round (my recommendation), or fix it first?
4. British English gets American number readings. Accept, or list it for the normalizer's next round?
5. The full round (~4.5-5 GPU hours) or the smaller variant (~2.5 hours)?

## Results
None yet.
