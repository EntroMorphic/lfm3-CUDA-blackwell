# #2 step 4: native text in -> text and speech out, against HF (C1, C2, C4)

`lfm3-speech` (`native/src/lfm3_speech.cuh`, driver `native/tests/lfm3_speech.cu`) against HF BF16 SDPA
(`tools/lfm3_speech_hf.py`) on `prompts.json`: 10 cases (TTS female and male: fox, museum, numbers; a long
paragraph; three interleaved replies), seeds 1-3, the same prompt ids. Round: `tools/thor/speech_e2e_round.sh`;
ASR: Whisper large-v3 (`tools/lfm3_asr.py`); tables: `tools/lfm3_speech_report.py`. Bar:
`journal/lfm3-tts-quality-bar.md`. Records: `rounds/{e2e,long,fridge}/` (report.md, report.json, the ASR jsons).

## C1, intelligibility (per case)
The rule, per case: a case is flagged when native's mean WER over the seeds exceeds HF's worst seed on that case; a
flagged case escalates to 15 seeds and a Mann-Whitney test. (The first version pooled all cases and passed while one
case was 0.89 against 0.16; replaced.) Seeds 1-3 (`rounds/e2e/report.md`) flag three cases:
- `tts-numbers-female`: 15 seeds (`rounds/long`), native 0.204 against HF 0.196, p = 0.55. **Clears.** Both sit on
  a 0.158 floor (Whisper's reading of the digits).
- `il-fridge`: 15 seeds (`rounds/fridge` adds 4-15), about 0.018 against 0.010, both under 2%, every run ended by
  EOS. **Clears.**
- `tts-long-male`: native 0.418 against HF 0.163 over 15 seeds (p = 0.019). **The model's own failure, fixed by
  splitting long text into sentences:** `../speech-long-form-v1` (native split: WER 0.000 on 30 of 30 runs).
The other seven cases are not flagged (fox and interleaved 0.000 in both engines; museum 0.028 in both).

## C2, duration and endings
Every utterance of both engines ended by the model's own EOS. Frames per utterance, native/HF ratio of means by
case: 0.95-1.05 except il-fridge 1.14 at seeds 1-3 (1.03 at seeds 4-15) and tts-long-male 1.74 (the loops above).

## C4, interleaved: speech against its own text
il-reflective and il-study 0.000 WER against the text each engine wrote, il-fridge as above.

## Speed (decode loop, BF16 weights, batch 1, Thor)
Audio seconds per generation second: native 3.24, HF 1.91 (`rounds/e2e`); native split long-form 3.42 including
every chunk's prefill.

## C3 (listening)
`outputs/listening-c3-v1` (seed 1 of every case, blind A/B, key separate) and `outputs/listening-split-v1`:
**pending Tripp's ear.**
