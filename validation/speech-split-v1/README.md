# #2 step 4: choosing the splitter's length rule by test (2026-11-09)

The step-4 red-team found that the sentence splitter has no length cap. A chunk ends only at `. ! ?` followed by
whitespace, so long text without those stays one request and can fail as unsplit long-form does
(`../speech-long-form-v1`). Before choosing a fix, the options were measured. All runs are native `lfm3-speech`
speech-v4 in MAXN, male TTS voice, 8 seeds per text; Whisper large-v3 with compound folding (`tools/lfm3_asr.py`).
Rounds: `tools/thor/speech_native_round.sh`. Rules: `tools/lfm3_speech_chunked_hf.py` (`LFM3_SPLIT_RULE`,
`LFM3_SPLIT_CAP`; the default is still the validated sentence rule).

## Phase A: how long can one chunk be? (`asr/phase-a-unsplit.json`, prompts `prompts-split-phase-a.json`)
Rule, set before the data: a length is safe if every run has WER <= 0.05, and the cap is the longest safe length.
Single natural sentences, two texts per length, unsplit:

| words | mean WER | runs > 0.05 | first-error onsets (word) |
|---|---|---|---|
| 25, 26 | 0.000, 0.000 | 0/8, 0/8 | none |
| 34, 38 | 0.000, 0.000 | 0/8, 0/8 | none |
| 49, 49 | 0.000, 0.000 | 0/8, 0/8 | none |
| 59, 64 | 0.013, 0.002 | 1/8, 0/8 | 53, 55 |
| 71, 71 | 0.173, 0.023 | 5/8, 1/8 | 53-69 |
| 90, 87 | 0.176, 0.039 | 4/8, 2/8 | 41-84 |

By the rule the cap is 49 words. The first failure is at 59, so 49 leaves little margin, and 8 seeds is thin evidence.
A cap of 30 was added to Phase B as a sensitivity check after seeing these numbers, labelled as such. A single sentence
holds longer than the 97-word paragraph did (its onsets were at word ~41), which fits the window model: what matters
is how far the speech gets past the start of the text being read.

## Phase B: which rule? (`asr/phase-b-*.json`, `chunk-endings.json`, prompts `prompts-split-phase-b.json`)
Rule, set before the data: a rule passes an input if its mean WER is <= 0.05 and no run is above 0.3; among the rules
that pass every input, the simplest wins. Five inputs that defeat the sentence rule: quoted dialogue (85 words, every
sentence ending inside its quotes), paragraphs separated by blank lines without end punctuation (75), one 90-word
sentence with commas, one 85-word sentence without commas, and abbreviations (75 words: Dr., Mr., a.m., Jan., U.S.).
The rules:
- **sentences:** the current rule.
- **cap-comma N:** sentences, then any chunk over N words is cut at its last clause mark (`, ; :` or a dash) at or before
  word N.
- **cap-hard N:** the same, but the chunk is cut into equal pieces of at most N words.
- **full N:** blank lines and closing quotes also end chunks, and titles and abbreviations do not; then cap-comma N.

| rule | dialogue | paragraphs | long, commas | long, no commas | abbreviations | all |
|---|---|---|---|---|---|---|
| sentences | 0.040 | **fail** 0.178 | **fail** 0.318 (max 0.83) | **fail** 0.073 | 0.008 | no |
| cap-comma 49 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | yes |
| cap-hard 49 | 0.004 | 0.000 | 0.001 | 0.000 | 0.008 | yes |
| full 49 | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 | yes |
| full 30 | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 | yes |

Every chunk of every run ended by its own EOS (`chunk-endings.json`). Any cap fixes the failure. On WER the two plain
caps tie for simplest, so the choice goes to listening (Phase C), since WER cannot hear where a cut falls in a phrase.
