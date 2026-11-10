# #2 step 4: choosing the splitter's length rule by test (2026-11-09)

The step-4 red-team found that the sentence splitter has no length cap. A chunk ends only at `. ! ?` followed by
whitespace, so long text without those stays one request and can fail as unsplit long-form does
(`../speech-long-form-v1`). Before choosing a fix, the options were measured. All runs are native `lfm3-speech`
speech-v4 in MAXN, male TTS voice, 8 seeds per text; Whisper large-v3 with compound folding (`tools/lfm3_asr.py`).
Rounds: `tools/thor/speech_native_round.sh`. Rules: `tools/lfm3_speech_chunked_hf.py` (`LFM3_SPLIT_RULE`,
`LFM3_SPLIT_CAP`). The experiment's baseline was the first rule, which is now named `sentences-v1`. Since speech-v5
the name `sentences` (the default) means the rule this experiment chose, so its baseline rounds
(`phase-b-sentences-0`) must be rerun as `LFM3_SPLIT_RULE=sentences-v1`. The other candidates are built on
`sentences-v1`, as when they ran.

## Phase A: how long can one chunk be? (`asr/phase-a-unsplit.json`, prompts `prompts-split-phase-a.json`)
Rule, set before the data: a length is safe if every run has WER <= 0.05, and the cap is the longest safe length.
Single natural sentences, two texts per nominal length (the rows below), unsplit:

| words | mean WER | runs > 0.05 | first-error onsets (word) |
|---|---|---|---|
| 25, 26 | 0.000, 0.000 | 0/8, 0/8 | none |
| 34, 38 | 0.000, 0.000 | 0/8, 0/8 | none |
| 49, 49 | 0.000, 0.000 | 0/8, 0/8 | none |
| 59, 64 | 0.013, 0.002 | 1/8, 0/8 | 53, 55 |
| 71, 71 | 0.173, 0.023 | 5/8, 1/8 | 53-69 |
| 90, 87 | 0.176, 0.039 | 4/8, 2/8 | 41-84 |

How the rule was applied: per row, so a length counted as safe only if both of its texts were. Read per text, the
64-word text is also safe (max 0.016), but the 59-word text below it is not. The cap was therefore taken as the longest
length below the first failure: **49**. That reading was not written down before the data; it is the more cautious of
the two. Lengths 50-58 were not tested, so the real limit lies somewhere between 49 and 58, measured on 8 seeds.
A cap of 30 was added to Phase B as a sensitivity check after seeing these numbers, labelled as such.
The onsets are first errors of any kind, so they mix mishearings with the window failure. In len90-b, for example:
- the errors at word 41 are one word ("squares") misheard in seeds 1, 2 and 5, and seed 5 is a passing run;
- seeds 2 and 8 garble words 43-47 ("markets had once" heard as "demokrats"; "once been held" as "died");
- seed 1 drops out from word 67.
So these data neither show nor rule out that a single sentence holds longer than the 97-word paragraph did (~41).

## Phase B: which rule? (`asr/phase-b-*.json`, `chunk-endings.json`, prompts `prompts-split-phase-b.json`)
Rule, set before the data: a rule passes an input if its mean WER is <= 0.05 and no run is above 0.3; among the rules
that pass every input, the simplest wins. Five inputs that defeat the sentence rule: quoted dialogue (85 words, every
sentence ending inside its quotes), paragraphs separated by blank lines without end punctuation (75), one 90-word
sentence with commas, one 85-word sentence without commas, and abbreviations (75 words: Dr., Mr., a.m., Jan., U.S.).
The rules:
- **sentences-v1** (named "sentences" when it ran; the tables below use that name for this rule): the first rule.
- **cap-comma N:** sentences-v1, then any chunk over N words is cut at its last clause mark (a word ending in `, ; :` or
  a dash, or followed by a lone dash) at or before word N that leaves both pieces at least 6 words long. With no such
  mark, the chunk is cut into the fewest equal pieces; this repeats on the rest.
- **cap-hard N:** sentences-v1, then the chunk is cut into the fewest equal pieces of at most N words.
- **full N:** blank lines also end chunks, as does `. ! ?` followed by a closing quote or bracket. Titles (Mr. Dr. ...)
  never end a chunk; abbreviations (a.m. U.S. ...) end one only before a word starting with a capital. Then
  cap-comma N.

| rule | dialogue | paragraphs | long, commas | long, no commas | abbreviations | all |
|---|---|---|---|---|---|---|
| sentences | 0.040 | **fail** 0.178 | **fail** 0.318 (max 0.83) | **fail** 0.073 | 0.008 | no |
| cap-comma 49 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | yes |
| cap-hard 49 | 0.004 | 0.000 | 0.001 | 0.000 | 0.008 | yes |
| full 49 | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 | yes |
| full 30 | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 | yes |

Every chunk of every run ended by its own EOS (`chunk-endings.json`). The cap of 49 fixes the failure on all five inputs,
both alone (cap-comma, cap-hard) and in the full rule. Cap 30 was tested only inside the full rule. On WER the two plain caps tie for simplest, so the choice went to listening (Phase C), since WER
cannot hear where a cut falls in a phrase. Note that the cap-comma runs also used even cuts where a chunk had no
clause mark: the paragraphs and the sentence without commas.

## Phase C: listening (Tripp, 2026-11-09; `outputs/listening-split-rules-v1`, key beside it, not in the repo)
Seed 1, blind A/B, five pairs that each isolate one choice. The order was drawn from OS entropy, and the key was
opened only after every call was in. One seed, and four of the five calls were "slightly": these are preferences that
guided the choice, not measured effects.

| pair | A | B | Tripp | reading |
|---|---|---|---|---|
| long sentence | cap-comma 49 | cap-hard 49 | A good; B has an odd inflection at "eye" | The even cut fell at "as far as \| the eye could see", exactly there, and Whisper capitalized "The". So when a clause mark exists, cut there rather than evenly. (Even cuts remain the fallback for chunks without one.) |
| two or four pieces | full 30 (4) | cap-comma 49 (2) | both good, B slightly preferred | A preference for fewer seams: cap 49 over 30 |
| dialogue | full 49 (line by line) | cap-comma 49 (2 pieces) | both good, B slightly better | No preference for line-by-line breaks at closing quotes |
| paragraphs | full 49 (each blank line) | cap-comma 49 (2 pieces) | both good, B slightly better | No preference for breaks at each blank line. Confound: cap-comma's even cut happened to land on a blank line (38 words = 21 + 17), so this compared 2 paragraph-aligned pieces against 4. |
| abbreviations | full 49 (guarded) | sentences (cuts after Mr. / a.m.) | both struggle with abbreviations and numbers; B slightly worse | A slight preference for the guard. WER is consistent with no difference (1 run with errors against 3, of 8). |

## The decision: the default since speech-v5 (refined in speech-v7)
1. **Sentence ends.** A sentence ends at `. ! ?` followed by whitespace, with two exceptions:
   - never after a title (Mr. Mrs. Ms. Dr. Prof. St. Mt. vs.);
   - after an abbreviation (a.m. p.m. U.S. U.K. e.g. i.e. etc. No. Jr. Sr. and the months) or an ellipsis (`...`),
     only before a word that starts with an ASCII capital, after any opening quote or bracket.
2. **Short parts.** A part under 6 words joins the next one.
3. **Normalizing and the cap.** Every chunk becomes its words joined by single spaces. Any chunk over 49 words is cut at
   its last clause mark at or before word 49: a word ending in `, ; : --` or an em or en dash, or followed by a lone
   dash. With no mark, it is cut into the fewest equal pieces.
4. **No breaks at quotes or blank lines.**

Python reference: `sentences()` in `tools/lfm3_speech_chunked_hf.py`. The first rule is kept as `sentences_v1`, and
the experiment's candidates are frozen as they ran (`x_cap_comma`, `X_TITLES`, `X_ABBREVIATIONS`). Native:
`native/include/lfm/speech_text.hpp`.

**Refined by the second red-team (speech-v7).** None of these changes the chunks of any validated prompt or of the
five Phase B inputs (tested):
- **Bug, fixed.** The cap step recursed and copied the rest of the text at every cut. Natively, 200,000 words with no
  clause mark took 11-12 s and 8-9 GB; Python crashed with a RecursionError from about 49,000 words. Both now use one
  loop: natively, 1,000,000 words take about 0.2 s (the host tool's run time, JSON included, plus 0.6 s to load the
  tokenizer); Python takes 0.1 s for 200,000. `test_size_is_linear` bounds 300,000 words at 20 s, which the old quadratic version exceeds.
- `no.`, `jr.`, `sr.` were titles, so "told him no. Then ..." never cut. That made the chunk longer and pushed the cap's
  cut into the middle of the next sentence. They are now abbreviations. `vs.` moved the other way.
- An abbreviation followed by an opening quote or bracket, then a capital (`etc. "Next`), now cuts. An ellipsis before
  a lowercase word ("say something... but I forgot") no longer does. ASCII `--` counts as a dash.
- Chunks under the cap kept their original whitespace (newlines included), while cut pieces did not. Every chunk is
  now normalized.

**Verification:**
- **Parity.** Native equals Python on:
  - edge cases, every list entry in three cases before lowercase, capital, quoted and bracketed words, and dashes,
    ellipses and whitespace;
  - 5,000 fuzzed texts in the test file, plus the reviewer's two fuzzers on four fresh seeds (40,000 inputs, 0
    mismatches);
  - every chunk length from 40 to 110 words, with one clause mark at every position in each of its 10 forms;
  - 300,000-word inputs.
- **Mutation testing.** 46 planted bugs across both red-teams are all caught, 4 of them only after the second pass
  added tests (stacked and curly openers; a sanitizer build that catches undefined behaviour, such as a scan past the
  end). Four more give identical output and cannot be caught, because the normalization or the 6-word join absorbs
  them.
- **Thor.** 17 tests pass in lfm3-omni-dev, HF prompt ids included. The sanitizer test needs g++, which the container
  lacks; it runs and passes on the host with all the others. speech-v7's hash is in
  `../speech-long-form-v1/binaries.sha256`; it is the same binary as speech-v6, whose only source difference is an
  `#include`.
- **Unchanged.** Every validated prompt splits exactly as under the first rule (tested). The speech-v7 regression check
  is bit-identical (36 of 36 files), with all 8 refusals clean and stale outputs removed (`build/speech-v7-regression.log`
on Thor).
- **Phase B, natively.** With speech-v7, the five Phase B inputs (seeds 1-2) are bit-identical to the experiment runs
  whose chunks they reproduce: cap-comma 49, and full 49 for the abbreviations. Those experiment runs scored WER
  0.000-0.003 over seeds 1-8. Seeds 3-8 were not rerun; the engine is deterministic, so they carry over.

**Limits of the rule (documented, not fixed):**
- **Closing quotes.** No cut after them (`"Stop." Then`). The listening calls slightly preferred fewer seams.
- **Abbreviations before a proper noun.** One of the abbreviations followed by a proper noun is cut ("the U.S. | Army"
  when the parts are long enough). Telling these apart needs more than a capital letter.
- **Where clause-mark cuts can land.** A cut can fall inside a quoted list. A lone ` - ` used as a hyphen counts as a
  mark. And "the last mark" can make a 6 + 49 split where an even one would be about 28 + 27.
- **Words without spaces.** A word is a run without whitespace, so text without spaces (Chinese, URLs, long numbers)
  is not limited by the cap. The rule is for English text, as are all the system prompts here.
- **Headings and paragraph breaks** without a full stop join the next sentence.
- **The Unicode ellipsis** `…` (U+2026) is not `. ! ?`, so it never ends a sentence; the cap still bounds the length.

**Still open:**
- The cap's margin: lengths 50-58 are untested, and the first failure was at 59, on 8 seeds.
- The model's reading of numbers and abbreviations.
