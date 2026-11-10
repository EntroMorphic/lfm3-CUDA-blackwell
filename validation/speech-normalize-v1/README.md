# Numbers and abbreviations: rules, the model, or both, chosen by test

Tripp (2026-11-10): "Can we just have the model or a preprocessor convert abbreviations and numbers to words?", then
"Do it." The TTS quality bar found the model stumbling on digits ("389", "7:45"; journal C3). Two ways to give it
words instead: a deterministic preprocessor, or the model itself rewriting the text first. Measured here before
choosing.

## Arms
All four feed the same native TTS path (lfm3-speech audioin-v5, the default sentence split, the female American English
voice, `Perform TTS in American English using a female voice.`), seeds 1-15.
- **none:** the text as written.
- **rules:** `tools/speech_normalize.py` (a Python reference; ported to native only if chosen).
- **model:** the model rewrites the text first (native lfm3-speech, text mode, greedy; the prompt is fixed in
  `tools/speech_normalize_round.py`), its rewrite used whatever it says.
- **hybrid:** the model's rewrite when it passes the guard, the rules' text otherwise. The guard: aligning the
  rewrite's words with the original's (difflib, lowercased, punctuation stripped), every changed, deleted or inserted
  region of the original must touch a target word (one holding a digit or a symbol, or a known abbreviation) and keep
  its other words, in order, in the rewrite's region; and the rewrite must hold no digit.

## Sets
- **Dev (D, `cases-dev.json`):** 20 sentences I wrote to cover the cases (times, money, years against counts, decimals,
  percentages, ordinals and dates, large numbers, the "389"/"7:45" failure, titles, an address, units, temperatures, a
  phone number, a fraction and a range, Latin abbreviations, a decade, math symbols, "No."/"#"). The rules are
  developed against D.
- **Held out (H):** after the rules and the model prompt are frozen (their commit recorded here), the model writes 12
  sentences of its own (text mode, greedy, the prompt in `tools/speech_normalize_round.py`); I write their spoken
  references by the style guide below before any arm runs on them.

Spoken references follow one style guide, written now: American cardinals without "and" (1,234 "one thousand two
hundred thirty-four"); years in pairs (1984 "nineteen eighty-four", 1905 "nineteen oh five", 1900 "nineteen hundred",
2000 "two thousand", 2005 "two thousand five", 2026 "twenty twenty-six") and decades likewise ("nineteen nineties");
house numbers in pairs (1600 "sixteen hundred"); decimals with "point" and digits; money as "three dollars and fifty
cents", "one point two million dollars"; "percent"; times "seven forty-five", "seven oh five", "nine o'clock";
"a.m."/"p.m." and acronyms as written; dates "November ninth, twenty twenty-six"; phone numbers digit by digit with
"oh" for zero; titles and abbreviations in full (Doctor, Mister, Missus, Saint or Street, versus, et cetera, for example,
that is, number); units and symbols in words (kilometers, degrees Celsius, plus, equals).

## Measures
- **Text accuracy** (no audio): an arm's text equals the spoken reference after light normalization (lowercase, every
  character but letters, digits, apostrophes and spaces removed, hyphens as spaces, spaces collapsed).
- **WER** of each run against the spoken reference (Whisper large-v3, `tools/lfm3_asr.py`'s scoring: Whisper's
  English normalizer, compound folding). A case's score is its mean over the 15 seeds. Scoring artifacts depend only
  on the transcript, so identical speech scores identically across arms.
- **Fidelity** (model): the guard's pass rate.
- **Latency** (model): the rewrite's end-to-end time per case, native.

## Rules, set before any run
1. **Rules are adopted if they help on the held-out set:** the rules arm's case WERs on H are lower than none's (paired
   Wilcoxon signed-rank, one-sided, p < 0.05).
2. **The model (or hybrid) replaces the rules only if all three hold:** its text accuracy on H is higher than the
   rules'; its mean WER on H is not higher than the rules'; and Tripp prefers it in a blind listening.
   Otherwise the rules stand: deterministic, instant, and unable to change other words.
3. **If the rules do not help on H**, nothing is adopted by default; the result is reported and discussed.
D is reported, but it does not decide anything: the rules were tuned on it.

## Phase 2: interleaved replies (reported, not gated)
The preprocessor cannot reach interleaved replies, where the model speaks as it writes. Ten questions whose answers
hold numbers, three seeds, with and without `Write every number and abbreviation out in words.` appended to the system
prompt: the share of replies whose text holds a digit, and the speech's WER against the reply's own text.

## Results
(pending)
