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

## Results (rules, prompts and dev set frozen at f7b6e47; held-out sentences and references at 31a37b0)
The model wrote 11 held-out sentences, not the 12 asked for: times with "AM"/"PM", prices, dates, percentages, a
weight, an emergency number, an address and a ZIP code, and no abbreviations. Abbreviation coverage therefore rests
on the dev set, which the rules were tuned on.

| | dev (20) | held out (11) |
|---|---|---|
| text accuracy: none / rules / model / hybrid | 0 / 20 / 2 / 19 | 0 / **8** / 1 / 8 |
| model rewrites passing the guard | 3 | 1 |
| rewrite latency, mean (max) | 0.57 s (4.2 s) | 0.41 s (0.48 s) |
| mean WER: none / rules / model / hybrid | 0.342 / **0.150** / 0.294 / 0.150 | 0.232 / **0.162** / 0.208 / 0.162 |
| rules lower than none, one-sided Wilcoxon | p = 0.0001 | **p = 0.003** |

- **Rule 1 PASS:** the rules help on the held-out set (p = 0.003). **The rules are adopted.**
- **Rule 2:** the model fails it. With the fixed prompt it returned most sentences unchanged, digits included; its
  text accuracy on H is 1 against the rules' 8. Hybrid's texts are the rules' whenever the guard fails, so hybrid
  equals rules (8). One prompt was tested; a few-shot prompt was not.
- **The rules' three held-out misses** are the ones visible before any run: "9:00 AM" (the time rule knows only
  lowercase "a.m."; read "nine o'clock AM"), "911" ("nine hundred eleven") and a ZIP code ("twelve thousand three
  hundred forty-five").
- **The WER understates the gain.** Whisper writes numbers as digits ("7.45", "$42.50", "8,336,817") and its normalizer
  does not always match them to the spoken words, so a correct reading still scores errors (both "seven forty-five"
  readings score 0.17 on d01). Identical speech scores identically, so the paired comparison stands, but the
  absolute values are mostly artifacts. Read from the transcripts, the rules fix real misreadings: "389" was heard
  as "309" without them, "1,500 residents" as "one Feimtund residence", "8,336,817" as "8,3260's 807", "911" as
  "919".
- **Found, not normalization:** in d14 the voice drops the aside "for example apples and nuts" in both arms
  ("Bring snacks to the third floor.").
- Every chunk of every run ended by EOS; 990 runs, Thor MAXN with locked clocks, unchanged (`power-*.json`).

### Phase 2: interleaved replies (`interleaved-score.json`)
| | replies with digits | speech WER against the reply's text |
|---|---|---|
| plain | 25 of 30 | 0.022 (0.016 on replies with digits) |
| instructed to write numbers in words | 19 of 30 | 0.041 |

Interleaved speech already reads its own digits well, and the instruction is only partly obeyed without making the
speech clearer: nothing is added for interleaved replies.

### Next
Tripp's blind listening (none against rules, seed 1 of every held-out sentence, the key kept on Thor). Then the
rules go native (C++, checked against this Python reference) ahead of the sentence split in lfm3-speech and
/speech; the three held-out misses are fixed there and recorded as changes made after this test.
