# Normalizer correctness audit (audio-assurance-v1, task 2)

Tripp (2026-11-11): "Do all four." The gap from the audit: a wrong number or abbreviation reading in
`tools/speech_normalize.py` (and its byte-for-byte native port, `native/include/lfm/speech_normalize.hpp`) is caught
today only by ear. This audit writes down what a careful human reader would say for a broad corpus, compares, and
fixes what is wrong.

## Rules, set before the corpus is scored
Written before any audit input was run through the normalizer. Failures stay recorded here, fixed or not.

1. **Corpus.** `tests/data/normalizer_audit.json`: at least 300 short sentences across the categories below, each with
   the reading a careful American English reader would give (`expect`), written by the style guide of
   validation/speech-normalize-v1 (cardinals without "and", years in pairs, "point" decimals, "three dollars and fifty
   cents", "seven forty-five", phone numbers digit by digit with "oh", titles in full, "a.m."/"p.m." and acronyms as
   written). Where two readings are both defensible, the entry lists both (`also`) and is marked ambiguous up front.
   Categories: cardinals, ordinals, decimals, negatives, ranges, years against counts, currencies, percentages, times,
   dates, fractions, phone numbers, Roman numerals, units, abbreviations and titles, acronyms, versions,
   URLs and emails, hashtags, math symbols, grouped numbers, scientific notation, sports scores, mixed sentences.
2. **Comparison.** Output and `expect` are compared after light normalization (lowercase; every character but
   letters, digits, apostrophes and spaces removed; hyphens as spaces; spaces collapsed), as in speech-normalize-v1's
   text accuracy.
3. **Classes.**
   - **fine:** equal to `expect` (or to an `also`).
   - **left as written:** the normalizer does not touch a form outside its rules (an acronym, a URL, a version string,
     a Roman numeral), and the text is unchanged. Not a normalizer error; reported, because the TTS model then reads it
     unaided.
   - **wrong:** the output changes the meaning (a different value, unit, date, person or place: "St." read "Street" before
     a saint's name, 3.10 read as a different number, a minus lost), or is broken (words glued, digits left inside
     words it wrote, a partial conversion that mangles a form such as an IP address or a version, a reading no human
     would give such as "one point two point three").
   - **ambiguous:** the output differs from `expect` but is one of two defensible readings (marked before scoring).
4. **Fixes.** Only "wrong" entries are fixed, each by the smallest change to the reference and the same change to the
   native port, with a test that fails on the old code. A fix keeps every other corpus entry's class, keeps the native
   port byte for byte equal to the reference (tests/test_speech_normalize.py: experiment sentences, real text, 30,000
   fuzz strings), and stays linear in the text with no recursion that grows with the input (the 64 KiB /speech limit is
   not relied on). A fix that would change an existing validation reference's text is listed with the reference, so
   the bit-exact audio references it touches are regenerated before they are compared again.
5. **Out of scope.** British English readings: the normalizer is American English only; the server applies it by
   default for a system prompt that names American or British English. Where a British reader would differ (dates
   D/M/Y, "and" in cardinals), the entry notes it, but the American reading is the expectation.

## Results (2026-11-11)
Corpus: 373 entries in 24 categories (`tools/normalizer_audit_corpus.py` writes `tests/data/normalizer_audit.json`;
`tools/normalizer_audit.py` scores it). Each "differs" entry was judged by hand under rule 3. The 7 entries marked
ambiguous up front were judged too: an output that is neither of the two defensible readings (broken, or a changed
value) counts as wrong.

| class | before | after |
|---|---:|---:|
| fine | 307 | 353 |
| wrong (meaning changed or broken) | 50 | 2 (not fixed, below) |
| style (meaning kept, not misleading; not fixed) | 6 | 8 |
| ambiguous (a defensible reading) | 1 | 1 |
| left as written | 9 | 9 |

No entry that was fine before changed.

**Wrong before the fixes (50).** Seven were marked ambiguous up front but were wrong:
- "$3.999" became "three dollars and ninety-nine cents", a changed value.
- "1:23:45" became "one twenty-three:forty-five".
- "2:1" became "two:one".
- "2024-03-15" became "two thousand twenty-four-zero three-fifteen".
- "15/03/1990" became "fifteen/zero three/one thousand nine hundred ninety".
- "9/1" became "nine firsts".

The other 44:
- Value changed:
  - "1,234.5" became "one,two hundred thirty-four point five".
  - "$3.2B", "$5M" and "$40k" became "three dollars and twenty centsB", "five dollarsM" and "forty dollarsk".
  - "-$3 million" and "+5%" lost their sign.
  - "2-3%" and "$10-$20" lost their range.
  - "10 - 3 = 7" became "ten to three equals seven".
  - "Oak St. Then" became "Oak Saint Then".
  - "1 000 000" became "one zero zero zero zero zero zero".
- Broken output:
  - "3:30pm" became "three:thirtypm".
  - "3:1", "16:9" and "1:100" kept their colons.
  - "50/50" and "24/7" became "fifty fiftieths" and "twenty-four sevenths".
  - "2 1/2" became "two one half".
  - "555.123.4567" and "+1 415 555 0123" were half converted.
  - "5'10\"" and "6' 2\"" kept their quote marks.
  - "1.2.3", "2.4.1" and "192.168.1.1" became "one point two.3" and similar.
  - "3e8", "6.02e23", "2E-5" and "10^6" were half converted.
  - "'90s" became "'ninetys".
  - "2+2" and "8 / 2" kept their symbols.
  - Numbers glued to the words written after them: "22B", "3D", "4G", "5k", "3a", "5x", "100m", "3x", "#2024goals".
  - "4x4" became "fourx4".
- A reading no one gives: "Dec. 31, 1999" became "one thousand nine hundred ninety-nine".

**Fixes.** Each change was made in `tools/speech_normalize.py` and the same change in
`native/include/lfm/speech_normalize.hpp`, marked "audit" there. Nothing else changed.
- ISO dates: 2024-03-15.
- Day-first dates when the day is above 12: 15/03/1990.
- International numbers of 8 digits or more: +1 415 555 0123.
- "." between a phone number's last groups: 555.123.4567.
- "+" on a number or between digits, and "=" between digits: +5%, 2+2, 1+1=2.
- Feet and inches, under 12 inches and not after "°" or "deg." (angles): 5'10".
- Dotted numbers of three groups or more: an IPv4 address digit by digit with "dot"; otherwise a version with "point".
- Scientific notation ("times ten to the eighth") and powers ("squared", "cubed", "to the tenth").
- Money:
  - a minus sign: -$3;
  - ranges: $10-$20;
  - k/K/M/B/bn scales;
  - more than two decimals read as a decimal: $3.999.
- Durations h:mm:ss in hours, minutes and seconds.
- A time with am/pm written on: 3:30pm.
- Ratios: 3:1. "24:00" and other zero-led right sides are left as before.
- Percent ranges: 2-3%.
- Two-digit decades: '90s, 80s.
- Mixed numbers whose fraction is proper: "two and a half", "one and an eighth".
- Fractions: n/n reads "fifty fifty"; 24/7 reads "twenty-four seven"; n/1 reads "nine to one".
- Dimensions written without spaces: 4x4, 1920x1080.
- In an equation, where `= <number>` follows: " - " reads "minus" and " / " reads "divided by", not a range.
- A grouped whole part in a decimal: 1,234.5.
- A year right after a month name, or after "Month <word>,": May 2020, Dec. 31, 1999.
- A space between a number and the letters written right after it: 22B, 6am, 3D.

**Rejected or narrowed after checking real text.** All 40,015 clean wikitext-2 and PG-19 lines were run through the old
and new reference, and the changed spans were read:
- "St." read "Street" after a capitalized word mid-sentence (to fix "Oak St. Then"): reverted. Real text had "Sallie St.
  Clair", "West St. Paul" and "Bury St. Edmund's", all wrong under it. "Oak St. Then" stays wrong (no safe rule).
- " / " between digits always read "divided by": narrowed to equations. Wikitext has "2012 / 13" seasons and "9 / 11".
- A range before "=" not read as a range: narrowed to `= <number>`. Wikitext headings "1941 – 1942 = =".
- Feet and inches: narrowed to under 12 inches and no angle before. PG-19 coordinates read "5 deg. 49' 30\" W."

The final real-text effect, 496 of 40,015 lines changed:
- 397 changed spans only add a space between a number and letters ("3b" became "three b");
- 166 now read a year after a month name as a year ("April nineteen eighty-two");
- 14 read ranges before "%" or decades ("'fifties").

None was judged worse. One neutral case: PG-19's "30s." means thirty shillings; it read "thirtys." and now reads
"thirties.". Both are wrong, and the old reading was no closer.

**Not fixed: wrong (2).**
- "Go down Oak St. Then turn.": no rule found that keeps the saints above.
- "1 000 000" with spaces between the groups: a rule would join adjacent numbers in ordinary text ("In 2024 100
  people").

**Not fixed: style (8).** The meaning is kept:
- "four hundred one(k)", "F-one hundred fifty", "Boeing seven hundred forty-seven";
- "HTTP/two";
- "#two thousand twenty-four goals";
- "m/s" and "100m" left with the letter;
- "six feet two inches man".

**Ambiguous (1).** "extension 4021" reads "four thousand twenty-one".

**Left as written (9).** These are outside the rules: Roman numerals (World War II, Henry VIII; "I" makes a rule risky),
".5" and the malformed "15,00".

**Tests.** In `tests/test_speech_normalize.py`, on the plain and the ASan+UBSan builds:
- `test_audit_fixes`: 32 cases. 28 fail on the old code; the other 4 are real-text guards.
- `test_audit_corpus`: every non-fine corpus entry must be one listed above, and the native port must equal the
  reference on the corpus.
- 11 new hostile seeds in `test_hostile_text_is_linear`.
  - Two of the new passes were quadratic in the native port: a grouped-decimal chain (",000" repeated) and the
    month-year `[\w-]` run ("2024-01-" repeated). Both are cached now, linear like `run_lo`/`run_hi`.
- 90 new fuzz pieces.

Results:
- `tests/test_speech_normalize.py`: 56 passed.
- 200,000 extra fuzz strings (seeds 101, 202, 303, 404): 0 differ between the native port and the reference.

**Existing references.** Only `/speech` in audio mode normalizes (`speech_text.hpp normalize_user_text`). The engine
(lfm3-speech) never does. None of these texts normalizes differently after the fixes:
- every user text in the 4,666 request files on Thor (build/ and validation/: 117 distinct texts, 1,577 TTS);
- the check suites' NUMBERS text (numbers-normalized-s1);
- the 32 sentences of validation/speech-normalize-v1.

No bit-exact audio reference changes.

## Red-team round (2026-11-11)
An independent red team checked the fixes above. Its repro harnesses were `h.py` (old, new and native output for a
file of sentences), `fuzz.py`, `fuzz2.py`, `fuzz3.py`, `timing.py` and `wiki2.py` (a sentence-level wikitext diff).

The native port equalled the reference in every finding, so these are rule errors, not port errors. The principle for
remediation: where a new rule cannot be made right, the output falls back to what the code read before the audit,
never a new wrong meaning. Every finding and what was done:

1. **Digits left after a decimal part.** "2^0.5" read "two to the zeroth.5"; the same happened with "10^1.5",
   "1:1.85", "1:2.5", "2:21:45.3" ("…seconds.3") and "1.5e3.5". **Fixed: the decimal part is read.**
   - Powers and scientific notation read a decimal exponent as a number: "two to the zero point five", "times ten to
     the three point five".
   - h:mm:ss reads decimal seconds: "forty-five point three seconds".
   - A ratio reads a decimal right side: "one to one point eight five".
   - The old time rule also left a lap time's fraction: "1:23.456" read "one twenty-three.456". It now reads "one
     twenty-three point four five six".
   - New invariant test: for 3,000 generated inputs across every audit rule (decimals, signs, plurals, context words),
     the output holds no digit.
2. **Plurals split from their "s"** (a regression): "747s" read "seven hundred forty-seven s"; also "F-16s", "6s",
   "under-21s", "AR-15s", "100s", "500s", and "49ers" read "forty-nine ers". **Fixed.**
   - No space is added before a lone "s" or "ers".
   - A plural pass reads "747s" "seven hundred forty-sevens", "6s" "sixes", "100s" "hundreds", "under-21s"
     "under-twenty-ones".
   - "49ers" reads "forty-nineers", as before the audit.
   - "4x4s" reads "four by fours".
3. **Clock times with seconds read as durations:** "at 07:15:00 sharp" read "seven hours fifteen minutes".
   **Fixed.** h:mm:ss reads as follows:
   - As a clock ("seven fifteen", with "and N seconds" when the seconds are not zero) after "at", "by", "until", "till",
     "from" or "since", with a leading zero, or at 13 hours or more.
   - As a duration only after "in", "ran", "time of", "lasted", "took", "finished", "clocked", "record", "split" and
     similar words, or above 23 hours.
   - Otherwise as before the audit.
4. **The ratio rule caught race times, timestamps, verses and ports:** "ran 27:15", "Skip to 45:12", "Genesis 1:1",
   "Psalm 119:105", "port 22:80". **Fixed.** A ratio now needs a ratio word among the three words before it or the
   word after it: ratio, aspect, odds, scale, mix, mixed, mixture, proportion. Everything else reads as before the
   audit, colon included. A "chapter verse" reading for verses was optional and is not done.
5. **False positives in prose.** **Fixed; each falls back to the reading before the audit:**
   - Mixed numbers ("the 300 2/3 voted") now need a whole number of one or two digits and a denominator of 2, 3, 4,
     8 or 16. "Week 3 4/5" also stays as before.
   - n/1 ("Starting 9/1, … 12/1", US dates) reads "to one" only after "odds"; otherwise "nine firsts" as before.
   - Money ranges no longer apply to a subtraction ("$5 - $3 = $2", "£3.50 - £2.00 = £1.50").
   - Money ranges and percent ranges need no space or a space on both sides of the dash, so "100 -3% lower" reads
     "minus three percent".
   - "3-1=2" read "three to one equals two": the early "=" rule ran before the range rule. "=" between digits now runs
     after it, and the equation rule takes an unspaced "-": "three minus one equals two". (Corrected in red-team
     round 2: this used to say "an unspaced '-' or '/'". An unspaced "8/2=4" is read by the fraction rule first, as
     "eight halves=four", the same as before the audit. Only a spaced " / " in an equation reads "divided by".)
   - "49ers": see item 2.
6. **Rules inside IDs, hex, links and emails.** **Fixed, or left no worse than before the audit:**
   - Every audit rule now leaves a match alone when it starts inside a link or address token. That is a
     whitespace-delimited token holding "@", "://", "www." or a domain (".com", ".org", ".net", ".edu", ".gov", ".io",
     ".dev", ".ai", ".app", ".co", ".uk", ".de"). This covers the ISO date inside "example.com/2024-03-15/post" and
     the email "sales-2024-03-15@corp.com", and the dotted rule inside "http://192.168.1.1:8080".
   - Scientific notation is not read after "#" or "-", or before "-", "." or a letter. A capital E without a decimal
     point or a sign is left as written.
     - The UUID "123e4567-e89b-…", "#3e5", "2E5-100" and "Room 2E1" no longer read as scientific notation.
     - A lowercase "1e100", "1e10" or "1e9" in prose ("Hash 1e100") still reads "one times ten to the …". It is
       well-formed, and nothing tells it from a number.
   - "0x10" is not a dimension; hex stays as before the audit.
   - Dotted numbers:
     - "15.03.2024" and "2024.03.15" read as dates.
     - "1.234.567" reads as a grouped number.
     - Four valid octets read as an IP address only with a three-digit group or a network word before them (ip,
       address, ping, connect, server, host, mask, gateway, dns, subnet, router). So "4.2.0.1" and "1.0.0.0" read as
       versions, while "255.255.255.0", "192.168.1.1" and "ping 8.8.8.8" read as addresses.
   - "2H2O" reads "two H2O" (before the audit: "twoH2O"). That is the chemist's reading, so it is kept.
7. **Reference (Python) only.** **Fixed:**
   - The feet-and-inches angle test searched the whole text before each match; 64 KiB of `1'1" ` took 5.4 s. It now
     scans back over spaces only, under 2 s in the test, as the native port does.
   - The reference lifts Python's 4,300-digit limit (`sys.set_int_max_str_digits(0)`).
   - `cardinal()` builds "trillion" groups in one pass, as the native port does. The red team's 64 KiB "555." input
     reached the grouped-thousands reading and overflowed the old recursion. 1,197 random numbers of 1 to 400 digits
     give the same words as the old recursion.

**Corpus after the round.**

| class | round 1 (above) | after the red-team round |
|---|---:|---:|
| fine | 353 | 351 |
| wrong | 2 | 4 |
| style | 8 | 8 |
| ambiguous | 1 | 1 |
| left as written | 9 | 9 |

"The video is 1:23:45 long." and "The score at halftime was 2:1." have no duration or ratio word, so they read as
before the audit ("one twenty-three:forty-five", "two:one") and count as wrong again. "The odds were 9/1." stays fine:
it has the "odds" cue.

**Real text after the round.** Of the 40,015 clean wikitext-2 and PG-19 lines, 487 change from the code before the
audit:
- 381 changed spans only add a space;
- 167 read a year after a month name;
- 20 are decades ("'fifties", "seven hundred thirties"), ranges before "%" and plurals ("sixes").

None reads worse. The red team's `wiki2.py`, 3,777 wikitext sentences, gives:
- native port equal to the reference on all of them;
- 197 changed from before the audit: 140 years, 64 space-only, and 18 others, all improvements (for example
  "seven,two" became "seven thousand two", and "'ninetys" became "nineties").

**Tests added this round** (plain and ASan+UBSan builds):
- `test_redteam`: 33 cases covering every finding. All 33 fail on the round-1 code, checked against a native build of
  round 1's header, which matched round 1's reference byte for byte.
- `test_rule_outputs_hold_no_digit`: the invariant in item 1.
- `test_reference_feet_and_long_numbers`: item 7.
- `test_audit_fixes` now uses forms with a cue: "the half in 1:35:10", "the aspect is 16:9", "odds of 9/1".
- `AUDIT_RESIDUAL` adds the two corpus entries above.

**Checks rerun on the final code:**
- `tests/test_speech_normalize.py`: 61 passed, plain and ASan+UBSan.
- `tests/test_speech_text.py` and `tests/test_lfm3_http_speech.py`, which normalize through the server's real HTTP
  layer: 153 passed, 1 failed. The failure is the known `test_every_validation_prompt_matches_hf_ids`, which can't
  load LFM3's HF model code (`modeling_lfm3.py` is missing from the transformers module cache). It failed the same
  way before the audit.
- Native port against the reference:
  - the red team's `fuzz.py`, `fuzz2.py` and `fuzz3.py`, seeds 11–14: 520,000 strings, 0 differ;
  - the same fuzzers on the ASan build: 40,000 strings, 0 differ, no sanitizer error;
  - the 200,000 extra strings of round 1: 0 differ;
  - the red team's six regression files (`h.py`): 0 diverge.
- The red team's `timing.py` (64 KiB of each of 56 hostile seeds, and 4× that): every native time scales linearly.
  - The new rules cost a constant factor over the old code. The largest is about 9×, on "555." and on a 2,000-digit
    "9…e9…" (0.16 s and 0.21 s for 64 KiB).
  - The Python reference no longer fails on any seed.
  - Two reference-only slow inputs predate the audit: "1,111," takes 71 s before and after, and "May 2020 " went from
    4.1 to 6.0 s. Both come from the reference's old year test, which scans the whole text before each number. Only
    the reference has them; the native year test is linear.
- Stored references, rechecked on the final code: none of the 117 distinct user texts in the 4,666 request files on
  Thor normalizes differently. That set includes NUMBERS (numbers-normalized-s1) and the speech-normalize-v1
  sentences. No bit-exact audio reference changes.

## Red-team round 2 (2026-11-11)
A second independent red team found the native port equal to the reference everywhere: no crashes, no slowness, and
`cardinal()` identical to HEAD. It found rule-level regressions against HEAD. Its repro files are `r1`–`r7` (`h.py`),
`fuzz4.py`, `timing2.py` and `uc.py` (141,984 ultrachat sentences). The same principle applies: where a rule cannot be
made right, it falls back to HEAD's reading. Every finding and what was done:

**HIGH**
1. **Comma-grouped numbers split at the comma** ("2,560×1,440" read "two,five hundred sixty by one,440"; also
   "1,000:1", "1,000^2", "1,234e5", "1,000'6\""). **Fixed.**
   - Dimensions read grouped numbers whole: "two thousand five hundred sixty by one thousand four hundred forty".
   - Ratios, powers, scientific notation, feet and inches, dotted numbers and mixed numbers no longer fire after a
     comma, so they read as HEAD did ("one thousand:one", "one thousand^two").
2. **IPv4 addresses with three-digit groups read as one big number** ("192.168.100.100", "255.255.255.255",
   "128.0.0.0 to 191.255.255.255"). **Fixed.**
   - Four valid octets are tested before grouped thousands, and can never be grouped thousands.
   - Grouped thousands ("1.234.567") also need no version word (version, release, v, build, update, upgrade, patch)
     and no network word near. "Version 1.100.200" reads as a version, and "IP address 123.345.128" (not a valid
     address) reads as a version, not as millions.

**MEDIUM**
3. **A range broken by a mixed number** ("1-1 1/2 hours" read "one-one and a half"). **Fixed.**
   - A range to a mixed number reads first: "one to one and a half hours", for "1-1 1/2", "2-2 1/2" and "1 - 1 1/2".
   - A mixed number right after a dash is left as HEAD read it: "-1 1/2" reads "minus one one half".
4. **A decimal and an "s" read as a plural** ("2.5s" read "two point fives"). **Fixed.** The plural exception applies
   only after whole numbers, so "2.5s" reads "two point five s" and "1.5ers" reads "one point five ers". There is no
   units rule for "s", so "seconds" is not read.
5. **h:mm:ss gating.** **Fixed:**
   - "of" is no longer a duration word, so "As of 10:15:30" and "night of" read as HEAD did.
   - a.m. or p.m. after the time forces a clock, without "o'clock": "nine thirty a.m.", "eleven thirty PM".
   - A duration word beats a leading zero: "finished in 02:15:30", "runs 00:03:25", "Duration 00:45:10" and
     "Runtime: 01:45:00" read as durations, zero hours omitted ("three minutes twenty-five seconds"). "runs" was added
     to the duration words, and a colon after the word is allowed.
   - Above 23 hours is never a clock ("at 24:00:00" reads "twenty-four hours"; this was already so).

**LOW**
6. Fixed:
   - "7.30pm" and "12.35am" read "seven thirty pm" and "twelve thirty-five am".
   - "911s" reads "nine one ones", as HEAD did.
   - Three-part dimensions read whole: "four by four by four". Letters after the last number are split off: "6.3cm"
     reads "six point three cm".
   - n/n above 4 reads "out of": "ten out of ten", "five out of five", "one hundred out of one hundred". "4/4" stays
     "four four" (time signatures), and 50/50 and 24/7 are unchanged.
   - "3.15.2024" reads "March fifteenth, twenty twenty-four" (month first when the second group is above 12).
   - "Version 1.100.200": see item 2.
   - A month right before a number reads it as a year only when no lowercase word follows. So "In March 1234 people"
     reads "one thousand two hundred thirty-four", as HEAD did, and "May 2020," still reads "twenty twenty". As a
     result the corpus entry "In May 2020 we stayed home." reads "two thousand twenty" again (style: a valid reading).
   - Percent ranges must ascend: "fell 30 - 2%" reads as HEAD did.
7. **Accepted, documented:**
   - "8/2=4" reads "eight halves=four", the same as HEAD. The claim in the first red-team section is corrected above:
     the fraction rule reads "8/2" first, and only a spaced " / " in an equation reads "divided by".
   - "2^31-1" reads "two to the thirty-first-one". The minus is lost, as in HEAD ("two^thirty-one-one").
   - "1 2/2" reads "one two two" (HEAD: "one two halves"). The input is malformed.

**The real-text regression gate** is `tools/normalizer_gate.py`. It covers 182,000 sentences:
- 40,015 clean wikitext-2 and PG-19 lines;
- the 141,984 ultrachat sentences that hold a digit (the red team's `uc.py` split).

Each sentence is read by the current reference and by HEAD's (commit fd349b5, the code before the audit, read with
`git show`). Every sentence whose reading changed is in `validation/audio-assurance-v1/normalizer-gate.jsonl.gz`, with
HEAD's reading, the reviewed reading, its categories and a verdict. The gate fails on any sentence that changed and is
not in the record, or that reads differently from its reviewed reading. `--write` rewrites the record after a review.

The review went category by category, reading every distinct changed span in each category. There were 20
categories, from space-only splits (2,115 distinct splits, grouped by the letters split off) to dimensions, dotted
numbers, money, plus, times, ranges, decades, powers, equations, dates, h:mm:ss, feet and fractions. Two automatic
checks backed it up: a list of sentences with more digits than HEAD, which was empty, and every sentence with fewer
words than HEAD, which was read.

It found seven regressions, all fixed:
- "8-9:30am" lost its range: the new "9:30am" rule claimed the time first. A range before a time with a.m./p.m. now
  reads first: "eight to nine thirty am".
- "5 – 4 = 1" with an en dash: the equation rule now takes the en dash: "five minus four equals one".
- "IP address 123.345.128" read as millions: see item 2.
- A DOI path "/15332691.2017.1351050" read as a version: dotted numbers after "/" are left as HEAD read them.
- "1/2^3" read "one/two cubed": powers after "/" are left as HEAD read them.
- "[-10^9" lost its minus: a minus sign before a power or scientific number reads "minus".
- Dewey "the 100s" read "the hundreds": after "the", a plural keeps "one hundreds"; elsewhere "100s of millions"
  reads "hundreds".

**Final count against HEAD:** 7,860 sentences change (7,464 ultrachat, 396 wikitext and PG-19). 7,857 are
improvements, 3 are neutral (a DOI, a race time written with dots, a regular expression in code: unreadable before
and after), and **0 are regressions**. By category (a sentence can fall in several):

| category | sentences |
|---|---:|
| a space between a number and the letters after it | 5,097 |
| decades and plurals ("thirties", "sixes") | 389 |
| dimensions | 380 |
| dotted numbers and decimals (versions, units) | 326 |
| money (scales, ranges) | 208 |
| fractions (24/7, 4/4, 2/2) | 199 |
| ranges and ratios | 183 |
| "+" | 138 |
| times with a.m./p.m. | 77 |
| dates and years | 86 |
| powers and scientific notation | 55 |
| IPv4 addresses | 54 |
| equations | 32 |
| h:mm:ss, feet and inches, n/n "out of", percent | 47 |
| other changed spans: mixed numbers, years, phone numbers | 788 |

**Corpus after round 2:** fine 350, wrong 4, style 9 ("In May 2020 we stayed home.", item 6), ambiguous 1, left as
written 9.

**Tests added:** `test_redteam2` has 30 cases covering every finding and the gate's regressions. 29 fail on the round-2
code (checked against a native build of round 2's header). The 30th, "at 24:00:00", was already right and stays as a
guard. `test_audit_fixes` now uses "In May 2020, we stayed home." (with the comma), and `AUDIT_RESIDUAL` adds the
corpus entry above.

**Checks rerun on the final code:**
- Native port against the reference:
  - the red team's seven regression files: 0 diverge;
  - `fuzz4.py`, `fuzz.py`, `fuzz2.py` and `fuzz3.py`, seeds 21–24: 640,000 strings, 0 differ;
  - the ASan+UBSan build, seed 31: 70,000 strings, 0 differ, no sanitizer error.
- `timing2.py` (41 hostile seeds, 64 KiB and 4× that): every native time scales linearly. The largest constant factor
  over HEAD is about 5× ("1e1.1"); the Python reference has no failures.
- The gate: 7,860 changed, 7,860 match the review, 0 new, 0 differ.
- Stored references: none of the 117 distinct user texts normalizes differently.
- `tests/test_speech_normalize.py`: 63 passed, plain and ASan+UBSan.
- `tests/test_speech_text.py` and `tests/test_lfm3_http_speech.py`: 153 passed, 1 failed. The failure is the known
  `test_every_validation_prompt_matches_hf_ids` (the HF model code is missing from the module cache), the same as
  before the audit.

## Red-team round 3 (2026-11-11)
A third independent check judged the work ready to merge after small fixes. It brought a fresh 303,996-sentence
corpus (ag_news, a wikipedia shard, amazon_polarity, gsm8k, imdb and scifact, built by its `build.py`), its changed
sentences grouped by shape, a 621-entry audit of the round-2 gate's verdicts (`sample.txt`) and adversarial files.
Same principle: where a rule cannot be made right, it reads as HEAD did.

**Findings and dispositions:**
1. **n/n fractions in arithmetic** ("Gap size = 2/2 = 1", "60/60 = 1", "6/6", 61 gsm8k sentences) and the n/n ≤ 4
   "two two" form. **Fixed.**
   - In a link, or with = * × + next to the fraction (or a parenthesis with one of them beyond it, "(3/3)*4"),
     every fraction reads as HEAD did ("two halves", "sixty sixtieths").
   - Elsewhere n/n also reads as HEAD did, except with a clear cue:
     - "out of" after a score word (rated, rating, score, scored, star, stars, grade, review): "rated ten out of ten";
     - "4/4" after a music word (time, signature, rhythm, meter, beat, measure, groove, tempo): "four four";
     - 20/20 near "vision": "twenty twenty";
     - 50/50 and 24/7 as said: "fifty fifty", "twenty-four seven".
   - "He went 3/3", "the 90/90 standard", "Tupelo Press' 30/30 Project", "No. 2/2" and "the 20/20 tournament" are
     back to HEAD's reading.
2. **Zero-led version parts lost the zero** ("Ubuntu 22.04.3"). **Fixed.** Zero-led parts read digit by digit, in
   HEAD's style for decimals: "twenty-two point zero four point three".
3. **Money subtraction read as a range** ("$450 - $75 - $120 = $255", "$100 - $18 leaves $82", "$60-$13-$6-$9";
   44 gsm8k sentences). **Fixed.** A money range now needs:
   - ascending amounts;
   - no third amount chained after it, and no amount-and-dash before it (a bullet "- $10,000 - $20,000" is still a
     range);
   - no "=" later in the clause (before ; ! ? a newline or ". ", within 120 characters).

   Otherwise it reads as HEAD did.
4. **Optional items, both done:**
   - "3x10(8)" and "24×7*3": dimensions do not fire next to * = + or before "(", so they read as HEAD did.
   - "PX-100s": a plural after a dash keeps "one hundreds", as HEAD did.

**Regressions this round's own changes caused, found by the gate and fixed before the final count:**
- "4/4 rhythm" and "4/4 meter" lost "four four": the music words above were added.
- "2) 24/7 Availability" was taken as arithmetic because of the ")": a parenthesis now counts only with an operator
  beyond it.
- A bullet list "- $10,000 - $20,000" lost its range: a chain now needs an amount before the dash.
- Chained subtraction "<<8-3-3=2>>" became "eight to three minus three" after a reordering. The reordering was reverted.
  Instead, a range followed by "= <number>" now reads as a subtraction with its "=": "20 - 1 = 19" reads "twenty
  minus one equals nineteen", and "3-1=2" reads "three minus one equals two".
- Numbers of 7 digits or more in arithmetic: HEAD read "2000000-400000" whole ("two million") through its range
  rule. They read whole again when an operator word follows them or "equals" comes before them. "-81712345" (an ID
  or a coordinate) is still digit by digit.
- "3.5x=1050." read 1050 as a year ("ten fifty"). A year never follows + * ×, or "=" after a digit or a one-letter
  variable. "established=1900" and URL paths keep HEAD's reading.
- Decimal ranges: "1-1.25" now reads "one to one point two five", where HEAD read "one to one.25" and left digits.
  A decimal range must ascend, have the same spacing on both sides of the dash, and have no year at either end.
  So "position=\"0 1.5 -5\"", "40.7128 -74.0060" (coordinates), "IEEE 610.12-1990" and the chess move "5.0-0" read
  as HEAD did.
- "2^31-1" reads "two to the thirty-first minus one": the minus is no longer lost.

**The auditor's verdict audit.** It disagreed with 2.7% of the 621 sampled verdicts: 16 "improvement" verdicts it
judged neutral, and "Gap size = 2/2 = 1" a regression. Its list of the 16 was not in the repro files, so I re-read all
621 entries. I found 15 overstated verdicts, all now resolved:
- **Now neutral:**
  - the baseball box scores "Cato 1x3" and "Arps 2x2" (#94, #98);
  - "£30-38k" (#315);
  - "+2-4" (#408);
  - the "0AA-BBBB" / "(\+81|0)" regex sentence (#417);
  - "5:00-6:00pm" (#588).
- **Back to HEAD's reading:**
  - "Tupelo Press' 30/30" (#221);
  - "90/90" (#222, #229, #230);
  - "20/20 tournament" (#223);
  - "No. 2/2" (#258);
  - "24×7*3" (#101).
- **Now an improvement:** "2^31-1" (#341), which reads "minus one".
- "Get 20/20 vision" (#227) now reads "twenty twenty vision".
- "Gap size = 2/2 = 1" reads as HEAD did ("two halves"), so it is no longer a changed sentence.

The auditor's accepted leftovers, compared with HEAD, are not regressions:
- "6:30-8:00pm": "six thirty-eight pm" (HEAD "six thirty-eight:zero zeropm");
- "3x daily": "three x daily" (HEAD "threex daily");
- "120/80 mmHg": "one hundred twenty eightieths", as HEAD;
- "5.25%-5.50%": as HEAD;
- ZIP codes (as HEAD);
- legal-citation years ("42 U.S.C. 1983", as HEAD);
- "Wait 30s": "thirties" (HEAD "thirtys").

**The gate now covers 485,135 distinct sentences:**
- 39,193 wikitext-2 and PG-19 lines;
- 141,976 ultrachat sentences;
- the round-3 corpus: 83,995 ag_news, 40,000 gsm8k, 59,999 amazon, 19,999 imdb, 79,973 wikipedia and 20,000 scifact.

`tools/normalizer_gate.py` rebuilds the round-3 corpus from the local Hugging Face cache exactly (checked: 303,996 of
303,996 sentences identical, in order).

**Review:**
- Every changed span was grouped by shape, with digits and number words masked.
- 26,926 spans followed four mechanical patterns and were classified without reading: words no longer glued to a
  number (14,910), "=" read (9,246), "+" read (2,412), a year after a month (358).
- The other 1,397 shapes (21,668 spans) were read: one example of every shape outside the operator, decade and
  money-scale patterns, and the largest shapes of those patterns.
- Every distinct span pair that appeared in the later passes of this round was read as well (about 1,100).

**Final count against HEAD:** 39,888 sentences change.
- 39,744 are improvements.
- 144 are neutral, by reason:
  - 85 descending decimal pairs: feet and inches ("47-6.75") or a subtraction without "=". HEAD read "to" and left
    digits; the dash now stays, with no digits.
  - 22 operands read whole instead of in pairs (".20*1800"). Both readings are valid.
  - 15 baseball box scores.
  - 7 counts after "n=" (both readings valid).
  - 5 chess scores.
  - 10 single cases: "+2-4", a regex, "5:00-6:00pm", "£30-38k", a subtraction without "=", a race time written with
    dots, a DOI, "9.11.01", and a year before a dotted date.
- **0 are regressions.**
- The gate passes: 39,888 match the review, 0 new, 0 differ.
- The 18 recorded sentences whose reading changed this round were re-read: all are improvements or one of the
  neutral cases above.

**Corpus:** unchanged from round 2 (fine 350, wrong 4, style 9, ambiguous 1, left as written 9).

**Tests:** `test_redteam3` has 22 cases. 17 fail on the round-3 code (a native build of round 3's header). The
other 5 are guards:
- an ascending money range;
- a subtraction with "=";
- three readings that were right before this round, which this round's own changes broke and the gate caught: "4/4
  rhythm", "2) 24/7", and the bullet money range.

The earlier tables (`AUDIT_FIXES`, `REDTEAM`, `REDTEAM2`) still hold.

**Checks rerun on the final code:**
- `tests/test_speech_normalize.py`: 65 passed, plain and ASan+UBSan.
- `tests/test_speech_text.py` and `tests/test_lfm3_http_speech.py`: 153 passed, 1 failed. The failure is the known
  `test_every_validation_prompt_matches_hf_ids` (the HF model code is missing from the module cache), the same as
  before the audit.
- Native port against the reference, 0 differences everywhere:
  - all 303,996 round-3 corpus sentences, on the plain and the ASan+UBSan builds (no sanitizer error);
  - the round-2 fuzzers (`fuzz4.py`, `fuzz.py`, `fuzz2.py`, `fuzz3.py`; 160,000 strings plain, 50,000 on ASan);
  - the seven round-2 regression files;
  - the round-3 adversarial files.
- `timing.py` and `timing2.py` (64 KiB and 4× that of every hostile seed): every native time scales linearly. The
  largest constant factor over HEAD is about 10× ("555.", 0.2 s for 64 KiB). The Python reference has no failures.
- The gate (`tools/normalizer_gate.py`): 485,135 sentences, 39,888 changed, all 39,888 matching the review, 0 new,
  0 differ.
- Stored references: none of the 117 distinct user texts in the request files on Thor (and NUMBERS and the
  speech-normalize-v1 sentences) normalizes differently. No bit-exact audio reference changes.
