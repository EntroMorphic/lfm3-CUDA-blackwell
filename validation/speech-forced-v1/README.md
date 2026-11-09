# #2 step 4, the deterministic gate: the native speech loop with forced inputs (A3 and the LM in speech context)

`native/src/lfm3_speech.cuh` (driver `native/tests/lfm3_speech.cu`, `lfm3-speech`) replays each traced case's text
tokens and raw frames (tts-trace-v1) with no RNG anywhere. `tools/lfm3_speech_replay.py` rebuilds the identical input
sequence (the prompt with its mode marker, then every decoded step's fed token or frame) and runs it through HF in one
pass. Quality bar: `journal/lfm3-tts-quality-bar.md` (A3; the backbone's tier-2a rule applied in speech context).

## A3: the speech-frame feedback embedding: PASS, bitwise
The sum of a frame's 12 codebook rows (`model.codebook_embedding`), accumulated in FP32 in codebook order and rounded
to BF16 once, against HF's `get_audio_out_embeds`: **1,021 of 1,021 fed frames bitwise identical** across the six
cases, and the 175 fed text tokens' embeddings too (1,196 decoded steps checked in all).

## The language model in speech context: PASS (text tier 2a's rule)
At every frame step: the final-normed hidden state the depth decoder reads, relative L2 from exact FP32 (HF eager,
TF32 off). Medians over each case's frame steps:

| case | frames | native | HF BF16 SDPA decode (the trace) | HF BF16 SDPA one pass | HF BF16 eager one pass | verdict |
|---|---|---|---|---|---|---|
| ref-interleaved | 373 | 0.01136 | 0.01131 | 0.01118 | 0.01217 | PASS |
| interleaved-study | 265 | 0.01102 | 0.01098 | 0.01111 | 0.01183 | PASS |
| tts-fox-female | 41 | 0.01084 | 0.01030 | 0.01088 | 0.01147 | PASS |
| tts-fox-male | 46 | 0.01068 | 0.01077 | 0.01060 | 0.01182 | PASS |
| tts-museum-female | 143 | 0.01072 | 0.01071 | 0.01065 | 0.01170 | PASS |
| tts-museum-male | 153 | 0.01066 | 0.01076 | 0.01086 | 0.01163 | PASS |

The rule (text tier 2a): native's median is at most the worst of HF's BF16 references.
- **Eager included:** native passes all six. HF eager is the farthest from FP32 in every case.
- **SDPA only:** the tool's first version left eager out. Under that reading native passes 4 of 6, and misses on
  ref-interleaved by 0.4% and tts-museum-female by 0.1%. All three BF16 paths sit about 1.1% from FP32, within a few
  tenths of a percent of each other: a tie at BF16 noise.
- Native vs HF's decode directly: a median 1.24-1.32% (the two paths' BF16 rounding differs).

The forced replays also reproduce the traced audio lengths exactly (372, 264, 40, 45, 142, 152 emitted frames).
Evidence: `replay.json`, `CASE.native.json` (each with its step trace: kind, decoded, the fed token or codes).
