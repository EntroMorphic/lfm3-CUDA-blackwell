# #3 audio in: speech input on LFM3-Omni-2B, natively

Tripp (2026-11-10): "Let's complete the Audio before we move to Images." The engine speaks; it now has to hear. This
round brings the audio encoder into the native engine and checks it against HF on speech recognition.

## What it is
- **Front end** (host): `lfm/wav.hpp` reads WAV (mono 16-bit PCM or 32-bit float, as soundfile reads them; a streamed
  data size runs to the end of the file); `lfm/base64.hpp` is a strict decoder; `lfm/audio_features.hpp` is the
  log-mel front end (Lfm3OmniFeatureExtractor) in double, with the token count per clip.
- **Encoder** (`lfm/audio_encoder.cuh`): the downsampler, the 17-layer LFM2 encoder and the projector, in FP32
  activations over the stored BF16 weights (`lfm/fp32_ops.cuh`); the output is rounded to BF16, the language model's
  input. Its checkpoint contract (`Encoder::check_contract`) is checked at worker startup from the config and the
  safetensors header (`lfm/safetensors_header.hpp`). Scratch is released after each clip.
- **Prompt:** `Runner::forward(prompt, injected, placeholder)` puts the encoded rows in place of the `<|audio|>`
  placeholders at prefill, as HF's masked_scatter does. The native renderer takes user audio parts
  (`lfm3_text::parse_messages(…, allow_audio)`) and expands each clip's `<|audio|>` to its token count
  (`lfm3_text::expand_audio`); `lfm3_speech::check_audio_spans` checks every span against its clip.
- **Session and driver:** `lfm3_speech::Options::audio`; every output mode works with audio input. `lfm3-speech`
  takes `"audio": [WAV paths]` with prompt_ids, or audio parts (paths) in messages; `"record_audio"` dumps every
  stage; `--batch LIST` loads the model once; with messages, its JSON adds the decoded `"text"` as the server decodes
  it.
- **Server** (`POST /speech`): user audio parts `{"type": "audio", "audio": base64 WAV}` (16 kHz mono, at most 64
  clips; the 4 MiB body holds about 90 s of 16-bit audio) and a new `"mode": "text"`. Audio input is not split. A
  checkpoint without the encoder this build implements gets 501 `audio_input_unsupported` while speech output keeps
  working. The worker refuses exactly what the HTTP layer refuses (shared `clip_from_base64` and
  `check_audio_spans`).

## Data
LibriSpeech test-clean (openslr.org/resources/12/test-clean.tar.gz, md5 `32fa31d27d2e1cad72775fee3f4849a9` as openslr
publishes it, sha256 `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`): 2,620 utterances, 40 speakers, about 5.4 hours, read English
speech. The FLAC files become 16-bit WAV (`tools/librispeech_cases.py`), the same samples. Prompt as the model card's
ASR example: system `Perform ASR in English.`, the clip as the user turn, `generation_mode="text"` (greedy).

## Rules, set before any run
1. **Features match HF.** On the 20 dumped clips (every 131st utterance), the native log-mel features are within 1e-4
   of HF's everywhere (max absolute difference).
2. **The encoder is as accurate as HF's.** On each of the 20 dumped clips, the native projected features' relative L2
   error to HF's FP32 encoder and projector (from the same features, TF32 off) is at most 1.25 times HF BF16's.
3. **Recognition is as good as HF's.** Over all 2,620 utterances, the corpus WER difference (native minus HF, in
   points, Whisper's English normalizer) has a paired-bootstrap 95% upper bound of at most +0.25.
4. **Coherency.** I read the transcripts of the dumped clips and of every utterance where the two engines differ in
   errors, and spoken questions answered in text and in speech.
5. **Refusals are clean.** A wrong sample rate, stereo, an empty clip, a clip/placeholder count mismatch, a span
   without its markers, audio with chunks, and image/video placeholders are refused before any state changes.

Speed is reported (encode seconds, end to end), not gated.

## Round 1 (build audioin-v1, encoder at HF's BF16 rounding points): FAIL on rules 1 and 2
On the 20 dumped clips (HF and native, same requests):
- **Rule 1 FAIL:** the features' max |native - HF| was 1.6e-4 (two clips over 1e-4). The native features are computed in
  FP64 and HF's in float32, so the rule cannot tell whose error it is; round 2 measures both against FP64 arithmetic.
- **Rule 2 FAIL, narrowly:** native/HF error ratio to the FP32 features had median 1.02 and range 0.83-1.252; one clip
  (4970-29095-0005) was over 1.25. The two engines were tied on accuracy, and the per-clip spread is BF16 noise.
- Recognition on these clips: 18 of 20 token-identical; WER 4.66% native against 4.41% HF (one word: "Raoul" became
  "Rao"; the other difference is capitalization only).

The change: the encoder now keeps FP32 activations throughout over the stored BF16 weights (`lfm/fp32_ops.cuh`, the
convention of the native codec and depth decoder), and rounds only its output, the language model's BF16 input. It
is the more accurate arithmetic, not a tuning, and costs little (the encoder is about 100M parameters).

## Round 2 rules (set before any round-2 run)
The 20 dumped clips are new, held out from round 1: every 131st utterance from the 66th (`cases[65::131]`), midway
between round 1's.
1. **Features are at least as exact as HF's.** On every clip, the native features' max absolute difference from the
   FP64 features (tools/lfm3_audio_in_hf.py `mel_fp64`) is at most HF's.
2. **The encoder is at least as accurate as HF's.** On every clip, the native projected features' relative L2 error
   to the FP32 model's is at most HF BF16's (ratio <= 1.0).
3. **Recognition is as good as HF's** (unchanged): over all 2,620 utterances, the WER difference's paired-bootstrap
   95% upper bound is at most +0.25 points.
4. and 5. unchanged.

## Round 3 rules (edge cases, set before the run; red-team finding 7)
`tools/audio_in_edge_cases.py`: clips of 1, 300, 399, 400, 401 and 1,681 samples (a single feature frame, the window
and hop edges, a 2-token clip), a 32-bit float WAV, two clips with text after them, and three clips of mixed lengths.
On every case, with all of a request's clips compared together: the run succeeds in both engines; the native features
are at least as close to the FP64 features as HF's (max absolute difference); and the native projected features'
relative error to the FP32 model's is at most HF BF16's.

## Results
### Round 2 (build audioin-v2): PASS
- **Rule 1 PASS:** on all 20 held-out clips the native features are closer to the FP64 features than HF's (max
  absolute difference at most 1.1e-4, HF's at most 4.4e-4). Round 1's 20 clips, run again as a diagnosis: the same, 20
  of 20.
- **Rule 2 PASS:** native/HF ratio of the projected features' error to the FP32 model's: 0.15-0.19 (median 0.18) on
  the held-out clips; round 1's clips 0.16-0.23. Before its final rounding the native output's error is below 5e-5;
  what remains (0.0017) is the BF16 rounding of the language model's input. HF's BF16 encoder: about 0.009.
- **Rule 3 PASS** (`round2-full-check.json`): all 2,620 utterances, 53,027 reference words.

  | | WER |
  |---|---|
  | HF (BF16, SDPA) | 2.606% |
  | native | 2.591% |
  | native - HF | -0.015 points, 95% CI [-0.038, +0.006] |

  Prompts identical 2,620 of 2,620; generated tokens identical 2,554; text identical 2,558. Of the 19 utterances
  where the error counts differ, native has fewer errors on 11 and more on 8.
- **Rule 4 (coherency):** I read all 19 differing utterances. Each is one hard word heard differently, in both
  directions ("Raoul" became "Rao" in native; HF wrote "motifs" for "motives" three times and "pirate" for "parrot").
  Spoken questions answered in text are grounded: the meal described in a clip, both speakers of two clips named, a
  2-second clip and its full version compared correctly.
- **Rule 5 PASS** (`refusals-render-check.json`, final build): 18 refusals, each with its own message and no outputs
  left: wrong rate, stereo, empty, 8-bit, not a WAV, missing file, too few or too many placeholders, a span without
  its markers, more clips than spans and the reverse, an image placeholder, audio with chunks or split, audio both in
  messages and in "audio", audio in an assistant message, over capacity, and an input clip that is also an output
  (found here: the driver would have deleted it).
- **Speed** (not gated; decode loop and end to end, locked clocks): native ASR decodes at 65 tokens/s (one stream);
  HF generate runs at 42 tokens/s including its encoder and prefill. Over the 5.4 hours of audio, native takes 1,360 s
  end to end (features and encoder 246 s, about 94 ms per clip, mostly the host DFT), HF's generate 1,621 s.

### Round 3, edge cases (final build audioin-v5): PASS
`round3-edge-check.json`: all 9 cases run in both engines; native features at least as close to FP64 as HF's on
every case (equal, at 0, on the 1-sample clip); error ratio 0.19-0.41; all 9 token-identical to HF, including the
two- and three-clip prompts and the float WAV.

### Server and regressions (final builds audioin-v5, server-audioin-v2)
- **Audio over HTTP** (`server-audio-in.json`, `tools/check_lfm3_audio_in_server.py`): on eight cases the reply's
  text and WAV equal lfm3-speech's byte for byte: three ASR clips, a question about a clip, two clips, a spoken question
  answered interleaved and in speech, and a 90-second recording (1,149 prompt tokens) transcribed in 7.1 s. The same
  request twice gives the same reply; a pinned /generate is unchanged; the server exits 0.
- **The native renderer equals HF's processor** on 2,624 conversations (`render-vs-hf-2624.txt`,
  `native/tests/audio_prompt_check.cpp`): all LibriSpeech requests and four multi-part conversations.
- **TTS is unchanged:** lfm3-speech's regression check, 36 of 36 outputs bit-identical to the validated runs and every
  refusal clean (`speech-regression.log`); the server's TTS WAVs identical on all five validated cases, worst cases
  144.5 s and 48.4 s (`server-tts-check.json`).
- **Text is unchanged:** the full text-server GPU gate passes (`text-gate.json`), control text sha256 `7ea05068…` as
  before.
- **The final build equals round 2's** on the 40 dumped clips: all 280 output files byte-identical, all 40 token
  sequences equal (`final-build-vs-round2.txt`). The red-team's changes did not touch the arithmetic.
- **Host tests:** 636 passed and 2 skipped (as before) over the HTTP, framing, chat, context, batch and cancellation
  gates, speech, text, and the new audio-input suites, on the plain, ASan+UBSan and TSan builds. The 3 warnings are
  from a shutdown test and occur without this change too. One test,
  `tests/test_speech_text.py::test_every_validation_prompt_matches_hf_ids`, cannot run on this workstation
  (transformers 4.57 cannot load the checkpoint's code); it fails the same way without this change.

### Speech-only replies to speech
Asked a question in speech with `generation_mode="audio"` (with or without a system prompt), the model answers with
unusable speech in HF as in the native engine ("Set, set.", "Zix. Zay. Lay.", echoes of the question). The model card's
voice chat uses interleaved, which answers well: the speech says what the text says. The server documents it; Tripp's
listening set is the interleaved answers.

## Red-team (one pass, an independent reviewer with the code only)
No memory-safety bug (libFuzzer with ASan+UBSan, 428k runs over the parsers and the front end), no request the HTTP
layer accepts that the worker refuses, and no divergence from HF in the front end, encoder, projector, rendering or
expansion. Found and fixed:
1. the round-2 clip selection was made by an inline script, not by the tool: `tools/librispeech_cases.py` now takes
   offsets and reproduces the run's cases exactly (all 40 dump flags included);
2. the encoder's contract was checked only on the first audio request (an incompatible checkpoint would have failed
   the server closed): now checked at startup, with 501 `audio_input_unsupported`, and before any session state;
3. the encoder kept up to about 0.5 GB of scratch after one long clip: released after each clip;
4. two comments were wrong (features "rounded to BF16"; the text-mode WAV is a 44-byte header, not empty);
5. text parts on either side of an audio part were checked as if adjacent (a false refusal): each run is checked
   alone;
6. WAV: a streamed data size is accepted, block align and byte rate are checked, and chunk ids in messages are made
   printable (found by the new mutation test);
7. tests: the parser, decoder and front-end host tests, the float WAV, the 64-clip cap (now in the server and worker
   too), the edge round, the renderer on every request, and the build's source list (base64, fp32_ops,
   safetensors_header).

Not changed: a hard link to an input under an output's name is still removed by the driver (paths are compared, not
inodes); the batch LIST file is not guarded.

## Files
`binaries.sha256` (audioin-v5's lfm3-speech; server-audioin-v2's server and worker), `sources-*.sha256` (identical to
this commit's files), the check records named above, and `round*-power-*.json`/`final-power-*.json` (MAXN, clocks
locked, unchanged across each run).
