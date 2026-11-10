# Is LFM3-Audio-2B a better reader of long text than Omni? (2026-11-09) No.

**Question (Tripp):** could a specialist audio model handle long-form speech output while Omni does perception?
The candidate is LFM3-Audio-2B, `LiquidAI/LFM3-Audio-2B` at revision `2d6c1f99`
(`../../download_lfm3_audio_2b.py`).

**What the cards and configs show.**
- **Not a TTS specialist.** LFM3-Audio-2B is Omni's audio sibling without vision: voice chat, ASR (including live),
  audio understanding, translation, tool calling and TTS.
- **Same context.** Both cards state a 32K-token context; both configs list `max_position_embeddings` 128,000.
- **Identical backbone:** 24 layers, the same 256-key sliding window at layers 4, 8, 12, 16 and 20 (the cause of
  Omni's long-form failure, `../speech-long-form-v1`), and the same depth decoder (3 layers, 12 codebooks x 2017).
- **Byte-identical** codec weights and tokenizer.
- **Same speech quality** on the vendor's own TTS score (UTMOSv2: 3.83 against Omni's 3.80).

The prediction was that it reads long text no better, unless training changed that.

**The test.** The rule, set before the data, is Phase B's (`../speech-split-v1`): it passes if mean WER <= 0.05 and no
run is above 0.3.
- **Material:** the 97-word lighthouse paragraph unsplit, seeds 1-15, HF BF16 SDPA, each model as its own card shows.
  - Audio-2B prefills `<|audio_start|>`; Omni writes `<think></think>`, then its audio mode adds the same marker.
  - The prompts are identical up to the assistant turn.
- **Tool:** `tools/lfm3_audio2b_tts_hf.py`, run by `tools/thor/speech_audio2b_round.sh` on Thor in MAXN with clocks
  locked (`power.json`).
- **Two sampling arms:**
  - **eval:** the settings both cards evaluate with, which are Omni's (temperature 0.4, top-k 0, penalty 1.2, window 64);
  - **card:** Audio-2B's recommended speech settings (1.0, top-k 128, penalty 1.1, window 4096).
- **ASR:** Whisper large-v3 with compound folding (`asr/`).

| unsplit, 15 seeds | WER mean | median | max | runs > 0.3 | vs Omni BF16 (95% CI), Mann-Whitney p | first-error onsets (word) |
|---|---|---|---|---|---|---|
| Omni, HF BF16 | 0.163 | 0.134 | 0.39 | 1 | — | 32, 41x2, 46x2, 48x2, 51, 64, 65x2, 72x2, 80x2 |
| Omni, HF FP32 | 0.433 | 0.258 | 2.09 | 6 | +0.270 (+0.076, +0.551), 0.005 | 39, 40x2, 41x2, 46, 47x2, 48, 49, 63x2, 64, 65x2 |
| **Audio-2B, eval** | **0.195** | 0.175 | 0.47 | 4 | +0.032 (-0.053, +0.119), 0.755 | 39, 40, 46, 63x2, 64, 65x5, 71, 72x2 |
| **Audio-2B, card** | **0.209** | 0.216 | 0.38 | 2 | +0.046 (-0.020, +0.109), 0.170 | 46x3, 47x2, 48x2, 63x2, 64, 65x3, 72, 76 |
| Omni native split | 0.000 | 0.000 | 0.00 | 0 | -0.163 (-0.214, -0.116), <0.001 | none |

(The table is produced by `tools/lfm3_speech_long_form_report.py`'s `table()` over the committed ASR files. The Omni
rows are `../speech-e2e-v1` and `../speech-long-form-v1`.)

**Verdict: FAIL in both arms.**
- Audio-2B reads the paragraph no better than Omni (p = 0.76 and 0.17), and both arms fail the rule.
- Its errors begin at the same words (40-48 and 63-65), where the window's trailing edge passes the text being read.
  It is the same mechanism.
- Its recommended sampling, with a longer repetition window, does not help.
- No run hit the token cap: the failures are skips and garbles, as with Omni.

The sentence split remains the long-form fix for either model. Moving speech output to Audio-2B buys nothing for
long-form. Its measured advantages (VoiceBench 73.1 against 64.7; lower ASR error) are on audio input.
