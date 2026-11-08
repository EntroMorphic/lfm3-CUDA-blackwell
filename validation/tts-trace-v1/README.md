# #2 step 1: measured trace of the HF text → text + audio path, and its ASR baseline

`tools/thor/tts_trace_round.sh` (log `run.log`) on Thor (120 W): `tools/lfm3_audio_trace.py` runs the reference model
exactly as `test_interleaved.py` does (BF16, SDPA, `lfm3-omni-dev`: torch 2.10.0+cu130, transformers 5.22.1) with
instrumentation that changes no computation; `tools/lfm3_asr.py` transcribes each output with Whisper large-v3
(`download_asr_model.py`, pinned; weights' sha256 equal to HF's record). Quality bar: `journal/lfm3-tts-quality-bar.md`.

Per case: `summary.json` (everything below), `response.txt`, `response.wav` (24 kHz, 16-bit PCM, for listening).
The golden per-frame dumps (`frames.pt`: LM hidden state, the three depth passes' revealed masks, input codes and
BF16 logits, raw codes, Philox offsets), `sequences.pt` and the codec's exact FP32 output stay on Thor (6–54 MB
each); their hashes are in `thor-only-files.sha256.txt`.

## The reference reproduces itself exactly

`ref-interleaved` is `test_interleaved.py`'s case (seed 42). Its text and all 714,240 PCM samples equal the committed
`outputs/reflective-recursion/` bit for bit. The HF path is deterministic run to run on Thor, so golden dumps from
it are stable.

## Answers to the source map's open questions (`journal/lfm3-tts-source-map.md`)

1. **Attention.** All three components run SDPA in BF16.
   - Backbone: 11 attention layers (6 full, 5 sliding, window 256). Prefill is causal with no mask and GQA (11
     calls). Decode: 473 steps × 11 = 5,203 calls; 3,923 with no mask and GQA, and 1,280 (the 5 sliding layers on
     the 256 steps after the sequence passes the window) with an explicit boolean mask and KV heads expanded to 16.
   - Depth decoder: bidirectional, no mask, GQA 32/8, head dim 32; 373 frames × 3 passes × 3 layers = 3,357 calls.
     Kernel: `pytorch_flash::flash_fwd_kernel`.
   - Codec: 3 sliding layers over the whole sequence at once (not streaming), boolean mask, 8/8 heads, head dim 64.
     Kernel: CUTLASS memory-efficient attention (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`).
   - The model's causal conv (backbone and codec) is a **Triton kernel** (`_prefill`, `lfm3_conv.py`), JIT-compiled;
     Triton's bundled ptxas does not know sm_110a, so runs need `TRITON_PTXAS_PATH` (round 1 failed without it).
2. **Codec runtime dtype: BF16** (all 73.6M parameters; 2,688 FP32 buffer elements: the ISTFT window and rotary
   tables). The ISTFT runs a cuFFT kernel (`regular_fft<640…>`) and overlap-add by `col2im`.
3. **Codec window: 30 keys including the query itself** (29 past + self), causal: the first rows allow 1, 2, …, 30,
   then 30 for every later row.
4. **RNG: exactly 64 Philox offsets per frame**, in every frame of every case. How they divide between the `randperm`
   and the 12 multinomial draws was not measured. Constant consumption makes replay positions predictable; bitwise
   replay would still need PyTorch's exact use of them.
5. **EOS:** in each utterance, exactly one frame contains EOS codes, and only in some codebooks (4, 4, 4, 2, 1, 1 of
   12 in the six cases); every other frame has none. The whole-frame rewrite to 2016 is therefore always triggered
   by a partial EOS. A native engine must reproduce "any codebook" detection, not "all".
6. **Golden dumps:** recorded for every frame of every case (above). They allow checking the depth decoder (A1) and
   the codec (A2) with forced codes, no RNG involved.

**Cadence, measured** (`slot_runs`): interleaved is `T7 A12 (T6 A12)…`, then a short text run, then one long audio
tail (A180, A120) until the EOS frame, then the final text slot. The first text run is 7, consistent with the marker slot not
being counted in the cadence (as the source reads). TTS mode is `T1 A… T1`: the audio-start marker, the frames, one closing slot.

## ASR baseline of the HF reference (C1; `asr.json`)

| Case | Mode | Audio | WER | Remaining errors (normalized) |
|---|---|---|---|---|
| ref-interleaved | interleaved | 29.8 s | **0.000** (0/86) | — |
| interleaved-study | interleaved | 21.1 s | **0.000** (0/64) | — |
| tts-fox-female | TTS | 3.2 s | 0.111 (1/9) | "the" heard as "a" |
| tts-fox-male | TTS | 3.6 s | 0.000 (0/9) | — |
| tts-museum-female | TTS | 11.4 s | 0.040 (1/25) | "2024" heard as "1944" |
| tts-museum-male | TTS | 12.2 s | 0.000 (0/25) | — |

- In interleaved mode the speech says exactly what the text says (150 of 150 words).
- The two errors are both in the female voice, and both are the HF reference itself. Whether the model said them
  or Whisper misheard needs listening (Tripp's gate). They are part of the baseline a native engine is measured
  against, not defects to fix.
- **The measure was corrected twice during this run** (the WER lines in `run.log` are from the first version):
  typographic punctuation (curly apostrophes, U+2011) made identical words count as errors (interleaved 0.035 and
  0.016 → 0.000), and an ordinal kept as "3rd" against Whisper's "3" for spoken "third". Both folds are applied to
  both sides and tested (`tests/test_lfm3_asr.py`).
- This is one seed per case and six cases: a baseline for the method, not yet the C1 bar, which needs a fixed prompt
  set and several seeds to measure seed-to-seed variation.
