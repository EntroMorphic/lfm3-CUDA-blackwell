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
     calls). Decode: 473 steps × 11 = 5,203 calls; 3,923 with no mask and GQA, and 1,280 = 5 sliding layers × 256
     steps with an explicit boolean mask and KV heads expanded to 16. The masked calls begin at the decode step whose
     key count reaches 255, in both cases that get that far (prompt 37 + 217 unmasked steps; prompt 33 + 221). The
     decode masks' contents were not recorded (the trace keeps only the first multi-query mask per component). The
     backbone is the text engine's, already gated.)
   - Depth decoder: bidirectional, no mask, GQA 32/8, head dim 32; 373 frames × 3 passes × 3 layers = 3,357 calls.
     Kernel: `pytorch_flash::flash_fwd_kernel`.
   - Codec: 3 sliding layers over the whole sequence at once (not streaming), boolean mask, 8/8 heads, head dim 64.
     Kernel: CUTLASS memory-efficient attention (`fmha_cutlassF_bf16_aligned_64x64_rf_sm80`).
   - The model's causal conv (backbone and codec) is a **Triton kernel** (`_prefill`, `lfm3_conv.py`), JIT-compiled;
     Triton's bundled ptxas does not know sm_110a, so runs need `TRITON_PTXAS_PATH` (round 1 failed without it).
2. **Codec runtime dtype: BF16** for every parameter (73.6M, encoder included). It also has 2,688 FP32 buffer
   elements, not itemized; they include the 1,280-sample synthesis window. The trunk and `lin` run in BF16. So the
   641 log-magnitudes and phases are **rounded to BF16 before** `exp`/`polar` (FP32 from there on). This is
   probably the codec's largest BF16 error, and it matters for A2. The ISTFT runs a cuFFT kernel
   (`regular_fft<640…>`) and overlap-add by `col2im`. The output waveform is FP32: `summary.json`'s
   `waveform.dtype_before_float` is the codec's *parameter* dtype, a misnamed field (renamed for later runs).
3. **Codec window: 30 keys including the query itself** (29 past + self), causal: the first rows allow 1, 2, …, 30,
   then 30 for every later row.
4. **RNG: exactly 64 Philox offsets per frame**, in every frame of every case. How they divide between the `randperm`
   and the 12 multinomial draws was not measured. Constant consumption makes replay positions predictable; bitwise
   replay would still need PyTorch's exact use of them.
5. **EOS** (half answered). Generation stops at the first frame with any EOS code, so "one EOS frame per
   utterance" holds by construction and is not a finding. The finding is that the EOS frame is **always partial**
   (4, 4, 4, 2, 1, 1 of 12 codebooks in the six cases). So the whole-frame rewrite to 2016 is always triggered by
   a partial EOS, and an "all codebooks" rule would never stop. The EOS frame is not decoded (`audio_frames` =
   sampled frames − 1). The other half of the question, how stable EOS placement is under a native LM with forced
   codes, needs the native depth decoder (step 3).
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
- All six clips are shorter than 30 s. Whisper's long-form path in `lfm3_asr.py` is therefore **not yet exercised**;
  the C1 prompt set must include a longer clip.

## Red-team (2026-11-07)

What was checked:
- **No attention call escaped the patch.** The counts close: 5,203 = 473 × 11, 3,357 = 373 × 9, and the codec's 3
  calls equal its 3 attention layers.
- **The RNG is CUDA only.** `randperm` runs on the hidden state's device and `multinomial` on CUDA logits. The CPU
  generator plays no part, so the CUDA offset is the whole RNG state of a frame.
- **The sampling settings are the model card's.** `audio_temperature=0.4` and `audio_repetition_penalty=1.2` are
  given for speech, TTS and interleaved alike, and `generation_config.json` holds the same values plus
  `audio_repetition_window=64`.
- **Determinism rests on two runs** (this one and the committed output), on the same machine and container.

What was corrected above:
- Q5's "one EOS frame" holds by construction, and the question was half open.
- The backbone's masked decode calls were an interpretation; they are now stated as measured onset counts.
- A field name made the waveform's dtype look like BF16; it is FP32.
- The BF16 rounding of the codec's log-magnitudes is now recorded.
- The ASR long-form path is untested.
