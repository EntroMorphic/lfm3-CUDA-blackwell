# #2 step 2: native codec decoder (A2): PASS

The native codec decoder is `native/include/lfm/codec.cuh`; `liquid-codec` (`native/src/codec_main.cu`) runs it on a
file of codes. It is checked against HF anchors decoded from the reference's own codes (forced codes, no RNG), for
the six cases of `validation/tts-trace-v1`. Quality bar: `journal/lfm3-tts-quality-bar.md`, A2. Runs on Thor (120 W):

- `tools/thor/codec_anchor_round.sh` → `tools/lfm3_codec_anchor.py`; output in `anchor/`.
- `tools/thor/codec_round.sh` → `tools/lfm3_codec_check.py`; output in `native/` and `run.log`.

The waveforms, the intermediates and the binary stay on Thor; their hashes are in `thor-only-files.sha256.txt`.

## Design

- **FP32 activations; BF16 weights used at their stored values** (exact in FP32).
- **Rotary angles and the polar step in FP64.**
- **Inverse FFT:** a direct DFT against a [1280 × 1282] basis built in FP64 with exact argument reduction, the
  synthesis window folded in. Like `irfft`, it ignores the imaginary parts of bins 0 and 640.
- **One tiled FP32 GEMM** for every linear layer and for the DFT.
- **Attention:** a warp per (row, head) over the 30-key window.
- **Overlap-add:** streaming, with a 960-sample carried tail.
- **Links neither cuBLAS nor cuFFT** (checked by `readelf`).
- **Streaming:** each call takes any number of frames. Every kernel computes a row the same way whatever the
  chunk, so a streamed decode is meant to equal a whole one bit for bit; that is gated below.

## The anchors (`anchor/anchor.json`)

| Case | Frames | HF FP32 | HF BF16 SDPA (shipped) | HF BF16 eager | HF BF16 streamed |
|---|---|---|---|---|---|
| interleaved-study | 264 | 102.2 | 31.7 | 31.0 | 34.7 |
| ref-interleaved | 372 | 101.2 | 32.2 | 33.2 | 30.5 |
| tts-fox-female | 40 | 113.6 | 31.0 | 31.4 | 31.5 |
| tts-fox-male | 45 | 109.9 | 25.5 | 24.1 | 27.4 |
| tts-museum-female | 142 | 102.3 | 31.9 | 33.4 | 34.4 |
| tts-museum-male | 152 | 99.6 | 25.6 | 28.7 | 24.3 |

SNR in dB against FP64. FP64 runs on the CPU, with the model's FP32 islands widened to FP64: RMSNorm, the rotary
tables and their application, and the ISTFT.

Each anchor run first checks itself:
- HF BF16 reproduces the trace's waveform bit for bit.
- The step-by-step decode copy equals `codec.decode` bit for bit.
- The FP64 widening leaves BF16 unchanged.
- Eager really differs from SDPA.

Findings:
- **FP32 is a sound anchor**: 100–114 dB from FP64.
- **The shipped BF16 codec is only 24–33 dB** from FP64. The male voice is the worst (max abs error up to 0.24 at
  peak 0.6). Its largest single rounding is probably the BF16 log-magnitudes before `exp` (tts-trace-v1, Q2).
- **HF's streaming decode is not bit-identical to its whole-sequence decode**: 32–36 dB apart, though its docstring
  says the chunks "decode to the same audio". That holds only to BF16 noise.

## Native against the anchors (`native/a2.json`)

| Case | Native vs FP64 | Native vs FP32 | Bar (best HF BF16 vs FP32) | Streamed 12 = whole | Streamed 1 = whole | A2 |
|---|---|---|---|---|---|---|
| interleaved-study | 109.6 | 100.7 | 31.7 | bitwise | bitwise | PASS |
| ref-interleaved | 107.8 | 99.3 | 33.2 | bitwise | bitwise | PASS |
| tts-fox-female | 108.3 | 109.2 | 31.4 | bitwise | bitwise | PASS |
| tts-fox-male | 101.8 | 104.2 | 25.5 | bitwise | bitwise | PASS |
| tts-museum-female | 106.5 | 107.2 | 33.4 | bitwise | bitwise | PASS |
| tts-museum-male | 96.6 | 104.1 | 28.7 | bitwise | bitwise | PASS |

- **Native is at FP32 quality**: 97–110 dB from FP64, against HF FP32's 100–114. That is **71–78 dB above HF's
  shipped BF16**. The bar (native ≥ the better of HF BF16 SDPA and eager) is met by a margin of 66–79 dB.
- **Streamed decodes equal the whole decode bit for bit** in 12-frame and 1-frame chunks. HF's do not.
- **Stage by stage** (`native/stages.txt`, tts-fox-female, dB against FP64):

  | Stage | Native | HF FP32 | HF BF16 |
  |---|---|---|---|
  | Latents to `lin` | 120.6–140.3 | 125.5–140.3 | 41.7–53.5 |

  Native sits 0–7.5 dB below HF FP32 at these stages, probably because the GEMM sums K sequentially where cuBLAS
  sums in blocks (not profiled). At the waveform native is within 8 dB of HF FP32 either way: from 8.1 dB below (tts-fox-male) to 7.4 dB above
  (interleaved-study).
- **Coherency: the native audio says the same words.** Whisper large-v3, run the same way as in tts-trace-v1, gives
  the same normalized transcript as HF for all six cases (`native/asr.json`).
  - Interleaved: 0/150 errors.
  - The same two female-voice TTS errors as HF ("the" heard as "a", "2024" heard as "1944"). They come from the
    codes, not the decoder.
- WAVs for listening: `native/<case>/response.wav`.

## Cost, unoptimized (`native/runs.jsonl`)

**Speed:**
- Whole sequence: 29.8 s of audio in 90 ms (330× real time).
- 12-frame chunks (the interleaved cadence): 383 ms (78×).
- 1-frame chunks: 4.5 s (6.6×). Each call costs about 12 ms whatever its size. Profiled in the red-team below: the
  time is GPU time in the linear kernel at 6 rows, not copies or launches.

**Device memory:**
- 154.3 MB for streaming. 147 MB of that is the codec checkpoint as uploaded, **encoder included** (about half),
  plus the 6.6 MB DFT basis.
- A whole-sequence decode of 372 frames adds about 108 MB of row buffers.

## Open

- Not yet in the single binary; the codec is still a separate tool.
- The checkpoint upload includes the encoder, which text → audio does not need. Uploading only the decoder would
  save about 74 MB.
- 1-frame streaming is slow in the linear kernel at few rows (red-team). The fix is a small-row GEMM, which the
  depth decoder needs too.
- C3 (listening) is still Tripp's.

## Red-team (2026-11-07, `redteam/`)

`tools/thor/codec_redteam.sh` rebuilt the decoder with a self-test added to `liquid-codec`.

What held:
- **Output unchanged.** All six whole decodes equal codec-v2's bit for bit.
- **Untested stream paths.** Reset and re-decode, 7- and 5-frame chunks (the last chunk partial), finalizing with
  a separate empty call, and a whole decode after a reset all equal the first decode bit for bit, for all six
  cases.
- **Refused inputs.** The EOS code 2016, negative codes, a file that is not whole frames, an empty file, and chunk
  sizes `-1`, `x` and `3x` are all refused.
- **The FP64 anchor's widening takes effect** (`widening.json`). Without the patches the FP64 decode differs, but
  only at 123 dB. The patched decode equals the saved anchor. Native is 108.3 dB from either, so the FP32 islands
  sit below native's own error and the verdict does not depend on them.

What was wrong:
- **The cause of the slow 1-frame streaming was guessed wrongly.** nsys, tts-fox-female in 1-frame chunks:

  | Where | Share | Detail |
  |---|---|---|
  | GPU, linear kernel (BF16 weights) | 92% of kernel time | 2,080 calls at 195 µs each, about 10 ms of each 12 ms call |
  | Host, `cudaMemcpy` | — | mostly waiting for those kernels |
  | Host, kernel launches | 35 ms total | 6.7 µs each |

  At 6 rows the 64×64-tiled kernel has 8 to 36 thread blocks for 20 SMs. Each block streams its weight tile with
  32-byte reads, so a call stays far above the weight-bandwidth floor (72 MB of weights, about 0.4 ms).
- **The fix:** a small-row kernel, one warp per output column over a group of rows. It keeps a reduction order
  that does not depend on the row count, so stream equals whole bit for bit still holds by construction. The
  depth decoder (step 3) runs at 12 rows, so the kernel is built there and the codec switched to it.
