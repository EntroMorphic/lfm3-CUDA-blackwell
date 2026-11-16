# Changelog

Milestones, newest first. Each links the validation round that holds its evidence. Dates are when the result was
committed.

## Unreleased

- **Images in.** Native JPEG/PNG decoding and preprocessing bit-identical to HF's on 1,982 images, with a
  stricter-than-reference policy for damaged files; native vision encoder and projector more accurate than HF's BF16
  path on all 31 test images; image parts in `POST /speech` (bodies up to 16 MiB with images). POPE 89.06 vs HF 89.11;
  MMStar answers agree with the FP32 model's 94.8% of the time, HF BF16's 93.9%. ([images-v1](validation/images-v1/))

## 2026-11-14

- Text requests no longer wait for speech to drain: sliced text interleaved with speech steps.
  ([text-wait-v1](validation/text-wait-v1/))

## 2026-11-13

- Audio assurance before images: fuzzing, soak, sanitizers and evidence audit; allocator retention found and fixed.
  ([audio-assurance-v1](validation/audio-assurance-v1/))

## 2026-11-11

- Faster concurrent speech, every output bit-identical (1.24× at 8 rows). ([speech-speed-v1](validation/speech-speed-v1/))

## 2026-11-10

- Concurrent `/speech` requests batched on the GPU, each byte-identical to its reply alone.
  ([speech-batch-v1](validation/speech-batch-v1/))
- Streamed `/speech` replies: audio and text sent as the engine makes them. ([speech-stream-v1](validation/speech-stream-v1/))
- Speech normalization for numbers and abbreviations, chosen by a held-out test and blind listening.
  ([speech-normalize-v1](validation/speech-normalize-v1/))
- Audio input: native speech recognition and spoken questions (LibriSpeech WER 2.59% vs HF's 2.61%).
  ([audio-in-v1](validation/audio-in-v1/))

## 2026-11-09

- `POST /speech` in `liquid-lfm3-server`. ([speech-server-v1](validation/speech-server-v1/))
- Speech listening gate passed: native matches HF by ear on every case and beats it on long-form text.
  ([speech-e2e-v1](validation/speech-e2e-v1/), [speech-long-form-v1](validation/speech-long-form-v1/))

## 2026-11-08

- Native text in → text and speech out, with long-form TTS by sentence requests.
  ([speech-e2e-v1](validation/speech-e2e-v1/))
- Tree-verify decode: exact single-request speculation, up to 4.1×. ([tree-verify-v1](validation/tree-verify-v1/))
- Batched text decode: 1,754 tok/s at batch 32 (decode loop). ([batch-decode-v1](validation/batch-decode-v1/))

## 2026-11-07

- Native depth decoder, frame sampler and mode machine. ([depth-v1](validation/depth-v1/),
  [audio-control-v1](validation/audio-control-v1/))
- Native audio codec decoder. ([codec-v1](validation/codec-v1/))
- The default engine runs without cuBLAS (native BF16 GEMM) and passes every gate; per-token NLL 12.8–14.4% closer to
  an FP32 model than HF SDPA's. ([g4-gates-v2](validation/g4-gates-v2/))

## 2026-11-06

- `liquid-lfm3-server`: text serving in one executable. ([native-server-v1](validation/native-server-v1/))
- Native LFM3 text front end (tokenizer and chat template), bit-exact. ([g2-text-v1](validation/g2-text-v1/))

## 2026-11-04

- Text forward pass and generation bit-exact against Transformers eager (cuBLAS build, through 4,096 tokens); 32K
  bounded-memory attention.
  ([audit/FORWARD.md](audit/FORWARD.md), [audit/ACCEPTANCE.md](audit/ACCEPTANCE.md))
