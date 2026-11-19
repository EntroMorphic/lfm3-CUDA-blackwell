# The audio encoder on the tensor cores: measure first, then keep FP32's accuracy

Written 2026-11-16, before any code. Nothing below has been run. The model is `validation/images-speed-v1/`.

## What it will be
The audio encoder (`lfm/audio_encoder.cuh`, audio-in-v1) runs every matrix multiply through `fp32::linear` (the
staged few-row CUDA-core kernel, `linear_staged`) and its window attention with one (query, head) per warp. This round
would move the multiplies onto the tensor cores with the vision encoder's split-BF16 method, keeping the FP32
residual stream, norms, short convolutions, softmax and nonlinearities, and keeping the encoder at least as accurate
as it is now. It only goes ahead if rule 0 says the encoder's GPU time is worth the work.

Out of scope: the host log-mel front end (`lfm/audio_features.hpp`), the language model, `lfm::gemm` itself, and
`tcgen05`/TMEM/TMA.

## Found before writing these rules (from the records and the code, not from new runs)
- **Most of the "encode" time is probably the host front end, not the GPU encoder.** `encode_seconds` in lfm3-speech
  times `log_mel` (host) and `Encoder::encode` (GPU) together. Across audio-in-v1's 2,620-utterance run
  (`round2-full-check.json`, audioin-v2, locked clocks), it is a flat **12.0–12.8 ms per second of audio** from 4.7 s
  to 31.6 s, with almost no fixed part, and it is 75–88% of `prefill_seconds`. Examples: 0.133 of 0.165 s at 10.8 s;
  1.09 of 1.24 s for the 90-second recording (`server-audio-in.json`). audio-in-v1 already noted "mostly the host
  DFT". `log_mel` is a direct DFT in double on one thread: 201 bins × 400 samples × 100 frames is about 16 M
  multiply-adds per second of audio. That alone is about 12 ms per second of audio on one core.
- **The GPU encoder's work is small.** Its multiplies hold about 115 M weights (encoder 106.2 M: 10 conv layers, 7
  attention layers and 17 SwiGLU blocks; downsampler 3.3 M; projector 5.8 M), so about 0.23 GFLOP per audio token
  (12.5 tokens per second of audio). That is about 29 GFLOP for a 10-second clip, plus about 1.3 GFLOP for the
  downsampler's 1×1 convolutions. At the CUDA-core kernel's measured efficiency in images-v1 (about 31% of FP32
  peak), that is on the order of 10 ms. That is an estimate; rule 0 measures it.
- **Attention does not matter here.** Each query sees 32 keys (head dim 96, 8 heads): about 9 MFLOP per layer at 375
  tokens, roughly 0.1% of the encoder's work. `attention_tc` (head dim 64, bidirectional, all keys) does not fit
  anyway. Attention stays on the CUDA cores unless rule 0 shows it at 10% or more of the encoder's GPU time; moving
  it would need an amendment first.
- **The front end runs on the engine thread.** In a batch, an audio request's `log_mel` and encode block every row's
  decode for their duration: about 12 ms per second of audio, so about 0.37 s for a 30-second clip. That matters for
  batch throughput, but the front end, not the encoder, is the larger part.

## Design (only if rule 0 says go)
- **Multiplies**: the vision encoder's `mul()`. `lfm::gemm` (`mma.sync` m16n8k16, BF16 in, FP32 accumulate, FP32
  out via `c_f32`) against the stored BF16 weights, then one kernel adding the bias where one exists (`lin_mid`,
  `lin_end`, `lin_out`, the projector). Variants:
  - `x1`: the FP32 activation rounded to BF16, one GEMM;
  - `x2`: `hi = bf16(x)`, `lo = bf16(x - hi)`, one GEMM over a doubled K: A = [hi | lo] per row against
    B = [W | W], with one FP32 accumulation (images-speed-v1 amendment 1).
- **Where**: every `f::linear` call: the downsampler's `lin_mid`, `lin_end` (256 → 256 over 32 and 16 rows per
  token) and `lin_out` (4096 → 768); the encoder's `in_proj`, `out_proj`, `q/k/v/out_proj` and `w1/w3/w2`; the
  projector's `linear_1` and `linear_2`. `conv_in`, the depthwise convolutions, RMSNorm/LayerNorm, rotary, the window
  attention, SwiGLU, GELU and the residual adds do not change. The output is rounded to BF16 as before.
- **Shared code, not copied**: `split_rows`, `double_rows` and `add_bias` move from `vision_encoder.cuh` to a shared
  header used by both encoders. The vision encoder's device code must not change. Check: `cuobjdump -sass` of
  lfm3-vision-check and lfm3-speech, vision kernels, identical to `vspeed-a5`/`vimg-a5`.
- **Memory**: the `[W | W]` cache is built on first use and kept. That is 2 × 115 M × 2 bytes ≈ **0.46 GB**, counting
  both halves (the images-speed-v1 red-team's correction). Scratch is still freed after every clip
  (audio-in-v1 red-team finding 3). If rule 0 or 2 shows allocation on the critical path, any change to that comes
  as an amendment.
- **Selectable**: `lfm3_audio_in::Precision { linear }` (`fp32`, `x1`, `x2`), as in the vision encoder, so the FP32
  path stays runnable and checkable (rule 6). The default is set by rule 1.
- **Ladder**: `x1` first, then `x2`. The fastest variant that passes rule 1 ships. No other variant is added without
  amending these rules first. A K-doubled GEMM is the only split form. An M-stacked split ([hi; lo] rows against W,
  halves summed) would avoid the 0.46 GB, but it is not a variant of this round unless Tripp asks for it before rule 1.

## Rules, set before any code
0. **Measure first** (Tripp runs it on a quiet Thor, MAXN, clocks locked, warm median of 5; encode-only and end to
   end labeled). Clips: short, `4446-2275-0004` (about 2.5 s); typical, `1089-134686-0000` (10.8 s); long, the
   longest test-clean utterance (about 35 s); and the 90-second joined recording (`asr-joined-90s`). Per clip:
   - native host `log_mel`, the GPU encoder alone (CUDA events around `Encoder::encode`), the language model's prefill
     and the time to first token, from the current build with the timing harness this round adds
     (`native/tests/audio_encoder_bench.cu`; it times the engine's own calls and changes no engine code);
   - HF's time to first token on the same requests (BF16, SDPA; the method of `tools/time_lfm3_vision_hf.py`);
   - nsys kernel shares inside the encoder: multiplies, attention and the rest.
   **The round goes ahead only if the GPU encoder is at least 15% of native time to first token at the typical or
   long length.** Otherwise the round stops, is recorded here as not worth doing, and the measured front-end share
   goes to Tripp as the case for a front-end round instead.
1. **Accuracy**, on audio-in-v1's 40 dumped clips (360 stage files; HF FP32 and BF16 dumps in
   `build/audioin-v2/dump`) and its 9 edge cases (93 files; `build/server-stream-v1-check/edge`), using
   `tools/check_lfm3_audio_in.py` plus a per-token cosine added to it (a tool change only). On every clip and case,
   with all of a request's clips compared together:
   - a. the projected features' (the BF16 output's) relative L2 error to HF FP32, divided by HF BF16's (the ratio of
     audio-in-v1 rule 2), is at most the FP32 path's ratio on the same clip **+ 0.01** (FP32 path: 0.15–0.23 on the
     dumped clips, 0.19–0.41 on the edge cases; the output's BF16 rounding, about 0.0017, dominates it);
   - b. the minimum per-token cosine of the projected features to HF FP32 is at least the FP32 path's on that clip
     **− 1e-6**;
   - c. the encoder stage (`encoded`, FP32, before the projector) has relative L2 error to HF FP32 of at most
     **0.01 × HF BF16's** on the same clip. FP32 path: about 7e-5 × (2.3e-7 against 3.1e-3). This is the gate that
     the output's BF16 rounding cannot hide. The downsampler stage is reported, not gated.
2. **Speed** (Tripp, quiet, as rule 0, on the shipped variant):
   - the GPU encoder alone (encode-only) is at least **2× faster** than the FP32 path at the typical, long and
     90-second lengths, and not slower at the short one;
   - time to first token (end to end) is not slower than the FP32 path at any length;
   - the first audio request's one-time `[W | W]` build is reported (cold start), and so are nsys's kernel shares and
     the time to first token against HF's.
   No bar against HF is set here: rule 0 decides whether the encoder can move that number at all.
3. **Benchmark**, LibriSpeech test-clean, all 2,620 utterances, one at a time (audio-in-v1 rule 3's tooling). The
   FP32 path's replies are `build/audioin-v2/native-full`, which audio-assurance-v1 rule 2 showed are the current
   build's, 2,620 of 2,620.
   - a. WER against HF BF16: paired-bootstrap 95% upper bound of (new − HF) at most **+0.25** points (unchanged);
   - b. WER against the FP32 path: paired-bootstrap 95% upper bound of (new − FP32 path) at most **+0.10** points;
   - c. agreement: normalized transcripts identical to the FP32 path's on at least **2,558 of 2,620** utterances,
     i.e. at least as often as HF BF16's transcripts agree with the FP32 path (audio-in-v1: 2,558).
   - d. the corpus in a batch (`--concurrent 8`, staggered): every output byte-identical to the new build's run alone
     (audio-assurance-v1 rule 3).
4. **Coherency**: I read every utterance whose transcript differs from the FP32 path's, and the new replies to the
   audio-in server cases (the spoken questions answered in text, interleaved and in speech; two clips; the 90-second
   recording). Changed spoken answers go to `outputs/audio-speed-v1/` for Tripp to listen to. Listening is the final
   gate for speech.
5. **Sanitizers**: memcheck (leak check on), racecheck, synccheck and initcheck on lfm3-speech with the 1-sample edge
   clip (one token: `lfm::gemm`'s M ≤ 8 path), a typical clip, the 90-second recording and a two-clip request.
   0 errors and outputs byte-identical to the plain run.
6. **Nothing else changes**:
   - with `Precision::linear = fp32`, the 360 + 93 audio-encoder files are byte-identical to audio-assurance-v1
     rule 8's;
   - lfm3-speech's regression (36 outputs), e2e and split (90 + 90);
   - images: the 31 images' 713 stage files and the vision kernels' SASS identical to images-speed-v1's;
   - the text gate (control `7ea05068...`);
   - the server checks (batch, stream, TTS, images): identical. The audio-in server check's replies must equal the new
     lfm3-speech's byte for byte (their content is judged by rules 3 and 4, not by identity with the old ones);
   - the host suites pass.

Failures are recorded here, not discarded. Every speed figure is labeled encode-only, decode loop or end to end.

## Before rule 0 (2026-11-16): the harness, and two clip lengths corrected
No rule changes. Rule 0's clips were mislabeled above. Their lengths from `build/audioin-v2/cases.json`:
- 4446-2275-0004 is 2.11 s, not about 2.5 s;
- 1089-134686-0000 is 10.4 s, not 10.8 s, and is longer than typical. The test-clean median is 5.79 s, so the clip
  nearest the median (121-127105-0007) is added and reported. The go/stop test stays at 10.4 s and 35 s as written.
- The longest clip is 4507-16021-0047 (34.96 s).

The harness:
- `native/tests/audio_encoder_bench.cu` (build: `tools/thor/build_audio_bench.sh`);
- `tools/time_lfm3_audio_hf.py`;
- `tools/thor/audio_speed_rule0.sh BENCH_TAG SPEECH_TAG OUT_TAG`: the bench, lfm3-speech, HF and host nsys, a
  summary with the go/stop verdict.

Built as `audio-bench-a1` (sha256 `274af23b...`). Checked functionally, not timed:
- the bench's `log_mel` equals the recorded audioin-v2 dump of 4446-2275-0004 byte for byte;
- the HF tool ran one pass on that clip.

A single smoke run (idle but shared GPU; **not** a measurement) showed the encoder at 15.8 ms on the 2.1 s clip. That
is more than the estimate above, so fixed per-clip costs (13 scratch allocations, zeroed and freed per clip) may weigh
at short lengths. Rule 0's run decides.

## Rule 0 results (2026-11-16, Tripp's quiet session): **STOP**, as the rule is written
Tripp ran `tools/thor/audio_speed_rule0.sh audio-bench-a1 vimg-a5 quiet-v1-audio` on a quiet Thor
(`build/quiet-v1-audio/summary.json`). The GPU encoder is under 15% of native time to first token at the typical
(10.4 s) and long (35 s) clips, so the round stops here. Recorded from Tripp's report; the full per-clip table is to
be copied from `summary.json` after the quiet window.

| clip | native time to first token | HF time to first token | native `log_mel` alone (front end only) |
|---|---|---|---|
| 10.4 s | 0.173 s | 0.051 s | 0.104 s |
| 35 s | 0.485 s | 0.072 s | 0.349 s |
| 90 s | 1.20 s | 0.141 s | 0.898 s |

HF's feature extractor takes 3–5 ms. The GPU encoder alone takes 0.166 s at 90 s (encode-only), more than HF's whole
time to first token. The rule's verdict stands. In absolute terms, native is 3–9× slower than HF to its first token:
- most of that is `log_mel` (audio-frontend-v1);
- what remains (encoder and the language model's prefill: about 0.07 s at 10.4 s and 0.30 s at 90 s) is still over
  HF's whole time to first token.

So this encoder round may return under audio-frontend-v1 amendment 1's bar (time to first token at least as fast as
HF's). It would need new rules; nothing here is overridden.
