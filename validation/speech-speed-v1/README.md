# Faster concurrent speech, every output bit-identical

Tripp (2026-11-10): "Make concurrency faster next, keep it bit-identical." speech-batch-v1 batched /speech on the GPU
(8 rows: 13.15 audio seconds per wall second; its rule 2, 4 times one row's, failed at 3.60).

## Where an 8-row step goes today (lfm3-speech batch-a4, 8 TTS requests, unprofiled)
44.9 ms a step: the language model's step 23.4 ms (16.3 ms at one row), the depth decoder 14.8 ms, host sampling
1.5 ms, codecs 3.5 ms (0.44 ms per row). Under nsys (steady state): the GPU is busy 89% of the time; the depth
decoder's FP32 linears are 31% of the GPU's time and run at about 1.7 TFLOPS (Thor's FP32 FMA peak measured here: 7.6);
the language model's head (the 65536-wide vocabulary projection) is 2.8 ms a step, computed for every row on every step
though only rows on a text step read it.

## Changes (written before the code)
1. **The language model's head on demand.** The speech batch's decoder defers the head to the first read of the logits
   (a text pick, or a row's load, which overwrites its row after the deferred head is computed). A step on which no row
   picks text skips it. When computed, it is the same GEMM over the same rows and inputs as before.
2. **Codecs off the critical path.** Each row's codec runs on its own non-blocking CUDA stream: a chunk is submitted
   (the same kernels in the same order) and its samples collected at a later step boundary, so the codec overlaps the
   next language-model step. Audio is delivered (streaming) when collected: up to one step later than today.
   *As built:* the chunk is submitted in the frame loop and collected at the end of the same step, after the
   language-model step it overlaps; so each row's audio still follows its slot within the step, and streamed audio is
   not delayed.
3. **Host sampling in parallel** across the rows (each row's sampler is its own), the depth logits copied into pinned
   memory.
4. **The FP32 linear kernel**, if a faster one computes every output as the same sum in the same order (lane l's chain
   over k = 256 j + 8 l + e, FMA from +0, then the warp's shuffle tree), checked bit for bit against fp32::linear on
   every shape the depth decoder and codec use. *As built:* fp32::linear is shared, so the audio encoder (speech
   input) takes it too; its shapes are in the check as well.

## Rules (set before any measured run of the changed engine)
Thor, MAXN, clocks locked, power recorded. Reference binaries: lfm3-speech batch-a4 and liquid-lfm3-server batch-s4
(speech-batch-v1), rebuilt from nothing.
1. **Exact.** Every output equals the reference's, byte for byte:
   - lfm3-speech's regression check, 36 of 36;
   - the engine check (check_lfm3_speech_concurrent.py): 41 requests alone and at 4, 8, 8 staggered and 16 staggered
     rows, every file identical to batch-a4's alone;
   - the server check (check_lfm3_speech_batch_server.py): the 42 cases in crowds, unstreamed and streamed, at once and
     in random order, equal to the single-request server's replies.
2. **Faster.** Rule 2's measurement of speech-batch-v1 (decode loop, aggregate audio seconds per wall second, the 21
   TTS requests looped, N = 1, 2, 4, 8, 16), run for the new engine and batch-a4 in the same session, alternating.
   Pass: at N = 8, the new engine's median at least 1.25 times batch-a4's. Also reported: every N, and the ratio of
   N = 8 to N = 1 against speech-batch-v1's bar of 4.
3. **No slower alone:** at N = 1 the new engine's median is at least 0.98 times batch-a4's.
4. **Streams stay real time:** no player runs dry at 8, 12 or 16 concurrent streams; first audio under load (a TTS
   request arriving while 7 stream) at most 1.0 s median of 10.
5. **Nothing else moves:** the server's streaming, TTS, audio-input checks and the text gate pass; host tests pass.

## Results (lfm3-speech speed-a4, server speed-s2; records in `build-final2/`)
Thor at MAXN, GPU clocks locked at 1.575 GHz at the start and the end. The reference binaries were rebuilt from
nothing and their hashes match speech-batch-v1's (`binaries.sha256`). The sources measured are the ones committed
here (`sources.sha256`, the same on Thor).

**Rule 1, exact: pass.**
- FP32 linear check: the new kernel equals fp32::linear's original on all 53 shapes (depth, codec, audio encoder,
  ragged and misaligned).
- Capacity check identical. lfm3-speech's regression check: 36 of 36 the same.
- The engine check: 41 of 41 requests alone identical to batch-a4's alone, and at 4, 8, 8 staggered and 16 staggered
  rows identical to alone.
- The server check: the 42 cases in all four crowds (unstreamed and streamed, at once and random) identical to the
  single-request server's replies.

**Rule 2, faster (decode loop, aggregate audio seconds per wall second, median of 4 runs each, alternating): fail.**

| N | batch-a4 | speed-a4 | ratio |
|---|---|---|---|
| 1 | 3.678 | 4.404 | 1.197 |
| 2 | 6.550 | 7.810 | 1.192 |
| 4 | 10.698 | 13.113 | 1.226 |
| 8 | 13.516 | 16.813 | **1.244** (bar 1.25) |
| 16 | 17.670 | 22.241 | 1.259 |

All runs were tight (the largest spread at N = 8 was 0.4%), so 1.244 is short of the bar, not noise. N = 8 to N = 1:
3.82 (batch-a4: 3.67; speech-batch-v1's bar was 4). Most of the gain came at every N, not just in crowds: the FP32
kernel and the codec overlap help a single row as much.

**Rule 3, no slower alone: pass** (1.197 at N = 1).

**Rule 4, streams real time: pass.** No stalls at 8, 12 or 16 streams, nor with 7 TTS streams while 5 spoken
questions join. First audio under load: median 0.376 s (batch-s4: 0.445).

**Rule 5, nothing else moves: pass.** The streaming check, the audio-input and TTS server checks and the text gate
pass. Host tests: 888 passed, 2 skipped; the one failure is the known transformers-version test. Text under speech
load: 26.6 s against an 11.3 s drain (batch-s4: 34.1 and 15.0), with identical ids.

**Where an 8-row step goes now:** about 37 ms (from 44.9): the language model 22 ms, the depth decoder 12.8 ms. The
language model's step, 16.3 ms at one row and 22 ms at eight, is now the larger share and the next lever.
