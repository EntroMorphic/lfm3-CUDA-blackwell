# audio-ttft-v1: audio in, time to first token at least as fast as HF's

## Why
Audio in is the largest measured gap against HF in what ships.
- audio-frontend-v1 amended rule 2 (quiet window 3, 2026-11-16) failed as recorded. F2's time to first token was 0.090 / 0.204 / 0.477 s at 10.4 / 35 / 90 s, against HF's 0.050 / 0.072 / 0.141 s, so 1.8–3.4× HF's.
- That native figure came from `lfm3-speech` (prefill plus one mean decode step, the request pre-tokenized), not from the server a client talks to.
- Since then the server computes features on the handler threads (audio-frontend-v1) and hands them to the engine by handle (audio-handles-v1). The server deployed on 2026-11-20 (`a681eab4c040`) has both.

The bar is audio-frontend-v1 amendment 1's, unchanged: time to first token at most HF's at the 10.4 s, 35 s and 90 s clips.

## Stage 0: measure first (set 2026-11-20, before any data; reported, not gated)
One quiet window on Thor (MAXN, the always-on service stopped, the 1 s sampler and the power state checked before every run as in audio-contention-v1). Clips: rule 0's five, which are short (2.11 s), the test-clean median, typical (10.4 s), long (34.96 s) and the 90 s joined recording.

**Measurements:**
1. **HF BF16** (`tools/time_lfm3_audio_hf.py`, audio-speed-v1's reference configuration): time to first token, from reading the file to the first new token, plus its stages (the processor, the feature extractor alone, the encoder and projector alone). 1 untimed pass, then 5 timed passes.
2. **The native server, as deployed** (`merge-srv`, sha256 `a681eab4…`, run fresh in `lfm3-profile-v1`), one request at a time (`tools/measure_lfm3_audio_ttft.py`, new). A streamed `/speech` request in text mode ("Perform ASR in English.", the clip, 32 new tokens), with two times:
   - **time to first token from the file:** from reading the WAV, base64 encoding and building the JSON included, to the first `text` event. This is the counterpart of HF's figure.
   - **from the request's first byte** to the first `text` event.

   The `done` event's `prefill_seconds` is recorded. 1 untimed pass, then 5 timed passes.
3. **The native stages, alone** (`audio-encoder-bench` built from the merged tree, F2 at 4 threads): `log_mel` alone (front end only) and the encoder alone (encode-only). 1 untimed run, then 5 timed.

**Figures:**
- the median of the 5 timed runs per clip, for each measurement;
- native ÷ HF per clip;
- the native budget: time to first token − `log_mel` − encoder = everything else (transport, parsing, waiting, the language model's prefill, the first step).

**What it decides.** The rules for any change are set after stage 0, before any code, from where the time goes. Candidates already known:
- the encoder on the tensor cores, as images-speed-v1 did for the vision encoder;
- `log_mel` on the GPU, which must keep audio-frontend-v1's correctly rounded features or be Tripp's call;
- overlapping the features with the encoder.

**Void if:** the sampler sees a live foreign process, the power state changes, or any request fails.

## Stage 0: results (2026-11-20, Thor, 21:28–21:30; `build/ttft-s0/`)
The window was clean:
- the sampler took 89 samples and saw no foreign process;
- the power state was the same before every step and at the end;
- every step exited 0;
- the server's two passes, at the start (A) and the end (D), agree to within 1.2 ms;
- each clip's transcript was identical across all passes.

Binaries: the server `a681eab4…` (deployed) and the bench `75f6b043…` (built from the merged tree).

**Time to first token, the median of 5 timed runs** (native: the mean of A and D, from reading the file):

| clip | audio | native (server) | HF | native ÷ HF |
|---|---|---|---|---|
| short | 2.11 s | 0.0451 s | 0.0475 s | 0.95 |
| median | 5.79 s | 0.0621 s | 0.0510 s | 1.22 |
| typical | 10.44 s | 0.0897 s | 0.0609 s | **1.47** |
| long | 34.95 s | 0.2280 s | 0.0800 s | **2.85** |
| 90 s | 90.00 s | 0.5433 s | 0.1512 s | **3.59** |

The bar (at most HF's at 10.4, 35 and 90 s) fails at all three, as expected. The short clip is already faster than HF.

**Where the time goes, per stage** (native: the bench alone and the server's `prefill_seconds`; HF: its tool's separate calls):

| clip | `log_mel` native / HF features | encoder native / HF | language model, native prefill − encoder | HF, generate to the first token |
|---|---|---|---|---|
| typical | 0.0199 / 0.0045 s | 0.0227 / 0.0113 s | 0.0379 s | 0.0528 s |
| long | 0.0667 / 0.0074 s | 0.0663 / 0.0125 s | 0.0641 s | 0.0682 s |
| 90 s | 0.1711 / 0.0150 s | 0.1663 / 0.0280 s | 0.1384 s | 0.1271 s |

HF's "generate to the first token" includes its own encoder pass, its language model's prefill and the first step. HF's encoder alone is in the encoder column.

**Reading (derived figures, labelled as such):**
- **The encoder is 6× HF's at 90 s** (0.166 against 0.028 s) and 5× at 35 s. This is the largest single gap and the most certain lever: the vision encoder made the same move (images-speed-v1: 1.87 to 0.389 s on the tensor cores).
- **`log_mel` is 11× HF's feature extractor at 90 s** (0.171 against 0.015 s). This is the correctly rounded F2 on 4 CPU threads; HF's is float32 numpy.
- **The language model's prefill:**
  - At 90 s, ours is about 0.138 s. HF's prefill plus first step is about 0.099 s (0.1271 − 0.0280, derived).
  - So our prefill alone is already about 40% over HF's prefill plus its first step.
  - The prefill GEMMs are what tcgen05-gemm-v1 speeds up (2.38× `lfm::gemm` at large M), held OFF by its rule 4b.
- **What is left:** at 90 s, about 0.056 s between the first byte and the first token, beyond `log_mel` and the server's prefill (0.532 − 0.171 − 0.305, derived). That is the first decode step (about 0.025 s) plus the handler's own parse and decode of the 3.8 MB request, which audio-handles-v1 moved off the engine thread but which is still in this path.
- **At 90 s, closing the bar needs all three.** Even with the encoder at HF's 0.028 s, `log_mel` at 0.015 s and nothing else changed, time to first token would be about 0.24 s against HF's 0.151 s. The language model's prefill and the remaining parse also have to come down. At 10.4 s, the encoder and `log_mel` together would be enough: about 0.027 s to save, from 0.043 s of those two stages.
