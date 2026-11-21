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
