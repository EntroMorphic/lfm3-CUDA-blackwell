# pinned-uploads-v1: stage the engine's large uploads through pinned memory

**Why.** alloc-effect-v1 (stages 0–4, branch `alloc-effect-v1`) found the cause of the state effect:
- A host-to-device `cudaMemcpy` of 1 MiB or more from **pageable** memory makes the GPU's memory reads about half as fast. A streaming read is 2.3× slower.
- The state lasts until the GPU is idle for 1 ms or more.
- The same copy from **pinned** memory does nothing.
- Under the deployed dispatch it slows the language model's `w1w3` and `w2` GEMMs by:
  - 4.3× and 3.0× at 32 rows;
  - 1.7× and 1.8× at 128 rows;
  - 2.0–2.7× for `w2` at 1,149 rows on tcgen05.
- While serving, the engine makes two such copies:
  - the audio encoder's mel features (`audio_encoder.cuh`), 4.6 MB for a 90 s clip and 1 MiB or more from about 20 s;
  - the vision encoder's pixel values (`vision_encoder.cuh`), megabytes per image.
- The encoder and the prefill run after them with no idle gap.

Tripp's go: item 2 of the remaining list ("the allocation effect"), 2026-11-22.

## The change (planned)
- `lfm::PinnedStaging` (`native/include/lfm/pinned_staging.hpp`): a 4 MiB pinned host buffer (`cudaMallocHost`), allocated on first use and kept for the owner's life.
- `upload(dst, src, bytes)` copies through it in chunks of at most 4 MiB: `std::memcpy` into the buffer, then `cudaMemcpy` from it. A host-to-device `cudaMemcpy` from pinned memory returns only when the copy is done, so the buffer is free for the next chunk.
- The audio encoder's feature upload and the vision encoder's pixel upload use it; each encoder owns one.
- The bytes reaching the device are the same, so every output must be byte-identical.
- Nothing else changes. The weights' upload at startup stays pageable: warm-up and idle follow it.

## Rules (set 2026-11-22, before any code)
1. **Byte identity.** The tcgen05-v2 rule-2 suite on the candidate (CMake build, tcgen05 ON, as deployed) gives outputs identical to main's:
   - TTS regression;
   - dump and edge (rule-6 files);
   - e2e and split;
   - the vision stage files;
   - the audio-in server;
   - 8 and 32 at once;
   - the image server and the image cache;
   - TTS;
   - the text gate.
2. **Speed, quiet Thor, ABAB** (main `f19f4ee40708`'s build A, the candidate B; the service stopped; the samplers):
   - **Audio:** `measure_lfm3_audio_ttft.py` on tcgen05-v2's five clips (short, median, typical, long, joined90), 1 untimed + 5 timed per process.
   - **Images:** `measure_lfm3_image_turns.py` on `cherry` and `two` (4 conversations, the first discarded). The measure is the time to the first text of turn 1, the turn that encodes.
   - **Pass:**
     - for every clip and image, max(B) ≤ mean(A) × 1.01;
     - and for joined90, mean(B) ≤ 0.97 × mean(A): the prediction, stated now. A 90 s request's upload is 4.6 MB, and its prefill and first decode steps are the slowed shapes.
3. **Sanitizers.** memcheck (with leak check), synccheck and initcheck on the candidate's `lfm3-speech` with the 90 s clip and an image request: 0 errors, 0 leaks, and outputs identical to the plain run.
4. **Host suites** pass (the same command as before; failures recorded).
5. **Memory.** The pinned buffers add at most 8 MiB per process (two encoders × 4 MiB), by construction. On the device stand-in (MemTotal − MemAvailable, ABAB, with the 90 s clip and an image), the candidate's increase over main is at most 16 MiB.
   - *Clarified 2026-11-22, before any data:* the measuring tool is `measure_lfm3_audio_device_memory.py` (the short clip, then the 90 s clip twice; ABAB). It has no image request. The vision encoder's staging is the same fixed 4 MiB buffer by construction, so the audio reading stands for both. The tool subtracts the server's RssAnon + RssShmem; whether pinned host memory lands there is not known, and is reported as found.
6. **Coherency.** Read the 90 s transcript and an image answer from the candidate end to end.

The merge and deploy are Tripp's call.

## Results
### Rules 1 and 3, and the unit check (`build/pu-exact/`, 22:54–23:32)
- **Source:** `build/pu-src` (081eab9), built by `build_tcgen05_engine.sh pu-on on`.
- **Binaries:**
  - `lfm3-speech` `987b407a…`
  - `liquid-lfm3-server` `d9b8c038…`
  - `liquid-lfm3-worker` `9c85bc0f…`
  - `lfm3-vision-check` `185aa050…`
  - `pinned-staging-check` `9d26ba44…`

**The unit check:** `pinned-staging-check` passed 72 of 72 cases. Each case uploads one of 8 sizes, from 0 bytes through 13 MiB + 5 across the 4 MiB chunk, at 3 × 3 unaligned source and target offsets. In every case the device held the source's bytes exactly, and every guard byte was unchanged.

**Rule 1, byte identity: PASS.**

| check | result |
|---|---|
| speech regression | 36 SAME, 0 differ |
| dump against main | 360 of 360 identical |
| edge against main | 93 of 93 identical |
| e2e | 90 of 90 identical |
| split | 90 of 90 identical |
| vision stage files against main's | 713 of 713 identical |
| audio-in server | PASS |
| 8 at once | 8 served, 8 identical to `lfm3-speech`, PASS |
| 32 at once | 32 served, 32 identical, PASS |
| images | IMAGE SERVER PASS |
| image cache | 24 of 24 identical, PASS |
| TTS | exit 0 |
| text gate | exit 0 |

**Rule 3, sanitizers: PASS.** The candidate's `lfm3-speech` ran on the 90 s clip, `cherry` and `two`.

| sanitizer | result | outputs against the plain run |
|---|---|---|
| memcheck (leak check) | 0 errors, 0 bytes leaked | 90 s, cherry, two: JSON and every file the same |
| synccheck | 0 errors | the same |
| initcheck | 0 errors | the same |
