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

### Rules 2 and 5 (`build/pu-quiet/`, quiet 23:47–23:54)
**The run:** sampler 331 samples, CLEAN; power state unchanged. A is `merge5-srv` (`f19f4ee4…`, the deployed build); B is `pu-on` (`d9b8c038…`).

**Rule 2: FAIL, as written.** Time to first text, in seconds:

| item | A1 | B1 | A2 | B2 | bound (mean A × 1.01) | mean B ÷ mean A | texts | |
|---|---|---|---|---|---|---|---|---|
| short | 0.0385 | 0.0384 | 0.0386 | 0.0385 | 0.0389 | 0.997 | same | |
| median | 0.0455 | 0.0459 | 0.0457 | 0.0455 | 0.0460 | 1.003 | same | |
| typical | 0.0569 | 0.0571 | 0.0574 | 0.0566 | 0.0577 | 0.995 | same | |
| long | 0.0881 | 0.0891 | 0.0895 | **0.0903** | 0.0897 | 1.010 | same | **OVER** |
| joined90 | 0.1854 | 0.1851 | 0.1867 | 0.1862 | 0.1879 | 0.998 | same | |
| image cherry (turn 1) | 0.2223 | **0.2288** | 0.2250 | 0.2059 | 0.2259 | 0.972 | same | **OVER** |
| image two (turn 1) | 0.0545 | 0.0534 | 0.0540 | 0.0540 | 0.0548 | 0.990 | same | |

- The long clip's B2 is 0.6 ms over its bound. Cherry's B1 is 2.9 ms over its bound, while its B2 is 8.5% under A's mean.
- **The joined90 prediction is NOT MET:** 0.998 against ≤ 0.97.

**Why no gain, by the record:**
- In the deployed engine, the audio encoder releases its scratch after every clip (audio-ttft-v1 amendment 6): a `cudaDeviceSynchronize` and a set of `cudaFree` calls between the encoder and the prefill.
- alloc-effect-v1 stage 4 found that any GPU idle of 1 ms or more ends the slow state. That release is what ended it. It is the same mechanism by which releasing the scratch "fixed" audio-ttft's 40 ms.
- So in the deployed server, the pageable upload's slow state covers only the encoder itself, which this run shows is not measurably slowed. The prediction assumed the state reached the prefill. It no longer does, and I did not check that before setting the prediction.
- What the change removes is a hidden coupling: today the prefill's speed depends on the encoder's release happening to leave a gap. That has no measured speed value now.

**Rule 5, memory: PASS.** Device stand-in after the 90 s clip, in MiB:

| | A1 | B1 | A2 | B2 |
|---|---|---|---|---|
| after 90 s | 7,747 | 7,761 | 7,757 | 7,716 |

- B − A = (7,761 + 7,716)/2 − (7,747 + 7,757)/2 = **−13.5 MiB**, against a bound of +16 MiB.
- VmRSS after 90 s: 5,443 MiB (A), 5,447 MiB (B).
- Whether the tool's RssAnon + RssShmem subtraction counts the pinned buffers cannot be told at this resolution: the 4 MiB is under the run-to-run spread of about 40 MiB.

### Rule 6, coherency: PASS
I read B1's texts.
- **joined90:** "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour" (32 tokens, as requested).
- **long:** "Yesterday you were trembling for a health that is dear to you. Today you fear for your own. Tomorrow it will be anxiety about money. The day".
- **cherry, all three turns:** on topic and consistent with the image (white blossoms, brown branches, soft diffused light). Turn 1 calls the stamens "pink-tinged … golden anthers" and turn 3 "orange". That is the model's own wording, byte-identical to main.

### Rule 4, host suites: PASS
`python3 -m pytest -q tests audit/test_host_contracts.py` (workstation): **2,590 passed, 3 skipped, 0 failed**, in 56 min 39 s. That is the same count as main's last run; this round adds no host test.
- A first attempt was stopped by my own 3,000 s `timeout` before the suite finished, which is not a test failure. I terminated its orphaned TSan harness and reran without the cap.

## Conclusion
| rule | result |
|---|---|
| 1 byte identity | **PASS** |
| 2 speed | **FAIL** as written: the long clip +1.0%, cherry B1 over; the joined90 prediction not met (0.998) |
| 3 sanitizers | **PASS** |
| 4 host suites | **PASS** |
| 5 memory | **PASS** (−13.5 MiB) |
| 6 coherency | **PASS** |

- **The change is correct and costs nothing measurable, but it buys no measurable speed in the deployed server.** Amendment 6's per-clip scratch release already leaves the 1 ms gap that ends the slow state before the prefill.
- What it would buy is robustness: the prefill would no longer depend on that release. Any later change that kept the encoder's scratch, or otherwise ran the prefill straight after the upload, would bring back audio-ttft's 40 ms at 90 s.
- Merging is Tripp's call.
