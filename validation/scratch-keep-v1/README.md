# scratch-keep-v1: the audio encoder keeps its scratch between clips

Branch `scratch-keep-v1`, from main a291502. That is the deploy record on top of 8120c19, the merges of encoder-wrap-v1, ttft-rest-v2 and pinned-uploads-v1, deployed 2026-11-25 as `0012be944a78`. Tripp's decision, 2026-11-25: "take the savings".

## Why now
- Since audio-ttft-v1 amendment 6, the audio encoder frees its whole scratch after every clip: a `cudaDeviceSynchronize`, then about 15 `cudaFree` calls. The next clip then allocates it again, about 30 `cudaMalloc` calls (`scratch()` and `lfm::split::Linear`).
- Amendment 1 had kept the scratch, up to 1 GiB, and saved those calls, about 10 ms at 90 s by that round's nsys reading. But the language-model prefill that followed was about 40 ms slower at 90 s, by an effect not then understood. Amendment 6 went back to releasing.
- alloc-effect-v1 found the effect. A pageable host-to-device copy of 1 MiB or more halves the GPU's read speed until the GPU next idles for 1 ms. The clip's feature upload was such a copy (3.84 MB at 90 s), and releasing the scratch happened to give the GPU the idle that ended the slow state.
- pinned-uploads-v1 (merged in 8120c19) stages that upload through pinned memory, so the slow state never starts. Keeping the scratch no longer needs the idle that releasing gave.
- encoder-wrap-v1 (merged in 8120c19) freed the encoder's doubled weights, 439.5 MiB for audio. Keeping a 90 s clip's scratch, about 623 MiB, leaves device memory after the 90 s clip within about 70 MiB of the server deployed today (f19f4ee40708).

## The change
- Amendment 1's keep returns, unchanged in substance. After each clip, the encoder adds up what its scratch holds (the 13 slots plus `lfm::split::Linear`'s two buffers). If the total is at most the cap, it keeps all of it for the next clip. Otherwise it releases all of it, as it does today.
- The cap is 1 GiB, as in amendment 1. At 90 s the scratch holds about 623 MiB (computed from the stage shapes), so it is kept. Any clip over about 148 s is released after encoding, so no clip leaves more than 1 GiB held for the life of the process. That keeps audio-in-v1's red-team point.
- `LFM3_AUDIO_SCRATCH_KEEP_MIB` overrides the cap (an integer number of MiB; 0 releases after every clip, as today). It exists so rule 4 can test the release path with the clips at hand. The default is 1024.
- The `cudaDeviceSynchronize` stays, so the host still waits for the encoder before it goes on. Dropping it would be a separate change with its own rules.
- Nothing else changes. The arithmetic, the kernels and the order of work on the GPU are all as they are now. The only differences are whether the `cudaMalloc` and `cudaFree` calls happen, and what memory the encoder holds between clips.

## Rules, set before any code
A is main 8120c19 (`build/merge6-srv`, the build checked for this deploy). B is the candidate, built the same way: `build_tcgen05_engine.sh`, tcgen05 ON.

1. **Byte identity.**
   - **1a. The checks every round runs:** run on B, every output must be identical to main's:
     - the speech regression;
     - the rule-6 suite (dump, edge, e2e and split);
     - audio in through the server, against `lfm3-speech`;
     - 8 and 32 requests at once, against `lfm3-speech`;
     - images through the server; the image cache;
     - TTS; the text gate.
     - The 713 vision stage files are not repeated. Rule 7 covers vision through machine-code identity, and the vision encoder's code does not change.
   - **1b. Order:** one `lfm3-speech --batch` list, run on A and on B: 90 s, short, typical, 90 s, cherry, short, 35 s. Under B a shorter clip reuses a longer clip's kept scratch, and an image runs between two audio clips. Every output file must be identical between A and B.
2. **Sanitizers:** run on B's `lfm3-speech`, over 1b's list:
   - memcheck with the full leak check: 0 errors, 0 bytes leaked;
   - synccheck and initcheck: 0 errors each. initcheck covers a kept buffer that is read before it has been written in this clip.
   - Under each tool, every output must equal B's plain run of the same list.
3. **Speed:** a quiet window: 30 quiet checks, the service stopped, the 1 s sampler, power checked, MONITOR CLEAN. Time to first token from the file, on tcgen05-v2's five clips (`measure_lfm3_audio_ttft.py`, 1 untimed and 5 timed passes per run), runs in the order A1 B1 A2 B2.
   - 3a. At 90 s, mean(B) is at most mean(A) − 0.005 s. The prediction is about 10 ms, from amendment 1's ~30 `cudaMalloc` at ~0.27 ms each plus the frees.
   - 3b. At every one of the five clips, mean(B) is below mean(A). Amendment 1 saved 1.6 ms on the short clip against amendment 6, and 2.9 ms on the 5.8 s clip.
   - 3c. Texts are identical to A's at every clip.
   - Reported, not ruled on: a trace run of B (`-DLFM3_TTFT_TRACE`, `trace_lfm3_audio_ttft.py`) splitting the encoder from the language-model call. It shows whether the language model's 40 ms has stayed away. If it came back, 3a would fail on its own.
   - Note added with the scripts, before any data: the trace runs on A's tree too, built the same way, so the two splits sit side by side in the same window. Scripts: `tools/thor/scratch_keep_exact.sh` (rules 1, 2 and 7) and `tools/thor/scratch_keep_quiet.sh` (rules 3 and 4).
4. **Memory:** `measure_lfm3_audio_device_memory.py`, the in-use stand-in, runs in the order A1 B1 A2 B2.
   - 4a. After the 90 s clip, B − A is between 550 and 700 MiB. The prediction is 623 MiB held; the window allows for the stand-in's granularity.
   - 4b. After the 90 s clip, B is at most the deployed server's figure from encoder-wrap-v1's run (A1 7,741 and A2 7,729 MiB, mean 7,735) + 128 MiB.
   - 4c. Release path: B run with `LFM3_AUDIO_SCRATCH_KEEP_MIB=256` (below the 90 s clip's ~623 MiB) has its after-90 s figure within ±32 MiB of A's.
5. **Host suites:** pass, with counts against main 8120c19's run (2,593 passed, 3 skipped).
6. **Coherency:** the texts of rule 3 and the at-once checks are read end to end.
7. **Nothing else changes:** `compare_sass.py`, A against B, on `liquid-lfm3-server`, `lfm3-speech` and `lfm3-vision-check`. Every kernel's machine code must be identical; only host code changes.

## Deploy
Merging and deploying are Tripp's call. If every rule passes, the deploy follows the always-on-v1 procedure.

## Results (2026-11-25)
Binaries:
- A: `build/merge6-srv`, server `0012be944a78…` (deployed).
- B: `build/keep-on` from d9981c2, server `bbf085b991b8…`.
- Trace builds: `keep-trace-a` `a2868f8c…` and `keep-trace-b` `e95b031a…`.

Exact runs: `build/keep-exact/`, 06:19–07:04. Quiet window: `build/keep-quiet/`, 07:22–07:28, with the service stopped; MONITOR CLEAN (316 samples, foreign in 0); power unchanged.

| rule | result |
|---|---|
| 1. byte identity | PASS |
| 2. sanitizers | PASS |
| 3. speed | PASS |
| 4. memory | **FAIL as written** (4a PASS; 4b and 4c FAIL) |
| 5. host suites | PASS |
| 6. coherency | PASS |
| 7. machine code | PASS |

**Rule 1: PASS.**
- **1a:**
  - The speech regression: 36 SAME.
  - The rule-6 suite against main: dump 360 of 360 and edge 93 of 93, none missing or extra; e2e and split 90 of 90 each.
  - Audio in: PASS. 8 and 32 at once: identical to `lfm3-speech`.
  - Images: PASS. The image cache: 24 of 24 identical.
  - TTS exit 0; the text gate exit 0.
- **1b, the order list on A and on B** (90 s, short, typical, 90 s, cherry, short, 35 s): all 7 items identical. That is, the JSON without its timing fields, plus 4 other files each, or 5 for cherry.

**Rule 2: PASS.**
- memcheck with the full leak check: 0 errors, 0 bytes leaked in 0 allocations. synccheck: 0 errors. initcheck: 0 errors.
- Under each tool, all 7 items are the same as B's plain run.

**Rule 3: PASS.** Time to first token from the file, median of 5 timed passes per run, in seconds:

| clip | A1 | B1 | A2 | B2 | saved | B/A | texts |
|---|---|---|---|---|---|---|---|
| short (2.1 s) | 0.0374 | 0.0360 | 0.0371 | 0.0360 | 0.0012 | 0.967 | identical |
| median (5.8 s) | 0.0441 | 0.0417 | 0.0442 | 0.0421 | 0.0023 | 0.949 | identical |
| typical (10.4 s) | 0.0556 | 0.0528 | 0.0552 | 0.0530 | 0.0026 | 0.954 | identical |
| long (35 s) | 0.0837 | 0.0778 | 0.0841 | 0.0784 | 0.0058 | 0.931 | identical |
| joined90 | 0.1744 | 0.1614 | 0.1741 | 0.1605 | **0.0133** | 0.924 | identical |

- 3a: 13.3 ms saved at 90 s, against a bound of 5 ms and a prediction of about 10.
- 3b: every clip is faster, from 1.2 to 13.3 ms.
- 3c: the texts are identical at every clip.

**Reported, the traces (medians, ms, A's tree / B's tree):**

| part | short | median | typical | long | joined90 |
|---|---|---|---|---|---|
| log_mel | 1.31 / 1.31 | 2.91 / 2.96 | 4.98 / 5.00 | 17.62 / 17.72 | 39.00 / 39.48 |
| encoder | 6.59 / 5.28 | 9.21 / 7.18 | 11.61 / 8.79 | 18.83 / 12.84 | **39.63 / 26.76** |
| lm_forward_call | 27.21 / 27.60 | 29.02 / 29.17 | 34.78 / 35.07 | 38.31 / 38.30 | 76.82 / 76.55 |
| total | 36.71 / 35.87 | 43.28 / 41.45 | 54.02 / 51.42 | 79.98 / 73.77 | 165.55 / 152.16 |

- All of the saving is in the encoder: 12.9 ms at 90 s and 6.0 ms at 35 s.
- The language model's call is unchanged at every clip: 76.82 against 76.55 ms at 90 s. The 40 ms that keeping cost in audio-ttft-v1 has not come back. That fits the cause found in alloc-effect-v1 and the fix in pinned-uploads-v1.

**Rule 4: FAIL as written.** These figures use the device in-use stand-in, in MiB:

| run | started | short | after_90 | after_image |
|---|---|---|---|---|
| A1 | 522 | 7,191 | 7,194 | 7,777 |
| B1 | 22 | 7,188 | 7,856 | 8,459 |
| A2 | 19 | 7,160 | 7,155 | 7,778 |
| B2 | 544 | 7,203 | 7,873 | 8,555 |
| R1 (B, cap 256 MiB) | 21 | 7,136 | 7,094 | 7,720 |
| R2 (B, cap 256 MiB) | 48 | 7,175 | 7,177 | 7,792 |

- **4a: PASS.** After the 90 s clip, B − A = 690 MiB, inside the window of 550–700. The prediction was 623 MiB, computed from the stage shapes. The measured figure is 67 MiB higher. Allocation granularity explains at most about 30 MiB of that (15 buffers rounded up to 2 MiB). The rest is not explained.
- **4b: FAIL as written.** After the 90 s clip, B is 7,865 MiB. The bound was the old deploy's 7,735 + 128 = 7,863, so B is 2 MiB over.
  - The bound took encoder-wrap-v1's saving (556 MiB in its run) and the predicted 623 MiB held. The measured 690 MiB is what breaks it.
  - Against the old deploy, B is about +130 MiB after the 90 s clip. Against today's deploy (A), it is +690 MiB.
- **4c: FAIL as written.** With the 256 MiB cap, after the 90 s clip, R − A = −39 MiB, outside the ±32 MiB set.
  - The release path does what it should. Had it kept the scratch, R would sit near B's +690 MiB; instead it sits at A's level, and slightly below it.
  - The bound was tighter than the stand-in's own spread in this window: A1 and A2 differ by 39 MiB, and R1 and R2 by 83 MiB. I set ±32 MiB without checking that spread. The figures are recorded as they came, with no rerun.

**Rule 5: PASS.** The host suites on d9981c2: 2,593 passed, 3 skipped. That is the same as main 8120c19.

**Rule 6: PASS.**
- The texts of rule 3 are identical between A and B at every clip, and coherent: the short clip "Alexander did not sit down."; the 90 s clip begins "He hoped there would be stew for dinner, turnips and carrots…".
- The at-once checks are identical to `lfm3-speech`, whose texts were read in earlier rounds.

**Rule 7: PASS.** `compare_sass.py`, A against B: `liquid-lfm3-server` 184 of 184 functions identical, `lfm3-speech` 184 of 184, `lfm3-vision-check` 121 of 121. None differ, are missing or are extra. Only host code changed.

## Conclusion
- **The change is correct and faster.** Every output is byte-identical, including clips in mixed order. The sanitizers are clean. Time to first token drops 13.3 ms at 90 s and 1.2 to 5.8 ms on the shorter clips. The language model's prefill is untouched.
- **It costs more memory than predicted.** After a 90 s clip, the server holds 690 MiB more than today's deploy, and about 130 MiB more than the server deployed until this morning. Rule 4b failed by 2 MiB on that, and 4c failed on a bound tighter than the measurement's own noise.
- A clip over about 148 s, or a lower cap, releases the scratch exactly as main does. 4c's figures show that.
- **Merging is Tripp's call.** The cap is one constant, and `LFM3_AUDIO_SCRATCH_KEEP_MIB` sets it at run time. A lower cap trades the 90 s saving for memory. For example, 448 MiB keeps clips up to about 65 s, so the shorter clips keep their savings and the 90 s clip releases as today.
