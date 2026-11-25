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
4. **Memory:** `measure_lfm3_audio_device_memory.py`, the in-use stand-in, runs in the order A1 B1 A2 B2.
   - 4a. After the 90 s clip, B − A is between 550 and 700 MiB. The prediction is 623 MiB held; the window allows for the stand-in's granularity.
   - 4b. After the 90 s clip, B is at most the deployed server's figure from encoder-wrap-v1's run (A1 7,741 and A2 7,729 MiB, mean 7,735) + 128 MiB.
   - 4c. Release path: B run with `LFM3_AUDIO_SCRATCH_KEEP_MIB=256` (below the 90 s clip's ~623 MiB) has its after-90 s figure within ±32 MiB of A's.
5. **Host suites:** pass, with counts against main 8120c19's run (2,593 passed, 3 skipped).
6. **Coherency:** the texts of rule 3 and the at-once checks are read end to end.
7. **Nothing else changes:** `compare_sass.py`, A against B, on `liquid-lfm3-server`, `lfm3-speech` and `lfm3-vision-check`. Every kernel's machine code must be identical; only host code changes.

## Deploy
Merging and deploying are Tripp's call. If every rule passes, the deploy follows the always-on-v1 procedure.
