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
- **Every bar clip needs more than the encoder and `log_mel`.**
  - At 90 s, with the encoder at HF's 0.028 s, `log_mel` at 0.015 s and nothing else changed, time to first token would be about 0.25 s (0.5433 − 0.138 − 0.156) against HF's 0.151 s. The language model's prefill and the remaining parse also have to come down.
  - At 10.4 s, 0.029 s must be saved, and bringing both stages to HF's level saves 0.027 s (0.0154 + 0.0114). That is just short, so there too something else has to move, or those two stages have to beat HF's.

## Stage 1: the audio encoder on the tensor cores (design and rules set 2026-11-20, before any code)
**Where the encoder's time goes** (nsys, `audio-encoder-bench` at the 90 s clip, 4 runs, the always-on service idle; shares, not a timing result; `build/ttft-nsys/`), per run about 156 ms of kernels:
- `fp32::linear_staged` (every multiply): 81%, about 126 ms;
- `fp32::rotary`: 8%, about 12.5 ms. Each call computes its FP64 angles (`pow`, `sincos`) for every element;
- `window_attention`: 4.5%, about 7 ms;
- `conv_in` and `depthwise` (the downsampler): about 6 ms.

**Design:**
- **Multiplies:** the vision encoder's split multiply (images-speed-v1, `x2`): `hi = bf16(x)`, `lo = bf16(x − hi)`, one `lfm::gemm` over a doubled K (`[hi | lo]` against `[W | W]`, built once per weight and kept), FP32 out, then the bias.
  - The kernels move from `vision_encoder.cuh` into a shared header unchanged, so the vision encoder compiles the same code.
  - Norms, the short convolutions, attention, SiLU, SwiGLU and residual adds stay as they are. The output is rounded to BF16 as before.
- **Rotary:** the cos/sin pairs come from a table computed once per sequence length by the same device code (`pow`, `sincos` in FP64), and the rotation is unchanged. So the rotated values are byte-identical to today's.
- **Fallback:** `LFM3_AUDIO_ENCODER=fp32` keeps today's multiplies exactly, the rotary table included.
- **Not in this stage:** attention, the downsampler's convolutions, `log_mel` and the language model's prefill.

**Rules:**
1. **Accuracy (gated), on audio-in-v1's 40 dumped clips** (rounds 1 and 2; `tools/check_lfm3_audio_in.py`, its HF FP32 and BF16 dumps):
   - on every clip, the ratio of the native projected features' relative L2 error to the FP32 model's, over HF BF16's, is at most **0.25** (today 0.15–0.23);
   - and at most today's native ratio on that clip + 0.02.
2. **Recognition (gated):** all 2,620 utterances through `lfm3-speech`, audio-in-v1's rule 3 unchanged: the WER difference against HF BF16 has a paired-bootstrap 95% upper bound of at most +0.25 points.
   - Reported: the count of utterances whose tokens differ from today's native run.
   - **Coherency:** I read every changed transcript.
3. **Speed (gated; quiet Thor, stage 0's method and sampler, HF re-measured in the same window):** the encoder alone (`audio-encoder-bench`) is at most HF's encoder alone at the 35 s and 90 s clips. The 10.4 s figure is reported, as are the server's time to first token at all five clips against HF's (stage 0's table, re-measured).
4. **Sanitizers (gated):** memcheck (leaks), racecheck, synccheck and initcheck on `audio-encoder-bench` at the short and 90 s clips: 0 errors, outputs byte-identical to the plain run.
5. **Nothing else changes (gated):**
   - the vision encoder's outputs on images-v1's 31 images (`lfm3-vision-check`) are byte-identical;
   - lfm3-speech's TTS regression and the text gate are byte-identical;
   - the server's audio-in replies are identical to the new `lfm3-speech`'s (the audio-in check, 8 and 32 at once);
   - `LFM3_AUDIO_ENCODER=fp32` reproduces today's 360 + 93 audio-in stage files byte for byte;
   - the host suites pass.

### Stage 1, development (2026-11-20, not a rule run; `build/ttft-dev/`)
- **The split multiply (51795de):** the encoder alone, rough timing on a non-quiet Thor, x2 against fp32:
  - 90 s: 0.0519 against 0.1538 s;
  - 35 s: 0.0232 against 0.0624 s;
  - 10.4 s: 0.0116 against 0.0214 s.

  Still over HF's encoder (0.028 and 0.0125 s at 90 and 35 s).
- **Accuracy on the 40 dumped clips** (`check_lfm3_audio_in.py`; development evidence, the gated run comes with the final build):
  - x2's ratio equals fp32's on every clip to 4 decimals (0.152–0.228);
  - the error before the BF16 output rounding rises from at most 5.0e-7 to 4.7e-6 (encoder output), far below the rounding (about 1.7e-3);
  - the 40-clip WER equals HF's (37 of 40 token sequences equal HF's; fp32: 38).
- **Where x2's 90 s encode goes** (nsys, per run): about 40 ms of kernels in 52 ms of wall time.
  - Kernels: multiplies 15.4 ms, `window_attention` 7.0, `split_rows` 4.6, `conv_in` and `depthwise` 6.5.
  - The other ~12 ms is about 30 `cudaMalloc` calls per encode (median 0.27 ms) and their frees: the scratch is released after every clip (audio-in-v1 red-team), about 630 MB at 90 s.
  - The doubled weights are built once (104 weights).

## Stage 1, amendment 1 (2026-11-20, before its code): two more changes, the rules unchanged but one added
1. **Scratch kept between clips, up to a cap.** The encoder keeps its scratch, the split multiply's included, after a clip when the total is at most **1 GiB**. A clip that needs more releases all of it at its end, as today. This adds a memory rule:
   - **5b (gated):** the server's idle device memory after a 90 s clip is at most 1 GiB above the deployed server's after the same clip.
   - Reported: the doubled weights' size.
2. **`window_attention` with its keys and values staged in shared memory.** A block takes consecutive queries of one head, which share most of their 32-key windows. Each (query, head) keeps today's arithmetic in today's order (the dot product over d, the warp softmax, the weighted sum over j), so the output is byte-identical to today's. Rule 5's `LFM3_AUDIO_ENCODER=fp32` stage files test this, since fp32 mode uses the same attention.

### Amendment 1, development results (2026-11-20, not a rule run)
- **`audio-attention-check`:** `window_attention_tiled` is bit-identical to `window_attention` on random inputs at 11 lengths (1, 2, 31, 32, 33, 63, 64, 65, 100, 1,125, 1,149), the tile edges included.
- **The encoder alone, rough timing on a non-quiet Thor** (x2 after the amendment, x2 before it, fp32 after it):

  | clip | x2 after | x2 before | fp32 after |
  |---|---|---|---|
  | 90 s | 0.0330 s | 0.0519 s | 0.1378 s |
  | 35 s | 0.0157 s | 0.0232 s | 0.0551 s |
  | 10.4 s | 0.0084 s | 0.0116 s | 0.0189 s |

  HF's encoder: 0.028 / 0.0125 / 0.0113 s.
- **Noted:** the bench repeats each clip, so its timed runs find the scratch already allocated. In the server, the first clip longer than any before it still allocates.

## Stage 1, amendment 2 (2026-11-20, before any x1 data): x1 as a candidate
x2's error before the output's BF16 rounding (at most 4.7e-6) is about 350 times below that rounding (about 1.7e-3), so a single BF16 part may meet rule 1. The vision round tried x1 first for the same reason; its x1 failed a stricter bar (0.15 × HF BF16's).
- **The variant:** x1, the activations rounded once to BF16 (`LFM3_AUDIO_ENCODER=x1`). It is one GEMM over K, not 2K, through the same `lfm::split::Linear`.
- **The choice, by the rules unchanged:** x1 ships as the default if it passes rules 1 and 2 (accuracy on the 40 clips, recognition on the 2,620). Otherwise x2 ships.
  - Rules 3–5 then run on the chosen variant.
  - No other variant is added without amending first.

### Rules 1 and 2 (2026-11-20, Thor, 21:47–22:50; `build/ttft-rule12/`; lfm3-speech `ttft-speech-a2`, `8586be33…`, from 165cc8c)
Each mode ran the 40 dumped clips and all 2,620 utterances, scored by `check_lfm3_audio_in.py`.

| mode | rule 1: ratio, 40 clips | vs fp32, per clip | WER (HF 2.6062%) | difference, 95% CI | token sequences equal to fp32's |
|---|---|---|---|---|---|
| fp32 (today's bytes) | 0.1524–0.2276 | — | 2.5911% | −0.0151 [−0.0381, +0.0057] | — |
| x1 | 0.3402–0.5188 | +0.177 to +0.310 | 2.6006% | −0.0057 [−0.0299, +0.0171] | 2,572 of 2,620 |
| x2 | 0.1524–0.2276 | +0.0000 on every clip | 2.5987% | −0.0075 [−0.0306, +0.0149] | 2,578 of 2,620 |

- **x1: rule 1 FAIL** (over 0.25 on every clip, and far over fp32 + 0.02). By amendment 2, **x2 ships**; x1 is not the default. Its rule 2 would have passed.
- **x2: rule 1 PASS; rule 2 PASS** (upper bound +0.0149 points against +0.25).
- **Coherency (x2):** I read all 42 utterances whose tokens differ from fp32's.
  - 1 differs in tokens only; the text is identical.
  - Most differences are punctuation and capitalisation ("Mrs." / "Missus", a comma, "said" / "Said").
  - Word changes go both ways:
    - better: "prayed" (was "pray"), "Independence" (was "independents"), "Ben-Zoof" (was "Ben-Zouf");
    - worse: "Leoceria" (was "Leocadia", twice), "king's sword" (was "king swore", for "soared"), "call off" (was "carf", for "carved"), "Bronte's".
  - On these 42 the errors are 32 (fp32) against 36 (x2).
  - Every transcript is a coherent English sentence; none is truncated, repeated or garbled.

## Stage 1, amendment 3 (2026-11-21, before any rule-4 data from the bench): rule 4 as written, and a deviation recorded
- **The deviation.** The exactness script (`tools/thor/audio_ttft_stage1_exact.sh`, ac21bae) ran rule 4's sanitizers on `lfm3-speech` (the whole engine, the shortest dumped clip and the 90 s clip), not on `audio-encoder-bench` as rule 4 says. Those runs are recorded below as extra evidence, not as rule 4.
- **The gap behind it.** `audio-encoder-bench` recorded only the mel features' hash, so on the bench "outputs byte-identical to the plain run" had nothing to compare for the encoder.
- **The amendment.** The bench now also hashes the encoder's BF16 output (`encoded_fnv1a`), copied to the host outside the timed span. The hash must be equal on every run of a clip, or the tool fails.
  - Rule 4 then runs as written: memcheck (`--leak-check full`), racecheck, synccheck and initcheck on this bench at the short and 90 s clips (`--repeat 1`). Each needs 0 errors and the same `mel_fnv1a` and `encoded_fnv1a` as the plain run.
  - The quiet window's timed bench (`ttft-bench-s1`, `bbcc9573…`) was built from 6ec2358, before this change; the change adds only the copy and hash after each timed span.

### Rule 5, the exactness run (2026-11-20 22:53 to 11-21 06:08, Thor, not timed; `build/ttft-exact/`)
Binaries:
- server `ttft-srv` `222ff9cf…`, worker `600d42d0…` (6ec2358);
- `lfm3-speech` `ttft-speech-a2` `8586be33…`;
- `lfm3-vision-check` from this round `b77467ce…` and from main `30afbe24…`.

| check | result |
|---|---|
| vision encoder, images-v1's 31 cases, x2/x2 | 713 of 713 stage files byte-identical to main's |
| lfm3-speech TTS regression | 36 of 36 SAME |
| `LFM3_AUDIO_ENCODER=fp32`, 40 dumped clips (360 files) | 355 identical; the 5 that differ are F2's known 5 (`8455-210777-0002` mel; `8555-292519-0007` mel, down, encoded, features) |
| `LFM3_AUDIO_ENCODER=fp32`, 93 edge files | 93 of 93 identical |
| `LFM3_AUDIO_ENCODER=fp32`, e2e and split | 90 of 90 and 90 of 90 identical |
| default (x2), 40 dumped clips | the down, encoded and features files of all 40 differ (by design); mel as fp32 (F2's 2); 3 of 40 transcripts differ (`4507-16021-0023`, `5105-28240-0024`, `7127-75946-0008`; rules 1 and 2 cover these) |
| default (x2), edge files | 57 of 93 identical; the 36 that differ are the encoder stages (down, encoded, features) of the 12 clips with audio |
| default (x2), e2e and split | 90 of 90 and 90 of 90 identical |
| server audio in against the new `lfm3-speech` | PASS (repeat identical, text control identical) |
| server, 8 and 32 at once | 8 of 8 and 32 of 32 identical to `lfm3-speech`; PASS |
| server TTS check; text gate; images through the server | exit 0; exit 0; PASS |
| host suites on the branch | 2,569 passed, 3 skipped, 0 failed |

**Two checks exited 1, by design.** The concurrent engine check and the batch-server check compare against references recorded before this round.
- In both, the only differences are the five audio-in cases `spoken-q0`–`spoken-q4`. The other 36 (engine) and 37 (server crowds, all four orders) are identical.
- The concurrent check at 4, 8 and 16 rows is 41 of 41 identical to its own run alone.
- **Confirmed in fp32 mode:** the concurrent engine check with `LFM3_AUDIO_ENCODER=fp32` (`engine-exact-fp32.log`) PASSES, 41 of 41 at every row count against the old references, so the five differences come from the encoder's multiplies alone.
  - It ran beside the sanitizers; it has no timing criteria.
  - The batch-server check in fp32 mode has timing sub-checks, so it runs quiet, after the quiet window (`build/ttft-post/`).

**Extra evidence, not rule 4 (amendment 3):** the sanitizers on `lfm3-speech` (the shortest dumped clip and the 90 s clip, whole requests):
- memcheck with `--leak-check full`: 0 errors, 0 bytes leaked; synccheck: 0 errors; initcheck: 0 errors.
- Every output file equals the plain run's (the JSON compared without its timing keys).
- racecheck: the short clip finished (outputs identical) but the 90 s clip did not end within the 6 h limit. I stopped it with SIGTERM, so its log has no summary.
- A script fault: its output comparison called `compare_speech_outputs.py` with file prefixes, not directories, and printed nothing. I compared the files directly afterwards (above).
- A harness lesson: `timeout` on `docker exec` ends the client, not the process in the container. The post-window script runs `timeout` inside the container.

### Rule 3 and the quiet window (2026-11-21, Thor, quiet 06:25–06:38; `build/ttft-s1q/`)
Binaries:
- server `ttft-srv` `222ff9cf…`; deployed server `merge-srv` `a681eab4…`;
- bench `ttft-bench-s1` `bbcc9573…` (6ec2358).

Window checks:
- the always-on service stopped for the window and started again at 06:38:22;
- power state unchanged before every step and at the end; every step exited 0;
- sampler: 724 samples; foreign processes in 0; more than one server in 0; defunct (allowed) in 1. **MONITOR: CLEAN.**

**Rule 3: FAIL.** The encoder alone is not at most HF's encoder alone at 35 s or at 90 s.

| clip | native encoder (bench, median of 5) | HF encoder | native ÷ HF | stage 0's native encoder |
|---|---|---|---|---|
| 10.4 s (reported) | 0.0083 s | 0.0112 s | 0.74 | 0.0227 s |
| 35 s (gated) | 0.0157 s | 0.0124 s | **1.27** | 0.0663 s |
| 90 s (gated) | 0.0330 s | 0.0279 s | **1.18** | 0.1663 s |

The encoder is 4.2× (35 s) and 5.0× (90 s) faster than stage 0's, but still over HF's at both gated lengths.

**Time to first token** (the server, from reading the file; runs A and D are this window's two passes; HF's is re-measured in this window; stage 0's figures come from another window, 2026-11-20):

| clip | audio | server A | server D | HF | native ÷ HF | stage 0's ratio | log_mel | encoder | rest |
|---|---|---|---|---|---|---|---|---|---|
| short | 2.11 s | 0.0434 | 0.0434 | 0.0475 | 0.91 | 0.95 | 0.0054 | 0.0056 | 0.0324 |
| median | 5.79 s | 0.0547 | 0.0546 | 0.0518 | 1.05 | 1.22 | 0.0111 | 0.0070 | 0.0365 |
| typical | 10.44 s | 0.0754 | 0.0750 | 0.0600 | 1.25 | 1.47 | 0.0199 | 0.0083 | 0.0470 |
| long | 34.95 s | 0.1834 | 0.1816 | 0.0803 | 2.27 | 2.85 | 0.0667 | 0.0157 | 0.1001 |
| 90 s | 90.00 s | 0.4539 | 0.4533 | 0.1508 | 3.01 | 3.59 | 0.1711 | 0.0330 | 0.2495 |

- Times are in seconds. log_mel and encoder are the bench's; rest = server A − log_mel − encoder. All texts are identical across passes.
- **Noted, under diagnosis:** at 90 s, the server's time to first token fell 0.089 s from stage 0's while the bench's encoder fell 0.133 s, so rest grew from 0.2060 to 0.2495 s. At 35 s rest grew only 0.005 s.

**Rule 5b as first measured (the quiet window, step E): not a valid measure; see amendment 4.**
- `measure_lfm3_audio_memory.py` read VmRSS: `merge-srv` idle 5,439 MiB, `ttft-srv` 5,440 MiB, a difference of 0.6 MiB.
- But VmRSS on Thor does not count GPU allocations. image-cache-v1 had recorded this; amendment 1 missed it when it chose this tool.
  - The 5.4 GB is almost all `RssFile`, the checkpoint's mapping. The deployed service reads `RssAnon` 117 MB, `RssFile` 5,422 MB.
  - So "0.6 MiB" says nothing about the kept scratch.

**A diagnostic, not a rule run (`build/ttft-diag/`, `tools/profile_lfm3_audio_server.py`).** `ttft-srv` under nsys, four 90 s requests in a row. It ran beside rule 4's racecheck, so its times are not comparable.
- Request 1's encode made 118 `cudaMalloc` calls (38.7 ms in the API) and built the doubled weights (`double_rows`, 7.2 ms).
- Requests 2–4 made none: the server keeps the scratch, as the bench does.
- **Found:** a server started without `--warmup`, as every measuring tool starts it, pays for the encoder's construction, its allocations and the doubled weights on its first audio request. *(Corrected 2026-11-21: this first read "the server's warm-up does not run the audio encoder". That is wrong: `--warmup`, which the always-on service uses, sends a 2 s audio clip (`lfm/lfm3_warmup.hpp`), so the service pays this at startup.)*

## Stage 1, amendment 4 (2026-11-21, before any data from it): rule 5b by the device stand-in
- **The measure.** It is image-cache-v1's device stand-in: Thor's in-use memory (MemTotal − MemAvailable), less the same reading before the server started, less the server's RssAnon + RssShmem.
  - Tool: `tools/measure_lfm3_audio_device_memory.py`.
  - Each server runs alone, the deployed `merge-srv` first, then `ttft-srv`, on a quiet Thor (the window's usual checks: the service stopped, the sampler, the power state).
  - Sequence: start (its warm-up), a reading; one short-clip request, a reading; two 90 s requests, a reading.
  - Each reading is the median of 5 samples, 0.5 s apart, after 3 s idle.
- **The gate, rule 5b's bound unchanged:** after the 90 s clip, `ttft-srv`'s device figure is at most 1 GiB above `merge-srv`'s.
  - This covers everything the round keeps: the scratch (kept up to 1 GiB) and the doubled weights (kept for the process's life).
  - The difference after the short clip is reported: roughly the doubled weights.
- **Expected before the data, from the code:** the 90 s scratch is about 0.6–0.75 GB, and the doubled audio-encoder weights about 0.45 GB, so the total may exceed the bound.
  - If it fails, the bound is not relaxed after the fact. A lower keep cap, or not doubling, would be a new amendment with its own run.

### Rule 4 (amendment 3) and the batch server in fp32 mode (2026-11-21, Thor; `build/ttft-post/`)
**Rule 4: PASS.**
- Binary: `audio-encoder-bench` `ttft-bench-a3` `1f18a50e…`, from 7ae8f22 (the same bench source as e611281).
- Clips: the short clip (27 tokens) and the 90 s clip (1,125 tokens), `--repeat 1`, 07:14–07:24.

| tool | result | hashes against the plain run (short: mel `86fd42e9…`, encoded `3b6b1d97…`; 90 s: mel `0de75340…`, encoded `c4399214…`) |
|---|---|---|
| memcheck `--leak-check full` | 0 errors; 0 bytes leaked in 0 allocations | identical, both clips |
| racecheck | 0 hazards (0 errors, 0 warnings) | identical, both clips |
| synccheck | 0 errors | identical, both clips |
| initcheck | 0 errors | identical, both clips |

Racecheck on the encoder alone finished in minutes; on whole `lfm3-speech` requests it had not finished the 90 s clip in 6 h (above).

**The batch-server check with `LFM3_AUDIO_ENCODER=fp32`: PASS.** It ran quiet, 06:53–07:14:
- the service was stopped; MONITOR CLEAN (1,201 samples; foreign processes in 0); power unchanged;
- all four crowds: 42 of 42 identical to the old references, `spoken-q0`–`q4` included;
- real time at 8, 12 and 16 rows, and 7 TTS + 5 spoken joining: no stalls.

So the server's five audio-in differences under the default come from the encoder's multiplies alone, as the engine's did.

### Rule 5b by amendment 4 (2026-11-21, Thor, quiet 07:40–07:41; `build/ttft-mem/`): FAIL
Window checks: the service stopped; MONITOR CLEAN (53 samples, foreign processes in 0); power unchanged; both servers exited 0.

| | before (no server) | started | after the short clip | after the 90 s clip |
|---|---|---|---|---|
| `merge-srv`: in-use / device figure | 4,627 MiB | 4,745 / 25 | 11,464 / 6,697 | 11,437 / 6,666 |
| `ttft-srv`: in-use / device figure | 4,117 MiB | 4,729 / 518 | 12,001 / 7,744 | 12,655 / 8,394 |
| difference in the device figure (the gate) | | 493 | 1,046 | **1,728 (bound 1,024): FAIL** |
| difference in plain in-use | −510 | −16 | 537 | **1,218** |

- **The measure's flaw, found in its own data.** The no-server baseline moved by 510 MiB between the two servers' runs. That moves the gated figure by the same amount. The plain in-use readings, taken while each server ran, differ by 1,218 MiB after the 90 s clip. That is over the bound too, so the verdict does not depend on the flaw.
- **What the code explains:**
  - the doubled weights: 104 weights, 112 M parameters in the 101 counted from the checkpoint's header, plus the downsampler's output projection. That makes about 439 MiB as `[W | W]` in BF16, kept for the process's life;
  - the 90 s scratch, kept under the 1 GiB cap: `ttft-srv` grew 654 MiB from the short clip to the 90 s clip, while `merge-srv` stayed flat (−27 MiB).

  These two account for the 1,218 MiB.
- So the round, as built, keeps about 1.2 GiB more than the deployed server after a 90 s clip. Rule 5b fails.

### A diagnostic: the two servers in one window (2026-11-21, quiet 07:57–07:59; `build/ttft-ab/`; MONITOR CLEAN)
Time to first token from the file, in seconds (1 untimed + 5 timed passes per run, run in the order A1 B1 A2 B2):

| clip | merge A1 | ttft B1 | merge A2 | ttft B2 | saved | engine prefill, merge | engine prefill, ttft |
|---|---|---|---|---|---|---|---|
| short | 0.0454 | 0.0432 | 0.0447 | 0.0434 | 0.0018 | 0.0373 | 0.0354 |
| median | 0.0619 | 0.0547 | 0.0622 | 0.0552 | 0.0071 | 0.0455 | 0.0384 |
| typical | 0.0895 | 0.0751 | 0.0899 | 0.0756 | 0.0143 | 0.0605 | 0.0461 |
| long | 0.2263 | 0.1828 | 0.2267 | 0.1831 | 0.0435 | 0.1301 | 0.0854 |
| 90 s | 0.5411 | 0.4523 | 0.5447 | 0.4534 | 0.0901 | 0.3058 | 0.2138 |

- The saving is stable across passes, and every text is identical across passes.
- The bench's encoder alone saved 0.133 s (90 s) and 0.051 s (35 s); the server saves 0.090 s and 0.044 s. **So at 90 s about 40 ms of the encoder's gain does not reach the server, within one window.** It is not a cross-window artefact.
- The GPU clock is pinned (`gpu-gpc-0` 1,575 MHz, min = max), so it is not frequency scaling. A quiet nsys comparison of the server's encode against the bench's follows (`build/ttft-prof/`).

## Stage 1, amendment 5 (2026-11-21, before any data from it): no doubled weights
**The change.**
- `lfm::gemm::Problem` gains `b_k_wrap`: for an N × K B, B's k index is read modulo it.
- The split multiply's x2 then runs `[hi | lo]` against W itself (`ldb = in`, `b_k_wrap = in`), not against a stored `[W | W]`. The GEMM reads the same values in the same order, so its output is byte-identical; the 439 MiB copy goes.
- Every `in` is a multiple of 8, so no 8-element load straddles the wrap.
- `check()` refuses a wrap that is not a multiple of 8, an `ldb` below it, a K × N B, and the K = 128 decode path (no caller needs it).
- Off (0) everywhere else. The language model's and the vision encoder's GEMMs get one uniform branch and keep their arithmetic.
- The vision encoder still builds its own `[W | W]`. Moving it to the wrap is a later change with its own checks.

**Rules (all gated):**
1. **The GEMM:** `gemm-wrap-check` compares the wrap against an explicit `[W | W]` bit for bit:
   - the encoder's 7 shapes × 11 row counts (1 to 1,125: the thin path and every wide tile), FP32 and BF16 outputs;
   - every case identical, and the 4 invalid uses refused.
2. **Byte-identical to the build that passed rules 1, 2 and 4:**
   - `lfm3-speech` from this amendment, by default (x2): the 40 dumped clips' 360 files, the 93 edge files, and e2e and split are identical to `ttft-speech-a2`'s x2 run (`build/ttft-speech-a2-rule6-x2`);
   - the bench's `encoded_fnv1a` equals rule 4's (`3b6b1d97…` short, `c4399214…` 90 s).

   So rules 1 and 2 carry over without a rerun.
3. **Nothing else changes:** amendment 3's exactness script again, on this amendment's binaries:
   - vision 713 of 713;
   - TTS regression 36 SAME;
   - fp32 mode as before (F2's 5 only);
   - the concurrent engine, the batch server (both in fp32 mode against the old references);
   - server audio in, 8 and 32 at once, TTS, the text gate, images.

   Also: rule 4's four sanitizers on the bench at both clips, and the host suites.
4. **Rule 5b, measured again.** Amendment 4's tool, with its flaw removed (the no-server baseline moved by 510 MiB between servers):
   - the servers run in the order `merge-srv`, new, `merge-srv`, new;
   - the gate uses the plain in-use reading after the 90 s clip: the new server's mean minus `merge-srv`'s mean is at most 1 GiB;
   - each pair's difference and the baselines are reported.

   Expected from the code: about 0.65–0.75 GiB, the kept 90 s scratch.
5. **No slower (quiet window, one window):** time to first token on the five clips, `ttft-srv` and the new server ABAB. At every clip the new server's median is at most `ttft-srv`'s + 2%.

Rule 3 stays FAIL; this amendment does not aim at it.

### A diagnostic: the server's 90 s encode against the bench's (2026-11-21, quiet 08:15–08:16; `build/ttft-prof/`)
nsys CUDA trace, `ttft-srv` with four 90 s requests and `ttft-bench-a3` with `--repeat 3`. Each encode runs from `conv_in`'s launch to `to_bf16`'s end.

| | first encode | encodes 2–4 |
|---|---|---|
| server | 47.2 ms span (118 `cudaMalloc`, `double_rows` 4.7 ms) | 33.9 / 33.9 / 34.0 ms; 353 launches; no allocations |
| bench | 46.3 ms span (the same) | 34.0 / 33.9 / 34.0 ms; 353 launches; no allocations |

Kernel time per warm encode, the same in both to 0.1 ms:
- `wide_kernel` 15.2 ms, `split_rows` 4.5, `conv_in` 3.7, `depthwise` 2.5;
- `rms_norm` 1.2, `add` 1.0, `window_attention_tiled` 1.0, `swiglu` 1.0, `add_bias` 0.9, `activate` 0.9.

- **So the server's encode is the bench's.** The 40 ms of the A/B is not lost in the new server. The next check is whether stage 0's bench overstated the deployed server's encode: the deployed pair, `merge-srv` and stage 0's `ttft-bench`, profiled the same way (`build/ttft-prof-merge/`).
- **The first audio request after a start:**
  - its time to first token was 2.2 s, against 0.45 s for the next ones;
  - it pays for the encoder's construction, 118 allocations (the scratch), the doubled weights (removed by amendment 5) and first-launch costs (`cuLibraryGetKernel`).

  *(Corrected 2026-11-21:)* the always-on service runs `--warmup`, whose audio request already moves this to startup. Only servers started without it, as the measuring tools start them, pay it on their first request.

### A diagnostic: where the 40 ms goes (2026-11-21, quiet 08:32–08:33; `build/ttft-prof-merge/`)
**The deployed pair, profiled the same way:**
- `merge-srv`'s 90 s encode equals its bench's: 156.9–158.6 ms of kernels (`linear_staged` 126.0, `rotary` 12.5, `window_attention` 7.0), plus about 9 ms of `cudaMalloc` and `cudaFree` per clip.
- So stage 0's bench did not overstate the deployed encode. The new encoder saves about 132 ms inside the server.

**It is the language model's prefill.** From the encode's end to the first token (the first `greedy_rows`), 489 kernels in the same sequence in both servers:

| | request 1 | requests 2–4 |
|---|---|---|
| `merge-srv` | 142.4 ms | 142.9 / 141.5 / 141.6 ms |
| `ttft-srv` | 141.0 ms | **181.8 / 182.5 / 182.4 ms** |

Kernel by kernel (request 3 of each), the difference is two of the language model's GEMM shapes:

| kernel | instances | `merge-srv` | `ttft-srv` |
|---|---|---|---|
| `wide_kernel` grid 60×9 | 48 | 45.9 ms | 81.6 ms (+35.7) |
| `wide_kernel` grid 48×9 | 13 | 9.4 ms | 16.9 ms (+7.5) |

The other shapes are unchanged (grid 16×9: 33.0 against 33.5 ms; the norms, rotary and attention to 0.1 ms).

- **So after the new server's first audio request, those GEMMs run about 75% slower**, while its own encode is untouched. It is the same arithmetic on the same weights (the texts are identical), so it is a speed effect, not a correctness one.
- Something the round keeps after a request causes it: the kept scratch or the doubled weights.
- Next, a quiet window (`build/a5-diag/`) profiles four servers the same way:
  - `merge-srv`, as control;
  - `ttft-srv` (scratch kept, doubled weights);
  - `ttft-srv` with `LFM3_AUDIO_ENCODER=fp32` (scratch kept, nothing doubled, no split);
  - amendment 5's server (scratch kept, nothing doubled, x2 by the wrap).

### The four-way diagnostic (2026-11-21, quiet 08:55–08:59; `build/a5-diag/`)
nsys on four 90 s requests per server. All texts are identical. TTFT and prefill are in seconds, as each request reported them; the rest come from the trace.

| server | encoder state kept | TTFT, requests 2–4 | encode, warm | encode end → first token, request 1 / 2–4 | GEMM grid 60×9, request 1 / 2–4 |
|---|---|---|---|---|---|
| `merge-srv` | nothing (freed per clip) | 0.541–0.544 | 156.6–157.6 ms | 141.9 / 141.1–142.1 ms | 45.7 / 45.4–45.6 ms |
| `ttft-srv` (x2) | scratch, doubled weights | 0.454–0.457 | 34.0 ms | 140.1 / 183.2–184.6 ms | 46.0 / 81.3–81.7 ms |
| `ttft-srv`, `LFM3_AUDIO_ENCODER=fp32` | scratch | 0.559 | 139.0 ms | 141.0 / 183.0–183.3 ms | 45.9 / 82.0–82.2 ms |
| `a5-srv` (amendment 5) | scratch | 0.461–0.463 | 39.4 ms | 141.9 / 185.8–186.1 ms | 46.9 / 82.6–83.0 ms |

- **The cause is keeping the scratch, not the doubled weights.** fp32 mode doubles nothing and slows the same way. Every server that keeps the scratch slows after its first request; the one that frees it never does.
  - Each server's first request is normal, though by then its scratch is already allocated.
  - So the trigger is not the memory being held during the prefill. It is something about a request that makes no allocation calls.
- **Amendment 5's wrap costs the encoder 5.4 ms** (39.4 against 34.0 ms warm). It fixes the memory, but the modulo in the B loader is slow.
- **Amendment 5's exactness run was stopped** (SIGTERM, at step 2). It was certifying code about to change.
  - Its step 1 stands as evidence for the wrap's arithmetic: `gemm-wrap-check` 154 of 154 cases identical, 4 of 4 invalid uses refused.
- **Next (`build/keep-diag/`):** `lfm3-speech` built three ways from `ttft-src`, four 90 s requests each, quiet:
  - the scratch kept;
  - released after every clip;
  - kept, plus a 2 MiB `cudaMalloc` + `cudaFree` after every encode.

  This tests whether any allocation call restores the speed.

### The keep diagnostic (2026-11-21, quiet 09:19–09:21; `build/keep-diag/`)
`lfm3-speech` built three ways from `ttft-src` (6ec2358's engine), four 90 s requests per process. `encode_seconds` here includes log_mel. "Language model" is `prefill_seconds − encode_seconds`.

| variant | language model, request 1 | requests 2–4 |
|---|---|---|
| keep (as built), first run | 0.1795 s | 0.1787 / 0.1787 / 0.1790 s |
| release (`kKeepBytes = 0`) | 0.1386 s | 0.1382 / 0.1382 / 0.1382 s |
| touch (kept, then a 2 MiB `cudaMalloc` + `cudaFree` after each encode) | 0.1391 s | 0.1382 / 0.1382 / 0.1381 s |
| keep, second run | 0.1383 s | 0.1783 / 0.1779 / 0.1781 s |

- **One allocation call restores the full speed.** Holding the memory does not cause the slowdown. A process that has made no `cudaMalloc` or `cudaFree` for a while runs the language model's large prefill GEMMs about 29% slower (0.178 against 0.138 s). One small allocate-and-free just before brings them back.
- The deployed encoder frees its scratch after every clip, so it was making that call by accident. This round's keep removed it.
- The mechanism is not yet known. Candidates: a memory-clock (EMC) boost on allocation that decays, or similar driver state.
  - The memory clock cannot be read without root (`tegrastats` omits it for a user; `/sys/kernel/debug/bpmp` is denied).
  - A probe follows (`build/alloc-probe/`): the same GEMM shape and a 512 MiB copy, after idle periods with no call, and at delays after one allocate-and-free.
- **Wider question:** a text prompt's prefill makes no allocation calls either. If the slow state is the default, long text prefills may be paying it too. To be measured.

### The allocation probe (2026-11-21, quiet 09:38–09:44; `build/alloc-probe/`, `native/tests/alloc_effect_probe.cu`): no effect in isolation
The slowed GEMM's shape alone (M 1,149, N 7,680, K 2,048) and a 512 MiB device copy, in a fresh process. The GPU sat idle 5 s between readings. There were two rounds each in the container and on the host.
- The GEMM's median was 0.830–0.882 ms in every condition: no call, at 0–8 s after a 2 MiB `cudaMalloc` + `cudaFree`, and after `cudaFree(nullptr)`, `cudaMemGetInfo`, `cudaDeviceSynchronize`, or a kept malloc and its later free.
- The copy ran 232–240 GB/s throughout.
- Only the first launch after an idle period without an allocation call is about 0.07–0.1 ms slower. That is far too small to explain 40 ms.

So this is not a global clock or bandwidth state that an allocation call raises. It appears only in the engine's own context, and its mechanism is open. Two things are ruled out:
- the weights are read from device memory: one `cudaMalloc` payload (`src/safetensors.cpp`), not the file mapping;
- frequency scaling: the GPU clock is pinned.

## Stage 1, amendment 6 (2026-11-21, before any data from it): release the scratch per clip; amendment 5 withdrawn
**The change.**
- The encoder frees its scratch after every clip, as the deployed encoder does. The 1 GiB keep (amendment 1, part 1) is removed.
- Keeping cost the language model's prefill about 40 ms at 90 s by an effect not understood (above). Releasing measured fast (`release`: 0.138 s; keep: 0.178 s).
- The cost: the encoder's `cudaMalloc` and `cudaFree` calls per clip, about 10 ms at 90 s by the earlier nsys reading.
- The doubled weights stay (built once, kept). They are 439 MiB, inside rule 5b's 1 GiB on their own.

**Amendment 5 is withdrawn.** Its purpose was rule 5b, which releasing meets without it. Withdrawing it also leaves the language model's GEMM untouched.
- The wrap and its check stay in history (c968b5d) for a later round, the vision encoder's `[W | W]` included.
- Its one result: `gemm-wrap-check` 154 of 154 identical.

**The engine is then 6ec2358's, except that the scratch is freed after every clip.** That changes no arithmetic.

**Rules (all gated):**
1. **Byte-identical to the build that passed rules 1, 2 and 4.** The default (x2) `lfm3-speech` matches `ttft-speech-a2`'s x2 run:
   - the 360 dumped-clip files;
   - the 93 edge files;
   - e2e and split, against the validated runs;
   - the bench's `encoded_fnv1a` against rule 4's (`3b6b1d97…`, `c4399214…`).

   Rules 1 and 2 carry over.
2. **Nothing else changes:** amendment 3's exactness script on the new binaries (vision 713, TTS regression 36 SAME, fp32 mode, the concurrent engine in fp32 mode, the server checks); rule 4's four sanitizers on the bench, memcheck's leak check included; the host suites.
3. **Rule 5b** by amendment 5's corrected measure: plain in-use after the 90 s clip, ABAB against `merge-srv`, the new mean minus `merge-srv`'s mean at most 1 GiB. Expected: about 0.45 GiB, the doubled weights.
4. **Faster than the deployed server, one quiet window:** time to first token on the five clips, `merge-srv` against the new server ABAB, plus `ttft-srv` once as a reference.
   - The new server's median is at most `merge-srv`'s at every clip.
   - Reported: the saving, and the language model's prefill time from the done event.
   - Rule 3 (the encoder at most HF's) is re-measured in the same window and recorded. It is expected to stay FAIL: the bench's encode now includes the allocations.

### Amendment 6, the quiet window (2026-11-21, quiet 10:07–10:33; `build/a6-run/`)
Binaries from 14068df (engine 2fdb883):
- server `a6-srv` `cd66cf5d…`; `lfm3-speech` `a6-speech` `530f162d…`; bench `a6-bench` `136c24f0…`;
- for comparison: deployed `merge-srv` `a681eab4…`, `ttft-srv` `222ff9cf…`.

Window checks: the service stopped; MONITOR CLEAN (1,457 samples; foreign processes in 0; defunct, allowed, in 3); power checked before every step.

**Amendment rule 4, faster than the deployed server at every clip: PASS.** Time to first token from the file, median of 5 timed passes per run, in seconds:

| clip | merge A1 | a6 B1 | merge A2 | a6 B2 | ttft-srv R (scratch kept) | saved against merge | engine prefill: merge / a6 / ttft |
|---|---|---|---|---|---|---|---|
| short | 0.0450 | 0.0449 | 0.0454 | 0.0446 | 0.0433 | 0.0004 | 0.0373 / 0.0368 / 0.0352 |
| median | 0.0619 | 0.0571 | 0.0625 | 0.0574 | 0.0545 | 0.0049 | 0.0457 / 0.0408 / 0.0380 |
| typical | 0.0896 | 0.0777 | 0.0895 | 0.0776 | 0.0748 | 0.0119 | 0.0605 / 0.0486 / 0.0457 |
| long | 0.2282 | 0.1830 | 0.2264 | 0.1817 | 0.1823 | 0.0449 | 0.1302 / 0.0854 / 0.0849 |
| 90 s | 0.5395 | 0.4247 | 0.5435 | 0.4247 | 0.4505 | **0.1168** | 0.3060 / 0.1851 / 0.2121 |

- All texts are identical across every run.
- At 90 s, releasing beats keeping by 0.026 s: the language model's prefill is back to the deployed speed.
- On clips up to 10 s, keeping was 2–3 ms faster (no allocations, and the slowdown is small there). Releasing is still faster than the deployed server at every clip.

**Rule 3, re-measured: FAIL.** The bench's encode now includes the allocations.

| clip | native encoder | HF encoder |
|---|---|---|
| 10.4 s (reported) | 0.0109 s | 0.0113 s |
| 35 s (gated) | 0.0219 s | 0.0123 s |
| 90 s (gated) | 0.0463 s | 0.0279 s |

**Amendment rule 3 (rule 5b): PASS.**
- Plain in-use after the 90 s clip: `a6-srv` 12,002 and 12,018 MiB; `merge-srv` 11,462 and 11,496 MiB.
- Difference of the means: **531 MiB** (bound 1,024); pairs +540 and +522 MiB. Baselines 4,122–4,178 MiB.
- That is about the doubled weights (439 MiB) plus allocation rounding.

**The batch server with `LFM3_AUDIO_ENCODER=fp32` on `a6-srv`: PASS** (timing sub-checks included).

The exactness script (amendment rules 1 and 2) is running (`build/a6-exact/`).

### Amendment 6, the exactness run (2026-11-21, 10:33–11:15; `build/a6-exact/`)
**Amendment rule 1, byte-identical to the build that passed rules 1, 2 and 4: PASS.**
- Default (x2) `lfm3-speech` `530f162d…` against `ttft-speech-a2`'s x2 run: 360 of 360 dumped-clip files and 93 of 93 edge files, none missing or extra.
- e2e and split: 90 of 90 each, against the validated runs.
- The bench's hashes equal rule 4's at both clips.

So rules 1 and 2 carry over unchanged.

**Amendment rule 2, nothing else changes: PASS.**

| check | result |
|---|---|
| vision, 31 cases (`vcheck-a6` `71601aba…`) | 713 of 713 stage files identical to main's |
| TTS regression | 36 SAME, 0 differ or failed |
| `LFM3_AUDIO_ENCODER=fp32`: dump, edge, e2e, split | F2's known 5 only; 93 of 93; 90 of 90; 90 of 90 |
| concurrent engine, fp32 mode | CONCURRENT PASS |
| server: audio in; 8 and 32 at once | PASS; 8 of 8 and 32 of 32 identical to `lfm3-speech` |
| server: TTS; text gate; images | exit 0; exit 0; PASS |
| batch server, fp32 mode (in the quiet window) | PASS |
| sanitizers on the bench (memcheck with leak check, racecheck, synccheck, initcheck), short and 90 s | 0 errors, 0 bytes leaked, 0 hazards; hashes identical to the plain run |
| host suites (`tests` and `audit/test_host_contracts.py`, as before) | 2,564 + 5 = **2,569 passed, 3 skipped, 0 failed**; the first attempt ran `tests` alone (2,564 + 3 skipped), and the audit file's 5 then ran with the test module it imports |

## Conclusion (2026-11-21)
**What ships, if merged (engine 2fdb883, tools to 9adca1f):** the audio encoder's multiplies on the tensor cores and its attention in shared memory.
- The split multiply is x2: hi + lo BF16 parts against `[W | W]`, FP32 accumulation, the doubled weights kept (439 MiB).
- Attention: `window_attention_tiled`, byte-identical to the old kernel.
- The rotary table is computed by the same device code.
- The scratch is still released after every clip.
- `LFM3_AUDIO_ENCODER=fp32` reproduces the old bytes.

| rule | verdict |
|---|---|
| 1, accuracy on the 40 clips | PASS (ratio unchanged on every clip, 0.152–0.228) |
| 2, recognition on 2,620 utterances | PASS (WER 2.5987% against HF's 2.6062%; upper bound +0.0149 points); 42 changed transcripts read, all coherent |
| 3, the encoder alone at most HF's at 35 and 90 s | **FAIL** (0.0219 against 0.0123 s; 0.0463 against 0.0279 s, allocations included) |
| 4, sanitizers | PASS (0 errors, 0 leaks, 0 hazards; hashes identical) |
| 5, nothing else changes | PASS |
| 5b, memory | PASS by the corrected measure (+531 MiB, bound 1 GiB) |
| amendment 6, faster than the deployed server at every clip | PASS |

**Time to first token, same window as the deployed server:**

| clip | deployed | this round | change |
|---|---|---|---|
| 90 s | 0.5415 s | 0.4247 s | −21.6% |
| 35 s | 0.2273 s | 0.1824 s | −19.8% |
| 10.4 s | 0.0896 s | 0.0777 s | −13.3% |
| 5.8 s | 0.0622 s | 0.0573 s | −7.9% |
| 2.1 s | 0.0452 s | 0.0448 s | −1.0% |

- Against HF (HF measured in the 06:25 window, not this one), native is still 2.8× slower at 90 s and 2.3× at 35 s, but already faster on the short clip.
- **Where the 90 s time to first token goes now** (each part measured on its own): log_mel 0.171 s, the language model's prefill about 0.139 s, the encoder 0.046 s, about 0.07 s else. log_mel is now the largest part (HF's front end: 0.015 s).

**Failures and lessons, kept:**
- The exactness script ran rule 4 on the wrong binary (amendment 3).
- Amendment 1 measured memory with VmRSS, which does not see GPU memory on Thor (amendment 4).
- Amendment 4's own baseline drifted by 510 MiB.
- Amendment 5's wrap cost 5.4 ms and was withdrawn.
- The kept scratch slowed the language model's prefill by an effect not understood: one small `cudaMalloc` + `cudaFree` after the encode restored it. An isolated probe could not reproduce it.
  - **This is open, and it may reach beyond audio.** A text prompt's prefill also makes no allocation calls.
  - Any future change that removes allocation calls from a request (a memory pool, for one) must time the language model's prefill too.

**Open, for later rounds:**
1. log_mel (0.171 s at 90 s; the DFT option is Tripp's call).
2. The allocation effect: its mechanism (the memory clock needs root to read), and whether text prefills pay it.
3. *(Withdrawn 2026-11-21:)* the first audio request after a start. The service's `--warmup` already sends an audio clip; the 2.2 s was measured on servers started without it.
4. The encoder against rule 3: `split_rows` (4.5 ms) fused into the producers, `conv_in` and `depthwise` (6.2 ms), and the allocations (~12 ms).
5. The wrap (c968b5d) for the vision encoder's and this encoder's `[W | W]`, if memory matters.
