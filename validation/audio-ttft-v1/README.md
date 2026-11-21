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
- **Found:** the server's warm-up does not run the audio encoder. The first audio request after a start pays for the encoder's construction, its allocations and the doubled weights. This is noted for the warm-up, not changed in this round.

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
