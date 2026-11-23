# tcgen05-v2: the tcgen05 GEMM path on current main, and rule 4b decided

**Goal.** Turn on `LIQUID_LFM3_TCGEN05` if it holds on today's engine. This is lfm::gemm's tensor-memory path for problems with at least 257 rows, documented as bitwise identical to mma.sync (validation/tcgen05-gemm-v1).
- **The payoff:** the language model's prefill is now the largest part of audio time to first token (about 0.139 s of 0.24 s at 90 s). tcgen05-gemm-v1 measured 2.38× at large M.
- **Tripp's go:** 2026-11-22, item 1 of the remaining list.

**Where tcgen05-gemm-v1 stopped (2026-11-17):**
- Rules 1, 2, 3, 4a, 5 and 6 passed.
- Rule 4b, the hour-long server soak, failed as recorded. The tcgen05 run passed every gate, but the OFF baseline (main's own path) had 4 of 56 unstreamed crowd replies differ in pair 2.
- That baseline failure was not reproduced in 20 crowd rounds (crowd-identity-v1), and its cause is unknown: the harness then could not tell a 429 that ran out of retries from a content difference. It can now (crowd-identity-v1 added status, retries and differing fields to every crowd difference).

**The merge.** tcgen05-gemm-v1 merged onto main 6943e84's line cleanly (fc8811f). The hooks are unchanged: `lfm::gemm::run` and `linear_tiled` try tcgen05 for M ≥ 257, and the option stays OFF by default.
- **New since then:** the audio encoder's split multiplies (audio-ttft-v1) go through `lfm::gemm::run` with FP32 output and a doubled K. They fit tcgen05's contract (N a multiple of 256, K of 64) once a clip gives at least 257 tokens (about 20.5 s). Their shapes are added to the entry-point check (`a.*`).

## Rules (set before any data)
1. **Exactness at the entry point (gated):** `lfm3-tcgen05-gemm-check`, every engine shape (the language model's, the vision encoder's, and now the audio encoder's x2 shapes) at M = 256, 257, 258, 4,096 and 32,768 (the LM head to 4,096). Integer, normal and split data; FP32 and BF16 out. Every case 0 elements differing, and tcgen05 launched exactly where its contract says.
2. **Byte identity, ON against OFF (gated), on Thor:**
   - `lfm3-speech`: the TTS regression 36 SAME; the rule-6 audio-in files (360 + 93) identical to `mel3-speech-rule6`; e2e and split identical to the validated runs;
   - the vision encoder, 31 cases: 713 of 713 stage files identical to main's;
   - the server: audio in, 8 and 32 at once, images, the image cache, TTS, the text gate, and (quiet) the stream check and the batch-server check.
3. **Sanitizers (gated):** on `lfm3-speech` ON: memcheck with leak check, synccheck and initcheck on a 4,096-token text prefill and on the 90 s clip (tcgen05 in the encoder and the prefill). 0 errors, outputs identical to the plain run. racecheck is not repeated: the kernel is unchanged since tcgen05-gemm-v1's clean racecheck.
4. **Rule 4b, the server soak pair (gated):**
   - tcgen05-gemm-v1's harness (`check_lfm3_tcgen05_soak.py`: 3,600 s, 12 clients, text share 0.25, images), ON first, then OFF, each on a quiet Thor;
   - the gates and reading of amendments 2 and 2a (`tcgen05_4b_pair_verdict.py`);
   - a crowd difference counts as before (a 429 that ran out of retries fails too), but each one's status, retries and fields are now recorded and reported.
5. **Speed (gated; quiet Thor, ON and OFF alternating ABAB in one window):**
   - text prefill (`lfm3-speech`, text-only prompts of 512, 1,149 and 4,096 tokens, each 4 times per process): ON at most OFF at every length;
   - audio time to first token on the five clips (the server, `measure_lfm3_audio_ttft.py`): ON at most OFF at every clip.

   Reported: the gains; the audio encoder alone (`audio-encoder-bench`) ON against OFF; each process's first request apart from the rest (alloc-effect-v1's finding).
6. **Coherency:** by rule 2 the outputs are the same bytes as validated outputs. Nothing new to read unless rule 2 fails.

If every rule passes, the option goes ON by default (CMake) and the server deploys as before.

**Deviation, recorded before any data from it (2026-11-22, 09:55):** rules 3 and 5's "4,096-token" text prompt is 4,000 tokens. `lfm3-speech`'s capacity of 4,096 includes the 8 new tokens, and the first request file (alloc-effect-v1's `L4096.req.json`) exceeded it. The file was rebuilt with a 4,000-token prompt before either rule's runs reached it; its name is unchanged.

## Rules 1–3 (2026-11-22, 09:54–10:32, not timed; `build/tc2-exact/`, `tools/thor/tcgen05_v2_exact.sh`)
Binaries from 55b5170's tree (`build/tc2-src`), by `build_tcgen05_engine.sh`:
- ON: `lfm3-speech` `18eac2e8…`, server `a023e8c2…`, worker `6bdf11d9…`, `lfm3-vision-check` `7b768202…`, `lfm3-tcgen05-gemm-check` `ab4ca6fa…`;
- OFF: `lfm3-speech` `7a14280c…`, server `c5cbd8db…`, worker `027dbefb…`.

**Rule 1: PASS.** `lfm3-tcgen05-gemm-check`: 563 cases, 0 failed. Every case has 0 elements differing and tcgen05 launches exactly where its contract says. 210 of the cases are the audio encoder's x2 shapes, new in this round.

**Rule 2: PASS** (the parts that are not timed; the stream and batch-server checks run in the quiet window):

| check | result |
|---|---|
| `lfm3-speech` ON TTS regression | 36 SAME, 0 differ or failed |
| rule-6 audio-in files, ON against main's (`mel3-speech-rule6`) | dump 360 of 360, edge 93 of 93 |
| e2e and split | 90 of 90 and 90 of 90 |
| vision encoder ON, 31 cases | 713 of 713 stage files identical to main's |
| server ON: audio in against main's `lfm3-speech` | PASS |
| server ON: 8 and 32 at once | 8 of 8, 32 of 32 identical; PASS |
| server ON: images; image cache | PASS; 24 of 24 identical |
| server ON: TTS; text gate | exit 0; exit 0 |

**Rule 3: PASS.** On `lfm3-speech` ON, with the 4,000-token text prefill and the 90 s clip:
- memcheck with leak check: 0 errors, 0 bytes leaked;
- synccheck: 0 errors; initcheck: 0 errors;
- every output identical to the plain run (the JSON without its timing keys, and the 4 other files).

## The quiet window (2026-11-22, 10:47–13:53; `build/tc2-quiet/`, `tools/thor/tcgen05_v2_quiet.sh`)
Window checks: the service stopped for the whole window; MONITOR CLEAN (10,688 samples, foreign processes in 0); power checked before every step.

**Rule 2, the quiet part: PASS.**
- The stream check on the ON server: PASS.
- The batch-server check, ON against OFF as the reference server: PASS. All crowds identical; no real-time stalls at 8, 12 or 16 rows; 7 TTS + 5 spoken joining all done.

**With rules 1–3 above, rule 2 passes in full.**

**Rule 4b, the soak pair: PASS** (`tcgen05_4b_pair_verdict.py`, amendments 2 and 2a). ON ran 11:19–12:35, then OFF 12:36–13:52, 3,600 s each.

| | ON | OFF |
|---|---|---|
| replies, all status 200 | 3,544 | 3,538 |
| in-soak differences; errors | 0; 0 | 0; 0 |
| crowd after, unstreamed / streamed | 56/56 / 56/56 | 56/56 / 56/56 |
| server RSS idle growth | +10.2 MiB | +0.4 MiB |
| cgroup anon growth, checkpoints 2 to 4 | +31.2 MiB | +6.9 MiB |
| 429 retries per reply | 18.66 | 18.11 |

- **Equivalence (2a):**
  - server RSS: ON +10.2 against the bound max(0.4, 0) + 32 = 32.4: pass;
  - cgroup anon: ON +31.2 against 6.9 + 32 = 38.9: pass.
- The retries ratio is 1.03, so no investigation is triggered.
- Each harness exited 1 on its own memory verdict, which amendment 2 made reported-only. The pair verdict is **rule_4b_pass: true**.

So tcgen05-gemm-v1's open rule passes on today's engine. Its window-7 OFF crowd failure did not recur on either side.

**Rule 5, speed: FAIL as written, on one cell.**

Text prefill (`lfm3-speech`), seconds, first request | median of requests 2–4:

| tokens | OFF 1 | ON 1 | OFF 2 | ON 2 | ON ÷ OFF |
|---|---|---|---|---|---|
| 512 | 0.0655 / 0.0600 | 0.0394 / 0.0382 | 0.0652 / 0.0601 | 0.0395 / 0.0383 | 0.638 |
| 1,149 | 0.1259 / 0.1257 | 0.0733 / 0.0730 | 0.1261 / 0.1263 | 0.0739 / 0.0730 | 0.580 |
| 4,000 | 0.6016 / 0.5998 | 0.4357 / 0.4356 | 0.6020 / 0.6008 | 0.4345 / 0.4361 | 0.726 |

Audio time to first token (the server, from the file), seconds:

| clip | OFF A1 | ON B1 | OFF A2 | ON B2 | saved | texts |
|---|---|---|---|---|---|---|
| 2.1 s | 0.0322 | 0.0315 | 0.0316 | 0.0316 | 0.0004 | identical |
| 5.8 s | 0.0403 | **0.0403** | 0.0401 | 0.0399 | 0.0001 | identical |
| 10.4 s | 0.0537 | 0.0529 | 0.0529 | 0.0529 | 0.0004 | identical |
| 35 s | 0.1099 | 0.0835 | 0.1080 | 0.0831 | 0.0257 (−23.6%) | identical |
| 90 s | 0.2391 | 0.1799 | 0.2394 | 0.1761 | 0.0613 (−25.6%) | identical |

- **The failing cell: the 5.8 s clip.** ON B1 0.0403 s is above OFF's mean of 0.0402 s, by 0.1 ms. The rule said "ON at most OFF at every clip", so rule 5 fails.
- **Why the cell measures noise:** the 5.8 s clip's prompt is about 100 rows. No GEMM in it reaches tcgen05's 257 rows, so ON and OFF run the same kernels there.
- **The rule was written without allowing for noise where the two builds run identical code. That is the design's fault, and it is recorded, not explained away.**
- **Reported:** the audio encoder alone, OFF / ON, with identical output hashes at every clip:

  | clip | OFF | ON |
  |---|---|---|
  | 2.1 s | 0.0052 s | 0.0053 s |
  | 5.8 s | 0.0082 s | 0.0088 s |
  | 10.4 s | 0.0109 s | 0.0108 s |
  | 35 s | 0.0203 s | 0.0177 s |
  | 90 s | 0.0450 s | 0.0382 s |

  It is faster with tcgen05 from 35 s, where the encoder's tokens pass 257.
- The first request of each process is not slower than the rest here (text 512, 1,149, 4,000).

**Standing:** rules 1, 2, 3, 4b and 6 PASS; rule 5 FAIL as written (one 0.1 ms cell where tcgen05 cannot run). The option stays OFF until Tripp decides.

## Amendment 1 (2026-11-22, Tripp's choice, before any data from it): rule 5's audio part, with noise allowed where the builds run the same code
**The amended gate, decided on a fresh run.** Time to first token on the five clips, OFF and ON alternating A1 B1 A2 B2, quiet, as before. For each clip:
- **If tcgen05 never launches on it** (a trace of the ON server shows 0 `gemm_1cta` / `gemm_2cta` kernels), ON's worst run may be at most **1%** above OFF's mean. The two builds then run the same kernels; the 1% covers run-to-run noise (the failing cell was 0.25%).
- **Otherwise**, ON's worst run must be at most OFF's mean, as before.
- Texts identical in every run.

**The launch check:** nsys (CUDA trace) on the ON server, two requests of each clip, after the timed window and not timed. It counts `gemm_1cta` and `gemm_2cta` kernels per clip. Expected 0 on the 2.1, 5.8 and 10.4 s clips (prompts under 257 rows) and more than 0 on the 35 and 90 s clips; that is the positive control, and if it shows 0 there the check is void.

**Unchanged:** the failed run stays in the record as a failure under the rule as first written. Rule 5's text part passed and is not rerun.

## Amendment 1's run (2026-11-22, quiet 16:02–16:04, launch check after; `build/tc2-r5/`, `tools/thor/tcgen05_v2_rule5b.sh`): FAIL as written
Window checks: the service stopped; MONITOR CLEAN (83 samples, foreign processes in 0); power checked before every run.

| clip | OFF A1 | ON B1 | OFF A2 | ON B2 | tcgen05 kernels (trace, 2 requests) | bound | saved |
|---|---|---|---|---|---|---|---|
| 2.1 s | 0.0388 | 0.0388 | 0.0394 | 0.0390 | 4 | 0.0391 | 0.0002 |
| 5.8 s | 0.0458 | 0.0462 | 0.0464 | **0.0463** | 4 | 0.0461 | −0.0001 |
| 10.4 s | 0.0576 | 0.0574 | 0.0573 | **0.0579** | 4 | 0.0575 | −0.0002 |
| 35 s | 0.1152 | 0.0890 | 0.1146 | 0.0895 | 464 | 0.1149 | 0.0256 |
| 90 s | 0.2512 | 0.1848 | 0.2500 | 0.1859 | 464 | 0.2506 | 0.0652 |

Seconds; texts identical in every run. The launch check is valid (464 kernels on the 35 and 90 s clips).

- **Rule 5 (amendment 1) FAILS:**
  - tcgen05 launched on every clip, so the 1% allowance applied to none;
  - ON's worst run is over OFF's mean on the 5.8 s clip (+0.2 ms) and the 10.4 s clip (+0.4 ms).
- **A correction to amendment 1's premise, and to what I told Tripp.** I wrote that the short clips run the same kernels ON and OFF. They do not.
  - The trace shows 2 tcgen05 launches per request inside the audio encoder: `gemm_2cta` with grids 20×1 and 10×1, right after `depthwise` and `split_rows`.
  - These are the downsampler's 1×1 convolutions (`lin_mid`, `lin_end`: N 256, K 512 under x2), over W2 × 32 rows, so they pass 257 rows even on a 2 s clip.
- **So the short clips' 0.1–0.4 ms is a real, small regression,** not noise. With N = 256, tcgen05 has one output tile per 128-row block and leaves most of Thor's SMs idle, where mma.sync spreads the same work over many small tiles.
  - The encoder bench in the quiet window agrees: 5.8 s clip 0.0082 s OFF against 0.0088 s ON.
- The option stays OFF. The fix is the dispatch: tcgen05 only where it is faster (amendment 2, next).

## The speed map (2026-11-22, quiet 16:25–16:35; development evidence; `build/tc2-speedmap/`, `native/tests/lfm3_tcgen05_gemm_speed.cu`)
Each engine GEMM shape at each M ≥ 257, `lfm::gemm::run` with tcgen05 on and off, the median of 15 runs (CUDA events). The service was stopped and power unchanged. Off time ÷ on time (above 1: tcgen05 faster):

| shape | N | K | M = 257 … 4,096 (or the downsampler's rows) |
|---|---|---|---|
| lm.in_proj | 6,144 | 2,048 | 2.94, 2.80, 3.37, 2.23, 2.21, 2.51, 2.83 |
| lm.q_o | 2,048 | 2,048 | 2.59, 2.34, 2.86, 3.10, 3.25, 3.32, 2.38 |
| lm.kv | 512 | 2,048 | 1.60, 1.59, 1.58, 2.54, 2.21, 2.83, 3.10 |
| lm.w1w3 | 7,680 | 2,048 | 2.18, 1.66, 1.77, 2.36, 2.40, 2.66, 2.92 |
| lm.w2 | 2,048 | 7,680 | 2.53, 2.34, 3.20, 2.12, 3.47, 1.83, 1.81 |
| v.patchifier | 1,024 | 3,072 | 2.41, 2.43, 2.39, 2.77, 2.66, 2.94, 2.60 |
| v.in_proj | 3,072 | 2,048 | 2.35, 1.92, 2.21, 2.61, 2.57, 1.99, 2.14 |
| v.q_out | 1,024 | 2,048 | 2.02, 2.06, 2.09, 2.34, 2.30, 2.59, 2.41 |
| v.kv | 512 | 2,048 | 1.26, 1.26, 1.26, 2.09, 1.96, 2.37, 2.61 |
| v.w1w3 | 2,560 | 2,048 | 2.00, 1.99, 2.18, 2.42, 2.42, 2.20, 1.99 |
| v.w2 | 1,024 | 5,120 | 2.91, 2.92, 2.93, 3.12, 3.08, 2.09, 1.62 |
| v.proj1 | 4,096 | 8,192 | 2.08, 1.82, 2.20, 2.38, 4.41, 2.55, 2.36 |
| v.proj2 | 2,048 | 8,192 | 1.96, 1.78, 1.81, 1.39, 2.90, 1.37, 1.45 |
| **a.lin_mid** (the downsampler's 1×1 convolutions) | **256** | 512 | M = 257, 512, 1,024: **0.73, 0.76, 0.76**; 1,700, 4,096, 8,192: 1.22, 1.20; 16,384: 1.06; 36,000, 72,000: **0.82, 0.82** |
| a.attn | 768 | 1,536 | 1.18, 1.20, 1.82, 1.73, 1.66, 1.81, 2.13 |
| a.conv_in | 1,536 | 1,536 | 1.56, 1.23, 1.73, 1.83, 2.07, 2.16, 2.29 |
| a.w1w3 | 1,792 | 1,536 | 2.22, 1.79, 1.72, 2.15, 2.05, 2.22, 2.31 |
| a.w2 | 768 | 3,584 | 1.62, 1.63, 2.56, 2.36, 2.30, 2.62, 2.37 |
| a.lin_out | 768 | 8,192 | 2.04, 2.01, 3.40, 2.84, 2.76, 1.31, 1.32 |
| a.proj1 | 2,048 | 1,536 | 2.17, 1.71, 2.02, 2.23, 2.16, 2.12, 2.26 |
| a.proj2 | 2,048 | 4,096 | 2.49, 2.44, 3.02, 3.11, 2.32, 2.17, 2.49 |

**Every shape with N ≥ 512 is faster on tcgen05 at every M tested** (1.18× to 4.41×). Only N = 256 loses, at small and very large M, which is what the short clips hit.

## Amendment 2 (2026-11-22, before any data from it): tcgen05 only for N ≥ 512
**The change.** `tcgen05::eligible` adds one condition: `p.n >= 2 * kBN` (512), at least two of tcgen05's 256-wide output tiles. Everything else stays as it is (M ≥ 257, the contract).
- The downsampler's N = 256 multiplies go back to mma.sync. The result is bitwise identical either way (rule 1).
- On this evidence no engine shape with N ≥ 512 loses. The narrow middle range where N = 256 wins (1.2×, about 1,700–8,000 rows) is given up for a simple rule.

**What is run again, on the new build:**
- **Rules 1–3:** the exactness script. The entry-point check now also covers `a.lin_mid` (256 × 512) and expects tcgen05 only where M ≥ 257 and N ≥ 512.
- **Rule 5, fresh:** text prefill and audio time to first token, under amendment 1's per-clip rule. Clips with 0 tcgen05 launches (by the trace) may be up to 1% over OFF; the others must be at most OFF. Now the short clips are expected to launch none.

**Not run again: rule 4b.** Its pass stands. The change only narrows which problems take the tcgen05 path that rule 4b soaked for an hour on each side; it adds no path and no new state.

If rules 1–3 and 5 pass, the option goes ON by default.

## Amendment 2's runs (2026-11-22)
**Rules 1–3, the N ≥ 512 build (16:38–17:15; `build/tc3-exact/`): PASS.**
- **Rule 1:** 593 entry-point cases, 0 failed. 240 are audio-encoder cases; `a.lin_mid` (N 256) takes mma.sync, as the contract now says.
- **Rule 2** (the parts that are not timed): every check as before.
  - TTS regression 36 SAME; audio-in dump 360 of 360 and edge 93 of 93 against main; e2e and split 90 of 90;
  - vision 713 of 713;
  - server audio in, 8 and 32 at once, images, the image cache 24 of 24, TTS, the text gate: all pass.
- **Rule 3:** memcheck (0 errors, 0 leaks), synccheck and initcheck clean on the 4,000-token prefill and the 90 s clip; outputs identical to the plain run.

**Rule 5 under amendments 1 and 2 (quiet 17:30–17:32, launch check after; `build/tc3-r5/`): FAIL as written.**
- MONITOR CLEAN (106 samples).
- **The launch check is valid:** 0 tcgen05 kernels on the 2.1, 5.8 and 10.4 s clips; 460 on the 35 and 90 s clips. Amendment 2 did what it was for.

Audio time to first token, seconds:

| clip | OFF A1 | ON B1 | OFF A2 | ON B2 | tcgen05 | bound | verdict |
|---|---|---|---|---|---|---|---|
| 2.1 s | 0.0388 | **0.0394** | 0.0387 | 0.0316 | 0 | 0.0391 (+1%) | over |
| 5.8 s | 0.0455 | **0.0463** | 0.0457 | 0.0410 | 0 | 0.0460 (+1%) | over |
| 10.4 s | 0.0569 | **0.0576** | 0.0570 | 0.0536 | 0 | 0.0575 (+1%) | over |
| 35 s | 0.1144 | 0.0917 | 0.1090 | 0.0844 | 460 | 0.1117 | pass (−20%) |
| 90 s | 0.2503 | 0.1885 | 0.2407 | 0.1781 | 460 | 0.2455 | pass (−25%) |

Text prefill, first request | median of requests 2–4, seconds:

| tokens | OFF 1 | ON 1 | OFF 2 | ON 2 | worst ON ÷ mean OFF |
|---|---|---|---|---|---|
| 512 | 0.0784 / 0.0758 | 0.0429 / 0.0435 | 0.0782 / 0.0656 | 0.0449 / 0.0433 | 0.615 |
| 1,149 | 0.1739 / 0.1702 | 0.0780 / 0.0784 | 0.1375 / 0.1371 | 0.0798 / 0.0781 | 0.510 |
| 4,000 | 0.8176 / 0.7621 | 0.4743 / 0.4730 | 0.6558 / 0.6540 | 0.4752 / 0.4756 | 0.672 |

**Reading:**
- **On the three short clips the two builds now run the same kernels** (0 tcgen05 launches, by the trace). Yet ON B1 was 0.8–1.5% over the bound, and ON B2 was 6–19% under OFF.
- **That spread is the state effect of alloc-effect-v1, at full size, on identical code.** The same build differs by up to 25% between processes (B1 0.0394 against B2 0.0316 s on the 2.1 s clip). OFF's own text prefill at 1,149 tokens was 0.170 s in its first process and 0.137 s in its second.
- **No 1% gate can be read through a ±20% state swing.** Rule 5 fails as written. What fails it is the measurement, not tcgen05: where tcgen05 runs it is 20–49% faster in every cell.
- **The option stays OFF.** The decision of what comes next is Tripp's.
