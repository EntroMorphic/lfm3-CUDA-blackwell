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
