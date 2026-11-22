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
