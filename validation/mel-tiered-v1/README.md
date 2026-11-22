# mel-tiered-v1: the correctly rounded log-mel front end, made fast by tiered precision

**Goal.** Cut log_mel, the largest part of audio time to first token, without giving up the guarantee that every feature is correctly rounded (validation/audio-frontend-v1 amendment 2).
- At 90 s, time to first token is 0.425 s (audio-ttft-v1 amendment 6) and log_mel is 0.171 s of it: F2 on 4 threads. HF's front end takes 0.015 s.
- Tripp's go: 2026-11-21, after audio-ttft-v1 was merged and deployed.

## Where F2's time goes (from audio-frontend-v1's record, before any code here)
- **F2** computes every frame by a double-double FFT (error-free TwoSum and TwoProduct), about 5–15× the arithmetic of a double FFT. Its bound decides nearly every output, so quad almost never runs: 0.680 s at 1 thread, 0.171 s at 4, for the 90 s clip.
- **F1** computes every frame by a double FFT. Its bound leaves a few outputs undecided, about 0.02% on the 2,620 utterances (43,477 quad recomputes). Each is recomputed in quad, a direct 400-term DFT per tap at about a millisecond. So F1 was slower than F2 (1.455 / 0.395 s), almost all of it quad.
- Both produce the same floats: every decided output is the correctly rounded value of the same exact mathematics, and that value is unique.

## The change: F3, tiered precision
Each frame:
1. F1's double FFT, power, mel sums and bound, as today's F1.
2. An output F1's bound cannot decide takes F2's path for its frame: the double-double FFT of that frame, computed at most once per frame, and that mel's sum and bound exactly as F2 computes them.
3. An output F2's bound cannot decide takes quad, as today. Undecided even in quad: counted and reported (expected never).

Every output is decided by a bound already validated (F1's or F2's, audio-frontend-v1 amendment 2 rule 1 on Thor and x86). So **F3's floats must equal F2's, byte for byte.**
- Expected cost: about the double path's on ordinary speech, and at most about F2's plus the double path's on clips where most frames fall back (near silence).
- `LFM3_AUDIO_FEATURES=exact-f3` selects F3. If rules 1–4 pass, F3 becomes the default. `exact-f2`, `exact-f1` and `dft` still select the others.
- The thread count stays `kMelThreads` = 4.

## Rules (set before any code)
1. **Byte-identical to F2 (gated).** `audio-mel-check --transform f3 --compare` against F2's features, on Thor (aarch64) and on the x86 workstation, at 1 and 4 threads:
   - every one of the 2,620 utterances, the 40 dumped clips and the 12 edge clips identical;
   - F3's fast, double-double and quad counts reported, and undecided 0;
   - `--repeat 2` (every run equal to the first).
2. **Speed (gated; quiet Thor, the usual window).** `audio-encoder-bench` `log_mel` alone, F2 and F3 in one window, median of 5, at 4 threads, on audio-ttft-v1's five clips:
   - F3 at 90 s at most **0.060 s** (F2: 0.171 s);
   - F3 at most F2 at every one of the five clips.

   Reported: F3 at 1 thread; the edge clip with the most fallbacks, F2 against F3.
3. **Time to first token (gated; the same window).** The new server against the deployed one (`4025f693822c`'s build), ABAB on the five clips (`measure_lfm3_audio_ttft.py`, 1 untimed + 5 timed):
   - the new server's median at most the deployed one's at every clip;
   - texts identical.

   Reported: the saving at each clip.
4. **Nothing else changes (gated).** The features are the same bytes, so every downstream output must be too:
   - `lfm3-speech`'s TTS regression 36 SAME;
   - rule-6 audio-in files (360 + 93), e2e and split identical to amendment 6's x2 run (`a6-speech-rule6`);
   - the server's audio in, 8 and 32 at once, the text gate;
   - the host suites (with `test_audio_input_host.py`'s F3 cases at 1–12 threads and from 8 callers, plain, ASan+UBSan and TSan);
   - `LFM3_AUDIO_FEATURES=exact-f2` and `dft` still give their bytes.

## Development (2026-11-21, Thor, not a rule run; `build/mel3-check/`, `build/mel3-diagbin/`)
- **F3 as first built (448c1c5): byte-identical to F2, but only 1.5× faster.** On the 90 s clip, F2 and F3 at 1 and 4 threads all gave the same bytes. Per run: F2 0.69 / 0.18 s, F3 0.45 / 0.12 s.
- **Why:** 4,483 of 8,999 frames needed the second tier, though only 8,861 of 1.15 M outputs did (about 2 per frame). So F3 still paid the double-double FFT on half the frames.
  - From the two runs: the double path costs about 12 µs a frame, and a double-double frame about 77 µs.
- **The bound is the lever.** A diagnostic build (not kept) scaled F1's bound down. The frames needing the second tier fell about linearly: 4,483 (×1), 2,266 (×0.3), 995 (×0.1), 332 (×0.03), 123 (×0.01).
  - Outputs were identical at every scale. That is expected, since a looser or tighter bound changes only which tier decides; but a scaled bound is not a proof, so it is not used.

## Amendment 1 (2026-11-21, before any rule data): F3's first tier under a 2-norm bound
F1's bound is a max-norm recurrence in A0 = max |x[n]|: 435,103 u·A0 per component, u = 2^-53. It assumes every rounding error lines up against every bin. A bound on the whole transform's error vector in the 2-norm is also rigorous, and it scales with the frame's energy ‖x‖₂ instead. F3's first tier takes the smaller of the two (both hold). F1 and F2 are unchanged.

**The derivation, for this code** (`detail::fft`: decimation in time, levels n = 5, 25, 100, 400 with radices 5, 5, 4, 4). Round to nearest throughout.
- **The state.** At each level, the computed state is the 400 outputs of all the length-n DFTs, and e is its error vector.
  - The exact state has ‖v‖₂ = √n ‖x‖₂. Each length-n DFT multiplies a subsequence's 2-norm by √n, and the subsequences partition x.
  - A level applies the twiddles T (diagonal, |t| = 1 in exact arithmetic), then the radix-r butterflies F (block-diagonal, ‖F‖₂ = √r).
  - So e' = F T e + F δt + δb, where δt is the twiddle multiply's rounding and δb the butterflies'.
- **Twiddles** (`twiddle()`: re' = fma(re, ĉ, fl(im·ŝ)); im' = fma(im, ĉ, −fl(re·ŝ))). The table values ĉ, ŝ are within δw of the exact cos and sin.
  - Expanding the roundings: |re' − re*| ≤ (|re| + |im|)·k1, with k1 = δw + u(1 + δw) + u(1 + δw)(1 + u); the same for im.
  - So |δt_i| ≤ √2·k1·(|re| + |im|) ≤ 2 k1 |a_i| = κ |a_i|, and ‖δt‖₂ ≤ κ‖ṽ‖₂.
- **Radix 4** (additions only; ±i exact). Each output component is a sum of 4 components of the block's inputs, with each term rounded at most twice, so its error is ≤ γ2·S4, where S4 = Σ(|re| + |im|) over the block ≤ 2√2‖t_b‖₂.
  - A block's 4 complex outputs then have ‖δb‖₂ ≤ √4·√2·γ2·2√2‖t_b‖₂ = 8 γ2 ‖t_b‖₂. So b4 = 8 γ2, with γk = k u / (1 − k u).
- **Radix 5** (`X_q = t0 + Σ_j fma(t_j.re, c, fma(t_j.im, s, ·))`). Each component is a chain of 8 fused operations over 9 exact terms, so its rounding error is ≤ γ8 times the sum of the terms' magnitudes, ≤ γ8 (1 + δw) S5. The constants' table error adds δw·S5. Here S5 ≤ √10‖t_b‖₂.
  - Over the 5 outputs: b5 = √5·√2·√10·(γ8(1 + δw) + δw) = 10 (γ8(1 + δw) + δw).
- **The recurrence**, with N0 ≥ ‖x‖₂ and V = √(n_prev)·N0 + E (a bound on the computed state's norm):
  - level 1 (n = 5, no twiddles: k = 0 only): E = b5·N0;
  - each later level: E' = √r·(E + κV) + b_r·(1 + κ)·V.
  - The final E bounds every bin's complex error, so each component's too. It replaces e in the existing power bound: dpower = e(2(|re| + |im|) + 4e) + 2.02 u·power.
- **δw** is measured, not assumed: the largest |table − exact| over the 400 entries of `turn_cos` and `turn_sin`. "Exact" is `quad_turn()`'s quad values (a few quad ulps), with 2^-200 added for them. In amendment 2's blanket bound it was 24u.
- **N0** = √(Σ x[n]²) computed in double, times (1 + 1e-9) (the sum's 400 roundings are under 1e-13 relative).
- **The result, with δw near u:** E ≈ 2,290 u·‖x‖₂. Windowed speech frames have ‖x‖₂ ≈ 3–9 A0, so this is about 20–60× tighter than 435,103 u·A0. By the diagnostic, that should leave about 3–6% of frames for the second tier.

**An empirical check, added to rule 1 (gated):** the bound must hold against quad on real and adversarial frames.
- The double FFT's actual per-bin error, measured against a quad direct DFT, divided by the bound, must be below 1 on every bin of every checked frame.
- Frames checked: impulses, a full-scale square wave, random full-scale noise, and near-silence (host test); and 2,000 frames from the 90 s clip and the edge clips (Thor).
- Reported: the largest ratio.

## Rule 1 (2026-11-21): PASS on Thor and x86
**Thor** (aarch64; `build/mel3-rule1/`, `tools/thor/mel_tiered_rule1.sh`):
- Binaries: `audio-mel-check` from main 717cca6 `28185279…` (the reference: F2 as deployed); from this round, e816690's tree, `9dd6e929…`.
- Built as the engine compiles the header: nvcc's host pass, `-O3 --fmad=false`.

**x86** (the workstation; scratchpad `mel3-x86/`):
- The same WAVs, copied from Thor.
- g++ 13, `-O3 -ffp-contract=off -mfma`: main's tree `fedfea71…`, this round's `66a06754…`.

**Results.** The counts below are summed over `--repeat 2` (two passes). Identity is checked on every pass.

| set | F3, 1 and 4 threads, against main's F2 | F3 decisions (two passes) | frames to the second tier |
|---|---|---|---|
| 2,620 utterances | 2,620 of 2,620 identical, both platforms | 496,885,140 fast; 271,416 second tier; 308 quad (F2's own 154 per pass); 0 undecided | 255,492 of 3,884,038 (6.6%) |
| 40 dumped clips | 40 of 40, both platforms | 8,777,690; 5,154; 4; 0 | 4,828 of 68,616 (7.0%) |
| 12 edge clips | 12 of 12, both platforms | 517,968; 176; 0; 0 | 172 of 4,048 (4.2%) |

- **The refactor changed neither F1 nor F2.** This round's F2 at 4 threads equals main's on every clip, on both platforms, and so does F1 (Thor).
  - F1 still sends 4,650,016 outputs to quad on the 2,620 (two passes). That is why it was slow.
- `dft` reproduces audio-frontend-v1's original bytes on all 2,672 clips (Thor).
- **The bounds against quad (amendment 1's check, Thor):** the largest actual per-bin error divided by the bound, on 2,254 frames of the 90 s clip and on every frame of the 12 edge clips (up to 200 each):
  - 2-norm bound: **0.012** at most;
  - max-norm bound: 0.0007 at most.

  The 2-norm bound is still 80× conservative on real frames, yet it holds everywhere.
  - The host test adds an impulse, a full-scale square wave, full-scale noise and near-silence: all under 1.
- **The table twiddles' measured error: 8.9u**, under amendment 2's assumed 24u. The bound uses the measured value.

## Rules 2 and 3 (2026-11-21, Thor, quiet 19:02–19:04; `build/mel3-run/`, `tools/thor/mel_tiered_run.sh`)
Binaries from e816690's tree:
- bench `mel3-bench` `bd90eace…`; server `mel3-srv` `658cd733…`; `lfm3-speech` `mel3-speech` `d9407b6a…`;
- deployed, for comparison: `merge2-srv` `4025f693…`.

Window checks: the service stopped; MONITOR CLEAN (99 samples, foreign processes in 0); power checked before every step.

**Rule 2: PASS.** log_mel alone, `audio-encoder-bench`, median of 5, in seconds:

| clip | F2, 4 threads | F3, 4 threads | F3, 1 thread (reported) | mel hashes F2 = F3 |
|---|---|---|---|---|
| 2.1 s | 0.0054 | 0.0012 | 0.0033 | yes |
| 5.8 s | 0.0111 | 0.0027 | 0.0092 | yes |
| 10.4 s | 0.0199 | 0.0045 | 0.0165 | yes |
| 35 s | 0.0666 | 0.0162 | 0.0607 | yes |
| 90 s | 0.1709 | **0.0365** (bound 0.060) | 0.1431 | yes |

F3 is 4.1–4.7× faster than F2 at every clip.
- **A defect:** the runner was to time the edge clip with the most second-tier frames too. It read rule 1's edge list before rule 1 had written it (the two ran in parallel), found no clips, and timed none. This reported item runs again in its own quiet window (`build/mel3-edge/`).

**Rule 3: PASS.** Time to first token from the file, the deployed server (F2) against this round's (F3), ABAB, in seconds:

| clip | deployed A1 | F3 B1 | deployed A2 | F3 B2 | saved | change |
|---|---|---|---|---|---|---|
| 2.1 s | 0.0446 | 0.0404 | 0.0446 | 0.0400 | 0.0044 | −9.9% |
| 5.8 s | 0.0570 | 0.0490 | 0.0577 | 0.0490 | 0.0084 | −14.6% |
| 10.4 s | 0.0779 | 0.0625 | 0.0779 | 0.0622 | 0.0156 | −20.0% |
| 35 s | 0.1807 | 0.1313 | 0.1824 | 0.1315 | 0.0502 | −27.6% |
| 90 s | 0.4220 | 0.2883 | 0.4241 | 0.2890 | 0.1343 | **−31.7%** |

Texts are identical in every run.

## Rule 4 (2026-11-21, 19:04–19:16, not timed)
| check | result |
|---|---|
| `lfm3-speech` TTS regression | 36 SAME, 0 differ or failed |
| rule-6 audio-in files against amendment 6's (`a6-speech-rule6`) | dump 360 of 360, edge 93 of 93; none missing or extra |
| e2e and split, against the validated runs | 90 of 90 and 90 of 90 |
| server: audio in against `mel3-speech` | PASS (repeat identical, text control identical) |
| server: 8 and 32 at once | 8 of 8 and 32 of 32 identical to `lfm3-speech`; PASS |
| text gate | exit 0 |
| `exact-f2` and `dft` still give their bytes | yes, at the feature level: rule 1 (F2 = main's F2, `dft` = audio-frontend-v1's bytes, all 2,672 clips) |
| host test file `test_audio_input_host.py` (plain, ASan+UBSan, TSan) | 91 passed (F3 = F2 = F1 at 1–12 threads and from 8 callers; the second tier reached; the bounds against quad) |

The full host suites are running.
