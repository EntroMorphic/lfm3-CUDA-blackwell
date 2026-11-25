# encoder-wrap-v1: the encoders' split multiply without the doubled weights

**Goal.** Drop the `[W | W]` copies that both encoders keep for their split multiply (x2), with every output byte for byte unchanged and no speed lost.
- Both encoders keep one: the audio encoder's `lfm::split::Linear` and the vision encoder's own `doubled_`. Each is built on first use and kept for the process's life.
- The audio encoder's copy is 439 MiB (audio-ttft-v1). The vision encoder's is of the same order.
- The housekeeping item from audio-ttft-v1's open list (5): "the wrap (c968b5d) for the vision encoder's and this encoder's `[W | W]`, if memory matters".

Tripp's go: 2026-11-23 ("Let's do 1, 2, 4, and the housekeeping items").

## History
- audio-ttft-v1 amendment 5 added `Problem::b_k_wrap`: the GEMM reads B's k index modulo it, so W acts as `[W | W]`.
  - `gemm-wrap-check` found it bitwise identical on 154 of 154 cases.
  - The modulo cost the audio encoder 5.4 ms at 90 s.
  - c968b5d replaced the modulo with one compare and subtract. That version was never timed: amendment 6 withdrew the wrap, because its only purpose then, the memory rule, was met another way.
- Since then, tcgen05-v2 sends every GEMM with M ≥ 257, N ≥ 512, N % 256 = 0 and K % 64 = 0 to the tcgen05 kernel. Those are most of both encoders' multiplies.
  - c968b5d's wrap is only in the mma.sync kernels.
  - So the wrap now has to reach the tcgen05 kernel too, or the tcgen05 calls would lose it.

## The change (set 2026-11-23, before any code)
1. **`lfm::gemm::Problem::b_k_wrap`** (0: off): for an N × K B, k at or past the wrap reads k − wrap.
   - `check()` refuses a wrap that:
     - is not a multiple of 8;
     - has an `ldb` below it;
     - has k over twice it;
     - comes with a K × N B;
     - comes on the K = 128 decode path.
   - These are c968b5d's refusals.
2. **mma.sync (`wide_kernel`, `thin_kernel`):** a `WRAP` template parameter.
   - The `WRAP = true` instantiations do c968b5d's compare and subtract.
   - The `WRAP = false` instantiations are today's code, unchanged. Every caller that sets no wrap, the language model and every text path included, runs the same machine code.
3. **tcgen05 (`gemm_2cta`):** a `WRAP` template parameter.
   - With a wrap, B's tensor map covers W itself (N × wrap).
   - The producer's k coordinate is `kt·64 − wrap` past the wrap. A coordinate is computed once per k tile by one thread.
   - Eligibility takes a wrapped B when wrap % 64 = 0, `ldb` = wrap and k = 2·wrap.
   - `WRAP = false` is today's kernel.
4. **Both encoders' x2:** `[hi | lo]` against W itself, with `ldb = in` and `b_k_wrap = in`. Their `doubled_` maps and the `double_rows` kernel go.
   - The GEMM reads the same values in the same order as against `[W | W]`, so every output is byte-identical.
5. Nothing else changes: x1, fp32, the attention kernels, the scratch.

## Rules (set before any code)
1. **The GEMM, bit for bit (gated):** `gemm-wrap-check` (c968b5d's, extended) compares each wrapped GEMM against the same GEMM over an explicit `[W | W]`. FP32 and BF16 outputs.
   - mma.sync: c968b5d's 154 cases (thin and each wide tile).
   - tcgen05: M ∈ {257, 300, 1125, 4500}; (in, out) from both encoders' x2 shapes with out ≥ 512 and out % 256 = 0.
     - The tcgen05 launch counter must rise on every such case: that is the positive control.
     - The same cases with `LFM3_TCGEN05=0` must also match (mma.sync).
   - The invalid uses are refused: c968b5d's 5, plus a tcgen05-eligible shape with a wrap that is not a multiple of 64, which must run on mma.sync and still match.
2. **Unchanged machine code (gated):** `tools/compare_sass.py`, main at the branch point (built the same way) against the candidate, for `liquid-lfm3-server`, `lfm3-speech` and `lfm3-vision-check`.
   - Every function in main's binary must have identical SASS in the candidate's (`cuobjdump -sass`, every instruction and its encoding).
   - A `WRAP = false` instantiation is matched to main's kernel of the same name without that argument.
   - Missing functions (expected: `double_rows`) and extra ones (expected: the `WRAP = true` instantiations) are listed.
   - This is what keeps the language model's speed out of question. No text timing rule is set.
3. **Byte identity (gated):** tcgen05-v2's rule-2 suite on the candidate against main's outputs:
   - speech regression;
   - dump and edge;
   - e2e and split;
   - vision's 713 stage files;
   - the server's audio-in, 8 and 32 at once, images, the image cache, TTS and the text gate.
4. **Sanitizers (gated):** memcheck with leak check, synccheck and initcheck on the candidate's `lfm3-speech` with the 90 s clip, cherry and two. 0 errors, outputs identical to the plain run. memcheck on `gemm-wrap-check`.
5. **Memory (gated):** device memory in use (`measure_lfm3_audio_device_memory.py`, with cherry after the 90 s clip), ABAB, main (A) against the candidate (B), each after the short clip, the 90 s clip and cherry.
   - B must be lower than A by at least 90% of the doubled bytes. That is Σ out × 2·in × 2 bytes over each encoder's x2 weights, computed from the checkpoint's shapes and reported.
   - Each pair is reported.
6. **Speed (gated; quiet Thor, the samplers):**
   - audio time to first token on the five clips, ABAB, main (deployed `merge5-srv`) against the candidate;
   - images cherry and two (`measure_lfm3_image_turns.py`), ABAB, each of the three turns, with the image cache off (`--image-cache-mb 0`).
     - With the cache on, only the first, discarded conversation would run the vision encoder.
   - Each: mean(B) ≤ mean(A) × 1.01.
   - This is pinned-uploads-v1 rule 2 with the mean in place of the larger B. One run varies by ±1% there: the long clip's A1 and A2 differed by 1.6%.
   - The encoders' own times are reported (`audio-encoder-bench` at 35 and 90 s, the image server's encode).
   - No speed change is predicted. W is now read twice per multiply, as `[W | W]` was, from half the bytes.
7. **Coherency:** outputs are byte-identical (rule 3), so it carries over. The texts are still read once.

**What would stop the round:** rule 2 failing (machine code on an untouched path changed), or rule 6 failing at either encoder.
- If rule 6 fails only on the mma.sync wrap (the audio downsampler, N = 256), the wrap is limited to tcgen05 calls, and the downsampler's `[W | W]` (2 × 256 × 2 × 256 × 2 bytes each, under 1 MiB) stays, as an amendment.

## The first exactness run (2026-11-23, Thor, 16:50–17:05; `build/wrap-exact/`, stopped)
**Builds:**
- candidate `wrap-on` (5f5498f): server `ca5e381b…`, `lfm3-speech` `3d2eeaea…`;
- main at the branch point `wrap-main` (e41e36c): server `9a31794f…`, `lfm3-speech` `8b772cbe…`;
- `gemm-wrap-check` `4de2f68d…`.

**Rule 1: PASS.**
- mma.sync: 154 of 154 identical.
- tcgen05: 104 of 104 identical, with the launch counter rising by two on each.
- tcgen05 off, and the wrap that is not a multiple of 64: 106 of 106 identical.
- Launch-count mismatches: 0.
- 5 of 5 invalid uses refused.

**Rule 2: FAIL as written.**
- Server and `lfm3-speech`: 74 of 161 functions identical, 50 differ.
- `lfm3-vision-check`: 19 of 98 identical, 48 differ.
- Two causes:
  1. **The comparison tool's error.** `cu++filt` prints the new template argument as `(bool)0`, not `false`. So no `WRAP = false` instantiation was matched to main's kernel; they were listed as missing and extra.
  2. **A real change.** Kernels the round does not touch differ too, for example `thin_k128_kernel<8>`: different register allocation, 5,214 against 5,310 lines.
     - The build is deterministic: the deployed `merge5-srv` and `wrap-main` (the same source) give byte-identical SASS for that kernel.
     - So the change is the round's own. `b_k_wrap` was appended to `lfm::gemm::Problem`, growing it from 128 to 136 bytes. Every GEMM kernel takes a `Problem` by value.
- Rule 2 did what it was set for: it caught a change in machine code on paths the round claimed not to touch.

The run was stopped (SIGTERM, during rule 3's e2e step; 36 of 36 speech regression SAME by then) because the build it was certifying had to change. Rule 1's result stands as evidence for the wrap's arithmetic. It is rerun on the amended build.

## Amendment 1 (2026-11-23, before the next run)
- **`Problem::b_k_wrap` moves into the 7 bytes of padding after `bool b_k_contiguous`** (offset 108; `workspace` stays at 112; size 128). That is main's layout.
- **`tc05::Args::b_k_wrap` moves into the 4 bytes of padding after `k`** (offset 12; size 64), also main's.
- `static_assert`s pin both layouts (size and the offsets around the new field), so a later edit cannot grow either struct by accident.
- **`compare_sass.py`** strips `, (bool)0>` as well as `, false>` from a wrapped kernel's name before matching.
- The rules are unchanged. The whole exactness run (rules 1–4) is repeated on the amended build.

## The amended run, rules 1 and 2 (2026-11-23, Thor, from 17:04; `build/wrap-exact/`)
**Builds:**
- candidate `wrap-on` (947d57b): server `75f8c5b4…`, `lfm3-speech` `eb9204e8…`, `lfm3-vision-check` `525b3ec1…`;
- main `wrap-main` (e41e36c): server `9a31794f…`, the same as the first run's;
- `gemm-wrap-check` `18341816…`.

Both `static_assert`s held: the build compiled.

**Rule 1: PASS.**
- mma.sync: 154 of 154 identical.
- tcgen05: 104 of 104 identical.
- tcgen05 off, and the wrap that is not a multiple of 64: 106 of 106 identical.
- 0 launch-count mismatches; 5 of 5 invalid uses refused.

**Rule 2: PASS** (`sass2-*.log`).
- `liquid-lfm3-server` and `lfm3-speech`: 160 of 161 of main's functions identical, 36 of them `wide_kernel`, `thin_kernel` and `gemm_2cta` instantiations. 0 differ.
- `lfm3-vision-check`: 97 of 98 identical (30 of those kernels). 0 differ.
- The one missing function is `lfm::split::kernels::double_rows`, which the round removes.
- The 24 extra are all `WRAP = true` instantiations:
  - `thin_kernel` R = 1–8;
  - the three wide tiles, for sm_110 and sm_110a;
  - `gemm_2cta` with FP32 and BF16 output.

The comparison tool needed one more fix first.
- The run's own comparison (`sass-*.log`) also showed 0 differ, but it matched no wrapped kernel: `cu++filt`'s names begin with `void `, and the tool tested for a leading namespace.
- That reading was vacuous for those kernels and is not counted.
- `compare_sass.py` now looks for the kernel's name anywhere. It was rerun on the same binaries.

## The amended run, rules 3 and 4 (2026-11-23, Thor, 17:13–17:49; `build/wrap-exact/`)
**Rule 3: PASS.**
- speech regression: 36 SAME, 0 differ;
- dump and edge: 360 of 360 and 93 of 93 identical to main's, none missing or extra;
- e2e and split: 90 of 90 each;
- vision: 713 of 713 stage files identical to main's;
- the server:
  - audio in: PASS;
  - 8 at once and 32 at once (2.4 s and 4.4 s): identical to `lfm3-speech`;
  - images: PASS;
  - the image cache: 24 of 24;
  - TTS and the text gate: exit 0.

**Rule 4: PASS.**
- On the candidate's `lfm3-speech`, with the 90 s clip, cherry and two:
  - memcheck: 0 errors, 0 bytes leaked;
  - synccheck: 0 errors;
  - initcheck: 0 errors;
  - every output (JSON and files) the same as the plain run's.
- memcheck on `gemm-wrap-check`: 0 errors, 0 leaks, and the check passes under it.

## Rules 5 and 6, the quiet window (2026-11-23, Thor, 18:04–18:12; `build/wrap-quiet/`)
**Binaries:**
- A = deployed `merge5-srv` `f19f4ee4…`;
- B = `wrap-on` `75f8c5b4…`;
- `audio-encoder-bench` from main's tree `b7c9e76a…` and the candidate's `70ac8942…`.

**The window:**
- service stopped 18:04:34, started again 18:12:00;
- power unchanged;
- sampler 397 samples, foreign in 0. **MONITOR: CLEAN.**

**Rule 6: PASS.** Time to first text in seconds; images with the image cache off; mean(B) ≤ mean(A) × 1.01; texts identical across all four runs.

| item | A1 | B1 | A2 | B2 | B/A |
|---|---|---|---|---|---|
| 2.1 s | 0.0386 | 0.0376 | 0.0387 | 0.0381 | 0.979 |
| 5.8 s | 0.0458 | 0.0452 | 0.0457 | 0.0450 | 0.986 |
| 10.4 s | 0.0575 | 0.0566 | 0.0572 | 0.0567 | 0.988 |
| 35 s | 0.0892 | 0.0875 | 0.0895 | 0.0877 | 0.980 |
| 90 s | 0.1840 | 0.1835 | 0.1867 | 0.1832 | 0.989 |
| cherry, turn 1 | 0.8754 | 0.8647 | 0.8592 | 0.8615 | 0.995 |
| cherry, turn 2 | 0.9308 | 0.9163 | 0.9166 | 0.9169 | 0.992 |
| cherry, turn 3 | 0.9360 | 0.9198 | 0.9159 | 0.9196 | 0.993 |
| two, turn 1 | 0.1290 | 0.1266 | 0.1280 | 0.1258 | 0.982 |
| two, turn 2 | 0.1317 | 0.1302 | 0.1317 | 0.1295 | 0.986 |
| two, turn 3 | 0.1339 | 0.1326 | 0.1350 | 0.1321 | 0.984 |

The candidate is at or below main on every item: 1–2% lower.

The encoder alone (`audio-encoder-bench`, reported, ms; main, candidate, main, candidate):

| clip | main 1 | candidate 2 | main 3 | candidate 4 |
|---|---|---|---|---|
| 2.1 s | 5.05 | 5.32 | 6.09 | 5.28 |
| 35 s | 17.75 | 18.09 | 18.54 | 17.67 |
| 90 s | 38.29 | 37.90 | 40.60 | 38.83 |

The encoded features' hash is the same in all four (`c4399214feec3835` at 90 s).

**Rule 5: PASS.**
- **The doubled bytes,** from the checkpoint's shapes (`out × 2·in × 2` bytes over every weight the encoders' `mul()` takes):
  - audio: 104 weights, **460,849,152 bytes (439.5 MiB)**. The three downsampler weights are stored as 1 × 1 convolutions and counted by every dimension.
  - vision: 85 weights, **757,071,872 bytes (722.0 MiB)**.
- **Device memory** (MiB, the in-use stand-in):

| reading | A1 | B1 | A2 | B2 | saved (pairs) | bound |
|---|---|---|---|---|---|---|
| short clip | 7,762 | 7,218 | 7,728 | 7,181 | 544, 547 | 395.6 (90% of audio's) |
| 90 s clip | 7,741 | 7,185 | 7,729 | 7,177 | 556, 552 | 395.6 |
| cherry | 9,096 | 7,785 | 9,090 | 7,845 | 1,311, 1,245 | 1,045.4 (90% of both) |

- The savings exceed the doubled bytes a little: about 110 MiB for audio, by allocation rounding.
- A1's `started` reading (539 MiB, against 22–28 for the other three) is a baseline drift in that one server's start. The gated readings are differences after requests.

**Rule 7 (coherency):** outputs are byte-identical (rule 3). Read once:
- the 90 s transcript ("He hoped there would be stew for dinner, turnips and carrots and bruised potatoes…");
- cherry's three turns (the white blossoms as the focal point; the soft, diffuse daylight);
- two's first turn (the red, multi-tiered pavilion).

All are right and coherent.

## Conclusion (2026-11-23)
**Every rule passes on the amended build. The first build failed rule 2 as written; that is recorded above.**
- Both encoders drop their `[W | W]` copies: **about 0.54 GiB less after audio, 1.25 GiB less once an image has been encoded.**
- Every output is unchanged byte for byte.
- Every kernel that takes no wrap runs main's exact machine code, so text and the language model are untouched.
- Audio and images are 1–2% faster.
- Merging and deploying are Tripp's call.

**What it opens:** keeping the audio encoder's scratch (about 10 ms at 90 s; audio-ttft-v1) is now roughly memory-neutral against today's server. That needs pinned-uploads-v1 first.
