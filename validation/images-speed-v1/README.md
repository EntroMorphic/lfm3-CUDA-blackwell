# The vision encoder on the tensor cores: meet or beat HF end to end, keep the accuracy lead

## What it will be (Tripp, 2026-11-16: "Do it.")
images-v1's speed round found native 30% slower than HF end to end on 4K photographs, and faster on every other
stage-0 request (`validation/images-v1/README.md`, "Speed against HF"). nsys traced it to the code: the encoder's
matrix multiplies run `fp32::linear_staged`, a few-row kernel that streams every weight matrix 512 times at 8,192
patches (memory-bound, ~31% of the FP32 cores' peak), and its attention holds one query per thread with a
shared-memory load per FMA (~30%). Even a perfect FP32 kernel (~0.55 s) cannot reach HF's 0.148 s encode: the tensor
cores are required. This round moves the encoder's multiplies and attention onto them with the engine's existing
`mma.sync` path, keeping its FP32 residual stream, norms, softmax and nonlinearities. (`tcgen05`/TMEM/TMA is a
separate, later round.)

## Design
- **Multiplies**: `lfm::gemm` (`wide`: `mma.sync` m16n8k16, BF16 in, FP32 accumulate), FP32 output, then one kernel
  adding the bias (and, split, the halves). The weights are BF16 already, exactly. Activations, two variants:
  - `x1`: the FP32 activation rounded to BF16 (one GEMM);
  - `x2`: split, `hi = bf16(x)`, `lo = bf16(x - hi)`: `x·W = hi·W + lo·W` to ~2^-16 of x (a batch of 2 GEMMs
    sharing W, summed in FP32).
- **Attention**: a new fused kernel (flash-attention style, head dimension 64, all keys: the encoder is
  bidirectional): a block's warps own 16 queries each; per key tile, S = Q·Kᵀ on the tensor cores, the online softmax
  in FP32 registers, O += P·V on the tensor cores. Variants:
  - `x1`: Q, K, V and P in BF16;
  - `x2`: each split into hi + lo, three products each (hi·hi + hi·lo + lo·hi; lo·lo is below FP32's resolution here).
- Everything else unchanged: the patchifier and projector use the same multiply path; norms, gate, spatial conv,
  rotary (FP64 tables), GELU (FP64), residual adds stay as they are. The output is rounded to BF16 as before.
- The variant is chosen by the rules below, in this order: (linear `x1`, attention `x1`), then the split for
  whichever part a failure points to, then both split. The fastest variant that passes rule 1 ships. No other
  variant is added without amending these rules first.

## Rules, set before any code
1. **Accuracy, on all 31 images of images-v1 rule 2** (11 stage-0, 20 held-out MMStar; the same HF FP32 and BF16
   dumps; `tools/compare_lfm3_vision_stages.py`): on every image, image_features' relative L2 error to HF FP32 is at
   most **0.15 x HF BF16's**, and the minimum per-token cosine to FP32 is at least **0.9999**. (images-v1: ratios
   0.023-0.112, cosines >= 0.9999983; the output's BF16 rounding alone is 1.66e-3.)
2. **Speed** (`tools/thor/vision_speed_round.sh`, quiet, MAXN, the method of images-v1): native's end to end is at
   most HF's on **every** stage-0 request, the two 4K photographs included. Also reported: the encode alone at
   7,920 patches, and nsys's kernel shares.
3. **Benchmarks** (the full MMStar and POPE runs of images-v1 rule 3, the new build): MMStar answers agree with HF
   FP32's at least as often as HF BF16's do (rule 3b's measure: >= 93.87%); POPE accuracy against HF BF16 with a
   paired-bootstrap lower bound >= -0.5 points.
4. **Coherency**: I read the 11 stage-0 replies and every benchmark item whose answer changed from images-v1's
   native run.
5. **Sanitizers**: memcheck (leaks), racecheck, synccheck and initcheck on `lfm3-vision-check`'s small, two-image and
   largest cases: 0 errors, outputs byte-identical to the plain run.
6. **Nothing else changes**: lfm3-speech's regression (36 outputs), the audio-in replies (360 + 93 files), e2e and
   split (90 + 90), and the server checks (batch, stream, audio in, TTS, text gate with control `7ea05068...`) all
   identical; images through the server identical to lfm3-speech; the host suites pass.

## Amendment 1 (2026-11-16), before rule 1's run: how the split runs, from two profiles
A first build (`vspeed-a1`), one image (china) and the largest case only, no rule run: every mode combination on
china (`--linear`, `--attention`): fp32/fp32 byte-identical to images-v1; x2/x2 0.028 of HF BF16's error, cosine
0.9999983 (fp32's, to the output's rounding), 0.047 s against 0.165 s; any x1 part fails rule 1's thresholds on its
own (x2/x1 0.214, x1/x2 0.297, x1/x1 0.396). At 8,192 patches x2/x2 took 0.685 s (fp32: 1.87 s), not yet enough.
nsys on it: the attention kernel 53% at ~14 TFLOPS (each block re-converts every key and value from FP32, a transposed
store with bank conflicts, 4 warps); the GEMMs ~137 ms at ~40 TFLOPS; the split and combine passes ~91 ms;
allocating and zeroing scratch per image ~58 ms. Changes, the mathematics unchanged:
- The split multiply is one GEMM over a doubled K: A = [hi | lo] per row, B = [W | W] (built once, on first use: the
  vision weights again, ~0.37 GB more GPU memory; **corrected by the red-team below: 0.76 GB**, both halves), FP32 out directly; a bias, the projector's only, added after.
  (The design said a batch of 2 summed in FP32: same products, one accumulation.)
- Attention: K and V split once per layer into BF16 buffers in the tiles' layouts (V transposed); 8 warps a block.
- Scratch kept between images (~0.5 GB after the largest), no longer freed and re-zeroed each time.

## Rule 1 results (2026-11-16): **PASS**, x2/x2 ships
`tools/thor/vision_accuracy_round.sh vspeed-a2 vspeed-rule1-a2 x2 x2` (`rule1/`): on all 31 images, image_features'
error 0.023-0.112 of HF BF16's (images-v1's FP32 path: the same range), minimum token cosine 0.9999966 (>= 0.9999).
In rule order: x1/x1 failed the thresholds on china alone (0.396, cosine 0.978), and so did each mixed variant
(x2/x1 0.214, x1/x2 0.297): x2/x2 is the only variant that passes, and the encoder's default
(`lfm3_vision::Precision`). The next build (`vspeed-a3`: attention fragments by `ldmatrix.x4`, tiles double-buffered by
`cp.async`, the products and their order unchanged) reproduced all 713 stage files of the 31 images byte for byte.

## Rule 2 results (2026-11-16): **PASS** in two rounds of three; the third slowed by the machine
Encode alone at 8,192 patches (`lfm3-vision-check`, warm median of 3): FP32 path 1.87 s; first split build 0.685 s;
amendment 1's 0.458 s; with ldmatrix and double buffering 0.389 s. nsys of the last split build (`profile/`): attention
~48% at ~23 TFLOPS, the GEMMs ~28% at ~43 TFLOPS. `lfm3-speech` build `vimg-a4` (sha256 `8af368b4...`), three rounds
of `tools/thor/vision_speed_round.sh` (MAXN, GPU clocks locked at 1.575 GHz in all three), `rule2/`:
| round | 4K photographs, HF / native end to end | other 9 requests | native decode | native faster on |
|---|---|---|---|---|
| vspeed-v2 07:27 | 0.91, 0.92 | 1.03-1.18 | ~20.2 ms/token | 9 of 11 |
| vspeed-v3 07:36 | **1.07, 1.08** | 1.12-1.29 | 16.0-16.3 ms/token | **11 of 11** |
| vspeed-v4 07:43 | **1.05, 1.07** | 1.11-1.28 | 16.1-16.3 ms/token | **11 of 11** |
- v2 is the same binary and the same work: its 44 native replies are token-identical to v3's; only its decode was
  slower (20.2, 20.2, 20.2, then 17.8 ms/token across its passes, recovering). The previous build run right after it
  decoded at 16.0-16.3, and probes of text before and after a 4K image on both builds showed no effect of the
  encoder's kept memory (`vdecode-probe`: 15.8-16.3 ms/token throughout). The cause was the machine at that time; the
  method's power check records GPU and CPU clocks, not the memory clock (unreadable without root) or other load: a
  gap, recorded. v3 and v4 meet the rule; v2 is kept here, not discarded.
- At 4K: image encode 0.364 s (HF 0.148), prefill with it 0.59 s (HF's first token 0.32), front end 0.32-0.42 s
  (HF 0.16), decode 16.3 ms/token (HF 21.7): native is ahead end to end on the faster decode. The encode, the language
  model's prefill and the CPU front end are still slower than HF's: the tcgen05 round and the GPU front end.

## Red-team (2026-11-16): no wrong output found; two host-side fixes, one record corrected
An independent review of 89d671c (a fresh reader given the code, not these results) and my own reading of the stage-0
replies (`vimg-e2e-a4`: 3 of 11 identical to HF FP32's as before, the others drifting at the same near-tie points as
images-v1's FP32 build, all coherent). Checked and found sound: partial key tiles and their masking (no row can see
only masked keys), query rows past `n`, `cp.async` group ordering, the `ldmatrix` fragment layouts against m16n8k16,
shared-memory and global alignment, launch grids, index overflow, stale scratch (every region read is written first in
the same call), reallocation with work in flight, the GEMM contract, the bias, and the `kFp32` path. Found:
1. **Memory held for good (record corrected, not changed):** the `[W | W]` cache is **0.76 GB** (189.3 M vision and
   projector parameters, twice, BF16), not amendment 1's ~0.37 GB (one copy), plus the kept scratch (~0.6 GB after
   the largest image): ~1.4 GB of Thor's 128 GB unified memory held after the first large image, by design
   (amendment 1). A first image after a long text-only run allocates it inside that request; a failed allocation
   throws before any state changes (the cache and scratch are assigned only after success). No test checks device
   retention for images; image-queue-v1 rule 3 bounds host memory.
2. **Fixed:** the attention's dynamic shared-memory opt-in was set once per process (a function-level `static`); a
   function attribute belongs to a device, so an Encoder on a second device would have failed at launch (caught by
   the error check at the end of the encode, not silently). Now set in each Encoder's constructor.
   `lfm::gemm` has the same once-per-process pattern; it is the engine's shared code, outside this round (rule 6), and
   Thor has one device: left as is, noted.
3. **Fixed:** `lfm3-vision-check --repeat` accepted `3x` as 3 and negatives as 1; now an integer 0-1000 or refused
   (`3x`, `-1`, `1001`, `abc` all exit 1).
4. Not changed: `kX1`'s linear path hands the checkpoint's weights straight to `lfm::gemm`, which refuses (throws on)
   weights not 16-byte aligned; `kX1` does not ship (rule 1), and this checkpoint is aligned (the x1 runs completed).

The fixes are host code only: `cuobjdump -sass` of the remediated builds (`vspeed-a5`, `lfm3-speech` `vimg-a5`) is
identical to `vspeed-a4`'s and `vimg-a4`'s (132,564 and 199,729 instruction lines), so rule 2's timings and rule 3's
benchmark (run on `vimg-a4`) carry over. Re-verified on the new builds: rule 1 PASS, all 713 stage files of the 31
images byte-identical to the validated run; memcheck (no leaks), initcheck and synccheck 0 errors on all three cases,
outputs identical (racecheck: the device code is `vspeed-a4`'s, whose racecheck is rule 5's run); lfm3-speech's
regression 36 of 36 identical; the audio-in replies 360 of 360 and 93 of 93, e2e 90
of 90 and split 90 of 90 identical. The server and worker rebuilt from these sources (`vimg-srv-a3`: server
`a29f808a`, worker `f66d3681`) carry rule 6's server checks; `liquid-lfm3` differs from `vimg-srv-a2`'s in 2 bytes
only, nvcc's temporary file name in the symbol table (`tmpxft_000000af` against `..b0`).

## Rule 3 results (2026-11-16): **PASS** (`bench/`)
The full benchmarks on `lfm3-speech` `vimg-a4` (`tools/thor/vision_bench_native.sh vimg-a4 vbench-a2`, 10,500 items, 8
concurrent rows, 07:47-13:25; scored by `tools/score_lfm3_vision_bench.py` and
`tools/score_lfm3_vision_fp32_agreement.py`):
- **MMStar against FP32: native agrees on 95.67% of items**, HF BF16 on 93.87% (+1.80 points, 95% CI [+0.60, +3.00];
  the bar: >= 93.87%). images-v1's FP32-path native agreed on 94.80%: the split path is closer to FP32 still. Replies
  identical to FP32's: 1,249 (HF BF16 1,169). Accuracy 59.13% (HF BF16 59.27%, FP32 59.07%).
- **POPE**: accuracy 89.06 against HF BF16's 89.11, -0.06 points, 95% CI [-0.13, +0.02] (bar: lower bound >= -0.5);
  8,987 of 9,000 replies identical to HF's.

## Rule 4 results (2026-11-16): **no engine fault** (`bench/changed.json.gz`; stage-0 in the red-team section)
I read all 97 items whose answer changed from images-v1's native run (`vbench-a1`): 89 MMStar, 8 POPE; 32 wrong to
right, 29 right to wrong. On the 89 MMStar items the new build agrees with FP32 47 times, images-v1's 34. The changes
are what images-v1's reading found: borderline items and fluent reasoning that drifts either way at a near tie
(geometry, food webs, counting, chart reading); POPE's 8 are three images flipping together across the three splits.
One reply loops on one sentence until its 512 tokens run out (`mmstar/val/1298`, "Try ACK = ..."). MMStar's
unanswered items rose to 25 (images-v1 native 18, HF BF16 14), toward FP32's own 23: 17 of the 25 are FP32's
unanswered items too (13 for images-v1's native and for HF), 18 of them stopped by the 512-token limit (FP32: 15). The
model in FP32 runs out of tokens on these long problems; an engine closer to FP32 does too.

## Rule 5 results (2026-11-16): **PASS** (`sanitizers/`)
`tools/thor/vision_sanitizers.sh` on `vspeed-a4` (`lfm3-vision-check` sha256 in `vspeed-a4.binary.sha256`): memcheck
(leak check), initcheck, synccheck and racecheck on the small, two-image and largest (8,192 patches) cases: 0 errors,
0 hazards, 0 bytes leaked, every output byte-identical to the plain run (racecheck at the largest: 6,464 s). The
red-team's build (`vspeed-a5`, identical device code) passed memcheck, initcheck and synccheck again
(`redteam-a5.log`).

## Rule 6, so far (2026-11-16)
lfm3-speech's regression 36 of 36, audio-in replies 360 of 360 and 93 of 93, e2e 90 of 90 and split 90 of 90
identical, on `vimg-a4` and again on `vimg-a5`. The server checks run on a quiet machine
(`tools/thor/quiet_session_v1.sh` step 5, `vimg-a5` with server `vimg-srv-a3`).
