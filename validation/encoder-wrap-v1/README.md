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
   - The invalid uses are refused: c968b5d's 4, plus a tcgen05-eligible shape with a wrap that is not a multiple of 64, which must run on mma.sync and still match.
2. **Unchanged machine code (gated):** in the candidate's `liquid-lfm3-server` and `lfm3-speech`, the SASS of every `WRAP = false` instantiation of `wide_kernel`, `thin_kernel` and `gemm_2cta` is identical to main's for the same kernel (`cuobjdump -sass`, function bodies compared with their names normalized).
   - This is what keeps the language model's speed out of question. No text timing rule is set.
3. **Byte identity (gated):** tcgen05-v2's rule-2 suite on the candidate against main's outputs:
   - speech regression;
   - dump and edge;
   - e2e and split;
   - vision's 713 stage files;
   - the server's audio-in, 8 and 32 at once, images, the image cache, TTS and the text gate.
4. **Sanitizers (gated):** memcheck with leak check, synccheck and initcheck on the candidate's `lfm3-speech` with the 90 s clip, cherry and two. 0 errors, outputs identical to the plain run. memcheck on `gemm-wrap-check`.
5. **Memory (gated):** device memory in use, ABAB, main (A) against the candidate (B), each after the short clip, the 90 s clip and cherry.
   - B must be lower than A by at least 90% of the doubled bytes. That is Σ out × 2·in × 2 bytes over each encoder's x2 weights, computed from the checkpoint's shapes and reported.
   - Each pair is reported.
6. **Speed (gated; quiet Thor, the samplers):**
   - audio time to first token on the five clips, ABAB, main (deployed `merge5-srv`) against the candidate;
   - images cherry and two (`measure_lfm3_image_turns.py`), ABAB.
   - Each: mean(B) ≤ mean(A) × 1.01.
   - This is pinned-uploads-v1 rule 2 with the mean in place of the larger B. One run varies by ±1% there: the long clip's A1 and A2 differed by 1.6%.
   - The encoders' own times are reported (`audio-encoder-bench` at 35 and 90 s, the image server's encode).
   - No speed change is predicted. W is now read twice per multiply, as `[W | W]` was, from half the bytes.
7. **Coherency:** outputs are byte-identical (rule 3), so it carries over. The texts are still read once.

**What would stop the round:** rule 2 failing (machine code on an untouched path changed), or rule 6 failing at either encoder.
- If rule 6 fails only on the mma.sync wrap (the audio downsampler, N = 256), the wrap is limited to tcgen05 calls, and the downsampler's `[W | W]` (2 × 256 × 2 × 256 × 2 bytes each, under 1 MiB) stays, as an amendment.
