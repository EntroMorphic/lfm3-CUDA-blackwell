# Lossless BF16 weights (single-request speed, decision 5): exact, not yet faster

`native/include/lfm/gemm_lossless.cuh`, opt-in with `LFM3_LOSSLESS=1` (off by default). Check tool:
`native/tests/lfm3_lossless_check.cu`. Thor, LFM3-Omni-2B.

## Format
Per 64-weight group along K (one BK = 64 tile): a sign+mantissa byte per weight, a 4-bit exponent code per weight
(exponent - base, base = group max - 14; 15 marks an exception), and an 8-byte header (base, count, up to 3
exceptions). That is 104 bytes per 64 weights, **81.25% of BF16** (2.81 GB vs 3.46). LFM3's groups never need more
than 3 exceptions at 4 bits: all 154 projection matrices plus the tied head pack. The streams are tile-major.
(The checkpoint's information content is 10.3 bits per weight, 64%; a 3-bit code would reach 76.6% but needs an
overflow path for 0.27% of groups.)

## Exactness
- Every packed matrix, compressed vs plain kernel, M = 1, 7, 32, 64, five tile shapes including the production
  lossless tiles, and split-K: **4,308 comparisons, 0 mismatches** (bitwise).
- End to end, plain vs lossless: batched decode at batch 1, 8, 16 and 32 (two runs each) and the tree decoder on
  code edit, document answering and chat: **tokens identical**.

## Speed: slower in the decoder today
Decode-loop tok/s, paired, the same binary (`r-*`, `t-*`):

| | plain | lossless |
|---|---|---|
| batch 1 | 49-62 (clock ramp) | 54 |
| batch 8 | 485, 487 | 435, 432 |
| batch 16 | 954, 969 | 856, 852 |
| batch 32 | 1,757, 1,763 | 1,610, 1,598 |
| tree, code edit | 254 | 231 |
| tree, chat | 66 | 59 |

What the diagnostics showed (a step's projections back to back, `lfm3_lossless_check`):
- **Loads only** (decode skipped): 14.1 ms vs plain 18.2, 77% of the time, close to the 81% byte ratio. The
  bandwidth saving is real.
- **The first layouts:** row-major streams read 32- and 64-byte fragments per row and ran at half of plain's DRAM
  throughput. Tile-major streams fixed that (40 → 29 ms).
- **The decode:** in the tensor-core loop it competes for issue with the mma pipeline. A branch-free decode,
  decoding one tile ahead, and more resident warps (16-row 4-warp and 32-row 8-warp tiles, 3 stages) brought it from
  2.4x slower to 1.13-1.18x faster than plain W32/C32 in isolation. That is still slower than the tuned plain table
  inside the decoder.
- A 2-stage tile broke the decode-ahead (k-tile k + 1 not yet loaded). The check tool now verifies exactly the tiles
  production uses, and the kernel requires ≥ 3 stages.

## Next, if pursued
Warp specialization (decoder warps fill shared memory, mma warps consume it); a byte-permute decode (two weights in
about four instructions); a 3-bit code once the decode is cheap.
