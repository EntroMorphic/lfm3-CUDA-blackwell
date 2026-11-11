# Faster concurrent speech, round 2, every output bit-identical

Tripp (2026-11-11): "Let's see what more we can squeeze out of it bef we move to images." Round 1
(speech-speed-v1, lfm3-speech speed-a4) reached 1.244 times speech-batch-v1 at 8 rows, short of its 1.25 bar.

## Where an 8-row step goes now (speed-a4, 8 TTS requests, nsys, steady state)
On the main stream, per step: the language model's projections 15.5 ms, run at about 205 GB/s (up projections),
170 GB/s (the 7680-deep down projection: 32 blocks on 20 SMs) and 196 GB/s (2048 x 2048); Thor reads at about
256 GB/s. The depth decoder about 10.5 ms, of which its FP32 linears are 8.5 ms in the trunk (3 passes x 3 layers
over 12 slots per frame). Host sampling between the depth passes 0.56 ms. The GPU is busy 93% of the time.

A tile sweep of the language model's projections at 1, 4, 8 and 16 rows (distinct weights per instance, every tile
bitwise equal): deeper pipelines reach 240-250 GB/s on the up, conv-input and 2048 x 2048 projections, and a
32-column tile 205-209 GB/s on the down projection (64 blocks).

## Changes (written before the code)
1. **The language model's tiles at 16 rows or fewer:** 16 x 64 x 64, 4 warps, 12 stages, for every projection but
   the down projection, which takes 16 x 32 x 64, 2 warps, 24 stages. Every tile shape adds each output over the same
   k16 steps in the same order (the tile contract, checked bitwise by the sweep), so no value changes. The table is
   shared with the text engine's batch, which takes them too at 16 rows or fewer.
2. **The depth decoder's last layer for the scored rows only.** A pass scores 4 of a frame's 12 codebooks; only
   their rows reach the heads. The last layer computes K and V for every slot (attention reads them), and the query,
   gate, attention, output projection, feed-forward and final norms only for the scored rows. Every one of those
   operators computes a row the same way whatever the rows beside it, so the scored rows' values are unchanged.

## Rules (set before any measured run of the changed engine)
Thor, MAXN, clocks locked, power recorded. Reference binaries: lfm3-speech speed-a4 and liquid-lfm3-server speed-s2
(speech-speed-v1), rebuilt from nothing; their hashes must match the recorded ones.
1. **Exact.** As speech-speed-v1's rule 1, every output byte for byte: lfm3-speech's regression check (36 of 36); the
   engine check (41 requests alone, at 4, 8, 8 staggered and 16 staggered rows), alone identical to speed-a4's
   alone; the server check's 42 cases in all four crowds equal to the single-request server's replies. Also the
   tile sweep with the production tiles: bitwise equal.
2. **Faster.** speech-speed-v1's paired measurement (decode loop, aggregate audio seconds per wall second, the 21
   TTS requests looped, N = 1, 2, 4, 8, 16), the new engine and speed-a4 alternating, two passes each. Pass: at
   N = 8, the new engine's median at least 1.08 times speed-a4's. Also reported: every N, and the cumulative ratio
   to batch-a4 (speech-speed-v1's 1.25 bar) from round 1's medians.
3. **No slower alone:** at N = 1, at least 0.98 times speed-a4.
4. **Text not slower, and the same:** lfm3-batch (the text batch's decode loop, batch 1, 8 and 16, 256 tokens, two
   runs each), old and new: every output token identical, and each batch's median rate at least 0.98 times the
   old one's.
5. **Streams stay real time:** no player runs dry at 8, 12 or 16 streams; first audio under load at most 1.0 s.
6. **Nothing else moves:** the server's streaming, TTS and audio-input checks and the text gate pass; host tests pass.

## Red-team (a fresh reviewer, the diff only), before the measured runs
It traced the new tiles (loads, swizzle, ldmatrix addressing, epilogue, shared memory, the stage ring) and the depth
decoder's last layer (the shared attention core against the old kernel expression by expression, positions,
per-codebook norm weights, buffer aliasing, the heads' order). It found no change to any output. Its findings:
1. **lfm3_lossless_check no longer compiled:** it used the removed S16 tile. **Fixed:** S16 is kept, named as the
   table's previous rows <= 16 tile.
2. **The tile sweep checked less than the rule implies:** the last weight instance only, the BF16 epilogue only,
   no output narrower than a tile, no M of 3 or 12. **Fixed:** every instance's output is compared, in both the BF16
   and the FP32 epilogue, at M = 1, 3, 8, 12, 16, 32 and 64, with the 16-wide gate projection added. All bitwise
   equal (Thor).
3. **The lossless kernel had no guard against a 32-column tile** (its packed streams are 64-row tile-major); none is
   passed today. **Fixed:** a static_assert.
4. **Accepted:** with no codebook scored, a pass now returns before any kernel (before, it ran the trunk into
   internal buffers only; the logits were never written in either case).

## Deviation: the reference hash rule
The first suite run stopped at its first gate: lfm3-speech and the worker rebuilt from 8f87d69 did not hash as
recorded (the server did). The difference is one byte, in the symbol string table: nvcc names its temporary file by a
process counter (`tmpxft_00000008` against `_00000007`), and the table is not loaded at run time. Rebuilt twice, the
binary is the same each time; with symbol tables stripped (`objcopy --strip-all`), the rebuilt references equal the
recorded binaries exactly. The rule, as written, cannot hold for an nvcc build; the suite now compares stripped
binaries (`reference-check.txt`) and checks that the recorded binaries still have their recorded hashes. Nothing had
been measured.

## Results (lfm3-speech speed2-a2, text tools speed2-t2, server speed2-s1; records in `build-final/`)
Thor at MAXN, GPU clocks locked at 1.575 GHz at the start and the end (`power-*.json`). References rebuilt from
8f87d69 equal the recorded binaries once stripped (`reference-check.txt`). The sources measured are in
`sources.sha256` (the same on Thor); the change itself is `speed2-a2.patch`.

**Rule 1, exact: pass.** Tile sweep bitwise at every shape, M and epilogue; FP32 linear check and capacity check
identical; regression 36 of 36; the engine check 41 of 41 alone identical to speed-a4's alone, and every crowd
identical to alone; the server's 42 cases identical in all four crowds.

**Rule 2, faster (decode loop, aggregate audio seconds per wall second, median of 4, alternating): fail.**

| N | speed-a4 | speed2-a2 | ratio | vs batch-a4 (round 1's medians) |
|---|---|---|---|---|
| 1 | 4.404 | 4.337 | 0.985 | 1.179 |
| 2 | 7.790 | 7.836 | 1.006 | 1.196 |
| 4 | 13.112 | 13.306 | 1.015 | 1.244 |
| 8 | 16.847 | 17.286 | **1.026** (bar 1.08) | 1.279 |
| 16 | 22.206 | 23.308 | 1.050 | 1.319 |

Spreads were at most 0.5%. The quick 8-request run that showed about +14% (`build/q2`) did not predict this; the
isolated tile sweep overstated the new tiles in the running model.

**Rule 3, no slower alone: pass, narrowly** (0.985 against 0.98).

**Rule 4, text not slower and the same: fail.** Tokens identical (greedy, sampled, and every rate run), but the
decode rate fell: 0.972 at batch 1, 0.972 at 8, 0.976 at 16 (bar 0.98). Text at rows <= 16 takes the new tiles.

**Rule 5, streams real time: pass.** No stalls at 8, 12 or 16 streams; first audio under load median 0.382 s.

**Rule 6, nothing else moves: pass.** Streaming check (first audio 0.239 s TTS, 0.334 s interleaved), audio-in and
TTS server checks, the text gate; host tests 888 passed, 2 skipped, the one known transformers-version failure.
Text under speech load: 25.5 s, identical ids.

**Disposition.** Two rules failed, so the change is not merged: the engine stays at round 1 (8f87d69's sources),
and `speed2-a2.patch` keeps what was measured. The patch holds two independent changes (the rows <= 16 tiles, and
the depth decoder's last layer computed only for the scored rows); a next round would measure them apart, and keep
text on its old tiles.
