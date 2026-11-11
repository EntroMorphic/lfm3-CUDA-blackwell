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
