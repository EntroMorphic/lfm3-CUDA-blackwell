# crowd-identity-v1: the 4 crowd differences after tcgen05 rule 4b pair 2's OFF soak

**Status: rules set 2026-11-17, before any data.**

## What happened
Window 7 (`build/quiet-tcgen05-4b-pair2/off/soak-4b.json`, server `54ee76ac`, worker `b211b296`, the tcgen05 build
with the option off, so main's GEMM path): after a 3,600 s soak, the crowd (56 cases at once, no other load) had
52/56 unstreamed identical. The four that differed were `budget-truncated-s1`, `numbers-normalized-s1`,
`image-mirror` and `image-noise`. Streamed was 56/56, and none of the 3,406 replies during the soak differed. The
same binary had 56/56 both ways in window 5.

The harness counted a non-200 as a difference and saved neither the status nor the reply. A 429 after the crowd's
retries ran out (3,000 tries 10 ms apart) can't be told from a content difference. This is recorded as a failure
whose cause is not known.

## The probe
`tools/check_lfm3_audio_soak.py` now records, for each crowd difference, its status, the tries it used and which
fields differ; it appends the differing replies to `<out>.crowd-differ.jsonl`. `CROWD_ROUNDS=N` repeats the crowd
N times (each round gated as before; with N=1 the verdict is the same as before). Run through the tcgen05 branch's
`check_lfm3_tcgen05_soak.py` (the same 56 cases) on the same OFF binaries: a 600 s soak at 12 clients and text share
0.25, then `CROWD_ROUNDS=20` (2,240 crowd replies). In a quiet window: 429s depend on contention.

## Reading (set before the run)
1. **Any difference with status 200: content.** The engine answers differently in a crowd than alone. That is an
   exactness bug in the default path and comes before all other work. The saved replies say where they part.
2. **Every difference non-200 (429 after 3,000 tries): the harness.** The crowd's retry budget is too small for 56
   requests through a front end that prepares one at a time. Fix: retry by deadline, not count. Pair 2's OFF failure
   stays recorded, with this cause.
3. **No difference in 20 rounds: not reproduced.** Nothing is concluded. The diagnostics stay in the soak, so the
   next long soak's crowd says which it was.
Mixed results read as 1.
