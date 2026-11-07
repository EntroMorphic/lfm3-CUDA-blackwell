# G4 prefill: query heads per attention call — measured (step 1 of journal/g4-prefill-memory_synth.md)

`tools/thor/g4_prefill_heads.sh` (log `run.log`, raw results `time-*.json`): the default (native) engine with
`LFM3_PREFILL_HEADS` = 16, 8, 4, in one session, alternating settings, median of 3; 128 forced decode steps.

| Case | 16 heads | 8 heads | 4 heads (v6 default) | Score scratch at 16 / 8 / 4 heads |
|---|---|---|---|---|
| capacity 4,096, 3,968-token prompt (one tile) | 0.756 s | 0.757 s | 0.761 s | 1 GiB / 512 MiB / 256 MiB |
| capacity 32,768, 4,096-token prompt (128-row tiles) | 0.771 s | 0.803 s | 0.866 s | 256 / 128 / 64 MiB |
| capacity 32,768, 32,000-token prompt | 7.62 s | 8.12 s | 9.10 s | 256 / 128 / 64 MiB |

- **Bit-invariance:** the exact FP32 logits of all 129 steps hash the same for every setting at every size
  (`run.log`, "bit-invariance").
- **The declared predictions held:** at capacity 32,768, 16 heads recovers v5's prefill (7.62 s; v5 measured 7.61 s
  in another session); at capacity 4,096, heads-per-call changes prefill by under 1% (predicted ≤ 10%).
- **Why:** in 128-row tiles, P·V's grid is 1 × 1 × heads — 16 CTAs with all heads, 4 with one KV group, on 20 SMs.
  In one 3,968-row tile the grid is large either way.
- **Correction to earlier evidence:** the "4,096" prefill timings in `../g4-native-gates-v1/` used 4,096 prompt tokens
  plus 128 decode steps, i.e. capacity 32,768 (the tiled path), not capacity 4,096.
- Decode is unaffected (one call with all heads in every setting).
