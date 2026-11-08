# Drafts for exact speculation (speed detour, parked): low-bit self-drafts and the trained head

Record: `../../../journal/lfm3-speed-detour.md`.

- `tools/ternary_draft_demo.py` (CPU, transformers 5.22.1): teacher-forced agreement of low-bit copies of
  LFM3-Omni-2B with the BF16 model on 6 held-out chat answers (1,470 tokens), plus chain acceptance and an estimate of
  single-request tok/s from Thor's measured pass times and each draft's bytes.
  `ternary.json`, `ternary-int.json` (adds int4/int8), `nvfp4.json`, `mxfp8.json`. HF BF16 agrees with the native
  engine on 99.05% of these tokens.
- `tools/head_data.py`, `tools/head_train.py`, `tools/head_eval.py`: the EAGLE-style draft head (data: 12,000 UltraChat
  prompts answered by the engine, 6.6M tokens; training: 2 epochs, 1,382 steps), and its offline tree-acceptance test
  (`head-eval.json`).
