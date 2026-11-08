# Batched decode for the LFM3 text engine (P0 speed): all gates PASS

`native/src/lfm3_batch.cuh`, driven by `native/tests/lfm3_batch.cu` (`lfm3-batch`). Not yet wired into
`liquid-lfm3` or the server: no user-facing path is faster yet. Run: `tools/thor/build_batch_tools.sh batch-v1`, then
`tools/thor/batch_round.sh batch-round-v1 batch-v1` (`run.log`, `evidence/`). Thor, 120 W, no cuBLAS, pool-free
driver (`EnableSystemMemoryPools: 0`), LFM3-Omni-2B.

## What it does

B sequences decode in lockstep, so each weight is read once per step for all of them. The design follows the LFM2.5
engine (`../liquid-CUDA-inference-engine/native/BATCH_DECODE.md`): per-row KV and conv state, a row-generic layer
pass, token selection on the GPU. Adapted to LFM3: head_dim 128 with 4 KV heads, sliding layers, QK norms, gates.
- Prefill stays the validated `Runner`. Each prompt's KV caches, conv states and last logits are copied into its row.
- Projections use `lfm::gemm`'s tensor-core kernel at every row count, in the tile shape measured fastest per
  projection (`evidence/tile-sweep.log`). Every tile shape and every M gives bitwise the same values, so no padding
  to fixed tiles is needed, unlike the LFM2.5 engine.
- Norms run every row with the one-row block width.
- Attention is split-K over fixed 256-key splits, combined in split order. Scores, the online softmax and P·V are FP32.
- Greedy and temperature sampling run on the device, one block per row; only B token ids return to the host.

## Gates

| Gate | Result |
|---|---|
| Row independence: 6 chat prompts, alone vs together, greedy and sampled (T 0.528, seed 42), up to 512 tokens | 6/6 identical in both modes |
| Every row of batch 8, 16, 32, 64 equals batch 1 (256 tokens) | yes (`run.log`) |
| Decode fidelity, teacher forced, 32 WikiText windows × 1,024 positions after a 3,072-token prefill, against the exact FP32 model; tier-2b rule (FAIL iff CI above +1e-5) | **better than HF SDPA**: mean \|NLL − FP32\| 0.01502 vs SDPA 0.01731; D = −2.29e-3, CI [−2.56e-3, −2.02e-3] (`evidence/fidelity.json`) |
| Same positions, the accepted serial decode (for comparison) | 0.01486, D = −2.44e-3; HF eager 0.02006 (worse than SDPA) |
| Perplexity on those positions | batched 11.2506, serial 11.2472, FP32 11.2461, SDPA 11.2499 |
| Device greedy vs the host sampler, 8,192 rows with planted ties and suppressed maxima | 0 mismatches |
| Device sampling vs the host sampler's exact distribution, 204,800 draws per case, T 0.1 to 1.0 | χ² p = 0.62, 0.38, 0.20, 0.13; no suppressed or zero-probability id drawn |
| Coherency (`evidence/coherency.md`, read in full) | coherent and correct: sky, Fibonacci with complexity, 3:40 + 2:35 = 6:15 pm, the poems, "Priya" recalled, the 512-token essay |
| The accepted engine untouched by this work's `lfm3_main.cu` change (the long-context softmax rewrite) | its 4K and 32K perplexity NLLs are bitwise identical (`build/speed-v1/regression`, 72 and 9 windows) |
| Memory | 2.9–3.4 GiB used before and after every stage (`evidence/memory.log`) |

The batched decoder is a new arithmetic path, not bitwise the serial decode. Serial uses `thin` GEMMs at M ≤ 8 and
rounds probabilities to BF16. On the coherency prompts the two agree for 19 to 128 tokens before diverging.

## Speed: what the numbers are, and are not

**Decode-loop aggregate**: tokens generated across all rows ÷ wall time of the decode loop, after one untimed
warm-up. Prompt processing and model loading are excluded. This is the metric the LFM2.5 engine reports.
Workload: the reflective-recursion chat prompt (25 tokens) in every row, exactly 256 tokens each (stop disabled),
capacity 4,096. Two runs each:

| Batch | Decode-loop tok/s | Per stream |
|---:|---:|---:|
| 1 | 61.9, 61.5 | 62 |
| 8 | 486.8, 488.0 | 61 |
| 16 | 957.2, 956.9 | 60 |
| 32 | **1,756.0, 1,751.5** | 55 |
| 64 | **3,226.3, 3,222.2** | 50 |
| 32, sampled (T 0.528) | 1,507.4, 1,483.2 | 47 |

This is a best case: one short prompt, equal lengths, short context. The same batch-32 run **end to end**
(8,192 tokens over 0.72 s of serial prefill plus 4.65 s of decode) is **~1,525 tok/s**. At 3–4K tokens of context,
the teacher-forced run decoded 32 rows at ~1,170 tok/s. **Serving throughput has not been measured**: it needs the
server path, with mixed prompts and natural lengths.

Where batch 32's 18.5 ms step goes (nsys): GEMMs 14.8 ms (80%; 233 GB/s, 91% of Thor's measured 256.5 GB/s read
ceiling), attention 1.1 ms at ~280 tokens of context, norms and elementwise kernels 1.2 ms, GPU idle 1.3 ms (7%).

## Measured along the way, and rejected

- `wide` at small M is limited by too few blocks (16 for the 2,048-wide projections). The best tiles fix the MLP up
  projections and the head, but not the 7,680-deep down projection (~160 GB/s).
- Split-K for the down projection (4 slices): no clear gain at batch 1, −10% at batch 32, −20% at batch 64. It is
  off by default (`LFM3_BATCH_SPLITK`, for experiments).
- Thor's GPU clock ramps over about a second, and the first run of a pair can be 30% slow. Every rate here uses a
  full untimed warm-up and two runs.

## Next (not done)

1. Batched prefill: one pass for all queued prompts, not one per prompt.
2. Continuous batching in the server: refill a finished row with the next request.
3. CUDA graphs for the step (the 7% idle).
4. An end-to-end HTTP benchmark: N concurrent clients, first request to last answer. Target: ≥ 1,750 tok/s at N ≥ 32.
