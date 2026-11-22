# alloc-effect-v1: the language model's prefill GEMMs run ~1.8× slower in some states, restored by an allocation call

**The finding being investigated.** In audio-ttft-v1, keeping the audio encoder's scratch between clips made the language model's prefill about 40 ms slower at 90 s (0.178 against 0.138 s).
- One 2 MiB `cudaMalloc` + `cudaFree` after the encode restored the speed (`build/keep-diag/`).
- An isolated probe (`native/tests/alloc_effect_probe.cu`) found no effect: the same GEMM shape and copy bandwidth were unchanged after idle periods, with or without allocation calls.
- At three deploys in a row, a time-to-first-token pass that followed the stream check's heavy GPU work came in 5–19% under the same source's earlier runs.
- Tripp's go: 2026-11-22, item 2 of the remaining list.

## Stage 0 (2026-11-22, a diagnostic; reading the existing traces and one text run)
**The four-way traces (`build/a5-diag/*.sqlite`, audio-ttft-v1):**
- **Nothing runs beside the slowed GEMMs.** Every kernel is on one stream (7), and no kernel overlaps any of the 48 large GEMMs in any request of `merge-srv`, `ttft-srv` x2 or fp32 mode.
- **The slowdown is uniform across instances.** `ttft-srv` request 2 against request 1: each of the 48 grid-60×9 GEMMs is 1.57–2.01× slower (median 1.78), from the first to the last of the 140 ms prefill.
- **It is specific to two shapes.** In the same requests:

  | kernel | slow request ÷ fast request |
  |---|---|
  | `wide_kernel` 60×9 | 1.78 |
  | `wide_kernel` 48×9 | 1.80 |
  | `wide_kernel` 16×9 | 1.05 |
  | `norm` | 1.02 |
  | `swiglu` | 0.94 |
  | `attention_partial` | 1.00 |

- **So it is not the GPU clock** (the 16×9 GEMMs and the attention would slow too) **and not DRAM bandwidth** (the memory-streaming norm and swiglu would).
- **What the two slowed shapes share:** each re-reads a 25–31 MB weight matrix (N = 7,680 or 6,144, K = 2,048) across 9 row tiles, about 9× its size in L2 traffic.
- **Hypothesis, not tested:** an L2 or memory-subsystem clock or mode that drops when no allocation call has happened, and that a driver allocation raises. Reading it needs root (`sudo tegrastats`; `/sys/kernel/debug/bpmp` is denied to a user).

**Text-only prefill (`build/alloc-text/`, `tools/thor/alloc_effect_text.sh`, quiet 09:52–09:53):**
- `lfm3-speech` from main with a diagnostic switch in a throwaway copy: `LFM3_DIAG_TOUCH` set means one 2 MiB `cudaMalloc` + `cudaFree` before each prefill. Binary `9df9a245…`.
- Text-only prompts of 128, 512, 1,149 and 4,096 tokens, 8 new tokens, each 4 times per process; processes off, on, off, on.
- **The 4,096-token prompts failed.** Prompt plus new tokens exceeds `lfm3-speech`'s capacity of 4,096, so every process exited 1 after its other requests. The file `alloc-text/req/L4096.req.json` now holds a 4,000-token prompt (the same construction: the 15-token header, filler, the 7-token tail) for later runs.

Prefill seconds, first request | median of requests 2–4:

| tokens | off-1 | on-1 | off-2 | on-2 |
|---|---|---|---|---|
| 128 | 0.0304 / 0.0302 | 0.0318 / 0.0315 | 0.0306 / 0.0318 | 0.0318 / 0.0317 |
| 512 | **0.0791 / 0.0792** | 0.0669 / 0.0668 | 0.0674 / 0.0673 | 0.0674 / 0.0671 |
| 1,149 | **0.1809** / 0.1390 | 0.1386 / 0.1388 | 0.1400 / 0.1388 | 0.1388 / 0.1387 |

**Reading:**
- **Text prefill is not generally in the slow state.** At 1,149 tokens it is 0.139 s with or without the switch, the fast audio figure.
- **The slow state did appear, once, without the switch:** in the first process only, on all four of its 512-token prefills (0.079 against 0.067 s, +18%) and on its first 1,149-token prefill (0.181 against 0.139 s, +30%). The same process's 1,149-token prefills 2–4 were fast. No process with the switch was slow.
- That process was the first GPU work after the window opened (the service stopped, the GPU idle for over 30 s). That fits a state that decays when idle and is raised by allocation calls or by sustained work; it does not prove it.

**Next:** the same run, repeated, with the first process's state controlled: an idle wait before each process, and the switch's allocation alone against heavy GPU work. To be designed after tcgen05-v2's window, which holds Thor until mid-afternoon.
