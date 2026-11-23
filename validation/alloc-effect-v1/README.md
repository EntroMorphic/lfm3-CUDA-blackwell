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

## Stage 1: does the slowdown follow the buffer or the time? (rules set 2026-11-22, before any data)
**The question.** Every sighting so far fits two different causes:
- **Placement:** the speed depends on where the driver put a buffer's memory (for example large pages against small ones, so address translation costs more on the GEMMs that re-read a 30 MiB weight 9 times and a 4.5 MiB activation 60 times). A new allocation lands somewhere else, which would explain why a 2 MiB `cudaMalloc` + `cudaFree`, or releasing the encoder's scratch, changed the speed.
- **State:** a clock or mode of the memory system that drops when the GPU is idle and recovers under work. That would explain why the first process after an idle window ran slow twice (stage 0, and tcgen05-v2's amendment-2 rule 5: `off-1` 1,149 tokens 0.170 against `off-2` 0.137 s).

One user-readable clue: `/proc/buddyinfo` shows how much free memory is in large blocks. The memory clock needs root and is not read.

**The tool:** `native/tests/alloc_placement_probe.cu` (`alloc-placement-probe`), built with nvcc as `alloc-effect-probe` was.
- One process, back to back, no idle waits:
  - **B:** 96 fresh 30 MiB weight buffers (N 7,680 × K 2,048), each kept, with one fixed activation buffer. lfm::gemm at M 1,149 (the slowed grid 60×9) is timed on each in allocation order (pass 1), then in reverse (pass 2), then in order again (pass 3). Pass 1 also times a one-thread pointer chase, one hop per 64 KiB region, shuffled (ns per hop: the cost of address translation).
  - **A:** 64 fresh 4.5 MiB activation buffers, with one fixed weight, timed in order and in reverse.
  - **Control:** the fixed pair, re-timed every 8 buffers: a time series of the GPU's state.
- Every time is the median of 5 launches after 1 untimed. `/proc/buddyinfo` is read before and after.

**The run** (`tools/thor/alloc_effect_stage1.sh`): quiet Thor, the service stopped (restarted by an EXIT trap), the 1 s sampler, `power_state.sh`.
1. 120 s idle, then probe **P1**.
2. **P2**, immediately after P1.
3. 120 s idle, then stage 0's `lfm3-speech` (`9df9a245…`, switch off) on the same 128 / 512 / 1,149 / 4,000-token prompts, 4 times each (**T**).
4. **P3**, immediately after T.
5. 120 s idle, then **P4**.

**Readings, set now.** Within a probe process, the reference is the median of its B pass-1 times. A buffer is slow if its time exceeds 1.25 × the reference.
- **Placement** if, in some process:
  - at least 2 buffers (B or A) are slow in pass 1;
  - at least 80% of them are slow again in every later pass;
  - at most 5% of the pass-1 fast buffers turn slow later;
  - no control reading exceeds 1.25 × that process's first control.
  
  The chase then says whether it is translation: placement plus every slow B buffer's ns per hop above every fast one's means translation is supported.
- **State** if any control reading in a process exceeds 1.25 × that process's first, or one process's reference exceeds 1.25 × another's.
- **Neither** if no process has 2 slow buffers and no state reading is met. The effect is then not reproduced by fresh allocations or by idle, and that is recorded as such.
- **T** is read against stage 0's off-2 (fast) figures: slow if its median of requests 2–4 at 512 or 1,149 tokens exceeds them by 10%. This is reported beside the probe; it is not a rule.

This is a diagnostic: it changes no engine code and gates nothing. What follows depends on the reading: placement points to how the engine allocates; state points to a pre-measurement warm-up for timing rules, and to Tripp's `sudo tegrastats` for the memory clock.
