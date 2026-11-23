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

### Stage 1 results (2026-11-22, `build/alloc-s1/`, quiet 21:27–21:34)
**The run:**
- Window 21:27–21:34. Sampler: 382 samples, CLEAN. Power state unchanged.
- Probe `5b50adbb…`, built from 7645e17's tree. T is `9df9a245…`.
- All five processes exited 0.

**Reading: STATE.** Placement is not met: no B buffer was slow in pass 1.

| process | B pass 1 (min / med / max ms) | B pass 2 | B pass 3 | A pass 1 | A pass 2 | first control | controls after |
|---|---|---|---|---|---|---|---|
| P1 | 0.944 / 1.056 / 1.212 | 1.362 / 1.547 / 1.788 | 1.442 / 1.546 / 1.748 | 1.600 / 1.677 / 1.781 | 1.590 / 1.690 / 1.829 | 0.866 | 1.64–1.89 |
| P2 | 0.935 / 1.088 / 1.279 | 1.394 / 1.543 / 1.836 | 1.446 / 1.546 / 1.828 | 1.637 / 1.723 / 1.848 | 1.637 / 1.735 / 1.855 | 0.843 | up to 1.83 |
| P3 | 1.003 / 1.084 / 1.232 | 1.257 / 1.559 / 1.784 | 1.474 / 1.556 / 1.726 | 1.598 / 1.641 / 1.743 | 1.580 / 1.648 / 1.697 | 0.862 | up to 1.75 |
| P4 | 0.996 / 1.087 / 1.280 | 1.362 / 1.544 / 1.733 | 1.456 / 1.553 / 1.759 | 1.574 / 1.647 / 1.849 | 1.568 / 1.659 / 1.812 | 0.850 | 1.61–1.75 |

- **The rule's own words.** Every A buffer is "slow" against the B pass-1 reference, and every B buffer "turns slow" in pass 2. That is the state moving under all of them, not placement. Per buffer, pass 1 and pass 2 do not track each other.
- **The pattern is the same in all four processes, whether or not idle came before:**
  - the fixed pair runs at about 0.85 ms right after setup;
  - it runs at 1.6–1.9 ms from the first control on, 8 buffers later, and stays there to the end;
  - B pass 1 runs at about 1.06 ms in the middle.
- **What precedes each timing differs.** In B pass 1, every timing follows a fresh `cudaMalloc`, a 30 MiB host-to-device copy, a chase and a fill kernel that wrote that buffer. In the passes that re-read, it follows only the previous GEMMs. At the start, both operands had just been written.
- **The translation chase is flat:** 61–70 ns per hop on every buffer.
- **buddyinfo:** each process consumed about 1,500 free order-9 (2 MiB) blocks and gave them back, with no visible effect.
- **T** (`lfm3-speech`, after 120 s idle) was slow throughout:

  | tokens | request 1 | median of 2–4 | stage 0's fast off-2 |
  |---|---|---|---|
  | 512 | 0.0762 | 0.0762 | 0.0673 |
  | 1,149 | 0.1740 | 0.1720 | 0.1388 |
  | 4,000 | 0.8036 | 0.8051 | (4,096 failed in stage 0) |

**An exploration after the window, not a rule** (`oc-explore.log`, the service idle, nothing else running):
- `/sys/class/hwmon` is readable by a user: soctherm over-current counters (oc2, oc3, throttling enabled), the VDD_GPU rail (ina3221) and VIN (ina238).
- A 24-buffer probe run of 1.1 s reproduced the step (control 0.837, then 1.53 ms).
- Neither over-current counter moved. VDD_GPU peaked at 1.62 A at 19.8 V (limit 5.04 A).
- **Over-current throttling is not the cause.**

**What this leaves:** the fast state follows a recent write of the operands, more than idle or allocation calls, and it is lost within a few hundred launches. Stage 2 tests that directly.

## Stage 2: what restores the fast state? (rules set 2026-11-22, before any data)
**The tool:** `native/tests/alloc_state_probe.cu` (`alloc-state-probe`), built twice:
- plain, which takes lfm::gemm's mma.sync path;
- with `-DLFM_GEMM_TCGEN05` and tc2-on's `gemm_tcgen05.o`, which takes the deployed tcgen05 path.

**Each process:**
- One fixed operand pair at the slowed shape.
- 300 back-to-back launches, each timed (the **sustained** series), right after both operands are written.
- Then 5 rounds. Each round runs ten conditions in a rotated order. Each condition gets 30 settling launches (**pre** is the median of the last 5), then the action, then 6 timed launches (**median6**).
- The conditions:
  - `none`;
  - `fill_b`: a kernel rewrites the weight, same values;
  - `fill_a`: the same for the activation;
  - `fill_other`: a kernel writes an unrelated 30 MiB buffer;
  - `copy_b`: the weight is copied to a spare and back by the copy engine;
  - `read_b`: a kernel reads the whole weight;
  - `idle_5ms`, `idle_100ms`, `idle_1000ms`: the host sleeps and the GPU is idle;
  - `alloc`: a 2 MiB `cudaMalloc` + `cudaFree`.

**The run** (`tools/thor/alloc_effect_stage2.sh`):
- Quiet Thor, the service stopped, the 1 s sampler, `power_state.sh`.
- The over-current counters, VDD_GPU current and VIN power sampled at 10 Hz.
- Processes in the order mma.sync, tcgen05, mma.sync, tcgen05, back to back, after 60 s idle.

**Readings, set now, per binary:**
- **Restores in a round:** median6 ≤ 0.80 × pre.
- **A condition restores** if it does so in at least 8 of that binary's 10 rounds (2 processes × 5).
- **Void:** if `none` restores in more than 2 of 10 rounds, the settling does not reach the slow state. The binary's readings are then void.
- The named readings:
  - **write of the operand:** `fill_b` or `copy_b` restores and `fill_other` does not;
  - **write in general:** `fill_other` restores;
  - **activation:** `fill_a` restores;
  - **read:** `read_b` restores;
  - **idle:** the shortest idle that restores;
  - **allocation:** `alloc` restores.
- **The sustained series:** the launch at which a time first exceeds 1.25 × the median of launches 1–5, and the median of launches 251–300.
- **Over-current:** whether either counter moved.

A diagnostic: no engine change, nothing gated.

### Stage 2 results (2026-11-22, `build/alloc-s2/`, quiet 21:52–21:55)
**The run:**
- Window 21:52–21:55. Sampler: 88 samples, CLEAN. Power state unchanged.
- Binaries: mma.sync `470a18a2…`, tcgen05 `1b34ee77…` (tc2-on `gemm_tcgen05.o` `3a8c563b…`). All four processes exited 0. tcgen05 launched 300 of 300 in T1 and T2.
- Hardware, sampled at 10 Hz (855 samples): the over-current counters moved 0 and 0; VDD_GPU peaked at 2.72 A; VIN at 135 W.

**As written, the reading is NONE RESTORES for both binaries: 0 of 10 rounds for every condition.** `none` restored in 0 of 10, so the run is not void by its rule. **The reading is still empty: the slow state never appeared.**

| binary | sustained: launches 1–5 | 251–300 | first launch over 1.25× | median pre across conditions |
|---|---|---|---|---|
| mma.sync M1 / M2 | 0.846 / 0.882 ms | 0.843 / 0.888 | none / none | 0.864–0.870 ms |
| tcgen05 T1 / T2 | 0.391 / 0.352 ms | 0.390 / 0.360 | none / none | 0.367–0.376 ms |

- The fixed pair ran at its fast speed through about 2,100 launches per process, with every condition between them. The rule did not foresee a slow state that never forms. That is the rule's gap, recorded as found.
- **What stage 2 removed from stage 1:** in stage 1, each B buffer's pass 1 allocated a new 30 MiB buffer (kept), uploaded 30 MiB from pageable host memory into it, ran the chase and filled it. The fixed pair was slow by the first control, 8 such buffers in. Stage 2 allocates five buffers at the start and nothing after; no fill, copy, read, idle gap or allocation call made it slow.
- **So the question turns around:** not what restores the fast state, but which of stage 1's per-buffer steps brings on the slow one. Stage 3.

## Stage 3: which of stage 1's steps brings on the slow state? (rules set 2026-11-22, before any data)
**The tool:** `native/tests/alloc_onset_probe.cu` (`alloc-onset-probe INGREDIENT`), built for mma.sync and for tcgen05 as in stage 2.

**One ingredient per process:**
- `none`;
- `malloc`: 30 MiB allocated, kept, untouched;
- `malloc_fill`: the same, written by a kernel;
- `h2d`: 30 MiB uploaded from pageable host memory into a spare allocated at the start;
- `h2d_pinned`: the same from `cudaMallocHost` memory;
- `malloc_h2d`: allocate, keep, and the pageable upload into it;
- `gemm_other`: 6 GEMMs on one of 8 other weights, allocated and written at the start;
- `stage1`: stage 1's whole step.

**The process:**
1. 30 launches of a fixed pair: the **baseline** median.
2. 8 steps of the ingredient, each followed by 6 timed launches of the fixed pair.
3. 30 launches: the **after** median.
4. Then, each followed by 6 launches: `idle_1000ms`, `fill_b`, `alloc` (2 MiB malloc + free), `free_kept` (free every buffer the steps kept), and `idle_1000ms` again.

**The run** (`tools/thor/alloc_effect_stage3.sh`):
- Quiet Thor, the service stopped, the 1 s sampler, `power_state.sh`, the 10 Hz hardware sampler.
- After 60 s idle, every ingredient twice per binary: 32 processes back to back, in a fixed shuffled order, binaries alternating.

**Readings, set now, per binary:**
- **Induces in a process:** after ≥ 1.25 × baseline.
- **An ingredient induces** if it does so in both of its processes.
- **Void:** the binary is void if `none` induces in either process.
- **Onset:** the first step whose 6-launch median is ≥ 1.25 × baseline, reported for inducing processes.
- **A restorer restores** in an inducing process if its 6-launch median ≤ 0.80 × after. This is reported per restorer; it is not a rule.
- **The start's footprint:** `gemm_other`'s baseline (8 extra 30 MiB buffers at the start) against `none`'s, reported.

A diagnostic: no engine change, nothing gated.
