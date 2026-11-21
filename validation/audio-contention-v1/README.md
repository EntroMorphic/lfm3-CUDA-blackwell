# audio-contention-v1: the handler's `log_mel` against the engine's decode steps

## Why
audio-handles-v1 rule 3 (the stall a clip's admission causes the other rows) still fails as written: the long clip is about 8 ms over its bar at 8 and 16 rows (window 19).

Window 16's trace (audio-frontend-v1) put about 15–20 ms of the 90 s clip's excess in slower decode steps while the handler's `log_mel` runs. The steps took 39.8 and 32.2 ms against 25.8 ms. One thread against four made no difference (window 16), so the cost follows the total work, not the thread count. That leaves the mechanism open. Three candidates:
- **scheduling:** a `log_mel` thread takes the engine thread's core, or delays its wake-ups;
- **shared resources:** memory bandwidth or the shared cache, which neither priority nor placement changes;
- **frequency:** ruled out for the record. The CPU clocks are locked at 2.601 GHz (min = max) in every measured window.

audio-handles-v1 listed three options. This round takes the two that are not Tripp's call: lower the `log_mel` threads' priority, and run them on cores the engine thread does not use. The third, the cheaper direct DFT, would reverse audio-frontend-v1 amendment 2 and stays Tripp's call.

**Thor's CPUs, as found on 2026-11-20:**
- 14 cores, one thread each, a 1 MiB L2 each, one NUMA node.
- The kernel isolates cores 0–2 (`isolcpus=0,1,2 nohz_full=0,1,2 rcu_nocbs=0,1,2`).
- Host processes, the always-on service included, may run on cores 0 and 3–13. Processes in `lfm3-profile-v1` may run on 0–13.
- The isolated cores are Tripp's configuration. No default of this round uses them.

## The switches (environment variables, read once; unset means today's behaviour exactly)
- **`LFM3_ENGINE_CPUS`** (a CPU list such as `3` or `3,5-7`): the in-process engine thread (`Service`'s owner) pins itself to these CPUs when it starts.
- **`LFM3_MEL_CPUS`** (a CPU list): `log_mel`'s worker threads pin themselves to these CPUs.
- **`LFM3_MEL_NICE`** (an integer 1..19): `log_mel`'s worker threads lower their own priority to this nice value. Linux sets nice per thread, and an unprivileged thread can raise its nice value but not lower it again.

**The threads.** With either `log_mel` switch set, the calling thread (the request's handler) computes nothing itself. It starts one worker per range, the same ranges as today, and waits. So no handler thread is ever pinned or niced, and none needs restoring.

**Exactness.** The features do not depend on which thread computes a range. That is audio-frontend-v1 rule 1, byte-identical at 1–12 threads.

**Refusals.**
- A malformed value is refused with the variable's name, as `LFM3_MEL_THREADS` is.
- The server checks all three switches at start-up, before it binds its port. It refuses a CPU outside the process's allowed set with the CPU's number. So a misplaced CPU fails the start, never a request.
- If a thread cannot be started, the range is computed on the calling thread, unpinned and unniced. The bytes are the same, as today.

**Thread names:** the engine thread is named `lfm3-engine` and `log_mel`'s workers `lfm3-mel`, for traces and `top`. That is the only change with the switches unset.

## Rules, set before any code (2026-11-20)
**Rule 1, exactness (gated):**
- The host suites in full, plain, ASan+UBSan and TSan, with new tests: each switch's parsing and refusals; `log_mel` byte-identical with each switch set (1, 4 and 12 threads, the edge lengths); a worker actually on its CPUs and at its nice value (read back from `/proc`).
- On Thor, the server's audio-in check and 32 requests at once, with the chosen setting: identical to lfm3-speech.

**Rule 2, the diagnostic (stage 1, reported; it chooses the setting):**
- **Method:** one quiet window, `measure_lfm3_audio_stall.py` on this round's server in `lfm3-profile-v1`, the 90 s clip and the long clip, 8 rows, 3 repeats. One throwaway run comes first. Monitoring as audio-handles-v1's third run: the 1 s sampler and the power state checked before every run.
- **Variants:**

  | variant | setting | purpose |
  |---|---|---|
  | V0 | none (today) | baseline |
  | V1 | `LFM3_MEL_NICE=19` | priority |
  | V2 | `LFM3_ENGINE_CPUS=3`, `LFM3_MEL_CPUS=10-13` | placement |
  | V3 | V1 and V2 | both |
  | V4 | `LFM3_ENGINE_CPUS=2` (isolated, `nohz_full`) | diagnostic only, never a default: the engine thread alone on a core |

- **Order:** 3 rounds, rotated: V0 V1 V2 V3 V4, then V2 V3 V4 V0 V1, then V4 V0 V1 V2 V3.
- **Figures:** per variant and clip, the median stall over the rounds and its spread (maximum − minimum). Also the median "worst gap without": the decode steps without a clip, to show whether a variant changes the steps themselves.
- **A variant helps** if, for both clips, its median stall is below V0's by more than max(3 ms, the larger of the two spreads).
- **Choice:** the first helping variant in the order V1, V2, V3. V1 is preferred because it assumes no core layout.
  - If only V4 helps, the mechanism is scheduling on the engine thread's own core. Using an isolated core is Tripp's call, so the round stops and asks him.
  - If none helps, scheduling is ruled out. The cost is in shared resources, and this round records that and stops. The remaining option, the DFT, is Tripp's call.
- **Void if:** the sampler sees a live foreign process, or the power state changes.

**Rule 3, the stall (gated; only if rule 2 chooses a setting):**
- The setting becomes the server's default, in code. On the host, where the always-on service runs, it must use only cores 0 and 3–13.
- Then audio-handles-v1's rule 3 as written, in one quiet window: the bar measured in that window, all five clips, 8 and 16 rows. The deployed merge (`merge-srv`) runs in the same window for the record.
- **Passes** if every cell's stall is at or under its bar.

**Rule 4, nothing else slower (gated, same window as rule 3):**
- The worst gap without a clip (the decode steps alone), at 8 and 16 rows, may not be above the merge's by more than max(3 ms, the larger spread).
- `log_mel`'s own time for the 90 s clip (`audio-encoder-bench`, front end only) is reported. Niced or pinned workers may be slower, and that is acceptable if rule 3 passes, because the features are computed before the request reaches the engine.

**Rule 5, the record:** every run kept, every failure recorded, nothing rerun to pass without a new rule set before its data.

## The code (2026-11-20, after the rules)
The code is on the branch `audio-contention-v1` (0ed1e7e, eb5b005), **not merged**. Whether to merge the switches is Tripp's call.
- `lfm/thread_place.hpp` holds the three switches: parsing, the allowed-CPU check, and the per-thread pin, nice and name. `log_mel`'s workers and the engine thread call it.
- `Service`'s constructor and the server's `main` check all three before the model loads. On Thor, the real server refuses each bad value in 0.04 s, exit 1, with the variable's name.
- `audio-input-tool mel-place` and `tools/check_lfm3_thread_place.py` read the threads' CPUs and nice values back from `/proc`.

## Rule 1, the host part: PASS
- **New tests:**
  - 18 in `test_audio_input_host.py`: bytes unchanged at 1, 4 and 12 threads with each switch, on 1, 400, 16,000 and 160,000 samples; workers placed and the caller untouched, read from `/proc`; 19 refusals;
  - 27 in `test_lfm3_http_warmup.py`: the engine thread named and pinned; 8 start-up refusals.
  - All pass in the plain, ASan+UBSan and TSan builds.
- **Host suites in full** (`tests` and `audit/test_host_contracts.py`, all three builds, workstation, 65 minutes): **2,614 passed, 3 skipped, 0 failed.**

## Stage 1 (rule 2): no setting helps (2026-11-20, Thor, 17:45–18:41; `build/cont-stage1/`)
The window was clean:
- the sampler took 3,164 samples and saw no foreign process, and never more than one server (defunct entries, which are allowed, in 3);
- the power state was the same before every run and at the end;
- every run exited 0.

**Placement as set: PASS.** In the real server, each variant's engine thread and `lfm3-mel` workers had exactly the variant's CPUs and nice value, and every request returned 200.

The stall in seconds, median over 3 rounds (spread), and the median worst gap without a clip:

| variant | 90 s clip | long clip | worst gap without |
|---|---|---|---|
| V0 none | 0.3174 (0.0007) | 0.1257 (0.0021) | 0.439 s |
| V1 mel nice 19 | 0.3172 (0.0013) | 0.1253 (0.0009) | 0.439 s |
| V2 engine on 3, mel on 10–13 | 0.3161 (0.0011) | 0.1317 (0.0007) | 0.451 s |
| V3 V1 + V2 | 0.3171 (0.0003) | 0.1318 (0.0048) | 0.451 s |
| V4 engine on 2 (isolated) | 0.3159 (0.0012) | 0.1316 (0.0002) | 0.451 s |

**Verdict, as set: none helps.**
- V1: −0.3 ms and −0.4 ms.
- V2, V3 and V4: about −1 ms on the 90 s clip, but about **+6 ms** on the long clip.

**What it shows:**
- **Scheduling is ruled out.** Lowering the workers' priority changes nothing. Even the engine thread alone on an isolated `nohz_full` core does not reduce the stall. So the `log_mel` cost reaches the decode steps through shared resources (memory bandwidth or the shared cache), as the rules anticipated for this outcome.
- **Pinning the engine thread hurts.** With the engine thread on one core (V2, V3 and V4 alike, isolated or not), the decode steps alone are slower: the worst gap without a clip is 0.451 s against 0.439 s, about +12 ms. That is unexplained. One untested hypothesis is that the unpinned thread benefits from moving between cores, for example away from interrupts or CUDA's helper threads.
- None of the switches becomes a default.

**As the rules set:** the round stops here. Rules 3 and 4 do not run, and rule 1's Thor part (exactness with a chosen setting) does not apply. The remaining option for the last ~8 ms of audio-handles-v1's rule 3 is the cheaper direct DFT, which is Tripp's call.

**Noted, not a result:** the long clip's V0 stall here, 0.1257 s, is under window 19's bar of 0.1413 s. That bar was measured in a different window, and window-to-window movement of about 25 ms has been seen (audio-handles-v1), so this is not evidence that rule 3 passes.
