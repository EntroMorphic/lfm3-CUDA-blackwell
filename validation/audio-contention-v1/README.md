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
