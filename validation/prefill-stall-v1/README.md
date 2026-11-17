# An intermittent 3-4 s stall in the engine's prefill

## How it was found (2026-11-16)
image-cache-v1's rule 0 saw turn 1 of a 4K image conversation alternate between 0.6 and 3.9-4.1 s of prefill. A scan
of earlier runs' per-request timings (the audio engineer, read-only) found 86 requests more than 1 s above their run's
trend in 12 runs; the intermittent +3 to +4 s kind in audioin-v2 (1 of 2,620, in `encode_seconds`),
speech-normalize-v1 (54: TTS prefills, no media encoder), images-speed-v1's `vspeed-v3`/`v4` (one each; the median of
three passes hid them) and audio-in-listen (1). So: engine-wide, not the encoders'.

## What is known
- **stall-v1** (`tools/thor/stall_probe.sh vimg-a5 stall-v1`, quiet Thor, speech-normalize-v1's 630 dev TTS
  requests, kernel counters and the process's own every 100 ms; `stall-v1-analysis.txt`): **15 stalls** (3.4-4.1 s of
  prefill against a 0.03 s median). Each one coincides with **10-12 million minor page faults in the engine process**
  and 3.1-3.7 s of its kernel CPU time inside the prefill window; normal requests: none. The system-wide reclaim and
  compaction counters did not move in those windows: not memory pressure, not swap (there is none); NUMA balancing is
  off (one node). Transparent huge pages are `always` (defrag `madvise`); the machine had 80 GB free and 33 GB of page
  cache.
- **stall-v2** (under `strace`, mmap/munmap/madvise/brk/mprotect, 300 requests): **no stall**; the tracer slowed the
  engine (generate 6 s instead of 1.5). No large mapping call during requests (the 4.3 GB unmap at exit).
- **stall-v3** (under `gdb` from the start): 6x slower, no stall in 145 requests; stopped.
- **stall-v4** (`tools/stall_catch.py`: full speed, `native/probes/ptracer_any.c` preloaded so gdb may attach the
  moment a thread faults over 1M/s): **no stall in 630 requests**, 16:35-16:55, three other engineers' exactness jobs
  sharing the GPU; 50 GB of page cache, 57 GB free.
- **quiet-w3** (`stall_probe.sh vimg-a5`, quiet window 3, 18:14-18:40, MAXN, nothing else on the GPU;
  `quiet-w3/`): **28 stalls in 630 requests** (3.3-4.1 s against a 0.033 s median), each again with 10-12 million
  minor faults in the engine (two smaller: 1.8 M and 3.5 M, 3.4 and 3.3 s) and 3.0-3.8 s of kernel time. So a quiet
  machine reproduces it reliably; stall-v4's load is what hid it. Which requests stall is not fixed: 3 of quiet-w3's 28
  were among stall-v1's 22 (same list, same order). They come in runs at a steady spacing (positions 50, 54, 58;
  283, 292, 301; 580, 586, 592, 598). Free memory fell 71.6 -> 64.5 GB and page cache rose 50.0 -> 57.6 GB during the
  run; buddyinfo shows the free order-8 blocks falling 2,409 -> 159 (order 9 and above unchanged): the kernel's
  supply of mid-sized free blocks was used up while the engine faulted.
So the trigger depends on the machine's state at the time, not only on the request sequence; GPU load from other
work suppresses it.

- **quiet-w4** (`tools/stall_catch.py`, quiet window 4, 22:48-23:12, nothing else running; `quiet-w4/`): **0 stalls in
  630** (prefill median 0.033 s, max 0.036 s), no catch. Like stall-v4, the engine ran **on the host**; every stall
  ever recorded (stall-v1, quiet-w3, the server runs) ran **inside the container `lfm3-profile-v1`**.
- **The container's memory cgroup** (read 23:20): `memory.max` 96 GiB, `memory.peak` = the limit, `memory.events`
  **max 33,326** (times the limit was hit), 43 GB of page cache (`file`) charged to it against 0.12 GB of anon,
  in-cgroup reclaim `pgscan` 6.9 M / `pgsteal` 6.4 M. The probe's global counters (`pgsteal_direct`, kswapd,
  compaction) do not see reclaim at a cgroup's limit, which is why they "did not move".
- **Hypothesis** (not yet tested): the runs' outputs and reads fill the container's page cache up to its 96 GiB limit;
  from then on allocations inside the container reclaim from its own cache synchronously, and the prefill pays
  seconds of kernel time and millions of faults. Window 5 (`tools/thor/quiet_session_v5.sh`, rule written in the
  script before the data): the probe samples the cgroup and runs container, host, container.

## Next
Window 5 tests the container hypothesis; if it holds, the fix is a container setting (the memory limit, or where
the outputs' page cache is charged): Tripp's to change, with the evidence. (Superseded: repeat stall_catch on a quiet
machine, done in window 4), with /proc/meminfo, buddyinfo and the THP counters snapshotted
around each stall; if caught, the backtrace names the code that touches ~40 GB of 4 KiB pages. Not yet a fix: no
change is proposed until the cause is known. A system setting (THP) would be Tripp's to change, with the evidence.

## Window 5 note, written before its data was read (2026-11-17 01:30)
Window 5 began at 01:12:57; a tcgen05 racecheck (1,024-token text, `build/tgemm-r3c/`, in the same container) ran
01:12:58-01:19:56, overlapping the first ~7 min of the container-1 probe. Since load has hidden the stall before
(stall-v4), container-1 is read only on requests whose prefill starts after 01:20:00; container-2 (after the host
run) is the clean container run. The pre-registered reading in `tools/thor/quiet_session_v5.sh` otherwise stands.

## Window 5 results (2026-11-17 01:13-02:31, quiet): the cgroup-limit hypothesis **falsified**; container vs host **confirmed** (`quiet-w5/`)
| run | where | stalls of 630 | cgroup `memory.events max` during the run |
|---|---|---|---|
| container-1 | lfm3-profile-v1 | **26** (all 26 after 01:20, none during the racecheck overlap) | 251,534 -> 251,534 |
| host | Thor host, same binary | **0** | (not in the cgroup) |
| container-2 | lfm3-profile-v1 | **30** | 251,534 -> 251,534 |
Each stall again 10.6-11.5 M minor faults (the cgroup's `pgfault` equal to the process's) and 3.2-3.6 s of kernel
time. The container's memory stayed far below its limit (file 4.5-5.7 GB, anon < 0.25 GB), its limit was never hit,
and its in-cgroup reclaim (`pgscan`, `pgsteal`) did not move once in the window. By the pre-registered reading the
hypothesis does not hold. The container/host split is now 56 of 1,260 against 0 of 1,890 (quiet-w4 and w5's host runs).
**Ruled out since**, by direct comparison: `libcuda.so.1` (same sha256 inside and out), the pinned CUDA libraries
(same sha256), glibc (2.39 both), the binary and model (the same files), THP (`always` both), CPU limits (none).
**One difference found**: RLIMIT_MEMLOCK is 16 GB on the host and **8 MiB (soft and hard) in the container**, root
included (Docker grants no CAP_IPC_LOCK). The engine pins host memory (`codec.cuh` `cudaMallocHost`) and the driver
pins pageable buffers for copies; a pin that fails or falls back under 8 MiB would fit 11 M repeated minor faults
in kernel time with no memory growth. Window 6 (`tools/thor/quiet_session_v6.sh`, its reading written in the
script before the data) runs the host at 8 MiB, the default, 8 MiB.

## Window 6 results (2026-11-17 07:27-08:39, quiet): the memlock hypothesis **falsified**
Host runs at the container's RLIMIT_MEMLOCK (8 MiB soft and hard, `limits.txt`), at the host default (16 GB), at 8 MiB
again: **0, 0, 0 stalls of 630** (median prefill 0.033 s each). By the pre-registered reading memlock is not the
trigger. Running total: container 56 of 1,260, host 0 of 3,780.
Still different inside the container: the engine runs as **root** (uid 0; the host runs are uid 1000); a seccomp
filter (mode 2); `/dev/shm` 64 MiB; the container image's own libstdc++/libgcc/libgomp; the mount namespace.
Next (window 8): the same probe inside the container as uid 1000 (`docker exec -u 1000`), order uid 1000, root,
uid 1000.
