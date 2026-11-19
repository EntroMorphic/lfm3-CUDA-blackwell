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

## Window 6 results (2026-11-17 07:27-08:39, quiet): the memlock hypothesis **falsified** (`quiet-w6/`)
| run | memlock (soft/hard) | stalls of 630 |
|---|---|---|
| host-8m-1 | 8,192 / 8,192 KiB (the container's) | **0** |
| host-default | 16,098,844 KiB | **0** |
| host-8m-2 | 8,192 / 8,192 KiB | **0** |
By the pre-registered reading, RLIMIT_MEMLOCK is not the trigger. The host is now 0 of 3,780 across five quiet runs;
the container 56 of 1,260. Remaining differences between the two: the user (root in the container, uid 1000 on the
host), the container's seccomp filter, its memory cgroup (`system.slice/docker-*.scope` against the user slice),
its mount namespace (the image's overlayfs for `/pinned-cuda` and the nvidia libraries; `/workspace` bind-mounted),
and the nvidia container runtime's device setup.
Next (window 8, `tools/thor/quiet_session_v8.sh`): the probe in the container as root (control), uid 1000, uid 1000;
a catch inside the container as uid 1000 if uid 1000 stalls.

## Window 8 results (2026-11-17 11:25-12:42, quiet): running as root **falsified** (`quiet-w8/`)
| run | user in the container | stalls of 630 | median prefill |
|---|---|---|---|
| container-root (control) | root | **34** | 0.033 s |
| container-u1000 | 1000 | **31** | 0.032 s |
| container-u1000-2 | 1000 | **26** | 0.032 s |
The control reproduces, and both uid 1000 runs stall (>= 5 each): by the pre-registered reading in
`tools/thor/quiet_session_v8.sh`, **the user is not the trigger**. The stalls look as before: 3.3-4.1 s, 1.5-12 M minor
faults each (`proc_minflt` = the cgroup's `pgfault`), 1.5-3.7 s of kernel time. The container is now 116 of 3,150
across five quiet runs; the host 0 of 3,780. Remaining differences: seccomp, the memory cgroup's place in the
hierarchy, the mount namespace (overlayfs for `/pinned-cuda` and the nvidia libraries) and the nvidia runtime's
device setup. **The catcher** (inside the container as uid 1000, host gdb attached, 12:42-13:08): 4 catches, each 0.1 s, the
engine exited 0. In all four the faulting thread (the main thread, tid 972316) was stopped at **the same instruction,
`0xfffe9f4f3344`**; the three other threads at `0xffffa7f30bdc` (waiting). Host gdb could not read symbols across
the PID namespace and the container has no gdb, so the PC is not yet tied to a library. One PC repeated through
~11 M minor faults reads as a fault-retry storm (the kernel resolves the fault, the mapping is invalidated, the same
instruction faults again), not code that walks new memory. Host state at the catches: `nr_foll_pin_acquired`
7.5 M (6,704 more than released), CMA 512 MiB (464 MiB free), 117 GiB available, no reclaim pressure. Window 11
(`tools/thor/quiet_session_v11.sh`) runs the catcher again with frame-0 PCs resolved against `/proc/PID/maps`.

**Correction (window 5 onward):** the `memlock` figure in `limits.txt` for container runs is the host shell's
(`ulimit -l` runs in `stall_probe.sh`, outside `docker exec`), so it reads 16,098,844 KiB. The container's own limit
is 8 MiB, as measured inside it for window 6; the container (created 2026-11-06, started 2026-11-15) was not changed.

## Window 11 results (2026-11-17 15:33-15:58, quiet): **the faulting instruction is in libcuda** (`quiet-w11-catch/`)
4 catches (container, uid 1000), the faulting thread the engine's main thread each time, frame 0 at the same address:
**`/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1.1`, file offset `0x3d3344`** (window 8's `...f3344` is the
same instruction at another load address). The library is the same file inside and outside the container (sha256
`fab2598e...`). The instruction, from `objdump -d` (stripped; the nearest exported symbol, `cuVDPAUCtxCreate`, is
0x91a70 bytes before it and unrelated):
```
3d331c: ldr x0, [x20, #5264]    ; a driver structure
3d3320: ldr w2, [x0, #40]       ; the value to write
3d3324: ldr x1, [x21, #64]
3d3328: ldr w3, [x1, #3152]
3d332c: ldr w0, [x1, #3488]     ; an offset
3d3330: ldr x1, [x1, #27280]
3d3334: cmp w3, #0x8 ; b.ls 3d335c
3d333c: ldr x1, [x1, #5408]     ; a base address
3d3340: add x0, x1, x0
3d3344: str w2, [x0]            ; <- every catch
3d3348: b 3d320c
```
One 32-bit store to base + offset, then a return: the shape of a doorbell write (the driver telling the GPU new work is
queued), not a copy. The process maps `/dev/nvidia0` write-only twice (64 KiB each), as a doorbell region would be.
**Working hypothesis, not established:** the page this store writes keeps being unmapped, so every store traps (11 M
minor faults at ~0.3 us each, all kernel time). The address it writes is not yet known. Window 14
(`tools/thor/quiet_session_v14.sh`) records the registers at the catch and resolves `x0` to its mapping, and compares
the container's device mappings with a host run's.

## Window 14 results (2026-11-17 17:46-18:36, quiet), plus checks the same evening: **the container cannot open `/dev/nvidia-uvm`** (`quiet-w14-catch/`)
**The store writes the doorbell.** In all 4 container catches (uid 1000), `x0 = 0xffffa0530090`, which sits in
`/dev/nvidia0 -w-s 0xffffa0530000-0xffffa0540000` at offset `0x90`; `x1` holds that mapping's base. The faulting
`str w2, [x0]` is the driver's 32-bit write to the GPU's write-only doorbell page.

**Host and container mappings differ in one line.** Maps snapshots at 120 s, the same engine and requests:

| Mapping | Container | Host |
|---|---|---|
| `/dev/nvidiactl`, `/dev/zero`, `/dev/nvidia0 -w-s 64 KiB` x2 | the same counts and sizes | the same |
| `/dev/nvidia-uvm rw-s 2048 KiB` | **absent** | 1 |

The host run had 0 catches again, so 0 stalls.

**Checks after the window (read-only):**
- The container has the node `/dev/nvidia-uvm` (c 488:0, mode 666), but `open(O_RDWR)` on it fails with **EPERM**, as uid 1000 and as root. `/dev/nvidia0` and `/dev/nvidiactl` open. EPERM on a mode-666 node means the container's device cgroup refuses it.
- libcuda is the same file inside and out (sha256 `fab2598e...`), so the driver takes a different path in the container because it has no UVM, not because it is another library.
- The container `lfm3-profile-v1` was created 2026-11-06 (`--runtime nvidia`, a GPU device request, no device-cgroup rules of its own, not privileged). Thor last booted 2026-11-14 21:12.
- `/dev/nvidia-uvm`'s major is assigned at module load (now 488). **Likely, not shown:** the container's device allowlist was fixed at creation with the major of that boot, and after the reboot no longer covers the new one. Not checked: the container's BPF device program, and whether stalls appeared before the 2026-11-14 boot.

**Reading:**
- In the container, CUDA runs without UVM. That is the one difference found between the runs that stall (container: 116 of 3,150 requests) and those that don't (host: 0 of 3,780).
- The stall is a write to the doorbell page that traps.
- Not yet shown: that giving the container UVM ends the stalls.

**The test that would show it** (a container setting: Tripp's call, not made):
- Run the window-8 stall reproduction in a fresh container from the same image, with the same mounts and runtime, so the GPU device allowlist is computed for this boot. The one difference to the current container is that `/dev/nvidia-uvm` opens.
- Prediction: 0 stalls, or the stall count drops to the host's, and the maps show `/dev/nvidia-uvm`.
- If it holds, the fix is to recreate the container, or to run the service on the host (always-on-v1 rule 5 already needs the host).
- Until then, every timing taken in `lfm3-profile-v1` since the reboot carries the stall. That includes window 15's first-request times, whose individual requests can include a stall.

## The UVM test, set before its data (2026-11-17 ~20:40; Tripp approved a throwaway container)
- **Container:** a fresh `lfm3-uvm-test-v1`, a copy of `lfm3-profile-v1`'s configuration. `lfm3-profile-v1` itself is untouched.
  - The same image ID (`sha256:2a9b0054...`) and `--runtime nvidia` with all GPUs.
  - Network none, the 96 GiB memory limit, `sleep infinity`.
  - The same three mounts (`/workspace` rw, `/pinned-cuda` ro, `/opt/nsys` ro) and the same environment.
- **Runs:** window 8's probe (`tools/thor/stall_probe.sh vimg-a5`, the 630 dev TTS requests, as uid 1000, so the engine's maps are readable from the host; uid 1000 had 31 and 26 stalls in window 8).
  - One control run in `lfm3-profile-v1`.
  - Two runs in the test container.
  - In each run, about 60 s in, the engine's `/proc/PID/maps` is checked for `/dev/nvidia-uvm`.
- **Precondition:** in the test container `open("/dev/nvidia-uvm", O_RDWR)` succeeds. If it doesn't, the test is void and nothing is concluded.
- **Prediction, if the missing UVM is the cause:**
  - The test container's runs map `/dev/nvidia-uvm` and have at most 2 stalls each. The host had 0 in 3,780 requests.
  - The control run maps no UVM and has stalls as before: 26-38 per 630 requests in windows 5-8 (uid 1000: 31 and 26).
- **Falsified if:** the test container's runs map UVM and still have stalls in the control's range (10 or more per run).
- **Anything in between** is recorded as unresolved.
- **Attempt 1 (20:35, `quiet-uvm-void1/`): VOID by the precondition.**
  - The fresh container, created by the nvidia runtime with the same configuration, has no `/dev/nvidia-uvm` node at all (ENOENT). `/dev/nvidia0` and `/dev/nvidiactl` open.
  - So on this Thor the runtime does not pass UVM to a container. How `lfm3-profile-v1` came to have the node, but no permission on it, is not known.
  - The container was removed, nothing ran in it, and nothing was measured.
- **Attempt 2:** the same test container with one addition, `--device /dev/nvidia-uvm`. Docker creates the node and allows it in the container's device cgroup.
  - The precondition, prediction and falsification are unchanged.
  - If it holds, the fix is a container created with `--device /dev/nvidia-uvm`, or the service on the host.

## The UVM test, attempt 2 (2026-11-17 20:37-22:09, quiet; `build/quiet-uvm/`): **the prediction holds; the stall is the missing UVM**
| Run | Container | `/dev/nvidia-uvm` mapped (60 s in) | Stalls (prefill > median + 1 s) | Median prefill | Span of the 630 requests |
|---|---|---|---|---|---|
| control | `lfm3-profile-v1` | 0 | **24** (3.6-3.9 s each, ~11 M minor faults each) | 0.033 s | 1,535 s |
| test 1 | `lfm3-uvm-test-v1` | 1 | **0** | 0.033 s | 2,530 s (see below) |
| test 2 | `lfm3-uvm-test-v1` | 1 | **0** | 0.033 s | 1,436 s |

- **Setup:** uid 1000, `vimg-a5/lfm3-speech`, all 630 requests done with 0 failed in each run. `/dev/nvidia-uvm` opened in the test container (`--device /dev/nvidia-uvm`) and was refused (EPERM) in the control.
- **By the pre-registered reading:** the test container maps UVM and has at most 2 stalls per run (it has 0 and 0); the control does not map UVM and stalls as before (24). **The cause is established:** in `lfm3-profile-v1`, CUDA runs without UVM, and that is what makes the doorbell write fault.
- **Test 2's span** is 99 s shorter than the control's, about the control's 24 stalls at ~3.7 s each.
- **Test 1 was slow for another reason, not explained:** between outputs it had gaps of up to 18 s (the control's largest is 7 s, test 2's 4 s), with normal prefills and 0 stalls. Nothing else was scheduled on Thor in that window. It does not bear on the stall count; it is recorded as found.

**The fix:** run the engine with UVM available. For a container that means `--device /dev/nvidia-uvm` (this Thor's nvidia runtime does not pass it). The always-on service runs on the host, which has UVM.
- **Recreating `lfm3-profile-v1` is Tripp's call.** Until then, timing measured in it can carry the stall.
- **Still open:** how `lfm3-profile-v1` got a `/dev/nvidia-uvm` node it may not open. The reboot tonight checks whether the major number moves.
- The test container is stopped, not removed.

## The fix applied (2026-11-18 17:41, approved by Tripp; `build/container-uvm-swap/`)
`lfm3-profile-v1` is now a new container with UVM. The old one is stopped and kept as `lfm3-profile-v1-pre-uvm`, for rollback.
- **Configuration:** compared field by field with the old one (image, runtime, GPUs, network, memory and swap limits, shm, restart policy, command, working directory, environment, mounts). The only difference is `--device /dev/nvidia-uvm`. Both inspect records are saved.
- **Old container's writable layer:** NVIDIA runtime files and caches; no package changes (`/var/lib/dpkg` untouched).
- **Checks on the new one:** `/dev/nvidia-uvm` opens as root and as uid 1000; the engine maps it; the 11 stage-0 image requests are **67 of 67 files identical** to `vimg-e2e-a4`.
- **Timing comparisons:** anything measured in this container before today carries the stall; anything after does not. Stall-sensitive numbers from the old container should not be compared with new ones without saying so.
- **Restart policy is "no", as before:** after a reboot the container must be started by hand (`docker start lfm3-profile-v1`).
