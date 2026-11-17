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

## Next
Repeat stall_catch on a quiet machine (window 4: quiet-w3 shows the quiet probe reproduces it), with /proc/meminfo, buddyinfo and the THP counters snapshotted
around each stall; if caught, the backtrace names the code that touches ~40 GB of 4 KiB pages. Not yet a fix: no
change is proposed until the cause is known. A system setting (THP) would be Tripp's to change, with the evidence.
