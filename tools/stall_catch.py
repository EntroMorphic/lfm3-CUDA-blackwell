#!/usr/bin/env python3
"""validation/prefill-stall-v1: catch the intermittent 3-4 s stall in the act. Runs lfm3-speech under gdb (Thor's
ptrace_scope 1: only a parent may trace) on a request list; a monitor thread samples every engine thread's minor faults
from /proc every 50 ms, and when one thread faults faster than RATE per second it records the thread's name and
signals the process (SIGUSR2): gdb stops it, prints every thread's backtrace, and continues. At most CATCHES catches,
10 s apart. Usage (Thor host): stall_catch.py LFM3_SPEECH MODEL_DIR LIST OUT_DIR [CATCHES] [RATE]
LD_LIBRARY_PATH as for nsys (the container's pinned CUDA libraries)."""
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

binary, model, lst, out = sys.argv[1:5]
catches = int(sys.argv[5]) if len(sys.argv) > 5 else 5
rate = float(sys.argv[6]) if len(sys.argv) > 6 else 2e6
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
cmds = out / "cmds.gdb"
cmds.write_text("\n".join(
    ["set pagination off", "set print thread-events off", "handle SIGUSR2 stop print nopass",
     "handle SIGPIPE nostop noprint pass", "run"]
    + [f"echo \\n=== CATCH {i} ===\\n\nthread apply all bt 30\ncontinue" for i in range(catches)]
    + ["echo \\n=== END ===\\n"]) + "\n")
gdb = subprocess.Popen(["gdb", "-q", "-batch", "-x", str(cmds), "--args", binary, model, "--batch", lst],
                       stdout=open(out / "gdb.txt", "w"), stderr=subprocess.STDOUT)
log = open(out / "monitor.txt", "w")


def inferior():
    for _ in range(600):
        r = subprocess.run(["pgrep", "-P", str(gdb.pid)], capture_output=True, text=True)
        if r.stdout.split():
            return int(r.stdout.split()[0])
        time.sleep(0.5)
    return None


def threads(pid):
    out = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            s = open(f"/proc/{pid}/task/{tid}/stat").read()
        except OSError:
            continue
        name = s[s.index("(") + 1:s.rindex(")")]
        f = s.rsplit(")", 1)[1].split()
        out[int(tid)] = (name, int(f[7]), int(f[11]), int(f[12]))  # minflt, utime, stime
    return out


pid = inferior()
print(f"inferior {pid}", file=log, flush=True)
caught, last_catch, prev, prev_t = 0, 0.0, {}, time.time()
while pid and gdb.poll() is None:
    try:
        now_threads, now = threads(pid), time.time()
    except FileNotFoundError:
        break
    dt = now - prev_t
    for tid, (name, minflt, ut, st) in now_threads.items():
        if tid in prev and dt > 0:
            r = (minflt - prev[tid][1]) / dt
            if r > rate / 4:
                print(f"{now:.3f} tid {tid} {name!r} minflt/s {r:.0f} stime+{st - prev[tid][3]}", file=log, flush=True)
            if r > rate and caught < catches and now - last_catch > 10:
                caught, last_catch = caught + 1, now
                print(f"{now:.3f} CATCH {caught}: tid {tid} {name!r}", file=log, flush=True)
                os.kill(pid, signal.SIGUSR2)
    prev, prev_t = now_threads, now
    time.sleep(0.05)
gdb.wait()
print(f"done, {caught} catches", file=log, flush=True)
print(f"stall_catch: {caught} catches; gdb exit {gdb.returncode}")
