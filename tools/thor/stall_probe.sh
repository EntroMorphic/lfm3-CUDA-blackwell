#!/usr/bin/env bash
# The intermittent 3-4 s prefill stall (found 2026-11-16 by image-cache-v1 rule 0 and a scan of earlier runs: 86
# requests in 12 runs, TTS prefills included): lfm3-speech BUILD on speech-normalize-v1's 630 dev TTS requests (38
# stalls in their recorded run), while a sampler records every 100 ms the kernel's reclaim, compaction and fault
# counters (/proc/vmstat) and the engine process's own faults (/proc/PID/stat). tools/stall_probe_analyze.py then lines
# each request's prefill (end: its output file's mtime, less its generate time) up against the counters.
# Usage (Thor host): stall_probe.sh BUILD OUT_TAG
set -uo pipefail
T=${1:?BUILD}; O=${2:?OUT_TAG}
W=${LFM3_WORKSPACE:?set LFM3_WORKSPACE}; B=$W/liquid-CUDA/build; C=/workspace/liquid-CUDA/build; M=/workspace/models/LFM3-Omni-2B
X="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
mkdir -p $B/$O/out && cd $B/$O
sed "s#$C/speech-normalize-v1/dev/tts/\([^ ]*\)\$#$C/$O/out/\1#" $B/speech-normalize-v1/dev/tts/list.txt > list.txt
head -1 list.txt
cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag > thp.txt
free -m > free-start.txt
python3 - > sampler.jsonl <<'PY' &
import json, os, subprocess, time
KEYS = ("compact_stall", "compact_fail", "compact_success", "allocstall_normal", "allocstall_movable", "pgmajfault",
        "pgsteal_direct", "pgscan_direct", "pgsteal_kswapd", "thp_fault_alloc", "thp_fault_fallback", "pswpin", "pswpout")
pid = None
end = time.time() + 6 * 3600
while time.time() < end:
    if pid is None:
        r = subprocess.run(["pgrep", "-f", "^/workspace/liquid-CUDA/build/.*/lfm3-speech .*stall"], capture_output=True, text=True)
        pid = int(r.stdout.split()[0]) if r.stdout.split() else None
    v = {}
    for line in open("/proc/vmstat"):
        k, n = line.split()
        if k in KEYS: v[k] = int(n)
    p = None
    if pid:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            p = dict(minflt=int(f[7]), majflt=int(f[9]), utime=int(f[11]), stime=int(f[12]), rss_pages=int(f[21]))
        except OSError:
            if os.path.exists("done.flag"): break
            pid = None
    print(json.dumps(dict(t=time.time(), vm=v, proc=p)), flush=True)
    time.sleep(0.1)
PY
SP=$!
echo "== run $(date +%T)"
$X $C/$T/lfm3-speech $M --batch $C/$O/list.txt > run.log 2>&1; echo "exit $?"
touch done.flag; sleep 1; kill -TERM $SP 2>/dev/null
free -m > free-end.txt
python3 $W/liquid-CUDA/tools/stall_probe_analyze.py $B/$O | tee analysis.txt
