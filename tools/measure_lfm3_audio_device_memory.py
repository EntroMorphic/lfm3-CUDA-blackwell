#!/usr/bin/env python3
"""validation/audio-ttft-v1 amendment 4, rule 5b: the server's device memory, by image-cache-v1's stand-in (on Thor the
GPU allocates from system memory, and no /proc field shows a process's share):
  device = (MemTotal - MemAvailable) - the same before the server started - the server's (RssAnon + RssShmem).
For each server in turn (one at a time, on a quiet Thor):
  0. before: the in-use reading with no server running;
  1. start the server (no extra options; its warm-up runs), then the in-use reading: `started`;
  2. one audio-in ASR request with the short clip (tools/measure_lfm3_audio_ttft.py's), then the reading: `short`;
  3. two requests with the 90 s clip (the first warm, as audio-frontend-v1's idle reading), then the reading: `after_90`;
  4. (validation/encoder-wrap-v1) with LFM3_MEMORY_IMAGE set to an image request (images-v1's stage-0 form), that
     request once (text mode, 16 new tokens; the vision encoder's first encode), then the reading: `after_image`.
Each reading: 3 s idle, then the median of 5 samples 0.5 s apart. Requests must return the done event.
Gate (rule 5b): device(after_90) of the round's server - that of the deployed server <= 1 GiB. The `short` difference
(about the doubled weights, built on the first clip, and a small scratch) is reported.
Usage: measure_lfm3_audio_device_memory.py SHORT_WAV LONG_WAV MODEL_DIR OUT_JSON NAME=SERVER...
"""
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lfm3_speech_stream import Server  # noqa: E402
from measure_lfm3_audio_ttft import one  # noqa: E402

GIB = 1 << 30


def meminfo():
    f = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        f[k] = int(v.split()[0]) * 1024
    return f


def proc_anon(pid):
    f = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("RssAnon:", "RssShmem:", "VmRSS:", "RssFile:")):
            k, v = line.split(":", 1)
            f[k] = int(v.split()[0]) * 1024
    return f


def reading(pid=None):
    time.sleep(3)
    used, anon, rss = [], [], []
    for i in range(5):
        m = meminfo()
        used.append(m["MemTotal"] - m["MemAvailable"])
        if pid:
            p = proc_anon(pid)
            anon.append(p["RssAnon"] + p["RssShmem"])
            rss.append(p["VmRSS"])
        if i < 4:
            time.sleep(0.5)
    return dict(in_use=statistics.median(used), anon_shmem=statistics.median(anon) if anon else 0,
                vmrss=statistics.median(rss) if rss else 0)


def main(short_wav, long_wav, model, out_json, *servers):
    out, failed = {}, False
    image = None
    if os.environ.get("LFM3_MEMORY_IMAGE"):
        from check_lfm3_image_server import as_client
        image = as_client(json.loads(Path(os.environ["LFM3_MEMORY_IMAGE"]).read_text()))
        image["mode"], image["max_new_tokens"] = "text", 16
    for spec in servers:
        name, binary = spec.split("=", 1)
        r = dict(before=reading())
        with tempfile.TemporaryDirectory() as scratch:
            s = Server(binary, model, scratch)
            try:
                r["started"] = reading(s.proc.pid)
                q = one(s, short_wav)
                failed |= not q["ok"]
                r["short"] = reading(s.proc.pid)
                for _ in range(2):
                    q = one(s, long_wav)
                    failed |= not q["ok"]
                r["after_90"] = reading(s.proc.pid)
                r["text_90"] = q["text"]
                if image:
                    st, events, _, _ = s.stream(image)
                    failed |= st != 200 or not any(e.get("event") == "done" for e in events)
                    r["after_image"] = reading(s.proc.pid)
                    r["text_image"] = "".join(e.get("text", "") for e in events if e.get("event") == "text")
            finally:
                code, err = s.stop()
            r["server_exit"] = code
            failed |= code != 0
        keys = ("started", "short", "after_90") + (("after_image",) if image else ())
        for k in keys:
            r[k]["device"] = r[k]["in_use"] - r["before"]["in_use"] - r[k]["anon_shmem"]
        out[name] = r
        print(name, {k: round(r[k]["device"] / 2**20) for k in keys}, "MiB device;",
              "VmRSS after 90 s", round(r["after_90"]["vmrss"] / 2**20), "MiB; exit", code, flush=True)
    names = [s.split("=", 1)[0] for s in servers]
    if len(names) == 2:
        dep, new = names
        d90 = out[new]["after_90"]["device"] - out[dep]["after_90"]["device"]
        ds = out[new]["short"]["device"] - out[dep]["short"]["device"]
        out["verdict"] = dict(after_90_difference=d90, short_difference=ds, bound=GIB, pass_=d90 <= GIB and not failed)
        print(f"RULE 5b (device, amendment 4): {new} - {dep} after the 90 s clip {d90 / 2**20:.0f} MiB (bound 1024); "
              f"after the short clip {ds / 2**20:.0f} MiB; {'PASS' if d90 <= GIB and not failed else 'FAIL'}")
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
