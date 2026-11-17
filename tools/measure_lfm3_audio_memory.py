#!/usr/bin/env python3
"""validation/audio-frontend-v1 rule 6's memory bound, as amendment 3 sets it: liquid-lfm3-server (8 speech rows,
--queue-size 40 --handlers 48) on a quiet Thor. Idle RSS: VmRSS 5 s after one warm request with the 90-second clip.
Peak: the most VmRSS, sampled every 50 ms from the first of 16 simultaneous requests to the last reply. The 16 clips
are the 90-second clip with its last sample set to 16 different values, so each is a distinct clip with its own
cache entry and key (amendment 3, as corrected after the red-team: one shared clip would have shared one entry).
Pass: peak - idle <= 1 GiB, and all 16 replies 200; how many replies carry the warm request's text is reported.
The server runs in this process's namespace, so its /proc is read directly.
Usage: measure_lfm3_audio_memory.py SERVER MODEL_DIR WAV OUT_JSON
"""
import base64
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_lfm3_audio_stall import RowsServer  # noqa: E402

N, GIB = 16, 1 << 30


def rss(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no VmRSS")


def main(server_bin, model, wav, out_json):
    clip = Path(wav).read_bytes()
    assert clip[:4] == b"RIFF" and clip[34:36] == (16).to_bytes(2, "little"), "a 16-bit WAV"

    def body(i=None):
        data = clip if i is None else clip[:-2] + int(i).to_bytes(2, "little", signed=True)  # the last sample: i
        return dict(messages=[{"role": "system", "content": "Perform ASR in English."},
                              {"role": "user", "content": [{"type": "audio", "audio": base64.b64encode(data).decode()}]}],
                    mode="text", max_new_tokens=2048)
    out = dict(n=N, wav=wav)
    with tempfile.TemporaryDirectory() as scratch:
        s = RowsServer(server_bin, model, scratch, ["--speech-rows", "8", "--queue-size", "40", "--handlers", "48"])
        try:
            status, warm = s.post(body())
            if status != 200:
                raise SystemExit(f"the warm request failed: {status} {warm}")
            time.sleep(5)
            idle = rss(s.proc.pid)
            bodies = [body(i) for i in range(N)]
            results = [None] * N
            done = threading.Event()
            peak = [idle]

            def sample():
                while not done.is_set():
                    peak[0] = max(peak[0], rss(s.proc.pid))
                    time.sleep(0.05)
            sampler = threading.Thread(target=sample)
            sampler.start()

            def run(i):
                results[i] = s.post(bodies[i], timeout=900)
            threads = [threading.Thread(target=run, args=(i,)) for i in range(N)]
            t0 = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            done.set()
            sampler.join()
            out.update(idle_rss=idle, peak_rss=peak[0], growth=peak[0] - idle, wall_seconds=time.monotonic() - t0,
                       statuses=[r[0] for r in results], served=sum(1 for r in results if r[0] == 200),
                       refusals=[r[1] for r in results if r[0] != 200][:4],
                       same_text_as_warm=sum(1 for r in results if r[0] == 200 and r[1].get("text") == warm.get("text")))
        finally:
            code, err = s.stop()
            out["server_exit"], out["server_stderr"] = code, err[-2000:]
    out["pass"] = out.get("growth", GIB + 1) <= GIB and out.get("served") == N and out["server_exit"] == 0
    Path(out_json).write_text(json.dumps(out, indent=1) + "\n")
    print(f"idle {out.get('idle_rss', 0) / 2**20:.0f} MiB, peak {out.get('peak_rss', 0) / 2**20:.0f} MiB, growth "
          f"{out.get('growth', 0) / 2**20:.0f} MiB (bound 1024); {out.get('served')} of {N} served, "
          f"{out.get('same_text_as_warm')} with the warm reply's text; server exit "
          f"{out['server_exit']}", "| PASS" if out["pass"] else "| FAIL")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
