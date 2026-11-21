#!/usr/bin/env python3
"""A diagnostic, not a rule check (validation/audio-ttft-v1): liquid-lfm3-server under nsys (CUDA API and kernels),
sent the same audio-in ASR request (tools/measure_lfm3_audio_ttft.py's) REPEATS times one after another, so the CUDA
calls each request makes (cudaMalloc and cudaFree above all) can be read per request from the report.
Each request's wall span (first byte to done) is printed with a monotonic and a wall-clock stamp to match the report.
nsys collects for a fixed --duration (DURATION seconds from launch), then ends the server with SIGTERM and writes the
report; stopping nsys early with SIGINT instead left the CUDA trace unflushed (1 event collected).
Timing figures from a profiled server are not comparable to unprofiled ones.
Usage: profile_lfm3_audio_server.py NSYS SERVER MODEL_DIR WAV REPEATS DURATION OUT_PREFIX
"""
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_lfm3_audio_ttft import one  # noqa: E402


class Profiled:
    def __init__(self, nsys, binary, model, scratch, out, duration):
        self.token = secrets.token_urlsafe(32)
        token_path = Path(scratch) / "profile.token"
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(self.token + "\n")
        self.proc = subprocess.Popen(
            [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--kill=sigterm", f"--duration={duration}",
             "--force-overwrite=true", "-o", out, binary, "--experimental", model, "--token-file", str(token_path),
             "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.port = None
        for line in self.proc.stdout:  # nsys writes its own lines too; the server's first JSON line has the port
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.port = json.loads(line)["port"]
                    break
                except (ValueError, KeyError):
                    pass
        if self.port is None:
            raise SystemExit(f"server did not start: {self.proc.stderr.read()[-800:]}")
        token_path.unlink()

    def stop(self):
        out, err = self.proc.communicate(timeout=900)  # nsys ends the server when the duration is up
        return self.proc.returncode, out[-1500:], err[-1500:]


def main(nsys, binary, model, wav, repeats, duration, out):
    with tempfile.TemporaryDirectory() as scratch:
        s = Profiled(nsys, binary, model, scratch, out, duration)
        try:
            for i in range(int(repeats)):
                t0, w0 = time.monotonic(), time.time()
                r = one(s, wav)
                print(json.dumps(dict(request=i, ok=r["ok"], wall_start=w0, seconds=time.monotonic() - t0,
                                      from_file_seconds=r["from_file_seconds"], prefill_seconds=r["prefill_seconds"],
                                      text=r["text"][:60])), flush=True)
                time.sleep(1.0)  # a gap between requests, to separate them in the report
        finally:
            code, o, e = s.stop()
    print(f"nsys exit {code}\n{o}\n{e}")


if __name__ == "__main__":
    if len(sys.argv) != 8:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
