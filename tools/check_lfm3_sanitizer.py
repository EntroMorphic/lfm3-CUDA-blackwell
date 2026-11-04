"""Fail-closed sanitizer preflight: never credit a library tracking failure as clean."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sanitizer", "probe", "binary", "checkpoint", "request", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--exclude-nvjet-synccheck", action="store_true")
    args = p.parse_args()
    if not 0 < args.timeout <= 3600:
        p.error("timeout must be in (0,3600]")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib/python3.12/site-packages/nvidia/cu13/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    records = []

    def run(name, tool, command, extra=()):
        log = args.output / (name + ".log")
        cmd = [str(args.sanitizer), "--tool", tool, "--error-exitcode", "86", "--print-limit", "4",
               "--log-file", str(log), *extra, *map(str, command)]
        timed_out = False
        with (args.output / (name + ".stdout.log")).open("w") as stream:
            try:
                process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        text = log.read_text() if log.exists() else ""
        clean = code == 0 and not timed_out and "Internal Sanitizer Error" not in text and (
            "ERROR SUMMARY: 0 errors" in text or "0 errors, 0 warnings" in text)
        record = dict(name=name, tool=tool, command=cmd, returncode=code, timed_out=timed_out, clean=clean)
        records.append(record)
        return record

    native = [args.binary, "--experimental", args.checkpoint, args.request]
    for tool in ("memcheck", "racecheck", "synccheck"):
        extra = ["--kernel-name-exclude", "kns=nvjet"] if tool == "synccheck" and args.exclude_nvjet_synccheck else []
        run(tool, tool, [*native, args.output / (tool + ".json")], extra)
    control = run("initcheck-kernel-control", "initcheck", [args.probe, "kernel", "257", "2048"])
    missing = run("initcheck-missing-control", "initcheck", [args.probe, "missing", "257", "2048"])
    producer = run("initcheck-cublas-producer", "initcheck", [args.probe, "gemm", "257", "2048"])
    missing_path = args.output / "initcheck-missing-control.log"
    missing_log = missing_path.read_text() if missing_path.exists() else ""
    controls_ok = control["clean"] and missing["returncode"] == 86 and "Uninitialized __global__ memory read" in missing_log
    if controls_ok and producer["clean"]:
        init = run("initcheck", "initcheck", [*native, args.output / "initcheck.json"])
        status = "passed" if init["clean"] else "failed"
    else:
        producer_log = args.output / "initcheck-cublas-producer.log"
        text = producer_log.read_text() if producer_log.exists() else ""
        stdout = (args.output / "initcheck-cublas-producer.stdout.log").read_text()
        reproduced = producer["returncode"] == 86 and "PASS: 128 sampled outputs" in stdout and "Uninitialized __global__ memory read" in text
        status = "blocked-cuBLAS-output-tracking" if controls_ok and reproduced else "failed-preflight-controls"
    passed = all(r["clean"] for r in records[:3]) and status == "passed"
    report = dict(passed=passed, initcheck_status=status, controls_ok=controls_ok,
                  synccheck_nvjet_excluded=args.exclude_nvjet_synccheck, records=records,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0 if passed else 86


if __name__ == "__main__":
    raise SystemExit(main())
