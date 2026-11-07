#!/usr/bin/env python3
"""Summarize a decode-rate run (tools/thor/g4_native_evidence.sh): prefill seconds and teacher-forced decode rate.

Each result is liquid-lfm3's output for req-L.json: L prompt tokens from ppl-v5/windows-32768.i32, the next 128
tokens teacher-forced, one warm-up run in process, three repetitions per build alternating default/native. Binary hashes are in
DIR/binaries.sha256. Usage: summarize.py DIR > summary.json
"""
import glob
import hashlib
import json
import statistics as st
import sys

d = sys.argv[1]
out = dict(definition=__doc__.split("\n\n")[0], lengths=[], files={})
for length in (86, 1024, 4096, 32000):
    row = dict(prompt_tokens=length, forced_decode_steps=128)
    argmax = {}
    for build in ("default", "native"):
        results = []
        for f in sorted(glob.glob(f"{d}/{build}-{length}-*.json")):
            raw = open(f, "rb").read()
            out["files"][f.split("/")[-1]] = hashlib.sha256(raw).hexdigest()
            results.append(json.loads(raw))
        argmax[build] = results[0]["teacher_argmax"]
        if any(r["teacher_argmax"] != argmax[build] for r in results):
            raise SystemExit(f"{build} {length}: repetitions disagree")
        row[build] = dict(repetitions=len(results), engine=dict(attention=results[0]["attention"], gemm=results[0].get("gemm", "cublas")),
                          prefill_seconds_median=st.median(r["prefill_seconds"] for r in results),
                          decode_tokens_per_second_median=st.median(r["decode_forward_calls"] / r["decode_seconds"] for r in results))
    row["teacher_argmax_agreement"] = [sum(a == b for a, b in zip(argmax["default"], argmax["native"])), len(argmax["default"])]
    out["lengths"].append(row)
print(json.dumps(out, indent=2))
