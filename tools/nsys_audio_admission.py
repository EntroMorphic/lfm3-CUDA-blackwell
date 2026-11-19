#!/usr/bin/env python3
"""validation/audio-handles-v1 rule 3 (reported): from an nsys trace (its SQLite export) of liquid-lfm3-server admitting
an audio clip, the engine thread's admission timeline. For each launch of the audio encoder's first kernel (`conv_in`):
- the engine thread: the thread that launched it;
- the CPU-only time before the encoder: the longest interval without a CUDA runtime call on that thread in the 200 ms
  before the launch (audio-frontend-v1's window 16 found 35.4 ms there: parsing the request line and decoding the
  clip's base64 and WAV);
- the encoder's GPU span (conv_in's start to to_bf16's end) and the decode-step interval before the admission.
Usage: nsys_audio_admission.py TRACE.sqlite OUT.json
"""
import json
import sqlite3
import statistics
import sys


def main(db, out_json):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id,value from StringIds"))
    ids = lambda pred: [i for i, v in S.items() if pred(v)]  # noqa: E731
    conv = ids(lambda v: v == "conv_in" or v.startswith("conv_in(") or v.endswith("::conv_in"))
    tobf = ids(lambda v: v == "to_bf16" or v.startswith("to_bf16(") or v.endswith("::to_bf16"))
    sync = ids(lambda v: v.startswith("cudaDeviceSynchronize"))
    q = f"select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where shortName in ({','.join(map(str, conv or [-1]))}) order by start"
    admissions = []
    for kstart, kend, corr in c.execute(q).fetchall():
        tid, launch = c.execute("select globalTid,start from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=?", (corr,)).fetchone()
        calls = c.execute("select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start between ? and ? "
                          "order by start", (tid, launch - 200_000_000, launch)).fetchall()
        gaps = [(b[0] - a[1], a[1], b[0]) for a, b in zip(calls, calls[1:])]
        gap, gap_from, gap_to = max(gaps, default=(0, launch, launch))
        end = c.execute(f"select min(end) from CUPTI_ACTIVITY_KIND_KERNEL where shortName in ({','.join(map(str, tobf or [-1]))}) "
                        "and start > ?", (kstart,)).fetchone()[0]
        syncs = [s for (s,) in c.execute(f"select start from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId in "
                                         f"({','.join(map(str, sync or [-1]))}) and start between ? and ? order by start",
                                         (tid, launch - 2_000_000_000, gap_from))]
        steps = [(b - a) / 1e6 for a, b in zip(syncs, syncs[1:])]
        admissions.append(dict(
            cpu_only_before_encoder_ms=gap / 1e6, cpu_only_ends_before_launch_ms=(launch - gap_to) / 1e6,
            encoder_gpu_ms=(end - kstart) / 1e6 if end else None,
            step_interval_before_ms=statistics.median(steps) if steps else None))
    out = dict(trace=db, admissions=admissions)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    for a in admissions:
        print(f"CPU-only before the encoder {a['cpu_only_before_encoder_ms']:.1f} ms; encoder (GPU) "
              f"{a['encoder_gpu_ms'] or 0:.1f} ms; decode step before {a['step_interval_before_ms'] or 0:.1f} ms")
    if not admissions:
        print("no conv_in launch in the trace")
    return 0 if admissions else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
