#!/usr/bin/env python3
"""Rules S1-S3 of validation/tcgen05-probe-v1 from three speed rounds (tools/thor/tcgen05_probe_speed.sh).

usage: tcgen05_probe_speed_summary.py ROUND_DIR ROUND_DIR ROUND_DIR [--variants 1,2,3]
--variants: the variants that passed every correctness rule and the sanitizers (only these may answer S1).

S1: on the 39 cases with M in {4096, 7920, 8192}, the fastest passing variant (highest geometric-mean TFLOPS over
    those cases, from the per-case median of the rounds) is above 56.7 TFLOPS on every case in at least 2 of 3
    rounds, and its geometric-mean speedup over lfm::gemm (per-case medians of the rounds) is at least 1.5x.
S2: its geometric mean against cuBLAS on the same cases (>= 0.9 is "parity").
S3: below M = 4096, its ratio to lfm::gemm per case, and the smallest M at which it is faster on every shape
    (in at least 2 of 3 rounds).
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CEILING = 56.7
BIG = {4096, 7920, 8192}

args = sys.argv[1:]
variants = ["1", "2", "3"]
if "--variants" in args:
    i = args.index("--variants")
    variants = args[i + 1].split(",")
    del args[i:i + 2]
rounds = [Path(a) for a in args]
if len(rounds) != 3:
    sys.exit("three round directories required")

data = []  # per round: {(shape, m): record}
for r in rounds:
    recs = {}
    for line in open(r / "speed.jsonl"):
        d = json.loads(line)
        if "shape" in d:
            recs[(d["shape"], d["m"])] = d
    data.append(recs)
    peak = [json.loads(l) for l in open(r / "peak.jsonl") if l.startswith("{")]
    print(f"{r.name}: {len(recs)} cases; peak " + ", ".join(f"{p['peak']} {p['tflops']} TFLOPS" for p in peak if "peak" in p))
cases = sorted(set.intersection(*(set(d) for d in data)), key=lambda c: (c[0], c[1]))

def med(case, impl, key="tflops"):
    return statistics.median(d[case][impl][key] for d in data)

def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

big = [c for c in cases if c[1] in BIG]
print(f"\nS1/S2 cases: {len(big)}")
scores = {v: geo([med(c, "V" + v) for c in big]) for v in variants}
for v, s in scores.items():
    print(f"  V{v}: geometric-mean TFLOPS {s:.1f}")
best = max(scores, key=scores.get)
bv = "V" + best
above = [c for c in big if sum(d[c][bv]["tflops"] > CEILING for d in data) >= 2]
speedup = geo([med(c, "lfm_gemm", "ms") / med(c, bv, "ms") for c in big])
vs_cublas = geo([med(c, "cublas", "ms") / med(c, bv, "ms") for c in big])
s1 = len(above) == len(big) and speedup >= 1.5
print(f"S1: fastest passing variant {bv}: above {CEILING} TFLOPS on {len(above)}/{len(big)} cases (2 of 3 rounds); "
      f"geometric-mean speedup over lfm::gemm {speedup:.2f}x -> {'YES' if s1 else 'NO'}")
for c in big:
    if c not in above:
        print(f"    below the ceiling: {c} {[d[c][bv]['tflops'] for d in data]}")
print(f"S2: {bv} against cuBLAS, geometric mean {vs_cublas:.2f}x ({'parity' if vs_cublas >= 0.9 else 'below parity'})")

print(f"\nS3 ({bv} time ratio lfm::gemm / {bv}, median of rounds; >1 = {bv} faster):")
by_m = defaultdict(dict)
for c in cases:
    by_m[c[1]][c[0]] = c
shapes = sorted({c[0] for c in cases})
print("  " + "M".rjust(6) + "".join(s.rjust(13) for s in shapes))
crossover = None
for m in sorted(by_m):
    row, all_faster = [], True
    for s in shapes:
        c = by_m[m].get(s)
        if not c:
            row.append("-".rjust(13)); continue
        row.append(f"{med(c, 'lfm_gemm', 'ms') / med(c, bv, 'ms'):13.2f}")
        all_faster &= sum(d[c]["lfm_gemm"]["ms"] > d[c][bv]["ms"] for d in data) >= 2
    print("  " + str(m).rjust(6) + "".join(row))
    if all_faster and crossover is None:
        crossover = m
    if not all_faster:
        crossover = None
print(f"  smallest M from which {bv} is faster on every shape at every larger M: {crossover}")

print("\nper case, median of rounds (TFLOPS):")
print(f"  {'shape':14s} {'M':>5s} " + " ".join(f"{('V' + v):>7s}" for v in variants) + f" {'lfm':>7s} {'cuBLAS':>7s}")
for c in cases:
    print(f"  {c[0]:14s} {c[1]:5d} " + " ".join(f"{med(c, 'V' + v):7.1f}" for v in variants)
          + f" {med(c, 'lfm_gemm'):7.1f} {med(c, 'cublas'):7.1f}")
