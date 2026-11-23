#!/usr/bin/env python3
"""Summarise a tcgen05 probe correctness run (validation/tcgen05-probe-v1): rules C1-C5 per data kind.

usage: tcgen05_probe_summary.py correct.jsonl [more.jsonl ...]
"""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    cases, fatal, summary = [], None, None
    for line in open(path):
        d = json.loads(line)
        if "fatal" in d:
            fatal = d
        elif "summary" in d:
            summary = d
        else:
            cases.append(d)
    by = defaultdict(lambda: defaultdict(int))
    worst = defaultdict(float)
    for d in cases:
        key = (d["data"], d["out"])
        b = by[key]
        b["cases"] += 1
        b["pass"] += d["pass"]["all"]
        b["c12"] += d["pass"]["c12"]
        b["c3"] += d["pass"]["c3"]
        b["c4"] += d["pass"]["c4"]
        c2 = d.get("c2")
        if c2:
            b["bitwise_identical_to_gemm"] += c2.get("identical_to_gemm", 0) == 1.0
            b["bound_violations"] += c2["bound_violations"]
            b["ref_cpu_mismatches"] += c2["ref_cpu_mismatches"]
            if "max_ratio" in c2:
                worst[key + ("max_ratio",)] = max(worst[key + ("max_ratio",)], c2["max_ratio"])
                worst[key + ("mean_ratio",)] = max(worst[key + ("mean_ratio",)], c2["mean_ratio"])
            else:
                b["beyond_1ulp_vs_gemm"] += c2["beyond_1ulp_vs_gemm"]
                b["not_rne_of_probe_f32"] += c2["not_rne_of_probe_f32"]
        else:
            c1 = d["c1"]
            b["bitwise_identical_to_gemm"] += c1["mismatches_vs_gemm"] == 0
            b["premise_mismatches"] += c1["premise_mismatches"]
    variant = cases[0]["variant"] if cases else "?"
    total = sum(b["cases"] for b in by.values())
    passed = sum(b["pass"] for b in by.values())
    print(f"{path}: variant {variant}, {passed}/{total} cases pass all of C1-C5"
          + (f"; FATAL {fatal}" if fatal else "") + (f"; summary {summary}" if summary else "; NO SUMMARY LINE"))
    for key in sorted(by):
        b = dict(by[key])
        extra = {k[2]: round(v, 4) for k, v in worst.items() if k[:2] == key}
        print(f"  {key[0]:6s} {key[1]:4s} {b} {('worst ' + str(extra)) if extra else ''}")
