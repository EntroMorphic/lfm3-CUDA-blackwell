#!/usr/bin/env python3
"""Summarize a tegrastats log (non-root: no memory clock or GPU load) for a speed round's record: per rail the median
and maximum power (mW), per sensor the maximum temperature (C), the CPU cores' median and maximum load (%) and the
frequencies seen. One JSON object on stdout. Usage: tegrastats_summary.py tegrastats.log"""
import json, re, statistics, sys

rails, temps, loads, freqs, n = {}, {}, [], set(), 0
for line in open(sys.argv[1], errors="replace"):
    if "RAM" not in line:
        continue
    n += 1
    for name, now in re.findall(r"(VDD_\w+|VIN\w*) (\d+)mW/\d+mW", line):
        rails.setdefault(name, []).append(int(now))
    for name, t in re.findall(r"(\w+)@(-?[\d.]+)C", line):
        temps.setdefault(name, []).append(float(t))
    m = re.search(r"CPU \[([^\]]*)\]", line)
    if m:
        cores = [c for c in m.group(1).split(",") if "@" in c]
        load = [int(c.split("%")[0]) for c in cores]
        if load:
            loads.append(statistics.mean(load))
        freqs.update(int(c.split("@")[1]) for c in cores)
print(json.dumps(dict(
    samples=n,
    rails_mw={k: dict(median=statistics.median(v), max=max(v)) for k, v in rails.items()},
    temps_max_c={k: max(v) for k, v in temps.items()},
    cpu_load_pct=dict(median=statistics.median(loads), max=max(loads)) if loads else None,
    cpu_mhz_seen=sorted(freqs)), indent=1))
