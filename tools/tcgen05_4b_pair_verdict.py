#!/usr/bin/env python3
"""validation/tcgen05-gemm-v1 rule 4b under amendment 2: the verdict of an ON/OFF soak pair.

usage: tcgen05_4b_pair_verdict.py PAIR_DIR      (PAIR_DIR holds on/ and off/, each with soak-4b.json and
                                                 soak-4b.readings.jsonl from tools/check_lfm3_tcgen05_soak.py)
Gated, per run: the harness's non-memory clauses (no differ, no errors, every status 200, alive, crowd identical, exit
0); the server RSS idle growth (memory_idle.server_rss, first to last checkpoint) <= 64 MiB; the container cgroup's
anon growth from checkpoint 2 to checkpoint 4 (medians of each checkpoint's 10 idle readings) <= 64 MiB.
Gated, across the pair: for each of the two memory figures, ON minus OFF < 32 MiB.
Reported, not gated: host_used_less_client (the harness's memory verdict), the client's RSS, and the 429 retries per
completed reply (flagged for investigation when ON's exceeds OFF's by more than 25%).
Exit 0 when rule 4b passes, 1 otherwise.
"""
import json
import statistics
import sys
from pathlib import Path

MIB = 2 ** 20


def checkpoints(readings_path):
    """Main-thread readings (each checkpoint's 10 idle readings), grouped by gaps of more than 5 s."""
    rows = [json.loads(l) for l in open(readings_path)]
    main = [r for r in rows if r["thread"] == "MainThread"]
    groups, cur = [], []
    for r in main:
        if cur and r["wall"] - cur[-1]["wall"] > 5:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    return groups


def run_figures(d):
    rep = json.loads((d / "soak-4b.json").read_text())
    groups = checkpoints(d / "soak-4b.readings.jsonl")
    anon = [statistics.median((r.get("cgroup_stat") or {}).get("anon", 0) for r in g) / MIB for g in groups]
    statuses = rep.get("statuses", {})
    replies = sum(v for k, v in statuses.items() if k == "200")
    non_memory = dict(
        no_differ=rep.get("differ_count", len(rep.get("differ", []))) == 0,
        no_errors=rep.get("error_count", len(rep.get("errors", []))) == 0,
        all_200=set(statuses) <= {"200"},
        alive=bool(rep.get("alive_throughout")),
        crowd_identical=all(not c.get("differ") for c in rep.get("crowd_after", {}).values()),
        exit_0=rep.get("server_exit") == 0)
    idle = rep["memory_idle"]
    return dict(
        non_memory=non_memory,
        server_rss_growth=idle["server_rss"]["growth_mib"],
        cgroup_anon_checkpoints=[round(a, 1) for a in anon],
        cgroup_anon_growth_2_4=round(anon[3] - anon[1], 1) if len(anon) >= 4 else None,
        reported_host_used_less_client_growth=idle["host_used_less_client"]["growth_mib"],
        reported_harness_pass=rep["pass_"],
        replies=replies,
        retries_429=rep["counts"]["retries_429"],
        retries_per_reply=round(rep["counts"]["retries_429"] / replies, 3) if replies else None)


def main(pair_dir):
    p = Path(pair_dir)
    runs = {side: run_figures(p / side) for side in ("on", "off")}
    ok = True
    for side, f in runs.items():
        gated = dict(f["non_memory"], server_rss_growth_le_64=f["server_rss_growth"] is not None and f["server_rss_growth"] <= 64,
                     cgroup_anon_growth_2_4_le_64=f["cgroup_anon_growth_2_4"] is not None and f["cgroup_anon_growth_2_4"] <= 64)
        f["gated"] = gated
        f["pass"] = all(gated.values())
        ok &= f["pass"]
    eq = {}
    for k in ("server_rss_growth", "cgroup_anon_growth_2_4"):
        on, off = runs["on"][k], runs["off"][k]
        eq[k] = dict(on=on, off=off, on_minus_off=None if on is None or off is None else round(on - off, 1),
                     pass_=on is not None and off is not None and on - off < 32)
        ok &= eq[k]["pass_"]
    rpr_on, rpr_off = runs["on"]["retries_per_reply"], runs["off"]["retries_per_reply"]
    retries = dict(on=rpr_on, off=rpr_off, ratio=round(rpr_on / rpr_off, 3) if rpr_on and rpr_off else None)
    retries["investigate"] = retries["ratio"] is not None and retries["ratio"] > 1.25
    out = dict(runs=runs, equivalence=eq, retries_per_reply=retries, rule_4b_pass=ok)
    print(json.dumps(out, indent=1))
    print("RULE 4B (amendment 2):", "PASS" if ok else "FAIL")
    (p / "verdict.json").write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
