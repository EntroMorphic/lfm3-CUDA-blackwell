#!/usr/bin/env python3
"""Distance of BF16 implementations from an anchor forward pass, per traced activation and per logit step.

Each candidate is NAME=ROOT/SUBDIR@IDS_ROOT: its traces are ROOT/CASE/SUBDIR/*.bf16, and the tokens it was
fed are IDS_ROOT/CASE/reference.json "generated_ids" (teacher-forced runs use the reference that made their
request). The anchor is ANCHOR_ROOT/CASE/reference with its own reference.json. A step k is compared only
while tokens 0..k-1 agree between anchor and candidate. Per file: relative RMS and max |error|. Per logit
step: KL(anchor || candidate) of the softmax over all logits, in nats, using each side's exact STEP.logits.f32
when present (its actual output), else its BF16 trace. Host numpy only.

Usage: compare_trace_spread.py ANCHOR_ROOT OUT.json --cases C1,C2 --candidate NAME=ROOT/SUBDIR@IDS_ROOT ...
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    raw = np.fromfile(path, np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64)


def distance(reference, other):
    error = other - reference
    scale = max(float(np.sqrt(np.mean(reference**2))), 1e-8)
    return float(np.sqrt(np.mean(error**2)) / scale), float(np.max(np.abs(error)))


def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def kl(anchor, other):
    p, q = log_softmax(anchor), log_softmax(other)
    return float(np.sum(np.exp(p) * (p - q)))


def load_exact_logits(bf16_path):
    """A logits trace's exact FP32 sidecar when it exists (the run's actual output), else its BF16 file."""
    exact = bf16_path.with_name(bf16_path.name[: -len(".bf16")] + ".f32")
    return np.fromfile(exact, np.float32).astype(np.float64) if exact.exists() else load(bf16_path)


def ids(root, case):
    return json.loads((root / case / "reference.json").read_text())["generated_ids"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("anchor", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--candidate", action="append", required=True)
    a = ap.parse_args()
    candidates = []
    for spec in a.candidate:
        name, rest = spec.split("=", 1)
        path, ids_root = rest.split("@", 1)
        root, subdir = Path(path).parent, Path(path).name
        candidates.append((name, root, subdir, Path(ids_root)))
    report = dict(definition=__doc__.split("\n\n")[0], anchor=str(a.anchor), cases=[])
    for case in a.cases.split(","):
        anchor_ids = ids(a.anchor, case)
        row = dict(case=case, candidates={})
        for name, root, subdir, ids_root in candidates:
            fed = ids(ids_root, case)
            agree = 0
            while agree < min(len(anchor_ids), len(fed)) and anchor_ids[agree] == fed[agree]:
                agree += 1
            rel, kls, logit_abs, files = [], [], [], 0
            for ref in sorted((a.anchor / case / "reference").glob("*.bf16")):
                step = int(ref.name.split(".")[0])
                other = root / case / subdir / ref.name
                if step > agree or not other.exists():
                    continue
                logit = ".logits." in ref.name
                r, o = (load_exact_logits(ref), load_exact_logits(other)) if logit else (load(ref), load(other))
                if r.shape != o.shape:
                    continue
                files += 1
                d, m = distance(r, o)
                if logit:
                    kls.append(kl(r, o))
                    logit_abs.append(m)
                else:
                    rel.append(d)
            row["candidates"][name] = dict(
                tokens_agree=agree, of_tokens=len(anchor_ids), files=files,
                activation_rel_rms_median=float(np.median(rel)) if rel else None,
                activation_rel_rms_max=max(rel) if rel else None,
                logit_steps=len(kls), logit_kl_mean=float(np.mean(kls)) if kls else None,
                logit_kl_max=max(kls) if kls else None, logit_max_abs=max(logit_abs) if logit_abs else None)
        report["cases"].append(row)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        for name, c in row["candidates"].items():
            f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
            print(f"{row['case']:<11} {name:<7} tokens {c['tokens_agree']}/{c['of_tokens']} files {c['files']:>4}"
                  f"  act rel_rms median {f(c['activation_rel_rms_median'], '.4f')} max {f(c['activation_rel_rms_max'], '.4f')}"
                  f"  logits KL mean {f(c['logit_kl_mean'], '.2e')} max {f(c['logit_kl_max'], '.2e')} ({c['logit_steps']} steps)"
                  f"  max_abs {f(c['logit_max_abs'], '.3f')}")


if __name__ == "__main__":
    main()
