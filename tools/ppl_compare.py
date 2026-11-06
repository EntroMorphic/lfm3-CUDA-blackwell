"""Compare engine perplexity against the HF SDPA reference (the agreed quality bar).

Bar (decided 2026-11-06): the forward pass passes "so long as the perplexity is equal or better",
read statistically (adopted 2026-11-06): a length PASSES unless the engine is statistically worse,
i.e. unless the whole paired 95% CI of dNLL lies above EQUAL_EPS. "Equal" = CI includes zero;
"better" = CI entirely below zero. With one window (no CI) the point estimate must be <= EQUAL_EPS.
The two point-estimate verdicts below are reported for information only.
  dNLL = mean NLL(engine) - mean NLL(hf-sdpa), in nats per token.
  EQUAL_EPS = 1e-5 nats/token: "equal". Measured scorer agreement (native vs HF eager, which the
            engine matches) is <= 2e-6 per token and ~1e-9 on the mean, so EQUAL_EPS is far above
            computation noise and 20x below the SDPA-eager gap.
  equal_or_better: dNLL <= EQUAL_EPS                                   (vs the configuration you run)
  within_standard: dNLL <= gap + EQUAL_EPS, gap = |mean NLL(eager) - mean NLL(sdpa)| at that length
            (no worse than the worse of the standard stack's two configurations). Where eager is not
            measured (32K) the gap from the longest measured length is used and labeled as borrowed.
Paired 95% CI: windows are scored on identical ids by every engine; bootstrap over windows (fixed seed).

Usage: python ppl_compare.py RESULTS_DIR [--engine native]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

EQUAL_EPS = 1e-5


def load(d, engine, length):
    p = d / f"nll-{engine}-{length}.f32"
    if not p.exists():
        return None
    x = np.fromfile(p, dtype=np.float32).astype(np.float64)
    if x.size % (length - 1):
        raise ValueError(f"{p.name}: size {x.size} is not a multiple of {length - 1}")
    return x.reshape(-1, length - 1)


def bootstrap_ci(window_diffs, seed=0, resamples=10000):
    rng = np.random.default_rng(seed)
    n = window_diffs.size
    idx = rng.integers(0, n, size=(resamples, n))
    means = window_diffs[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def passes_bar(delta, ci):
    """Pass unless statistically worse: the CI lower bound must not exceed EQUAL_EPS."""
    if ci is None:
        return delta <= EQUAL_EPS
    return ci[0] <= EQUAL_EPS


def verdict(delta, ci):
    if ci is None:
        return "equal-or-better (point)" if delta <= EQUAL_EPS else "worse (point)"
    if ci[1] < -EQUAL_EPS:
        return "better"
    if ci[0] > EQUAL_EPS:
        return "worse"
    return "equal"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--engine", default="native")
    a = p.parse_args()
    summary = json.loads((a.results / "summary.json").read_text())
    lengths = sorted(int(k) for k in summary["windows"])
    rows, gaps = [], {}
    for length in lengths:
        ref, eager, eng = (load(a.results, e, length) for e in ("hf-sdpa", "hf-eager", a.engine))
        if ref is None or eng is None:
            rows.append({"length": length, "missing": True})
            continue
        if ref.shape != eng.shape:
            raise ValueError(f"length {length}: shape mismatch {ref.shape} vs {eng.shape}")
        r = {"length": length, "windows": ref.shape[0], "tokens": ref.size,
             "ppl_sdpa": math.exp(ref.mean()), f"ppl_{a.engine}": math.exp(eng.mean())}
        d = eng - ref
        r["delta_mean_nll"] = float(d.mean())
        r["delta_ci95"] = bootstrap_ci(d.mean(axis=1)) if ref.shape[0] > 1 else None
        r["max_abs_token_delta"] = float(np.abs(d).max())
        if eager is not None:
            r["ppl_eager"] = math.exp(eager.mean())
            gaps[length] = abs(float(eager.mean() - ref.mean()))
            r["tolerance"], r["tolerance_source"] = gaps[length], "measured at this length"
            r[f"{a.engine}_minus_eager_mean_nll"] = float((eng - eager).mean())
        elif gaps:
            src = max(gaps)
            r["tolerance"], r["tolerance_source"] = gaps[src], f"borrowed from length {src} (eager not measured here)"
        r["passes_bar"] = passes_bar(r["delta_mean_nll"], r["delta_ci95"])
        r["verdict"] = verdict(r["delta_mean_nll"], r["delta_ci95"])
        r["equal_or_better"] = r["delta_mean_nll"] <= EQUAL_EPS
        r["within_standard"] = r["equal_or_better"] or ("tolerance" in r and r["delta_mean_nll"] <= r["tolerance"] + EQUAL_EPS)
        rows.append(r)
    report = {"engine": a.engine, "reference": "hf-sdpa", "corpus_sha256": summary["corpus_sha256"], "rows": rows,
              "equal_eps": EQUAL_EPS, "passes_bar": all(r.get("passes_bar") for r in rows), "all_equal_or_better": all(r.get("equal_or_better") for r in rows),
              "all_within_standard": all(r.get("within_standard") for r in rows)}
    (a.results / f"compare-{a.engine}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'length':>7} {'windows':>7} {'PPL sdpa':>10} {'PPL eager':>10} {'PPL ' + a.engine:>11} {'dNLL':>11} {'95% CI':>25} {'gap':>9}  {'verdict':>8} bar   (info: eq/better within-std)")
    for r in rows:
        if r.get("missing"):
            print(f"{r['length']:>7}  (missing results)")
            continue
        ci = r["delta_ci95"]
        print(f"{r['length']:>7} {r['windows']:>7} {r['ppl_sdpa']:>10.4f} {r.get('ppl_eager', float('nan')):>10.4f} "
              f"{r['ppl_' + a.engine]:>11.4f} {r['delta_mean_nll']:>+11.2e} "
              f"{('[%+.2e, %+.2e]' % ci) if ci else '':>25} {r.get('tolerance', float('nan')):>9.2e}  "
              f"{r['verdict']:>8} {'PASS' if r['passes_bar'] else 'FAIL'}   ({r['equal_or_better']}, {r['within_standard']})"
              + (f"   tolerance {r['tolerance_source']}" if 'borrowed' in r.get('tolerance_source', '') else ""))
    print(f"BAR (perplexity equal or better, 95% CI reading): {'PASS' if report['passes_bar'] else 'FAIL'}")
    print(f"equal eps {EQUAL_EPS:g} nats/token   info: all point-estimate equal-or-better: {report['all_equal_or_better']}   all within standard: {report['all_within_standard']}")


if __name__ == "__main__":
    main()
