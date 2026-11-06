"""Host checks for the perplexity bar (95% CI reading) in tools/ppl_compare.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ppl_compare import EQUAL_EPS, bootstrap_ci, passes_bar, verdict
import numpy as np


def test_equal_when_ci_includes_zero():
    assert passes_bar(7.9e-5, (-6.1e-5, 2.2e-4)) and verdict(7.9e-5, (-6.1e-5, 2.2e-4)) == "equal"


def test_better_when_ci_entirely_below_zero():
    assert passes_bar(-1e-3, (-2e-3, -5e-4)) and verdict(-1e-3, (-2e-3, -5e-4)) == "better"


def test_fails_only_when_statistically_worse():
    assert not passes_bar(1e-3, (5e-4, 2e-3)) and verdict(1e-3, (5e-4, 2e-3)) == "worse"


def test_eps_band_absorbs_numerical_noise():
    assert passes_bar(2 * EQUAL_EPS, (0.5 * EQUAL_EPS, 3 * EQUAL_EPS))


def test_single_window_uses_point_estimate():
    assert passes_bar(0.0, None) and not passes_bar(1e-3, None)


def test_bootstrap_is_deterministic_and_brackets_mean():
    x = np.random.default_rng(1).normal(0.0, 1e-3, 500)
    lo, hi = bootstrap_ci(x)
    assert (lo, hi) == bootstrap_ci(x) and lo < x.mean() < hi
