import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_depth_check as dc  # noqa: E402


def test_sampled_positions_reveal_each_codebook_once():
    order = np.random.default_rng(1).permutation(12)
    masked = np.ones((1, 3, 12), bool)
    masked[0, 1, order[:4]] = False
    masked[0, 2, order[:8]] = False
    picked = dc.sampled_positions(masked)
    assert picked.sum() == 12 and (picked.sum(1) == 1).all()
    assert set(np.nonzero(picked[0, 0])[0]) == set(order[:4])
    assert set(np.nonzero(picked[0, 2])[0]) == set(order[8:])


def test_log_sampling_matches_hf_penalty_then_temperature():
    logits = np.array([[2.0, -1.0, 0.5, 3.0]])
    history = np.array([[True, True, False, False]])
    s = np.array([2.0 / 1.2, -1.2, 0.5, 3.0]) / 0.4      # positive divided, negative multiplied
    expected = s - np.log(np.exp(s).sum())
    assert np.allclose(dc.log_sampling(logits, history), expected)


def test_verdict_uses_eps():
    assert dc.verdict([2e-5, 3e-5]) == "worse"
    assert dc.verdict([-3e-5, -2e-5]) == "better"
    assert dc.verdict([-2e-5, 5e-6]) == "equal"
    assert dc.verdict([5e-6, 9e-6]) == "equal"   # above zero but within eps
