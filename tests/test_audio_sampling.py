"""B2 (journal/lfm3-tts-quality-bar.md): the native speech-frame sampler (native/include/lfm/audio_sampling.hpp)
against an independent analytic oracle: softmax(penalize(logits) / 0.4), penalty 1.2 on the codebook's codes in
the last 64 accepted frames, as HF's RepetitionPenaltyLogitsProcessor then TemperatureLogitsWarper. The logits are
real depth-decoder logits (FP64 anchor, tts-museum-male; tests/data_audio_sampling_fixture.npz) at six positions
with 0 to 64 history codes. Seeds are fixed, so the chi-squared outcomes are reproducible.
"""
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = np.load(Path(__file__).with_name("data_audio_sampling_fixture.npz"))
POSITIONS = range(len(FIXTURE["codebook"]))
P_MIN = 1e-3  # every chi-squared test must clear this


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    exe = tmp_path_factory.mktemp("audio_sampling") / "oracle"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/tests/audio_sampling_oracle.cpp"), "-o", str(exe)], check=True)
    return exe


def run(oracle, args, stdin=""):
    return subprocess.run([str(oracle), *map(str, args)], input=stdin, capture_output=True, text=True,
                          check=True).stdout.split()


def case_input(i):
    history = FIXTURE[f"history{i}"]
    return " ".join(repr(float(v)) for v in FIXTURE["logits"][i]) + f"\n{len(history)} " + " ".join(map(str, history))


def analytic(i):
    logits = FIXTURE["logits"][i].astype(np.float64)
    seen = np.zeros(2017, bool)
    seen[FIXTURE[f"history{i}"]] = True
    s = np.where(seen, np.where(logits < 0, logits * 1.2, logits / 1.2), logits) / 0.4
    p = np.exp(s - s.max())
    return p / p.sum()


def chi2(counts, expected_p):
    n = counts.sum()
    expected = expected_p * n
    keep = expected >= 5
    obs = np.append(counts[keep], counts[~keep].sum())
    exp = np.append(expected[keep], expected[~keep].sum())
    if exp[-1] < 5:  # fold a small tail into the smallest kept bin
        j = np.argmin(exp[:-1]); obs[j] += obs[-1]; exp[j] += exp[-1]; obs, exp = obs[:-1], exp[:-1]
    return stats.chisquare(obs, exp * obs.sum() / exp.sum()).pvalue, int(len(obs))


@pytest.mark.parametrize("i", POSITIONS)
def test_exact_distribution_matches_analytic(oracle, i):
    out = run(oracle, ["dist", int(FIXTURE["codebook"][i])], case_input(i))
    p = np.zeros(2017)
    for code, prob in zip(out[::2], out[1::2]):
        p[int(code)] = float(prob)
    # FP32 logits widened to long double vs float64 here: agreement to ~1e-12 absolute
    assert np.abs(p - analytic(i)).max() < 1e-12


@pytest.mark.parametrize("i", POSITIONS)
def test_draws_follow_the_distribution(oracle, i):
    counts = np.array(run(oracle, ["codes", 60000, 1000 + i, int(FIXTURE["codebook"][i])], case_input(i)), np.int64)
    p, bins = chi2(counts, analytic(i))
    assert counts.sum() == 60000 and bins >= 2
    assert p > P_MIN, (p, bins)


def test_history_changes_the_distribution():
    # a fixture position whose history holds high-probability codes: the penalty must matter there
    i = next(i for i in POSITIONS if len(FIXTURE[f"history{i}"]))
    logits = FIXTURE["logits"][i].astype(np.float64)
    plain = np.exp((logits - logits.max()) / 0.4); plain /= plain.sum()
    assert np.abs(plain - analytic(i)).max() > 1e-3


def test_reveal_order_is_uniform(oracle):
    n = 120000
    out = np.array(run(oracle, ["order", n, 7]), np.int64)
    position, together = out[:144].reshape(12, 12), out[144:]
    assert (position.sum(0) == n).all() and (position.sum(1) == n).all()
    assert stats.chisquare(position.ravel(), np.full(144, n / 12)).pvalue > P_MIN
    assert len(together) == len(list(combinations(range(12), 2)))
    # Each permutation puts exactly 18 pairs in a shared pass, so only the per-pair counts are informative:
    # every pair shares a pass with probability 3/11 (Bonferroni over the 66 pairs).
    assert together.sum() == 18 * n
    assert min(stats.binomtest(int(k), n, 3 / 11).pvalue for k in together) > P_MIN / 66


def test_frame_protocol_and_marginals(oracle):
    # every codebook scored with the same logits, no history: each codebook's code follows the plain distribution
    n = 20000
    logits = FIXTURE["logits"][0].astype(np.float64)
    out = np.array(run(oracle, ["frame", n, 11], " ".join(repr(float(v)) for v in FIXTURE["logits"][0])), np.int64)
    counts = out.reshape(12, 2017)
    s = logits / 0.4
    p = np.exp(s - s.max()); p /= p.sum()
    for c in range(12):
        assert stats.chisquare(*map(np.asarray, _binned(counts[c], p))).pvalue > P_MIN / 12


def _binned(counts, p):
    exp = p * counts.sum()
    keep = exp >= 5
    obs = np.append(counts[keep], counts[~keep].sum())
    e = np.append(exp[keep], exp[~keep].sum())
    return obs, e * obs.sum() / e.sum()


def test_window_and_eos_refusal(oracle):
    out = run(oracle, ["window"])
    assert list(map(int, out[:-1])) == list(range(1, 65)) and out[-1] == "refused-eos"


def test_seeds_reproduce_and_differ(oracle):
    a = run(oracle, ["codes", 2000, 5, int(FIXTURE["codebook"][1])], case_input(1))
    b = run(oracle, ["codes", 2000, 5, int(FIXTURE["codebook"][1])], case_input(1))
    c = run(oracle, ["codes", 2000, 6, int(FIXTURE["codebook"][1])], case_input(1))
    assert a == b and a != c
