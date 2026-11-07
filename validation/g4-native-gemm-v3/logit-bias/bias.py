#!/usr/bin/env python3
"""Expected NLL bias from rounding logits to BF16, from traced bf16 logits.

Usage: bias.py TRACE_DIR WINDOW.i32 PROMPT_LEN
  TRACE_DIR holds STEP.logits.bf16 files from liquid-lfm3 with trace_logits_only; step i is the logits
  that predict position PROMPT_LEN + i. Prints the second-order formula and a Monte Carlo check.
"""
import glob
import sys

import numpy as np

trace, window, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(f"{trace}/*.logits.bf16"), key=lambda f: int(f.split("/")[-1].split(".")[0]))
w = np.fromfile(window, np.int32)
rng = np.random.default_rng(0)


def lse(z):
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))[..., 0]


formula, mc, nll, mags, ulp_top, pmax = [], [], [], [], [], []
for i, f in enumerate(files):
    if prompt + i >= len(w):
        break
    z = (np.fromfile(f, np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    e = np.floor(np.log2(np.maximum(np.abs(z), 1e-30)))
    ulp = 2.0 ** (e - 7)  # bf16: 7 explicit mantissa bits
    p = np.exp(z - z.max())
    p /= p.sum()
    formula.append(0.5 * np.sum(p * (1 - p) * ulp**2 / 12))
    eps = (rng.random((64, z.size)) - 0.5) * ulp  # uniform in +-ulp/2
    mc.append((lse(z + eps) - lse(z[None])[0] - eps @ p).mean())  # E[eps.p] = E[-eps_y] = 0 exactly
    nll.append(-np.log(p[w[prompt + i]]))
    mags.append(np.abs(z).max())
    ulp_top.append(ulp[np.argmax(p)])
    pmax.append(p.max())

formula, mc = np.array(formula), np.array(mc)
print(f"positions {len(formula)}; mean NLL {np.mean(nll):.4f}")
print(f"max |logit|: median {np.median(mags):.1f} max {np.max(mags):.1f}; ulp at argmax: median {np.median(ulp_top):.4f}; p_max median {np.median(pmax):.3f}")
print(f"bias, second-order formula: mean {formula.mean():.3e} median {np.median(formula):.3e} min {formula.min():.3e} max {formula.max():.3e}")
print(f"bias, Monte Carlo: mean {mc.mean():.3e} standard error {mc.std() / np.sqrt(len(mc)):.1e}")
