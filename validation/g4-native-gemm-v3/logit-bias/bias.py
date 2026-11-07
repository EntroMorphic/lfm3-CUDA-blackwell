#!/usr/bin/env python3
"""Expected NLL bias from rounding logits to BF16, from traced BF16 logits of one window.

Usage: bias.py TRACE_DIR WINDOWS.i32 LENGTH INDEX PROMPT_LEN
  WINDOWS.i32 holds windows of LENGTH tokens; INDEX selects one. liquid-lfm3 prefilled its first PROMPT_LEN
  tokens and teacher-forced the rest with trace_logits_only, so TRACE_DIR/STEP.logits.bf16 holds the logits
  that predict position PROMPT_LEN + STEP. Only steps whose target lies inside the window are used.

Model: each logit carries an independent rounding error, uniform in +-ulp/2 (BF16: 7 explicit mantissa bits,
ulp = 2^(floor(log2|z|) - 7)). The bias of E[logsumexp(z + e)] - logsumexp(z) is, to second order,
1/2 * sum_i p_i (1 - p_i) ulp_i^2 / 12 (the -e_y term has mean zero). The Monte Carlo draws e under the same
model and subtracts the control variate e.p, whose mean is exactly zero; it checks the truncation of the
Taylor series, not the error model itself.
"""
import glob
import sys

import numpy as np

trace, windows, length, index, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
w = np.fromfile(windows, np.int32).reshape(-1, length)[index]
files = sorted(glob.glob(f"{trace}/*.logits.bf16"), key=lambda f: int(f.split("/")[-1].split(".")[0]))
rng = np.random.default_rng(0)


def lse(z):
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))[..., 0]


formula, mc, nll, mags, ulp_top, pmax = [], [], [], [], [], []
for i, f in enumerate(files):
    if int(f.split("/")[-1].split(".")[0]) != i:
        raise SystemExit("trace steps are not contiguous from 0")
    if prompt + i >= length:
        break  # the last step predicts past the window's end
    z = (np.fromfile(f, np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(z), 1e-30))) - 7)
    p = np.exp(z - z.max())
    p /= p.sum()
    formula.append(0.5 * np.sum(p * (1 - p) * ulp**2 / 12))
    eps = (rng.random((64, z.size)) - 0.5) * ulp
    mc.append((lse(z + eps) - lse(z[None])[0] - eps @ p).mean())
    nll.append(-np.log(p[w[prompt + i]]))
    mags.append(np.abs(z).max())
    ulp_top.append(ulp[np.argmax(p)])
    pmax.append(p.max())

formula, mc = np.array(formula), np.array(mc)
print(f"window {index} of {windows} (length {length}); prompt {prompt}; scored positions {len(formula)}")
print(f"mean NLL of the scored positions {np.mean(nll):.4f}")
print(f"max |logit| per position: median {np.median(mags):.1f} max {np.max(mags):.1f}; "
      f"ulp at the argmax logit: median {np.median(ulp_top):.4f}; p_max median {np.median(pmax):.3f}")
print(f"bias, second-order formula: mean {formula.mean():.3e} median {np.median(formula):.3e} "
      f"min {formula.min():.3e} max {formula.max():.3e}")
print(f"bias, Monte Carlo (64 draws per position, control variate e.p): mean {mc.mean():.3e}; "
      f"standard error over positions {mc.std(ddof=1) / np.sqrt(len(mc)):.1e}")
