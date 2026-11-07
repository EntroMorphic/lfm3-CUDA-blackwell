#!/usr/bin/env python3
"""Compare one-window FP32 NLL files from the anchor probes (run on Thor in build/): plain eager, eager in blocks,
and SDPA, all FP32 with TF32 disabled, on window 0 of the 4,096-token windows."""
import hashlib
import sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "."
paths = {c: f"{d}/fp32-probe-{c}-4096-v2/nll-hf-{c}-fp32-4096.f32" for c in ("eager_blocked", "eager", "sdpa")}
x = {c: np.fromfile(p, np.float32).astype(np.float64) for c, p in paths.items()}
for c, p in paths.items():
    print(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p.split('/', 1)[-1] if d == '.' else p}")
for a, b in (("eager_blocked", "eager"), ("eager_blocked", "sdpa"), ("eager", "sdpa")):
    diff = np.abs(x[a] - x[b])
    print(f"{a} vs {b}: tokens {diff.size}, max |dNLL| {diff.max():.3e}, mean {diff.mean():.3e}, bitwise identical {bool((x[a] == x[b]).all())}")
