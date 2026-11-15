# Dump torch's uint8 resize output for chosen cases (to compare machines value by value).
import sys
import numpy as np, torch
from torchvision.transforms.v2 import functional as tvF
cases_txt, out_npz = sys.argv[1], sys.argv[2]
GOLDEN, C1, C2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
def noise(key, w, h):
    n = w * h * 3
    with np.errstate(over="ignore"):
        z = np.uint64(key) + np.arange(1, (n + 7) // 8 + 1, dtype=np.uint64) * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(C1); z = (z ^ (z >> np.uint64(27))) * np.uint64(C2); z = z ^ (z >> np.uint64(31))
    return z.astype("<u8").view(np.uint8)[:n].reshape(h, w, 3)
lines = open(cases_txt).readlines()
res = {}
for case in (1750, 330, 2459):
    iw, ih, ow, oh, key = map(int, lines[case].split())
    t = torch.from_numpy(noise(key, iw, ih)).permute(2, 0, 1).contiguous()
    res[f"c{case}"] = tvF.resize(t, [oh, ow], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True).numpy()
np.savez(out_npz, **res)
