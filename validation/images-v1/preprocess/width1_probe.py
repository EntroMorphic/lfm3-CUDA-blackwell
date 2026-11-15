# Thor: torch's uint8 bicubic-AA resize on the 12 width-1 cases vs its float32 path (rounded), and vs a transpose of
# the same problem (height 1 instead of width 1). A correct uint8 result stays within 1 of the float one.
import sys
sys.argv = [sys.argv[0]]
import numpy as np, torch
from torchvision.transforms.v2 import functional as tvF
GOLDEN, C1, C2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
def noise(key, w, h):
    n = w * h * 3
    with np.errstate(over="ignore"):
        z = np.uint64(key) + np.arange(1, (n + 7) // 8 + 1, dtype=np.uint64) * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(C1); z = (z ^ (z >> np.uint64(27))) * np.uint64(C2); z = z ^ (z >> np.uint64(31))
    return z.astype("<u8").view(np.uint8)[:n].reshape(h, w, 3)
B = tvF.InterpolationMode.BICUBIC
for case in (330, 403, 1104, 1675, 1692, 1750, 1836, 1969, 2406, 2459, 2844, 2905):
    iw, ih, ow, oh, key = map(int, open("liquid-CUDA/build/images-v1/preprocess/resize-cases.txt").readlines()[case].split())
    t = torch.from_numpy(noise(key, iw, ih)).permute(2, 0, 1).contiguous()
    u = tvF.resize(t, [oh, ow], interpolation=B, antialias=True).int()
    f = tvF.resize(t.float(), [oh, ow], interpolation=B, antialias=True).clamp(0, 255).round().int()
    tt = t.transpose(1, 2).contiguous()  # the same problem with height and width swapped
    ut = tvF.resize(tt, [ow, oh], interpolation=B, antialias=True).transpose(1, 2).int()
    print(case, [iw, ih, ow, oh], "uint8 vs float max|d|", int((u - f).abs().max()), "| transposed uint8 vs float max|d|",
          int((ut - f).abs().max()), "| uint8 == transposed", bool((u == ut).all()))
