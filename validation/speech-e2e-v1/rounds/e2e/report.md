| case | mode | native WER (mean over seeds) | HF WER | native frames mean [min-max] | HF frames | ratio | ended by EOS (native, HF) |
|---|---|---|---|---|---|---|---|
| tts-fox-female | audio | 0.000 | 0.000 | 43 [40-44] | 45 [42-48] | 0.96 | 3/3, 3/3 |
| tts-fox-male | audio | 0.000 | 0.000 | 44 [40-48] | 46 [44-50] | 0.95 | 3/3, 3/3 |
| tts-museum-female | audio | 0.028 | 0.028 | 152 [150-154] | 151 [146-158] | 1.01 | 3/3, 3/3 |
| tts-museum-male | audio | 0.028 | 0.028 | 155 [150-162] | 153 [150-154] | 1.02 | 3/3, 3/3 |
| tts-numbers-female | audio | 0.263 | 0.158 | 117 [112-124] | 117 [108-124] | 1.00 | 3/3, 3/3 |
| tts-numbers-male | audio | 0.158 | 0.175 | 121 [116-126] | 116 [112-118] | 1.05 | 3/3, 3/3 |
| tts-long-male | audio | 0.887 | 0.158 | 767 [493-1181] | 442 [399-492] | 1.74 | 3/3, 3/3 |
| il-reflective | interleaved | 0.000 | 0.000 | 322 [252-376] | 325 [259-397] | 0.99 | 3/3, 3/3 |
| il-study | interleaved | 0.000 | 0.000 | 266 [228-299] | 276 [267-293] | 0.96 | 3/3, 3/3 |
| il-fridge | interleaved | 0.037 | 0.009 | 324 [317-331] | 285 [218-325] | 1.14 | 3/3, 3/3 |

C1 pooled WER per seed: native {1: 0.5051020408163265, 2: 0.07808564231738035, 3: 0.15081206496519722}, HF {1: 0.053613053613053616, 2: 0.0812807881773399, 3: 0.0410958904109589}
C1 (native's best seed <= HF's worst seed): PASS
C2 mean frames ratio native/HF over cases: 1.080
real-time factor (audio seconds per generation second): native 3.24, HF 1.91
