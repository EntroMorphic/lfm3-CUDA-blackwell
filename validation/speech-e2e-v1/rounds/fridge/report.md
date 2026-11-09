| case | mode | native WER (mean over seeds) | HF WER (mean [worst]) | C1 | native frames mean [min-max] | HF frames | ratio | ended by EOS (native, HF) |
|---|---|---|---|---|---|---|---|---|
| il-fridge | interleaved | 0.013 | 0.010 [0.028] | ok | 304 [244-391] | 296 [176-390] | 1.03 | 12/12, 12/12 |

C1 pooled WER per seed: native {4: 0.0, 5: 0.013888888888888888, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.028985507246376812, 11: 0.02702702702702703, 12: 0.014705882352941176, 13: 0.0, 14: 0.0, 15: 0.06756756756756757}, HF {4: 0.02631578947368421, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.02702702702702703, 9: 0.014285714285714285, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.023529411764705882, 14: 0.028169014084507043, 15: 0.0}
C1 per case (native mean WER > HF's worst seed): flagged none
C2 mean frames ratio native/HF over cases: 1.028
real-time factor (audio seconds per generation second): native 2.85, HF 2.08
