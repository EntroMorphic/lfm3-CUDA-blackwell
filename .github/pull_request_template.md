## What changes

## Evidence

- Validation round: `validation/<round>/` (rules committed before the run: <commit>)
- Results per rule:
- Regressions re-run (earlier outputs byte-identical):
- Coherency read / listened:

## Checklist

- [ ] Host tests pass (`PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q`)
- [ ] Claims in docs and the round's README are worded to the data
- [ ] Speed numbers say decode loop or end to end, measured on an idle machine
