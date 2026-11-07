# Evidence behind the PLAN amendment (the tier 2b anchor)

- **FP32 SDPA does not fit at 32,768** (`memory-transcript.txt`, items 1 and 3): the full run hung Thor during
  length 32,768; one 32,768-token window then drove available memory from 107 GB to 11 GB (~96 GB) before a
  watchdog killed it. At 4,096 one window used ~13 GB (item 2). These figures were printed to the session, not to
  files; they are transcribed verbatim. Scripts: `fp32-probe-sigkill-watchdog.sh` (items 2–3; its SIGKILL left the
  memory held by the driver until a reboot) and `fp32-probe-sigterm.sh` (items 4–5).
- **FP32 SDPA is not exact FP32, and blocked eager is** (`compare-output.txt`, from `compare.py` over the v2 probe
  files on Thor; one 4,096-token window, 4,095 scored tokens, TF32 disabled):
  - eager in blocks vs plain eager: **bitwise identical** (the two NLL files have the same SHA-256, 7910638b…);
  - SDPA vs eager: max |ΔNLL| 5.05×10⁻⁵, mean 4.68×10⁻⁶.
- **Blocked eager runs at 32,768** (item 5; log tail in `compare-output.txt`): one window completed in 86 s, and the
  full anchor run (`../tier2b/run.log`) kept MemAvailable at or above 73 GB. MemAvailable dipped by only ~2 GB in
  the post-reboot probes, even at 4,096 where the earlier SDPA probe dipped by ~13 GB, so it may not capture every
  GPU allocation; it is a safety reading, not a measure of the run's footprint.

The probe NLL files themselves are on Thor (`build/fp32-probe-*-v2/`), identified by the hashes in
`compare-output.txt`.
