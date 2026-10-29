# Test plan (write before touching the test split)

Status: **DRAFT**. Lock date: ______ (before any test-set run).
Commit hash when locked: ______

Writing this down first keeps the final numbers honest: nothing gets tuned after seeing test results.

## What a good result looks like
- On real MLC pairs, the comparability-aware model raises fewer false change alerts than simply
  subtracting two cover estimates, while catching as many real changes.
- When the app answers (not `REVIEW`), its error rate stays within the targets set on the calibration
  split (α0 = 0.05 false change, α1 = 0.20 missed change).
- On the synthetic stress test, naive subtraction gets worse as the image degradation gets stronger;
  the full model degrades less.

## Metrics
- Patch classifier: macro recall (ReefNet's 39-class setup).
- Pairs: false-alert rate, recall, coverage (share not sent to `REVIEW`).

## Fixed choices
- Splits by group (source / quadrat), test IDs hashed in `data/splits/TEST_HASH.txt`.
- Seeds: 3 per configuration. Report mean ± std.

## What this project does not do
Explain why a reef changed, score reef health, or detect bleaching. Synthetic stress-test results are
shown separately and labelled synthetic.
