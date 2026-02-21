# Experiment B (Wave-1) Robust Artifacts

## What was validated
- Commutator/Hermitian-curvature construction for the integrated first-wave experiment.
- Spatial identity at `phi = 90 deg`: `Lambda_matter(|+>, x) = theta2(x)`.
- Antisymmetry between `|+>` and `|->`: `Lambda_matter(|->, x) = -Lambda_matter(|+>, x)`.
- Angular law: averaged `Lambda_matter(phi)` follows `sin(phi)`.

## Runs
- 120 randomized cases (`seed=20260220`).
- Each case writes full per-case artifacts in `cases/case_*/`.

## Key outcomes
- Fraction of cases passing all thresholds: `1.0` (120/120).
- Worst spatial error: `2.220446e-16`.
- Worst antisymmetry residual: `0.0`.
- Worst angular-law error: `4.440892e-16`.
- Minimum `R^2` for sinusoidal fit: `1.0`.

## Files
- `robustness_results.csv`
- `robustness_summary.csv`
- `worst_12_by_angle_err.csv`
- `cases/...` per-case outputs
