# Experiment G Robust Artifacts

## What was validated
- Profile invariance at fixed accumulated phase `varphi`:
  after rescaling each `omega(mu)` profile to the same `phi_target`,
  `Lambda` depends on `sin(varphi)` rather than profile shape.

## Runs
- 120 random parameter cases.
- Per case, 3 profiles (`constant`, `gaussian`, `oscillating`).
- Total rows: 360.

## Key outcomes
- Overall max relative error vs theory: `3.918375e-15`.
- Worst cross-profile spread in numerical `Lambda`: `8.437695e-15`.
- Fraction of cases with profile spread `< 1e-12`: `1.0`.

## Files
- `robustness_case_summary.csv`
- `robustness_all_rows.csv`
- `robustness_summary.csv`
- `cases/...` per-case outputs
