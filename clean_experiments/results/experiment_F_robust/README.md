# Experiment F Robust Artifacts

## What was validated
- Sinusoidal law in the U(2) toy model:
  `Lambda = r * <sigma_x> * sin(omega * Delta_mu)`.
- Numerical robustness across parameter and discretization sweeps.

## Runs
- 120 random parameter cases.
- Per case discretization sweep: `K = 4, 8, 16, 32, 64, 128`.
- Total rows: 720.

## Key outcomes
- Overall max relative error: `6.291431e-15`.
- 99th percentile relative error: `3.022760e-15`.
- Fraction of rows with `rel_error < 1e-12`: `1.0`.
- Fraction of cases with max error `< 1e-12`: `1.0`.

## Files
- `robustness_case_summary.csv`
- `robustness_all_rows.csv`
- `robustness_summary.csv`
- `cases/...` per-case outputs
