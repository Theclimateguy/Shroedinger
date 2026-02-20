# G2 Single-Qubit Robust Artifacts

## What was validated
- Clausius regression quality depends on the `omega(mu)` profile.
- Non-equilibrium profile (`oscillating`) should show lower `R2` than near-equilibrium profiles.

## Runs
- 48 parameter cases (`12 seeds x 2 Gamma x 2 DeltaS_hor`).
- 3 profiles per case: `constant`, `gaussian`, `oscillating`.
- Total profile runs: 144.

## Key outcomes
- Mean `R2` by profile:
  - `constant`: `0.937791`
  - `gaussian`: `0.336422`
  - `oscillating`: `0.047309`
- Fraction of cases where `oscillating` is lowest `R2`: `1.0`.
- Mean `R2(constant) - R2(oscillating)`: `0.890483`.

## Files
- `robustness_results.csv`
- `robustness_rank_by_case.csv`
- `robustness_summary.csv`
- `worst_12_const_minus_osc.csv`
- `cases/...` per-run timeseries/fit files
