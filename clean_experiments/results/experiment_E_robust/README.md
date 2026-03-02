# Experiment E Robust Artifacts

## What was validated
- In the coherence-driven regime (`eta = eta(coherence)`), coherence features should predict
  `|src_mu|` better than `|Lambda|`-only features.

## Runs
- 180 rows in the aggregated table (`10 seeds x 2 K x 3 gamma x 3 eta0`).
- Each run used 180 samples.

## Key outcomes
- Fraction with `R2_coh_features > R2_absLambda_only`: `1.0` (180/180).
- Fraction with `corr_coh_sum > corr_absLambda_sum`: `1.0` (180/180).
- Mean `R2` gap (`coh - lambda`): `0.04834`.
- Worst observed `R2` gap remained positive: `0.007044`.

## Files
- `robustness_results.csv`
- `robustness_summary.csv`
- `robustness_by_eta0.csv`
- `worst_15_by_r2_gap.csv`
- `cases/...` per-run raw outputs
