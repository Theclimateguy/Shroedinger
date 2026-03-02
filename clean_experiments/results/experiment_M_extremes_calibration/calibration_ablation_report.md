# Extreme Calibration: Ablation Sanity Check

## Purpose
Separate the effect of regime gating (`E`) from incremental contribution of `Lambda` under the same out-of-time test (2019).

## Setup
- Train: 2017-2018
- Test: 2019
- Thresholds: `q_extreme=0.85` (estimated on train only)
- Ridge: `alpha=0.1`

Models:
- `regime_no_lambda`: `ctrl + E + ctrl*E`
- `regime_with_lambda_v`: `ctrl + lambda_v + E + ctrl*E + lambda_v*E`
- `regime_with_lambda_h`: `ctrl + lambda_h + E + ctrl*E + lambda_h*E`

## Results
From `test_ablation_metrics.csv`:
- `regime_no_lambda`: gain_all = `0.132593`
- `regime_with_lambda_v`: gain_all = `0.139717`
- `regime_with_lambda_h`: gain_all = `0.140295`

From paired block-bootstrap (`test_ablation_incremental_lambda.csv`):
- Incremental gain (`with_lambda_v - no_lambda`) = `0.006959`, CI95 = `[0.000410, 0.014535]`
- Incremental gain (`with_lambda_h - no_lambda`) = `0.007522`, CI95 = `[0.000629, 0.016148]`

## Interpretation
- The main stabilization in noisy extremes comes from regime-aware calibration (`E` and interactions), not from lambda alone.
- `Lambda` still provides a small but positive incremental contribution on top of regime calibration, with CI above zero in paired block-bootstrap.
- This supports a conservative claim: calibration is doing real work in extremes while keeping a measurable lambda-specific signal.
