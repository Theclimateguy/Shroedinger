# Experiment M GKSL Hybrid Strict

## Protocol
- train years <= 2019
- holdout test year = 2020
- external holdout year = 2021
- phi candidate selected on train only (or locked by CLI)
- permutation tests on holdouts with fixed train-fitted coefficients

## Selection
- mode: `locked`
- selected bands: `0` / `1`
- selected feature: `raw`
- GKSL config: `G001`

## Holdout 2020
- ERA5+Lambda gain_vs_ctrl: `-0.002240`, p_abs=`0.405500`
- ERA5+Lambda+Phi gain_vs_ctrl: `-0.002307`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000067`, p_inc=`0.381000`
- shuffled control gain_vs_lambda: `0.000426`, p_inc=`0.193000`

## External 2021
- ERA5+Lambda gain_vs_ctrl: `-0.004088`, p_abs=`0.923000`
- ERA5+Lambda+Phi gain_vs_ctrl: `-0.004112`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000023`, p_inc=`0.335000`
- shuffled control gain_vs_lambda: `0.000242`, p_inc=`0.330500`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
