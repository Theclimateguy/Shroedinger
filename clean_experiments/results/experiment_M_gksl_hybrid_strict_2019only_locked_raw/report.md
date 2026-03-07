# Experiment M GKSL Hybrid Strict

## Protocol
- train years <= 2018
- holdout test year = 2019
- external holdout year = 2020
- phi candidate selected on train only (or locked by CLI)
- permutation tests on holdouts with fixed train-fitted coefficients

## Selection
- mode: `locked`
- selected bands: `0` / `1`
- selected feature: `raw`
- GKSL config: `G001`

## Holdout 2019
- ERA5+Lambda gain_vs_ctrl: `0.005368`, p_abs=`0.050000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.004594`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000779`, p_inc=`0.325000`
- shuffled control gain_vs_lambda: `0.000026`, p_inc=`0.310000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
