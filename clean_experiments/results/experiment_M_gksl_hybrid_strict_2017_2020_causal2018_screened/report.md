# Experiment M GKSL Hybrid Strict

## Protocol
- train years <= 2018
- holdout test year = 2019
- external holdout year = 2020
- phi candidate selected on train only (or locked by CLI)
- permutation tests on holdouts with fixed train-fitted coefficients

## Selection
- mode: `screened_train_only`
- selected bands: `0` / `3`
- selected feature: `eta`
- GKSL config: `G001`

## Holdout 2019
- ERA5+Lambda gain_vs_ctrl: `0.001552`, p_abs=`0.030000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.000898`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000655`, p_inc=`0.178000`
- shuffled control gain_vs_lambda: `-0.007294`, p_inc=`0.920000`

## External 2020
- ERA5+Lambda gain_vs_ctrl: `-0.000212`, p_abs=`0.372000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.000598`
- ERA5+Lambda+Phi gain_vs_lambda: `0.000810`, p_inc=`0.068000`
- shuffled control gain_vs_lambda: `-0.000400`, p_inc=`0.818000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
