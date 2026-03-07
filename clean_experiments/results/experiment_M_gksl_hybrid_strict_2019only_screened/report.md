# Experiment M GKSL Hybrid Strict

## Protocol
- train years <= 2018
- holdout test year = 2019
- external holdout year = 2020
- phi candidate selected on train only (or locked by CLI)
- permutation tests on holdouts with fixed train-fitted coefficients

## Selection
- mode: `screened_train_only`
- selected bands: `2` / `3`
- selected feature: `popdiff`
- GKSL config: `G001`

## Holdout 2019
- ERA5+Lambda gain_vs_ctrl: `0.005368`, p_abs=`0.050000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.004613`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000759`, p_inc=`0.675000`
- shuffled control gain_vs_lambda: `0.000134`, p_inc=`0.015000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
