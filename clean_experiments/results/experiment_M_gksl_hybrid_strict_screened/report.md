# Experiment M GKSL Hybrid Strict

## Protocol
- train years <= 2018
- holdout test year = 2019
- external holdout year = 2020
- phi candidate selected on train only (or locked by CLI)
- permutation tests on holdouts with fixed train-fitted coefficients

## Selection
- mode: `screened_train_only`
- selected bands: `0` / `2`
- selected feature: `pop_x_delta`
- GKSL config: `G001`

## Holdout 2019
- ERA5+Lambda gain_vs_ctrl: `0.002422`, p_abs=`0.190000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.002223`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000200`, p_inc=`0.370000`
- shuffled control gain_vs_lambda: `0.000861`, p_inc=`0.360000`

## External 2020
- ERA5+Lambda gain_vs_ctrl: `-0.006333`, p_abs=`0.965000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.009824`
- ERA5+Lambda+Phi gain_vs_lambda: `0.016054`, p_inc=`0.050000`
- shuffled control gain_vs_lambda: `-0.000590`, p_inc=`0.830000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
