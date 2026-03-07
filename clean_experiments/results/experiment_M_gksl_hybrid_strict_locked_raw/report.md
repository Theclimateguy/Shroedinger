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
- ERA5+Lambda gain_vs_ctrl: `0.002422`, p_abs=`0.190000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.003307`
- ERA5+Lambda+Phi gain_vs_lambda: `0.000888`, p_inc=`0.285000`
- shuffled control gain_vs_lambda: `-0.000221`, p_inc=`0.530000`

## External 2020
- ERA5+Lambda gain_vs_ctrl: `-0.006333`, p_abs=`0.965000`
- ERA5+Lambda+Phi gain_vs_ctrl: `-0.003216`
- ERA5+Lambda+Phi gain_vs_lambda: `0.003097`, p_inc=`0.140000`
- shuffled control gain_vs_lambda: `0.001111`, p_inc=`0.090000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
