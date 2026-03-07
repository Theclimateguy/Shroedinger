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
- ERA5+Lambda gain_vs_ctrl: `0.001552`, p_abs=`0.030000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.001949`
- ERA5+Lambda+Phi gain_vs_lambda: `0.000398`, p_inc=`0.021500`
- shuffled control gain_vs_lambda: `0.000063`, p_inc=`0.434500`

## External 2020
- ERA5+Lambda gain_vs_ctrl: `-0.000212`, p_abs=`0.388500`
- ERA5+Lambda+Phi gain_vs_ctrl: `-0.000398`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000187`, p_inc=`0.525500`
- shuffled control gain_vs_lambda: `-0.000941`, p_inc=`0.929000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
