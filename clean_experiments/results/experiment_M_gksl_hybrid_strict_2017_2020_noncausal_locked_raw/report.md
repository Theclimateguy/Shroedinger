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
- ERA5+Lambda gain_vs_ctrl: `0.001965`, p_abs=`0.164000`
- ERA5+Lambda+Phi gain_vs_ctrl: `0.001154`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.000812`, p_inc=`0.820000`
- shuffled control gain_vs_lambda: `0.000128`, p_inc=`0.498000`

## External 2020
- ERA5+Lambda gain_vs_ctrl: `-0.000673`, p_abs=`0.456000`
- ERA5+Lambda+Phi gain_vs_ctrl: `-0.002043`
- ERA5+Lambda+Phi gain_vs_lambda: `-0.001368`, p_inc=`0.954000`
- shuffled control gain_vs_lambda: `0.000458`, p_inc=`0.170000`

## Artifacts
- `strict_metrics_2019.csv`
- `strict_oof_2019.csv`
- `strict_active_calm_2019.csv`
- `strict_quarterly_2019.csv`
- `strict_phi_screening_train.csv`
- `strict_meta.json`
- `strict_metrics_external.csv`
