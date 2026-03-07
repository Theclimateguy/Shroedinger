# Experiment M Halo Boundary Strict

## Protocol
- train years <= 2019
- holdout test year = 2020
- external holdout year = 2021
- scoring target: core-only residual (`target_mode=physical_zscore`)
- halo ring width = 8 cells
- selected bands (fine/coarse) = 0/1 (centers 35.4/70.7 km)

## Model Ladder
- ERA_core
- ERA_window
- ERA_window_plus_Phi_H
- ERA_window_plus_Lambda_H (includes bath_c2f, bath_f2c)
- ERA_window_plus_Phi_H_plus_Lambda_H
- ERA_window_plus_Phi_H_plus_Lambda_H_shuffled

## Test 2020
- ERA_window gain_vs_core: `0.136707` p_abs=`0.000500`
- +Phi_H gain_vs_core: `0.136787` p_abs=`0.000500`
- +Lambda_H gain_vs_core: `0.136706` p_abs=`0.000500`
- +Phi_H+Lambda_H gain_vs_core: `0.136796` p_abs=`0.000500`

## External 2021
- ERA_window gain_vs_core: `0.092229` p_abs=`0.000500`
- +Phi_H gain_vs_core: `0.092290` p_abs=`0.000500`
- +Lambda_H gain_vs_core: `0.092270` p_abs=`0.000500`
- +Phi_H+Lambda_H gain_vs_core: `0.092359` p_abs=`0.000500`

## Artifacts
- `halo_metrics_test.csv`
- `halo_metrics_external.csv`
- `halo_bath_summary.csv`
- `halo_timeseries.csv`
- `halo_meta.json`
