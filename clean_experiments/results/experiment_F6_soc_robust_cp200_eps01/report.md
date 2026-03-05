# Experiment F6 Report: SOC Avalanche Signature

## Goal
- Test SOC signature in toy open system with slow drive + threshold relaxation in mu-space.
- Observable: avalanche sizes as coherence drops ΔLambda_coh.
- Theory target: alpha_pred = 1 + 1/y_rel, with y_rel from F2 linearized RG matrix.

## Theory Inputs
- epsilon* from F1: 1.000
- epsilon* used in F6 scan: 0.100
- y_rel from F2: 1.850000
- alpha_pred = 1 + 1/y_rel = 1.540541

## SOC Parameters
- transfer_fraction = 0.9600
- cap_power = 2.0000
- p_cap(epsilon) = 0.999 * balance(epsilon)^cap_power
- p_scale = 1.4000

## Main Fit at epsilon*
- alpha_measured (mean over seeds) = 2.110695
- alpha 95% seed interval = [2.062173, 2.139231]
- relative error |alpha_measured-alpha_pred|/alpha_pred = 37.010%
- fit R2 mean = 0.957676
- tail dynamic range (median) = 38.408
- failed seed share = 0.000

## Criteria
- epsilon_star_in_window: False
- tail_r2_gt_threshold (0.95): True
- tail_dynamic_range_ge_100 (100.0): False
- alpha_match_within_10pct (0.1): False
- failed_seed_share_below_limit (0.35): True
- PASS_ALL: False

## Files
- experiment_F6_avalanche_events.csv
- experiment_F6_powerlaw_fits.csv
- experiment_F6_epsilon_summary.csv
- plot_F6_star_tail_fit.png
- plot_F6_alpha_scan.png
- experiment_F6_verdict.json
