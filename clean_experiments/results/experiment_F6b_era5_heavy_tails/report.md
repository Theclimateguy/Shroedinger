# Experiment F6b Report: Heavy Tails of Structural |Lambda| in ERA5

## Protocol
- x_min is selected strictly by minimum KS distance (Clauset/Newman style scan).
- alpha is estimated by continuous-tail MLE, not by log-log linear regression.
- Tail model comparison uses LLR(Power-law vs Exponential) with Vuong-style p-value.
- Percentile scan: [70.0, 99.0] with 120 candidates.

## Global Best Fit
- xmin = 4.220869e-05
- alpha_emp = 2.906392
- alpha_pred(reference) = 1.540000
- KS distance = 0.026744
- dynamic range x_max/x_min = 15.464
- n_tail = 642
- LLR(PL-Exp) = 39.073800
- LLR p-value = 0.009391

## Strict PASS Criteria
- dynamic_range_ge_10: True
- llr_prefers_powerlaw: True
- alpha_in_universality_band_1p3_2p0: False
- PASS_ALL: False

## Stratified Diagnostics
- Best fits by regime are saved in `experiment_F6b_best_fits.csv`.
- Candidate scans are saved in `experiment_F6b_tail_metrics.csv`.
- convective_heavier_or_better: True

## Outputs
- experiment_F6b_tail_metrics.csv
- experiment_F6b_best_fits.csv
- experiment_F6b_verdict.json
- plot_F6b_empirical_tail.png
- plot_F6b_strata_tail_compare.png
