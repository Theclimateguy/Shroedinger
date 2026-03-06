# Experiment F6b-panel Report: Heavy Tails of |Lambda_local(t,y,x)|

## Protocol
- panel object is reconstructed exactly as in O spatial pipeline: Lambda_local(t,y,x)
- x_min is selected strictly by minimum KS distance over percentile scan
- alpha is estimated by continuous-tail MLE
- model comparison uses LLR(Power-law vs Exponential)
- fitting sample caps: global=3000000, regime=1200000

## Global Best Fit
- xmin = 5.113722e-05
- alpha_emp = 3.306688
- alpha_pred(reference) = 1.540000
- KS distance = 0.025088
- dynamic range x_max/x_min = 15.269
- n_tail = 30000
- LLR(PL-Exp) = 1173.549385
- LLR p-value = 0.000000

## Strict PASS Criteria
- dynamic_range_ge_10: True
- llr_prefers_powerlaw: True
- alpha_in_universality_band_1p3_2p0: False
- PASS_ALL: False

## Stratified Diagnostics
- convective_heavier_or_better: True
- per-regime best fits are saved in `experiment_F6b_panel_best_fits.csv`

## Outputs
- experiment_F6b_panel_tail_metrics.csv
- experiment_F6b_panel_best_fits.csv
- experiment_F6b_panel_verdict.json
- plot_F6b_panel_empirical_tail.png
- plot_F6b_panel_strata_tail_compare.png
