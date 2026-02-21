# Experiment K2 (Theory-Space Curvature)

This folder stores artifacts for Experiment 13 (`K2`), which checks whether
toy-model theory-space geometry tracks RG noncommutativity.

## Files

- `experiment_K2_scan.csv`: per-angle scan with source spread, commutator norm,
  FS/QGT diagnostics, and response-metric curvature summaries.
- `experiment_K2_gauge_checks.csv`: random gauge-invariance checks.
- `experiment_K2_convergence.csv`: coarse-grid stability check of headline correlations.
- `experiment_K2_summary.csv`: one-row headline metrics and pass/fail flags.

## Headline metrics (latest run)

- `corr(source_spread, ||[A1,A2]||_F) = 0.993129`
- `corr(source_spread, |Omega_FS|) = 0.993129`
- `corr(source_spread, det G_resp) = 0.327163`
- max gauge diff (source) `= 8.33e-16`
- max gauge diff (response metric) `= 2.00e-13`
- max coarse-grid correlation delta `= 8.50e-03`
- `pass_all = True`
