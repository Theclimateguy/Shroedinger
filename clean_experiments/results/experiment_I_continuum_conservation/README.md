# Experiment 10 (I): Continuum Conservation Extrapolation

This folder stores baseline outputs for:

- `clean_experiments/experiment_I_continuum_conservation.py`

## Reproduce

```bash
python clean_experiments/experiment_I_continuum_conservation.py \
  --out clean_experiments/results/experiment_I_continuum_conservation \
  --summary-only
```

## Main artifacts

- `experiment_I_case_summary.csv`
- `experiment_I_extrapolation.csv`
- (optional full table) `experiment_I_fd_balance_dataset.csv`

## Baseline headline (current run)

- `intercept_max_abs_residual = -2.317012e-03`
- `fit_r2_max_abs_residual = 0.911648`
- `max_residual_overall = 9.566557e-02`
- `intercept_scaled_tol = 7.653245e-03` (`max(3e-3, 0.08*max_residual)`)
- `pass_intercept_tol = True` and `pass_intercept_scaled_tol = True`
