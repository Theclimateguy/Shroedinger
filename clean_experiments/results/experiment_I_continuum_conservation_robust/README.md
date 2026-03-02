# Experiment 10 (I) Robust Sweep

This folder stores robust validation runs for:

- `clean_experiments/experiment_I_continuum_conservation_robust.py`

## Reproduce

```bash
python clean_experiments/experiment_I_continuum_conservation_robust.py \
  --cases 12 \
  --out clean_experiments/results/experiment_I_continuum_conservation_robust
```

## Main artifacts

- `robustness_results.csv`
- `robustness_summary.csv`
- `worst_12_by_intercept.csv`
- `cases/case_*/experiment_I_extrapolation.csv`

## Headline from current run

- `fraction_pass_hard_tol = 0.833333` (`10/12`)
- `fraction_pass_scaled_tol = 1.0` (`12/12`)
- `fraction_pass_all = 1.0` (`12/12`, scaled criterion + fit quality)
- `median_abs_intercept = 2.827205e-03`
- `max_abs_intercept = 1.045188e-02`
- `median_fit_r2 = 0.870574`
- `min_fit_r2 = 0.603078`
