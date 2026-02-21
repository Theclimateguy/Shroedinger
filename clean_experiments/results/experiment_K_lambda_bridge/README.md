# Experiment 12 (K): Lambda Bridge

Bridge validation between wave-1 matter-source observables and an effective thermodynamic Lambda proxy.

## Run command

```bash
python clean_experiments/experiment_K_lambda_bridge.py \
  --out clean_experiments/results/experiment_K_lambda_bridge
```

## Key outputs

- `experiment_K_dataset.csv`: per-case coupled wave1 + G2 observables.
- `experiment_K_coefficients.csv`: full-fit ridge coefficients.
- `experiment_K_splits.csv`: repeated stratified holdout diagnostics.
- `experiment_K_permutation.csv`: permutation null for median holdout `R2`.
- `experiment_K_bootstrap_coefficients.csv`: bootstrap coefficient diagnostics.
- `experiment_K_summary.csv`: headline metrics + pass flags.

## Latest headline metrics

From `experiment_K_summary.csv` (seed `20260225`):

- `test_r2_median = 0.454608`
- `test_r2_p25 = 0.376642`
- `perm_p_value = 0.007092`
- `boot_mean_sign_consistency_key = 0.968182`
- `pass_all = True`
