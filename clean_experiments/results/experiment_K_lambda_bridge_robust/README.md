# Experiment 12 (K) Robust Sweep

Robustness sweep for the Lambda bridge protocol under randomized hyperparameter settings.

## Run command

```bash
python clean_experiments/experiment_K_lambda_bridge_robust.py \
  --out clean_experiments/results/experiment_K_lambda_bridge_robust \
  --runs 4
```

## Key outputs

- `robustness_results.csv`: metrics for each robust run.
- `robustness_summary.csv`: aggregate robustness metrics.
- `worst_10_runs.csv`: lowest-scoring runs.
- `runs/run_*/`: full artifacts for each run.

## Latest robustness summary

From `robustness_summary.csv` (seed `20260226`, `n_runs=4`):

- `fraction_pass_all = 1.0`
- `median_test_r2_median = 0.356884`
- `min_test_r2_median = 0.313702`
- `median_perm_p = 0.008497`
- `max_perm_p = 0.009901`
- `min_boot_sign_consistency_mean = 0.665278`
