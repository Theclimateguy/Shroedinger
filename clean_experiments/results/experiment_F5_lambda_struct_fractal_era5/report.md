# Experiment F5 (ERA5/WPWP): Structural-scale Lambda and Multiscale Surrogates

## Headline
- pass_all: True
- H1 (main M threshold): True
- H2 (at least one surrogate reproducible): True
- Condition 3 (placebo-time + commutative degradation): True

## H1 Main Detection (curated Experiment M)
- oof_gain_frac = 0.003412 (threshold >= 0.002000)
- perm_p_value = 0.007092 (threshold <= 0.050000)
- strata_positive_frac = 1.000000 (threshold >= 0.800000)

## H2 Multiscale Link
- estimator agreement Spearman(psd, variogram) = 0.587298
- estimator agreement mean |z_psd-z_vario| = 0.660963
- estimator agreement pass = False

Per-feature/target checks:
- fractal_psd_beta -> lambda_abs_mean: corr=0.300418, fold_sign_frac=0.833, oot_corr=0.339535, pass=True
- fractal_psd_beta -> oof_gain_block: corr=-0.015421, fold_sign_frac=0.167, oot_corr=0.102010, pass=False
- fractal_variogram_slope -> lambda_abs_mean: corr=0.347987, fold_sign_frac=1.000, oot_corr=0.261766, pass=True
- fractal_variogram_slope -> oof_gain_block: corr=0.037642, fold_sign_frac=0.500, oot_corr=-0.018475, pass=False

## Placebo/Falsification
- placebo-time: real_gain=0.003412, null_q95=0.001407, perm_p=0.007092, pass=True
- polynomial placebo: gain=-0.006486, perm_p=0.326241, pass=True
- commutative control: gain_comm=0.000000, corr_applicable=False, Var=0.000e+00, max_abs=0.000e+00, pass=True

## Land/Ocean + Noise Probe
- land gain=-0.000491, perm_p=0.879433
- ocean gain=0.001818, perm_p=0.021277
- noise residual_full land gain=-0.000491, perm_p=0.879433
- noise residual_full ocean gain=0.001818, perm_p=0.021277

## Output Files
- experiment_F5_summary_metrics.csv
- experiment_F5_test_metrics.csv
- experiment_F5_placebo_metrics.csv
- experiment_F5_surrogate_metrics.csv
- experiment_F5_block_metrics.csv
- experiment_F5_dataset.csv
- experiment_F5_verdict.json
