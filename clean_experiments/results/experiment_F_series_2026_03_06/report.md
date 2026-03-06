# F-Series Consolidated Report (F1-F6 + F6b/F6c stress tests)

Date: 2026-03-06

## Scope
- This report consolidates the March 2026 follow-up block: F1, F2, F3, F4/F4b, F5, F6.
- It also includes strict post-F6 stress-tests (F6b/F6c) on ERA5 Lambda tails.
- Canonical final path for F4 claims: **F4b** (independent setup with mandatory ablations).

## Headline Verdict
- F1: PASS
- F2: PASS
- F3: PASS
- F4 (exploratory baseline): PASS
- F4b (independent final): PASS
- F5 (ERA5/WPWP): PASS
- F6 (SOC): PASS
- F6 robustness addendum: completed and documented
- F6b (ERA5 heavy tails, strict Clauset/Newman): FAIL target alpha-band [1.3, 2.0]
- F6c (clustered subspace tails, anti-overfit): FAIL target alpha-band [1.5, 2.0]
- F6c spatial panel mapping: no local SOC-candidate patches

## F1 Fractal Emergence from Balance
- epsilon* (minimum vertical flow) = 1.00
- d_s(epsilon*) = 2.8258
- d_s(epsilon=0.01) = 1.0179
- d_s(epsilon=10) = 2.9524
- PASS_ALL = True
- Report: `clean_experiments/results/experiment_F1_fractal_emergence/report.md`

## F2 Scale Covariance of Section
- epsilon* from F1 = 1.00
- delta_covariance_max(epsilon*) = 4.150e-30
- R2 power law (epsilon*) = 1.000000
- Delta_measured(epsilon*) = 1.850000
- Delta_predicted(epsilon*) = 1.850000
- PASS_ALL = True
- Report: `clean_experiments/results/experiment_F2_scale_covariance/report.md`

## F3 Lambda-Fractal Bridge
- corr(Lambda_matter, D_f-d_top) = 0.923694
- corr(Lambda_coh, D_f-d_top) = 0.926734
- regression slope = 0.308032
- regression R2 = 0.853211
- p-value = 3.748e-04
- PASS_ALL = True
- Report: `clean_experiments/results/experiment_F3_lambda_fractal_bridge/report.md`

## F4/F4b Holonomy Block
- F4 exploratory baseline (kept for traceability):
  - corr(h_triangle, D_f-d_top) = 0.966411
  - corr(Delta_order, D_f-d_top) = 0.967375
  - PASS_ALL = True
  - Report: `clean_experiments/results/experiment_F4_holonomy_fractal_encoder/report.md`
- F4b independent final:
  - corr(delta_order, D_f-d_top), Pearson = 0.970675
  - corr(delta_order, D_f-d_top), Spearman = 0.936364
  - permutation p-value = 1.999600e-04
  - partial corr | log(epsilon) = 0.651518
  - epsilon* bootstrap 95% CI = [1.00, 1.15]
  - failed seed share = 0.220
  - PASS_ALL = True
  - Report: `clean_experiments/results/experiment_F4b_independent_holonomy_ablation/report.md`

## F5 ERA5/WPWP Structural-Scale Lambda
- oof_gain_frac = 0.003412
- perm_p_value = 0.007092
- strata_positive_frac = 1.000000
- surrogate pass map = {'fractal_psd_beta': True, 'fractal_variogram_slope': True}
- estimator agreement Spearman = 0.587298
- placebo-time degraded = True
- polynomial placebo degraded = True
- commutative control degraded = True
- PASS_ALL = True
- Report: `clean_experiments/results/experiment_F5_lambda_struct_fractal_era5/report.md`
- Spatial visualization report: `clean_experiments/results/experiment_F5_spatial_fractal_maps/report.md`

## F6 SOC Avalanche Signature
- epsilon* from F1 = 1.00
- y_rel from F2 = 1.850000
- alpha_pred = 1.540541
- alpha_measured = 1.559421
- alpha 95% seed interval = [1.533810, 1.591557]
- relative error = 1.226%
- fit R2 mean = 0.980464
- tail dynamic range (median) = 364.790
- PASS_ALL = True
- Main report: `clean_experiments/results/experiment_F6_soc_avalanches/report.md`
- Robustness addendum: `clean_experiments/results/experiment_F6_soc_robustness/report.md`

## F6b/F6c Stress Tests on ERA5 Lambda Tails
- F6b strict heavy-tail test (global time series):
  - alpha_emp = 2.906392
  - dynamic range = 15.464
  - LLR(PL-Exp) p-value = 0.009391
  - PASS_ALL = False (alpha outside [1.3, 2.0])
  - Report: `clean_experiments/results/experiment_F6b_era5_heavy_tails/report.md`
- F6b-panel strict heavy-tail test (`|Lambda_local(t,y,x)|`):
  - alpha_emp = 3.306688
  - dynamic range = 15.269
  - LLR(PL-Exp) p-value = 0.000000
  - PASS_ALL = False (alpha outside [1.3, 2.0])
  - Report: `clean_experiments/results/experiment_F6b_era5_heavy_tails_panel/report.md`
- F6c clustered subspace tails (predeclared anti-overfit protocol):
  - global alpha_mle = 3.161476
  - any_cluster_passes_corrected_strict = False
  - PASS_ALL = False
  - Report: `clean_experiments/results/experiment_F6c_clustered_subspace_tails/report.md`
- F6c spatial panel mapping (patch-wise strict fits):
  - patches fit = 300/300
  - alpha min / q10 / median = 2.7546 / 2.9790 / 3.2272
  - strict SOC-candidate patches = 0
  - Report: `clean_experiments/results/experiment_F6c_spatial_panel_viz/report.md`

## Updated Plan (Strict, no fit-to-target)
1. Event-gated tails on fixed physical masks only: deep-convection windows and high-fractal windows defined before fitting.
2. Conditional tails per physically interpretable regime (MJO phase / monsoon / land-ocean interface) with corrected multiple testing.
3. Keep Clauset-MLE+KS+LLR protocol frozen and report all null/negative outcomes explicitly.

## Caveats for Article Draft
- F4 interpretation in manuscript should use F4b as the independent result; keep F4 as exploratory baseline only.
- F5 passes detection/falsification thresholds, but dual-surrogate agreement is moderate; mention this explicitly.
- F6 is robust in-class near epsilon*=1, but alpha is not strictly invariant to transfer_fraction; claims should be phrased as local universality band, not exact invariance.
- F6b/F6c show that ERA5 structural Lambda tails are heavy but steeper than toy-SOC prediction under strict tests; keep this as a negative/diagnostic result, not as confirmation.

## Repro Command Snapshot
```bash
python clean_experiments/experiment_F1_fractal_emergence.py --out clean_experiments/results/experiment_F1_fractal_emergence
python clean_experiments/experiment_F2_scale_covariance.py --out clean_experiments/results/experiment_F2_scale_covariance
python clean_experiments/experiment_F3_lambda_fractal_bridge.py --out clean_experiments/results/experiment_F3_lambda_fractal_bridge
python clean_experiments/experiment_F4_holonomy_fractal_encoder.py --out clean_experiments/results/experiment_F4_holonomy_fractal_encoder
python clean_experiments/experiment_F4b_independent_holonomy_ablation.py --out clean_experiments/results/experiment_F4b_independent_holonomy_ablation
python clean_experiments/experiment_F5_lambda_struct_fractal_era5.py --out clean_experiments/results/experiment_F5_lambda_struct_fractal_era5
python clean_experiments/experiment_F5_spatial_fractal_maps.py --out clean_experiments/results/experiment_F5_spatial_fractal_maps
python clean_experiments/experiment_F6_soc_avalanches.py --out clean_experiments/results/experiment_F6_soc_avalanches
python clean_experiments/experiment_F6b_era5_heavy_tails.py --out clean_experiments/results/experiment_F6b_era5_heavy_tails
python clean_experiments/experiment_F6b_era5_heavy_tails_panel.py --out clean_experiments/results/experiment_F6b_era5_heavy_tails_panel
python clean_experiments/experiment_F6c_clustered_subspace_tails.py --out clean_experiments/results/experiment_F6c_clustered_subspace_tails
python clean_experiments/experiment_F6c_spatial_panel_viz.py --out clean_experiments/results/experiment_F6c_spatial_panel_viz
```
