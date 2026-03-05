# Robust Validation Index

This directory stores lightweight documentation for robust validation artifacts.
Heavy generated outputs (CSV/JSON/PNG/NPZ) are produced locally and are not versioned by default.

0. `checklist_validations/` (cross-experiment checklist block)
   - Structural + wave-1 checklist validations used in the manuscript:
     unitarity/cocycle/isometry, Lindblad trace-preservation, classical limit,
     Bianchi identity, two-layer aggregate norm conservation, state-dependent
     source, exact sin-law, symmetrization reality, spatial modulation,
     mixed-state neutrality, and `Lambda_vac` linearity in `Delta alpha`.
   - See `checklist_summary.csv`, `classical_limit_scan.csv`,
     `two_layer_norm_timeseries.csv`.

1. `experiment_A_robust/` (Experiment 1)
   - Gauge invariance + noncommutativity checks.
   - See `robustness_grid_summary.csv`, `robustness_full_gauge_summary.csv`.
2. `experiment_B_wave1_robust/` (Experiment 2)
   - Wave-1 commutator + Hermitian projection + `Lambda_matter ~ sin(phi)` robustness sweep.
   - See `robustness_summary.csv`.
3. `experiment_D_robust/` (Experiment 3)
   - Full balance closure checks on `(t, x, mu)`.
   - See `combined_summary.csv`.
4. `experiment_E_robust/` (Experiment 4)
   - Coherence-vs-Lambda predictive power in coherence-driven rate regime.
   - See `robustness_summary.csv`.
5. `experiment_F_robust/` (Experiment 5)
   - Sinusoidal law robustness under random parameter sweeps.
   - See `robustness_summary.csv`.
6. `experiment_G_robust/` (Experiment 6)
   - Fixed-phase profile invariance robustness.
   - See `robustness_summary.csv`.
7. `experiment_G2_toy_chain_robust/` (Experiment 7)
   - Epsilon-scan robustness across seeds.
   - See `robustness_summary_by_eps.csv`, `robustness_summary_global.csv`.
8. `experiment_G2_single_qubit_robust/` (Experiment 8)
   - Clausius-regression profile ranking robustness.
   - See `robustness_summary.csv`.
9. `experiment_H_holographic_robust/` (Experiment 9)
   - Exponential layer-dimension growth with holographic truncation:
     regularization stability and Lindblad trace checks across random regimes.
   - See `robustness_summary.csv`.
10. `experiment_I_continuum_conservation_robust/` (Experiment 10)
   - Continuum extrapolation robustness for balance residual
     (`dt,dmu -> 0`) from microscopic evolution.
   - See `robustness_summary.csv`.
11. `experiment_J_berry_refinement_robust/` (Experiment 11)
   - Berry-phase refinement robustness across polar-angle scans and
     step-size refinement.
   - See `robustness_summary.csv`.
12. `experiment_K_lambda_bridge_robust/` (Experiment 12)
   - Bridge robustness for `Lambda_matter` to effective-`Lambda` proxy:
     repeated stratified holdout median-`R2`, permutation significance, and
     bootstrap sign-consistency diagnostics.
   - See `robustness_summary.csv`, `worst_10_runs.csv`.
13. `experiment_K2_theory_space_curvature_robust/` (Experiment 13)
   - Theory-space curvature robustness sweep:
     FS/QGT and response-metric geometry under random parameter regimes,
     with gauge-invariance and coarse-grid stability checks.
   - See `robustness_summary.csv`, `robustness_results.csv`.
14. `experiment_L_matter_fields_robust/` (Experiment 14)
   - Matter-field (`fermion + gauge`) embedding robustness:
     Ward commutator, local continuity residual, and total-charge drift.
   - See `robustness_summary.csv`, `robustness_results.csv`.
15. `experiment_M_cosmo_flow/` (Experiment 15)
   - Structural-scale cosmological-flow block on atmospheric fields:
     multiscale modal density matrices (`rho_mu`), interscale noncommutativity proxy,
     blocked-CV residual-closure diagnostics, and permutation significance.
   - See `experiment_M_summary.csv`, `experiment_M_timeseries.csv`,
     `experiment_M_splits.csv`, `experiment_M_permutation.csv`.
16. `experiment_M_cosmo_flow_robust/` (Experiment 15 robust)
   - Robustness sweeps for Experiment M across scale edges, window lengths,
     modal truncation, and flux scaling.
   - See `robustness_summary.csv`, `robustness_results.csv`.
17. `experiment_M_cosmo_flow_v3_calibrated/` (Experiment 15 calibrated)
   - Calibrated full-data run (best local configuration from parameter sweep):
     `residual_mode=physical_zscore`, `coherence_mode=offdiag_fro`,
     `n_modes_per_var=6`, `window=18`, `cov_shrinkage=0.1`, `ridge_alpha=1e-6`.
   - See `experiment_M_summary.csv`, `experiment_M_splits.csv`,
     `experiment_M_permutation.csv`, `experiment_M_strata.csv`.
18. `experiment_M_cosmo_flow_v3_calibrated_threshold/` (Experiment 15 calibrated, theoretical-detection threshold)
   - Same calibrated run with acceptance threshold set to
     `min_mae_gain=0.003` to evaluate theoretical-detection pass flags.
   - See `experiment_M_summary.csv`.
19. `experiment_M_calibration/`, `experiment_M_calibration_full/`, `experiment_M_calibration_local/`
   - Parameter-search artifacts for calibration.
   - See `search_results_screening.csv`, `calibration_full_compare.csv`,
     `local_sweep_summary.csv`.
20. (reserved)
   - Previously used for exploratory QFT-free-chain diagnostics.
   - Not part of the current canonical repository scope.
21. `experiment_M_cosmo_flow_v4_vertical_entropy/` (Experiment 15 diagnostic, excluded from core claim)
   - Raw vertical-entropy feature-set run (`lambda_entropy_vertical`).
   - Current status: non-significant/negative in core residual-closure metric.
   - See `experiment_M_summary.csv`.
22. `experiment_M_cosmo_flow_v4_macro_calibrated/` (Experiment 15 curated)
   - Macro-calibrated physically interpretable Lambda run (`feature_set=lambda_only`,
     `coherence_blend=0.6`, `coherence_floor=0.0`, `coherence_power=1.0`).
   - See `experiment_M_summary.csv`, `experiment_M_timeseries.csv`,
     `experiment_M_splits.csv`, `experiment_M_permutation.csv`.
23. `experiment_M_horizontal_vertical_compare/` (Experiment 15 curated analysis)
   - Consistency check between horizontal and vertical Lambda representations,
     plus combined-model and quartile-stratified diagnostics.
   - See `comparison_report.md`, `lambda_correlation_stats.csv`,
     `model_comparison.csv`, `quartile_gain_comparison.csv`,
     `artifact_diagnosis.json`.
24. `experiment_M_extremes_amplitude/` (Experiment 15 curated analysis)
   - Extreme-regime amplitude diagnostics (`union/intersection P90`, tail response,
     severity deciles).
   - See `extreme_amplitude_report.md`, `extreme_model_comparison.csv`,
     `extreme_amplitude_summary.csv`, `extreme_tail_response.csv`.
25. `experiment_M_extremes_calibration/` (Experiment 15 curated analysis)
   - Anti-overfit regime calibration (`train=2017-2018`, `test=2019`), one-SE
     model selection, out-of-time test, and lambda ablation.
   - See `calibration_report.md`, `selected_configs.csv`,
     `test_out_of_time_metrics.csv`, `calibration_ablation_report.md`.
26. `experiment_M_extremes_quarterly/` (Experiment 15 curated analysis)
   - Rolling-origin quarterly test over 2019 with train-only threshold estimation.
   - See `quarterly_report.md`, `quarterly_rolling_metrics.csv`,
     `quarterly_summary.csv`.
27. `experiment_N_navier_stokes_budget/` (Experiment 16)
   - Navier-Stokes moisture-budget closure with blocked-CV one-SE calibration,
     out-of-time 2019 test, permutation and bootstrap diagnostics.
   - See `test_metrics.csv`, `cv_stats.csv`, `selected_config.csv`,
     `quarterly_summary.csv`, `report.md`.
28. `experiment_N_followup/` (Experiment 16 follow-up)
   - Dual follow-up branches:
     `N11` source-proxy variant and `N12` localized multiscale Lambda variant.
   - See `experiment_N_followup_comparison.csv`,
     `experiment_N_followup_report.md`,
     `experiment_N11_source_proxy/test_metrics.csv`,
     `experiment_N12_localized_lambda/test_metrics.csv`.
29. `experiment_O_entropy_equilibrium/` (Experiment 17)
   - Clausius-consistent thermodynamic baseline (`dS_hor ~ (1/T_eff)dQ_in`)
     with Lambda correction and out-of-time diagnostics.
   - See `test_metrics.csv`, `cv_stats.csv`, `regional_metrics.csv`,
     `quarterly_summary.csv`, `report.md`.
30. `experiment_O_spatial_variance/` (Experiment 17 spatial decomposition)
   - Per-gridpoint baseline/full regressions, gain map diagnostics,
     convective-climatology correlation, and land/ocean stratification.
   - See `summary_metrics.csv`, `spatial_point_metrics.csv`, `report.md`.
31. `experiment_O_lambda_spatial_viz/` (Experiment 17 Lambda spatial maps)
   - Spatial visualization of reconstructed `Lambda_local(t,y,x)`:
     mean/std/abs maps, local corr maps, land-mask and west/east aggregates.
   - See `lambda_spatial_summary.csv`, `lambda_domain_timeseries.csv`, `report.md`.
32. `experiment_O_spatial_active_west/` (Experiment 17 masked subsets)
   - Masked diagnostics for `active`, `west`, `active_west` and land/ocean subsets
     to test whether quiet zones dilute the Lambda signal.
   - See `masked_summary.csv`, `top_hotspots_active_west.csv`, `report.md`.
33. `experiment_M_lambda_falsification_tests/` (M4 staged falsification suite)
   - Lambda necessity checks in three stages:
     `S1` placebo-mu scale permutation, `S2` `F^comm` scalar control,
     `S3` information-criteria necessity test (AIC/BIC).
   - See `falsification_report.md`, `placebo_mu_metrics.csv`,
     `comm_control_metrics.csv`, `info_criteria.csv`,
     `lambda_summary_real_placebo_comm.csv`.
34. `experiment_M_land_ocean_split/` (Experiment 15 surface split)
   - Same calibrated M setup, evaluated separately over land and ocean masks
     with blocked-CV and permutation diagnostics.
   - See `surface_summary.csv`, `surface_comparison.json`,
     `surface_splits.csv`, `surface_permutation.csv`, `report.md`.
35. `experiment_M_land_ocean_spatial_viz/` (Experiment 15 article maps)
   - Train/test local spatial decomposition (`train<=2018`, `test=2019`)
     for `gain_map = R2_full-R2_base`, `beta_lambda_map`, and land/ocean
     gain distributions for manuscript figures.
   - See `spatial_summary.csv`, `spatial_point_metrics.csv`,
     `plot_spatial_gain_map.png`, `plot_spatial_beta_lambda_map.png`, `report.md`.
36. `experiment_M_land_ocean_noise_probe/` (Experiment 15 detectability diagnostics)
   - Land/ocean target-noise probe across target variants
     (`residual_full`, `residual_no_pe`, `p_minus_e_only`, smoothed residual)
     to interpret closure detectability limits.
   - See `noise_probe_metrics.csv`, `noise_probe_components.csv`,
     `noise_probe_verdict.json`, `plot_noise_probe_gains.png`, `report.md`.
37. `experiment_F1_fractal_emergence/` (F1 follow-up)
   - Balance-driven fractal emergence scan in two-scale GKSL setting over
     `epsilon in [0.01, 10]`.
   - See `report.md`.
38. `experiment_F2_scale_covariance/` (F2 follow-up)
   - Scale-covariant section test near `epsilon*` with RG-trace prediction check.
   - See `report.md`.
39. `experiment_F3_lambda_fractal_bridge/` (F3 follow-up)
   - Operational bridge between `Lambda_matter` and excess fractal dimension
     (`D_f - d_top`) with regression diagnostics.
   - See `report.md`.
40. `experiment_F4_holonomy_fractal_encoder/` (F4 exploratory baseline)
   - Holonomy-path ordering diagnostics as fractal encoder baseline.
   - See `report.md`.
41. `experiment_F4b_independent_holonomy_ablation/` (F4 independent final)
   - Independent holonomy test with required ablations, bootstrap on `epsilon*`,
     gauge robustness, and failed-seed diagnostics.
   - See `report.md`.
42. `experiment_F5_lambda_struct_fractal_era5/` (F5 on ERA5/WPWP)
   - Structural-scale `Lambda` detectability and multiscale surrogate linkage
     with placebo/commutative falsifications.
   - See `report.md`.
43. `experiment_F5_spatial_fractal_maps/` (F5 spatial follow-up)
   - Spatial maps for PSD/variogram/composite fractal deltas conditioned on
     high/low `|Lambda|` regimes.
   - See `report.md`.
44. `experiment_F6_soc_avalanches/` (F6 SOC signature)
   - Avalanche-size power-law diagnostics for `DeltaLambda_coh` with
     `alpha_pred = 1 + 1/y_rel` coupling to F2.
   - See `report.md`.
45. `experiment_F6_soc_robustness/` (F6 robustness addendum)
   - Sensitivity diagnostics for `transfer_fraction` and `cap_power` with
     on-critical and off-critical checks.
   - See `report.md`.
46. `experiment_F_series_2026_03_06/` (F1-F6 consolidated package)
   - One-page consolidated summary of all March 2026 F-series runs
     and manuscript-facing conclusions/caveats.
   - See `report.md`.
