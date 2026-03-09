# Robust Validation Index

This directory stores lightweight documentation for robust validation artifacts.
Heavy generated outputs (CSV/JSON/PNG/NPZ) are produced locally and are not versioned by default.

Canonical numbering is alphanumeric and split by domain:

- `TOY_MODEL`: `T01 ... T20`
- `ATMOSPHERE_DATA`: `A01 ... A15`

## Cross-Experiment Checks

- `checklist_validations/`
  - Structural + wave-1 checklist validations used in the manuscript:
    unitarity/cocycle/isometry, Lindblad trace-preservation, classical limit,
    Bianchi identity, two-layer aggregate norm conservation, state-dependent
    source, exact sin-law, symmetrization reality, spatial modulation,
    mixed-state neutrality, and `Lambda_vac` linearity in `Delta alpha`.
  - See `checklist_summary.csv`, `classical_limit_scan.csv`,
    `two_layer_norm_timeseries.csv`.

## TOY_MODEL Block

- `[T01]` `experiment_A_robust/`
  - Gauge invariance + noncommutativity checks.
- `[T02]` `experiment_B_wave1_robust/`
  - Wave-1 commutator + Hermitian projection + `Lambda_matter ~ sin(phi)` robustness sweep.
- `[T03]` `experiment_D_robust/`
  - Full balance closure checks on `(t, x, mu)`.
- `[T04]` `experiment_E_robust/`
  - Coherence-vs-Lambda predictive power in coherence-driven rate regime.
- `[T05]` `experiment_F_robust/`
  - Sinusoidal law robustness under random parameter sweeps.
- `[T06]` `experiment_G_robust/`
  - Fixed-phase profile invariance robustness.
- `[T07]` `experiment_G2_toy_chain_robust/`
  - Epsilon-scan robustness across seeds.
- `[T08]` `experiment_G2_single_qubit_robust/`
  - Clausius-regression profile ranking robustness.
- `[T09]` `experiment_H_holographic_robust/`
  - Exponential layer-dimension growth with holographic truncation.
- `[T10]` `experiment_I_continuum_conservation_robust/`
  - Continuum extrapolation robustness for conservation residuals.
- `[T11]` `experiment_J_berry_refinement_robust/`
  - Berry-phase refinement robustness across polar-angle scans.
- `[T12]` `experiment_K_lambda_bridge_robust/`
  - Bridge robustness for `Lambda_matter` to effective-`Lambda` proxy.
- `[T13]` `experiment_K2_theory_space_curvature_robust/`
  - Theory-space curvature robustness sweep.
- `[T14]` `experiment_L_matter_fields_robust/`
  - Matter-field (`fermion + gauge`) embedding robustness.
- `[T15]` `experiment_F1_fractal_emergence/`
  - Balance-driven fractal emergence scan.
- `[T16]` `experiment_F2_scale_covariance/`
  - Scale-covariant section test near `epsilon*`.
- `[T17]` `experiment_F3_lambda_fractal_bridge/`
  - `Lambda_matter` to excess fractal-dimension bridge.
- `[T18]` `experiment_F4_holonomy_fractal_encoder/`, `experiment_F4b_independent_holonomy_ablation/`
  - Holonomy/path-ordering encoder with independent ablation finalization.
- `[T19]` `experiment_F6_soc_avalanches/`, `experiment_F6_soc_robustness/`
  - SOC avalanche signature and robustness diagnostics.
- `[T20]` `experiment_scale_gravity_einstein_box/`
  - Local synthetic Einstein-in-a-box closure check (`Lambda ~ Pi`) in forcing/inertial/dissipation zones.

## ATMOSPHERE_DATA Block

- `[A01]` `experiment_M_cosmo_flow/`, `experiment_M_cosmo_flow_robust/`, `experiment_M_cosmo_flow_v3_calibrated/`, `experiment_M_cosmo_flow_v3_calibrated_threshold/`, `experiment_M_cosmo_flow_v4_macro_calibrated/`, `experiment_M_cosmo_flow_v4_vertical_entropy/`, `experiment_M_calibration/`, `experiment_M_calibration_full/`, `experiment_M_calibration_local/`, `experiment_M_extremes_amplitude/`, `experiment_M_extremes_calibration/`, `experiment_M_extremes_quarterly/`
  - Structural-scale Lambda detectability core, calibration, extremes and curated variants.
- `[A02]` `experiment_M_horizontal_vertical_compare/`
  - Horizontal/vertical consistency and quartile diagnostics.
- `[A03]` `experiment_M_land_ocean_split/`, `experiment_M_land_ocean_noise_probe/`, `experiment_M_land_ocean_spatial_viz/`
  - Land/ocean split, detectability noise probe, and map diagnostics.
- `[A04]` `experiment_O_entropy_equilibrium/`
  - Clausius baseline vs Lambda correction (thermodynamic test).
- `[A05]` `experiment_O_spatial_variance/`, `experiment_O_lambda_spatial_viz/`, `experiment_O_spatial_active_west/`
  - Spatial decomposition of macro-signal detectability.
  - A05 scale-space continuation:
    - `experiment_P1_spatial_occupancy_cascade/`
    - `experiment_P2_theory_bridge_ablation/`
    - `experiment_P2_noncommuting_coarse_graining_calibrated/`
    - `realpilot_2024_p2dense_calibrated/`
    - `experiment_P2_noncommuting_coarse_graining_dense_calibrated/`
    - `experiment_P2_l8_diagnostic_block/`
    - `experiment_P2_memory/`
    - `experiment_P2_memory_x_viz/`
    - `experiment_P2_memory_geo_viz/`
    - `experiment_P2_memory_gksl_cptp/`
- `[A06]` `experiment_M_lambda_falsification_tests/`
  - Staged Lambda necessity falsification (S1/S2/S3).
- `[A07]` `experiment_F5_lambda_struct_fractal_era5/`, `experiment_F5_spatial_fractal_maps/`
  - Structural Lambda signal in ERA5/WPWP and fractal-surrogate spatial maps.
- `[A08]` `experiment_F6b_era5_heavy_tails/`, `experiment_F6b_era5_heavy_tails_panel/`
  - Strict Clauset/Newman heavy-tail tests on global and panel Lambda observables.
- `[A09]` `experiment_F6c_clustered_subspace_tails/`
  - Cluster-conditioned heavy-tail fits with anti-overfit protocol.
- `[A10]` `experiment_F6c_spatial_panel_viz/`
  - Patch-wise spatial alpha maps and strict SOC candidate masks.
- `[A11]` `experiment_M_phi_only_strict_causal2019_train2019_test2020_ext2021_perm1999/`, `experiment_M_gksl_hybrid_strict_2017_2020_causal2018_locked_raw/`
  - Strict chronological `Phi`-over-`Lambda` transfer check endpoint pair.
- `[A12]` `experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2/`
  - Strict halo-boundary core-only closure endpoint.
- `[A13]` `experiment_M_halo_boundary_widthscan_w0_causal2019_train2019_test2020_ext2021/`, `experiment_M_halo_boundary_widthscan_w4_causal2019_train2019_test2020_ext2021/`, `experiment_M_halo_boundary_widthscan_w6_causal2019_train2019_test2020_ext2021/`, `experiment_M_halo_boundary_widthscan_w8_causal2019_train2019_test2020_ext2021/`, `experiment_M_halo_boundary_widthscan_w10_causal2019_train2019_test2020_ext2021/`
  - Preregistered halo-width scan under fixed-core strict protocol.
- `[A14]` `experiment_M_halo_boundary_falsify_remote_w4_causal2019_train2019_test2020_ext2021/`, `experiment_M_halo_boundary_falsify_misaligned_w4_causal2019_train2019_test2020_ext2021/`
  - Halo-physics falsification controls (`local` vs `remote` vs `misaligned`).
- `[A15]` `experiment_scale_gravity_einstein_box_era/`
  - Local ERA5 atmospheric-column Einstein-box test (`Lambda ~ Pi`) with inertial pass criteria.

## Atmosphere Extensions Outside Canonical A01-A15

- `experiment_scale_gravity_einstein_box_era_window_scan/`
  - Time-window sensitivity scan for the ERA5 box closure relation.
- `experiment_N_navier_stokes_budget/`
  - Navier-Stokes moisture-budget closure with blocked-CV and out-of-time checks.
- `experiment_N_followup/`
  - `N11` source-proxy and `N12` localized-Lambda follow-up branches.
- `experiment_M_gksl_hybrid_bridge*/`
  - Non-strict GKSL bridge variants retained as auxiliary diagnostics.
- `experiment_M_gksl_hybrid_strict_locked_raw/`, `experiment_M_gksl_hybrid_strict_screened/`, `experiment_M_gksl_hybrid_strict_2017_2020_causal2018_nested/`, `experiment_M_gksl_hybrid_strict_2017_2020_noncausal_locked_raw/`
  - Additional strict hybrid variants outside canonical endpoint pair.
- `experiment_M_phi_only_strict_causal2018_locked_raw_perm1999/`, `experiment_M_phi_only_strict_causal2018_nested_perm1999/`
  - Additional phi-only strict variants outside canonical endpoint pair.
- `experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021/`
  - Pre-final halo strict run retained for reproducibility.
- `EXPERIMENT_M_GKSL_TRANSFER_HALO_BRANCH_2026_03_07.md`
  - Full March 7, 2026 run inventory and branch-level interpretation.

## Consolidated Program Docs

- `experiment_F_series_2026_03_06/`
  - Consolidated summary for F-series block and manuscript caveats.
- `research_programm_summary.csv`
  - Canonical experiment landscape table (`TOY_MODEL` + `ATMOSPHERE_DATA`).
