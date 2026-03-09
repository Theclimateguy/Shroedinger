# Clean Experiments

This folder contains cleaned, runnable scripts for the canonical experiment set.
Generated artifacts under `clean_experiments/results/...` are local-only and are not versioned in this repository (except markdown reports).

## Program landscape (alphanumeric)

Canonical mapping is defined in `research_programm_summary.csv` and split into two blocks:

- `TOY_MODEL`: `T01 ... T20`
- `ATMOSPHERE_DATA`: `A01 ... A15`

### TOY_MODEL

- `T01` **A**: `experiment_A.py`
  - Gauge invariance of `Lambda = Re Tr(F rho)` and noncommutativity diagnostics.
- `T02` **B / wave-1**: `experiment_wave1_user.py`
  - Commutator algebra, Hermitian curvature projection, spatial `Lambda_matter(x)`, and `Lambda_matter ~ sin(phi)`.
- `T03` **D**: `experiment_D.py`
  - Full balance closure on `(t, x, mu)` with explicit vertical flux/source terms.
- `T04` **E**: `experiment_E.py`
  - Coherence-driven vertical-rate diagnostics and predictor comparison vs `|Lambda|`.
- `T05` **F**: `experiment_F.py`
  - Sinusoidal law check and discretization convergence.
- `T06` **G**: `experiment_G.py`
  - Fixed-phase profile scan (`varphi = integral omega dmu`).
- `T07` **G2 (toy-chain)**: `experiment_G2_toy_chain.py`
  - Clausius-regression scan over `epsilon`.
- `T08` **G2 (single-qubit)**: `experiment_G2_single_qubit.py`
  - Clausius-regression profile scan.
- `T09` **H1**: `experiment_H_holographic.py`
  - Exponential layer growth with holographic truncation.
- `T10` **H2**: `experiment_I_continuum_conservation.py`
  - Continuum extrapolation of conservation residuals.
- `T11` **H3**: `experiment_J_berry_refinement.py`
  - Berry-phase refinement with robust angle sweep.
- `T12` **H4**: `experiment_K_lambda_bridge.py`
  - `Lambda_matter` linkage to effective-source proxy.
- `T13` **H4b**: `experiment_K2_theory_space_curvature.py`
  - Theory-space curvature diagnostics vs RG noncommutativity.
- `T14` **H5**: `experiment_L_matter_fields.py`
  - Matter-field embedding (`fermion + gauge`) with Ward/continuity checks.
- `T15` **F1**: `experiment_F1_fractal_emergence.py`
  - Fractal emergence under epsilon-balance scan.
- `T16` **F2**: `experiment_F2_scale_covariance.py`
  - Scale-covariant section and RG exponent consistency.
- `T17` **F3**: `experiment_F3_lambda_fractal_bridge.py`
  - Geometric Lambda bridge to excess fractal dimension.
- `T18` **F4/F4b**: `experiment_F4_holonomy_fractal_encoder.py`, `experiment_F4b_independent_holonomy_ablation.py`
  - Holonomy/path-ordering encoding with independent ablations.
- `T19` **F6**: `experiment_F6_soc_avalanches.py`
  - Toy SOC avalanche heavy-tail diagnostics.
- `T20` **EIB-synth**: `experiment_scale_gravity_einstein_box.py`
  - Local synthetic Einstein-in-a-box closure test (`Lambda ~ Pi`) by scale regimes.

### ATMOSPHERE_DATA

- `A01` **M1**: `experiment_M_cosmo_flow.py`
  - Macro detectability of Lambda in ERA5 moisture-budget closure.
- `A02` **M2**: `experiment_M_horizontal_vertical_compare.py`
  - Horizontal-vs-vertical Lambda consistency and placebo-style diagnostics.
- `A03` **M3**: `experiment_M_land_ocean_split.py`, `experiment_M_land_ocean_noise_probe.py`
  - Land/ocean detectability split and noise-limit probing.
- `A04` **O1**: `experiment_O_entropy_equilibrium.py`
  - Clausius baseline vs `+Lambda` thermodynamic test.
- `A05` **O2**: `experiment_O_spatial_variance.py`, `experiment_O_lambda_spatial_viz.py`, `experiment_O_spatial_active_west.py`
  - Spatial macro-signal diagnostics and regional masking analyses.
  - Scale-space continuation (P1/P2): `experiment_P1_spatial_occupancy_cascade.py`, `experiment_P2_noncommuting_coarse_graining.py`, `experiment_P2_theory_bridge_ablation.py`, `run_p2_calibrated_dense_ingest.py`, `experiment_P2_l8_diagnostic_block.py`, `experiment_P2_memory.py`, `experiment_P2_memory_gksl_cptp.py`.
- `A06` **M4**: `experiment_M_lambda_falsification_tests.py`
  - Staged Lambda necessity falsification (S1/S2/S3).
- `A07` **F5**: `experiment_F5_lambda_struct_fractal_era5.py`, `experiment_F5_spatial_fractal_maps.py`
  - Structural Lambda on ERA5 and multiscale surrogate linkage.
- `A08` **F6b**: `experiment_F6b_era5_heavy_tails.py`, `experiment_F6b_era5_heavy_tails_panel.py`
  - Strict heavy-tail tests for `|Lambda_struct|` and panel `|Lambda_local|`.
- `A09` **F6c**: `experiment_F6c_clustered_subspace_tails.py`
  - Clustered subspace heavy-tail fits under anti-overfit protocol.
- `A10` **F6c-spatial**: `experiment_F6c_spatial_panel_viz.py`
  - Patch-wise spatial tail exponent mapping.
- `A11` **M5**: `experiment_M_gksl_hybrid_strict.py`
  - Strict chronological `Phi`-over-`Lambda` transfer check on ERA5 closure holdouts.
- `A12` **M6**: `experiment_M_halo_boundary_strict.py`
  - Strict core-only closure with adjacent halo boundary context.
- `A13` **M7**: `experiment_M_halo_boundary_strict.py`
  - Preregistered halo-width scan (`w=0,4,6,8,10`) under fixed core protocol.
- `A14` **M8**: `experiment_M_halo_boundary_strict.py`
  - Halo-physics falsification (`local` vs `remote` vs `misaligned` context).
- `A15` **EIB-ERA**: `experiment_scale_gravity_einstein_box_era.py`
  - Einstein-in-the-atmospheric-column box test on ERA5 (`Lambda ~ Pi`) with inertial pass criteria.

### A05 run-level log (scale-space count-geometry continuation, ATMOSPHERE_EXTENSION)

- `A05.R1_p1_spatial_occupancy_cascade`
  - P1-lite occupancy cascade on sparse panel.
  - Artifacts: `clean_experiments/results/experiment_P1_spatial_occupancy_cascade/`.
- `A05.R2_p2_theory_bridge_c009`
  - P2 noncommuting bridge with ablation-selected C009 (`[occ,sq,log,grad]=[1.5,1,1,1]`, `lambda_scale_power=0.5`, `decoherence_alpha=0.5`).
  - Artifacts: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated/`, `clean_experiments/results/experiment_P2_theory_bridge_ablation/`.
- `A05.R3_p2_dense_c009`
  - Dense intra-event transfer test for calibrated C009 (`240 rows / 16 events`).
  - Artifacts: `clean_experiments/results/realpilot_2024_p2dense_calibrated/`, `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/`.
- `A05.R4_p2_l8_resolution`
  - Narrow `l=8` diagnostics (matched-event, operator attribution, resolution sensitivity, regime split).
  - Finalization report: `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`.
- `A05.R5_p2_memory`
  - Retarded density-matrix closure on dense panel under full operators and standard threshold.
  - Finalization report: `clean_experiments/results/experiment_P2_memory/report.md`.
  - Supporting visual diagnostics: `clean_experiments/visualize_p2_memory_x_profiles.py`, `clean_experiments/visualize_p2_memory_geo_maps.py`.
  - Visualization reports: `clean_experiments/results/experiment_P2_memory_x_viz/report.md`, `clean_experiments/results/experiment_P2_memory_geo_viz/report.md`.
- `A05.R6_p2_memory_gksl_cptp`
  - Full effective GKSL/CPTP memory model on dense panel (`rho_t = Reset o GAD o Dephase o U(rho_{t-1})`).
  - Finalization report: `clean_experiments/results/experiment_P2_memory_gksl_cptp/report.md`.

Atmosphere extensions outside canonical `A01-A15`:

- `experiment_N_navier_stokes_budget.py`
- `experiment_N_followup_dual.py`
- `EXPERIMENT_N_DATA_MANIFEST.md`, `download_N_data_era5.py`, `download_N_data_merra2.py`
- M transfer/halo auxiliary branch (outside canonical A11-A14 endpoints):
  - `experiment_M_gksl_hybrid_bridge.py`
  - non-strict bridge reports:
    - `results/experiment_M_gksl_hybrid_bridge/report.md`
    - `results/experiment_M_gksl_hybrid_bridge_locked_raw/report.md`
    - `results/experiment_M_gksl_hybrid_bridge_locked_raw_v2/report.md`
    - `results/experiment_M_gksl_hybrid_bridge_screened/report.md`
    - `results/experiment_M_gksl_hybrid_bridge_screened_v2/report.md`
  - additional strict variants:
    - `results/experiment_M_gksl_hybrid_strict_locked_raw/report.md`
    - `results/experiment_M_gksl_hybrid_strict_screened/report.md`
    - `results/experiment_M_gksl_hybrid_strict_2017_2020_causal2018_nested/report.md`
    - `results/experiment_M_gksl_hybrid_strict_2017_2020_noncausal_locked_raw/report.md`
    - `results/experiment_M_phi_only_strict_causal2018_locked_raw_perm1999/report.md`
    - `results/experiment_M_phi_only_strict_causal2018_nested_perm1999/report.md`
  - pre-final halo strict auxiliary run:
    - `results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021/report.md`
  - full branch summary: `EXPERIMENT_M_GKSL_TRANSFER_HALO_BRANCH_2026_03_07.md`
- granular ingest pilot (MRMS + GOES):
  - `download_mrms.py`
  - `download_goes.py`
  - `build_mrms_goes_aligned_catalog.py`
  - `download_matched_windows.py`
  - `download_matched_parallel.py`
  - `run_ultralight_mrms_goes_pilot.py`
  - `pilot_events_template.csv`
  - `EXPERIMENT_GRANULAR_MRMS_GOES_INGEST.md`
- Consolidated report: `clean_experiments/results/experiment_F_series_2026_03_06/report.md`

The full mapping table is in `clean_experiments/EXPERIMENT_NUMBERING.md`.
Stage-2 hypothesis-level protocol is in `clean_experiments/HYPOTHESIS_ROADMAP.md`.
Canonical Group A runbook is in `clean_experiments/EXPERIMENT_A_ATMOSPHERE_PIPELINE.md`.

Numbering policy:

- Canonical experiment codes are locked to `T01..T20` and `A01..A15`.
- Continuation runs are logged as run-level IDs in `ветка` (for example `A05.R*`, `A07.R*`, `A11.E*`) under `ATMOSPHERE_EXTENSION`.

## M-realpilot v1 (frozen prereg protocol)

Frozen script:

- `clean_experiments/experiment_M_realpilot_v1_frozen.py`
- SHA256: `aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2`
- locked positive run: `clean_experiments/results/experiment_M_realpilot_v1_expanded_positive/`

Preregistered rules:

1. Do not modify model code, feature definitions, CV design, thresholds, or permutation settings.
2. Expand only event list (`clean_experiments/pilot_events_real_2024_us_convective_expanded_v1.csv` and future supersets).
3. Keep fixed evaluation setup:
   - target: `next_p95`
   - model: `Ridge(alpha=10.0)`
   - CV: leave-one-event-out
   - placebo tests: `time_shuffle` and `event_shuffle`
   - permutations: `n_perm=499`
4. Compare only headline metrics across runs:
   - `mae_baseline`, `mae_full`, `mean_mae_gain`
   - `perm_p_time_shuffle`, `perm_p_event_shuffle`
   - `event_positive_frac`, `PASS_ALL`
5. Any additional targets (e.g., `delta_p95`) are reported as secondary checks, not replacement primary endpoints.

Current extension assets:

- independent seasonal event list: `clean_experiments/pilot_events_realpilot_v1_independent_seasonal_2024.csv`
- independent geographic event list: `clean_experiments/pilot_events_realpilot_v1_independent_geographic_southwest_2024.csv`
- unified panel builder: `clean_experiments/build_realpilot_unified_panel.py`
- satellite component stability check: `clean_experiments/experiment_M_realpilot_satellite_component_stability.py`
- applicability-map builder: `clean_experiments/summarize_m_realpilot_applicability.py`
- regime-detection package builder: `clean_experiments/build_m_realpilot_regime_detection_package.py`
- independent seasonal frozen run: `clean_experiments/results/experiment_M_realpilot_v1_frozen_independent_seasonal_2024/`
- independent geographic frozen run: `clean_experiments/results/experiment_M_realpilot_v1_frozen_independent_geographic_southwest_2024/`
- baseline-vs-independent headline compare: `clean_experiments/results/experiment_M_realpilot_v1_independent_extension_compare/report.md`
- applicability map (season/region + ABI-only vs ABI+GLM): `clean_experiments/results/experiment_M_realpilot_applicability_map/report.md`
- full diagnostic perimeter + regime conclusion: `clean_experiments/results/experiment_M_realpilot_regime_detection_package/report.md`

## A01/M1 decision policy

`A01` (`experiment_M_cosmo_flow.py`) uses a dual-threshold policy:

1. **Theoretical Detection Threshold**
   - `min_mae_gain >= 0.002`
   - `perm_p <= 0.05`
   - `strata_positive_frac >= 0.8`
   - Used to validate detectability of the structural-scale physical effect.
2. **Engineering Impact Threshold**
   - `min_mae_gain >= 0.03`
   - Used to decide whether to pursue heavy production/operational integration.

Reference methodology and calibrated runs:

- `clean_experiments/EXPERIMENT_M_METHODS_AND_RESULTS.md`

## Experiment M curated residual-closure package

For claims about physically meaningful `Lambda` signal in atmospheric residual closure,
use only the curated M block documented in:

- `clean_experiments/EXPERIMENT_M_METHODS_AND_RESULTS.md`

Curated scripts:

- `experiment_M_cosmo_flow.py` (core Lambda construction and baseline closure)
- `experiment_M_horizontal_vertical_compare.py` (horizontal vs vertical consistency)
- `experiment_M_extremes_amplitude.py` (extreme-regime diagnostics)
- `experiment_M_extremes_calibration.py` (anti-overfit regime calibration, out-of-time 2019 test)
- `experiment_M_extremes_quarterly.py` (2019 rolling-origin quarterly robustness)
- `experiment_M_lambda_falsification_tests.py` (M4 staged falsification: S1 placebo-mu, S2 F^comm control, S3 AIC/BIC necessity check)
- `experiment_M_land_ocean_split.py` (same calibrated M setup split by underlying surface)
- `experiment_M_land_ocean_spatial_viz.py` (article-ready spatial gain and `beta_lambda` maps)
- `experiment_M_land_ocean_noise_probe.py` (target-noise diagnostics for land/ocean detectability)
- `EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex` (standalone write-up, intentionally separate from the main manuscript)

Curated M addendum (March 2026 finalization):

- falsification tests keep Lambda above artifact thresholds in polynomial/placebo controls;
- over-ocean closure signal stays positive/significant in the split setup;
- over-land closure gain is weak in this data regime and interpreted as detectability-limited.

Excluded from core physical claim in current form:

- raw vertical-entropy run with `feature_set=lambda_entropy_vertical`
- pure extreme-only linear slices without regime calibration

## Checklist validations (non-numbered)

- `checklist_validations.py`
  - Unified reproducibility checks for the checklist claims used in the manuscript:
    unitarity/cocycle/isometry, Lindblad trace preservation, classical limit,
    Bianchi identity, two-layer total-norm conservation, and wave-1
    (`Lambda_matter`) laws.
  - Outputs: `clean_experiments/results/checklist_validations/`.

## Experiment N data acquisition manifest

Data manifest and download helpers:

- `clean_experiments/EXPERIMENT_N_DATA_MANIFEST.md`
- `clean_experiments/download_N_data_era5.py`
- `clean_experiments/download_N_data_merra2.py`

Dry-run request planning:

```bash
python clean_experiments/download_N_data_era5.py \
  --start-date 2017-01-01 \
  --end-date 2019-12-31 \
  --outdir data/raw/era5_n13

python clean_experiments/download_N_data_merra2.py \
  --start-date 2017-01-01 \
  --end-date 2019-12-31 \
  --outdir data/raw/merra2_n13
```

Execute downloads:

```bash
python clean_experiments/download_N_data_era5.py --download
python clean_experiments/download_N_data_merra2.py --download

python clean_experiments/download_mrms.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-02 \
  --dry-run

python clean_experiments/download_goes.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-02 \
  --satellites G19 G18 \
  --products ABI-L2-CMIPF GLM-L2-LCFA \
  --abi-channels C02 C08 C13 \
  --dry-run

python clean_experiments/build_mrms_goes_aligned_catalog.py \
  --mrms-manifest manifests/mrms_manifest.csv \
  --goes-manifest manifests/goes_manifest.csv \
  --out aligned_catalog.parquet

python clean_experiments/download_matched_windows.py \
  --aligned-catalog aligned_catalog.parquet \
  --mrms-manifest manifests/mrms_manifest.csv \
  --goes-manifest manifests/goes_manifest.csv

python clean_experiments/run_ultralight_mrms_goes_pilot.py \
  --events-csv clean_experiments/pilot_events_template.csv \
  --stage manifest_only
```

## Quick smoke runs

```bash
python clean_experiments/experiment_A.py --out out/experiment_A
python clean_experiments/experiment_wave1_user.py --out out/experiment_B_wave1
python clean_experiments/experiment_D.py --samples 20 --out out/experiment_D
python clean_experiments/experiment_E.py --samples 60 --out out/experiment_E
python clean_experiments/experiment_F.py --out out/experiment_F
python clean_experiments/experiment_G.py --out out/experiment_G
python clean_experiments/experiment_G2_toy_chain.py --quick --out out/experiment_G2_toy_chain
python clean_experiments/experiment_G2_single_qubit.py --quick --out out/experiment_G2_single_qubit
python clean_experiments/experiment_H_holographic.py --out out/experiment_H_holographic
python clean_experiments/experiment_I_continuum_conservation.py --out out/experiment_I_continuum_conservation --summary-only
python clean_experiments/experiment_J_berry_refinement.py --out out/experiment_J_berry_refinement
python clean_experiments/experiment_K_lambda_bridge.py --out clean_experiments/results/experiment_K_lambda_bridge
python clean_experiments/experiment_K2_theory_space_curvature.py --out clean_experiments/results/experiment_K2_theory_space_curvature
python clean_experiments/experiment_L_matter_fields.py --out clean_experiments/results/experiment_L_matter_fields
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow
python clean_experiments/experiment_M_horizontal_vertical_compare.py
python clean_experiments/experiment_M_extremes_amplitude.py
python clean_experiments/experiment_M_extremes_calibration.py
python clean_experiments/experiment_M_extremes_quarterly.py
python clean_experiments/experiment_M_lambda_falsification_tests.py
python clean_experiments/experiment_M_land_ocean_split.py
python clean_experiments/experiment_M_land_ocean_spatial_viz.py
python clean_experiments/experiment_M_land_ocean_noise_probe.py
python clean_experiments/experiment_M_halo_boundary_strict.py --input-nc data/processed/wpwp_era5_2017_2021_experiment_M_input.nc --outdir clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2 --train-end-year 2019 --test-year 2020 --external-year 2021 --halo-width-cells 8 --fine-band-idx 0 --coarse-band-idx 1 --n-perm 1999
python clean_experiments/experiment_F1_fractal_emergence.py --out clean_experiments/results/experiment_F1_fractal_emergence
python clean_experiments/experiment_F2_scale_covariance.py --out clean_experiments/results/experiment_F2_scale_covariance
python clean_experiments/experiment_F3_lambda_fractal_bridge.py --out clean_experiments/results/experiment_F3_lambda_fractal_bridge
python clean_experiments/experiment_F4_holonomy_fractal_encoder.py --out clean_experiments/results/experiment_F4_holonomy_fractal_encoder
python clean_experiments/experiment_F4b_independent_holonomy_ablation.py --out clean_experiments/results/experiment_F4b_independent_holonomy_ablation
python clean_experiments/experiment_F5_lambda_struct_fractal_era5.py --out clean_experiments/results/experiment_F5_lambda_struct_fractal_era5
python clean_experiments/experiment_F5_spatial_fractal_maps.py --out clean_experiments/results/experiment_F5_spatial_fractal_maps
python clean_experiments/experiment_F6_soc_avalanches.py --out clean_experiments/results/experiment_F6_soc_avalanches
python clean_experiments/experiment_N_navier_stokes_budget.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_N_navier_stokes_budget
python clean_experiments/experiment_N_followup_dual.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_N_followup
python clean_experiments/experiment_O_entropy_equilibrium.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_O_entropy_equilibrium
python clean_experiments/experiment_O_spatial_variance.py --input-nc /path/to/wpwp_vertical_data.nc --m-timeseries-csv /path/to/experiment_M_timeseries.csv --m-summary-csv /path/to/experiment_M_summary.csv --m-mode-index-csv /path/to/experiment_M_mode_index.csv --outdir clean_experiments/results/experiment_O_spatial_variance
python clean_experiments/experiment_O_lambda_spatial_viz.py --input-nc /path/to/wpwp_vertical_data.nc --m-timeseries-csv /path/to/experiment_M_timeseries.csv --m-summary-csv /path/to/experiment_M_summary.csv --m-mode-index-csv /path/to/experiment_M_mode_index.csv --outdir clean_experiments/results/experiment_O_lambda_spatial_viz
python clean_experiments/experiment_O_spatial_active_west.py --input-csv clean_experiments/results/experiment_O_spatial_variance/spatial_point_metrics.csv --outdir clean_experiments/results/experiment_O_spatial_active_west
python clean_experiments/experiment_P1_spatial_occupancy_cascade.py --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv --outdir clean_experiments/results/experiment_P1_spatial_occupancy_cascade --scales-cells 8 16 32 --mrms-downsample 16 --mrms-threshold 3.0 --n-perm 49
python clean_experiments/experiment_P2_theory_bridge_ablation.py --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv --outdir clean_experiments/results/experiment_P2_theory_bridge_ablation --scales-cells 8 16 32 --mrms-downsample 16 --mrms-threshold 3.0 --top-k 4 --n-perm-final 49
python clean_experiments/experiment_P2_noncommuting_coarse_graining.py --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv --outdir clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated --scales-cells 8 16 32 --mrms-downsample 16 --mrms-threshold 3.0 --lambda-weights 1.5 1.0 1.0 1.0 --lambda-scale-power 0.5 --decoherence-alpha 0.5 --n-perm 49
python clean_experiments/run_p2_calibrated_dense_ingest.py --top-events 16 --context-hours 6 --budget-gb 50
python clean_experiments/experiment_P2_l8_diagnostic_block.py --outdir clean_experiments/results/experiment_P2_l8_diagnostic_block
python clean_experiments/experiment_wave1_robust.py --cases 40 --out clean_experiments/results/experiment_B_wave1_robust
python clean_experiments/experiment_H_holographic_robust.py --cases 24 --out clean_experiments/results/experiment_H_holographic_robust
python clean_experiments/experiment_I_continuum_conservation_robust.py --cases 12 --out clean_experiments/results/experiment_I_continuum_conservation_robust
python clean_experiments/experiment_J_berry_refinement_robust.py --out clean_experiments/results/experiment_J_berry_refinement_robust
python clean_experiments/experiment_K_lambda_bridge_robust.py --out clean_experiments/results/experiment_K_lambda_bridge_robust --runs 4
python clean_experiments/experiment_K2_theory_space_curvature_robust.py --out clean_experiments/results/experiment_K2_theory_space_curvature_robust --cases 12
python clean_experiments/experiment_L_matter_fields_robust.py --out clean_experiments/results/experiment_L_matter_fields_robust --cases 10
python clean_experiments/experiment_M_cosmo_flow_robust.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow_robust --cases 8
python clean_experiments/checklist_validations.py --out clean_experiments/results/checklist_validations
```

Calibrated Experiment M run used in the latest results package:

```bash
python clean_experiments/experiment_M_cosmo_flow.py \
  --input data/processed/wpwp_era5_2017_2019_experiment_M_input.nc \
  --out clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated \
  --residual-mode physical_zscore \
  --coherence-mode offdiag_fro \
  --n-modes-per-var 6 \
  --window 18 \
  --cov-shrinkage 0.1 \
  --ridge-alpha 1e-6 \
  --min-positive-strata-frac 0.67 \
  --n-perm 140
```

Extended Experiment M run with entropy-curvature and vertical pressure-level channels:

```bash
python clean_experiments/experiment_M_cosmo_flow.py \
  --input data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc \
  --out clean_experiments/results/experiment_M_cosmo_flow_v4_vertical_entropy \
  --residual-mode physical_zscore \
  --coherence-mode offdiag_fro \
  --feature-set lambda_entropy_vertical \
  --n-modes-per-var 6 \
  --window 18 \
  --cov-shrinkage 0.1 \
  --ridge-alpha 1e-6 \
  --n-perm 140
```
