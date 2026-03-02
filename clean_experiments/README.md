# Clean Experiments

This folder contains cleaned, runnable scripts for the canonical experiment set.
Generated artifacts under `clean_experiments/results/...` are local-only and are not versioned in this repository (except markdown reports).

## Program numbering (1-20)

Numbering below follows `research_programm_summary.csv`.

1. **A**: `experiment_A.py`
   - Gauge invariance of `Lambda = Re Tr(F rho)` and noncommutativity diagnostics.
2. **B / wave-1**: `experiment_wave1_user.py`
   - Commutator algebra, Hermitian curvature projection, spatial `Lambda_matter(x)`, and `Lambda_matter ~ sin(phi)`.
3. **D**: `experiment_D.py`
   - Full balance closure on `(t, x, mu)` with explicit vertical flux/source terms.
4. **E**: `experiment_E.py`
   - Coherence-driven vertical-rate diagnostics and predictor comparison vs `|Lambda|`.
5. **F**: `experiment_F.py`
   - Sinusoidal law check and discretization convergence.
6. **G**: `experiment_G.py`
   - Fixed-phase profile scan (`varphi = integral omega dmu`).
7. **G2 (toy-chain)**: `experiment_G2_toy_chain.py`
   - Clausius-regression scan over `epsilon`.
8. **G2 (single-qubit)**: `experiment_G2_single_qubit.py`
   - Clausius-regression profile scan.
9. **H1**: `experiment_H_holographic.py`
   - Exponential layer growth with holographic truncation.
10. **H2**: `experiment_I_continuum_conservation.py`
   - Continuum extrapolation of conservation residuals.
11. **H3**: `experiment_J_berry_refinement.py`
   - Berry-phase refinement with robust angle sweep.
12. **H4**: `experiment_K_lambda_bridge.py`
   - `Lambda_matter` linkage to effective-source proxy.
13. **H4b**: `experiment_K2_theory_space_curvature.py`
   - Theory-space curvature diagnostics vs RG noncommutativity.
14. **H5**: `experiment_L_matter_fields.py`
   - Matter-field embedding (`fermion + gauge`) with Ward/continuity checks.
15. **M1**: `experiment_M_cosmo_flow.py`
   - Macro detectability of Lambda in ERA5 moisture-budget closure.
16. **M2**: `experiment_M_horizontal_vertical_compare.py`
   - Horizontal-vs-vertical Lambda consistency and placebo-style diagnostics.
17. **M3**: `experiment_M_land_ocean_split.py`, `experiment_M_land_ocean_noise_probe.py`
   - Land/ocean detectability split and noise-limit probing.
18. **O1**: `experiment_O_entropy_equilibrium.py`
   - Clausius baseline vs `+Lambda` thermodynamic test.
19. **O2**: `experiment_O_spatial_variance.py`, `experiment_O_lambda_spatial_viz.py`, `experiment_O_spatial_active_west.py`
   - Spatial macro-signal diagnostics and regional masking analyses.
20. **M4**: `experiment_M_lambda_falsification_tests.py`
   - Staged Lambda necessity falsification (S1/S2/S3).

Auxiliary (not in 1-20 numbering):

- `experiment_N_navier_stokes_budget.py`
- `experiment_N_followup_dual.py`
- `EXPERIMENT_N_DATA_MANIFEST.md`, `download_N_data_era5.py`, `download_N_data_merra2.py`

The full mapping table is in `clean_experiments/EXPERIMENT_NUMBERING.md`.
Stage-2 hypothesis-level protocol is in `clean_experiments/HYPOTHESIS_ROADMAP.md`.

## Experiment M decision policy

Experiment 15 uses a dual-threshold policy:

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
python clean_experiments/experiment_N_navier_stokes_budget.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_N_navier_stokes_budget
python clean_experiments/experiment_N_followup_dual.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_N_followup
python clean_experiments/experiment_O_entropy_equilibrium.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_O_entropy_equilibrium
python clean_experiments/experiment_O_spatial_variance.py --input-nc /path/to/wpwp_vertical_data.nc --m-timeseries-csv /path/to/experiment_M_timeseries.csv --m-summary-csv /path/to/experiment_M_summary.csv --m-mode-index-csv /path/to/experiment_M_mode_index.csv --outdir clean_experiments/results/experiment_O_spatial_variance
python clean_experiments/experiment_O_lambda_spatial_viz.py --input-nc /path/to/wpwp_vertical_data.nc --m-timeseries-csv /path/to/experiment_M_timeseries.csv --m-summary-csv /path/to/experiment_M_summary.csv --m-mode-index-csv /path/to/experiment_M_mode_index.csv --outdir clean_experiments/results/experiment_O_lambda_spatial_viz
python clean_experiments/experiment_O_spatial_active_west.py --input-csv clean_experiments/results/experiment_O_spatial_variance/spatial_point_metrics.csv --outdir clean_experiments/results/experiment_O_spatial_active_west
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
