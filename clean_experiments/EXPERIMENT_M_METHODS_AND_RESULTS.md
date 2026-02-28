# Experiment M: Methods, Calibration, and Latest Results

## Scope

Experiment M tests whether a structural interscale term `Lambda_struct(t)` improves closure of a moisture-balance residual beyond a baseline control model.

Core question:

- Does `ctrl + Lambda_struct` outperform `ctrl` out-of-sample on atmospheric reanalysis data?

## Dataset and input

- Source domain: WPWP (ERA5 subset).
- Time span: `2017-01-01` to `2019-12-31`, 6-hourly.
- Grid used in the calibrated run: `nt=4380`, `ny=81`, `nx=161`.
- Canonical input file:
  - `data/processed/wpwp_era5_2017_2019_experiment_M_input.nc`

Main variables:

- `iwv`, `ivt_u`, `ivt_v`, `precip`, `evap`, `u`, `v`, `temp`, `pressure`

## Methodological definition

### 1. Residual target

Experiment M now supports explicit residual construction modes:

- `component_zscore`: `z(dIWV/dt) + z(div IVT) + z(P-E)` (legacy mode)
- `physical_zscore`: `z(dIWV/dt + div IVT + (P-E))`
- `physical_raw`: `dIWV/dt + div IVT + (P-E)`

Calibrated runs use `physical_zscore`.

### 2. Scale coordinate and modes

- Scale bands: Fourier wavelength edges in km (default: `25,50,100,200,400,800,1600`).
- Scale coordinate: `mu = log2(L_ref / L_band_center)`.
- Per-band mode extraction from multivariate fields:
  - `iwv`, `ivt_u`, `ivt_v`, `p_minus_e`, `vorticity`.

### 3. Density matrices and structural term

- Rolling covariance per scale band yields `rho_mu(t)`.
- Optional covariance shrinkage:
  - `C <- (1-alpha) * C + alpha * diag(C)`, `alpha = cov_shrinkage`.
- Interscale noncommutativity proxy yields `lambda_mu(t)`.
- Coherence mode options:
  - `offdiag_fro`
  - `relative_offdiag_fro`
- Aggregate:
  - `Lambda_struct(t) = sum_mu w_mu(t) * coh_mu(t) * lambda_mu(t)`

## Validation protocol

- Blocked CV: `n_folds=6`.
- Permutation test with block shuffling:
  - default production evaluation: `n_perm=140`, `perm_block=24`.
- Stability diagnostics:
  - `lambda_sign_consistency`
  - stratum-level MAE gains via `strata_q=3`.

## Decision policy (fixed)

### Criterion 1: Theoretical Detection Threshold

- `min_mae_gain >= 0.002` (0.2%)
- `perm_p <= 0.05`
- `strata_positive_frac >= 0.8` (allows one stratum outlier)

Interpretation:

- sufficient to support theoretical detectability of nontrivial interscale physics (`F_{mu1,mu2} != 0`) and coherence contribution.

### Criterion 2: Engineering Impact Threshold

- `min_mae_gain >= 0.03` (3.0%)

Interpretation:

- threshold for heavy operational/production integration (Phase C).

## Calibration workflow

Three-stage calibration was run:

1. Random screening on a reduced time span (`search_results_screening.csv`).
2. Full-data comparison of top candidates (`calibration_full_compare.csv`).
3. Local sweep around the best region (`local_sweep_summary.csv`).

Best stable region:

- `residual_mode=physical_zscore`
- `coherence_mode=offdiag_fro`
- `n_modes_per_var=6`
- `window in [18,24]`
- `cov_shrinkage in [0.0,0.1]`
- `ridge_alpha=1e-6`

## Latest full-data results

### Baseline revised setup (v2)

- Folder: `clean_experiments/results/experiment_M_cosmo_flow_v2`
- Key metrics:
  - `oof_gain_frac = -0.002154`
  - `perm_p_value = 0.652482`
  - `strata_positive_frac = 0.0`

### Calibrated setup (v3)

- Folder: `clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated`
- Config:
  - `physical_zscore`, `offdiag_fro`, `n_modes_per_var=6`, `window=18`,
    `cov_shrinkage=0.1`, `ridge_alpha=1e-6`.
- Key metrics:
  - `oof_gain_frac = +0.003379`
  - `perm_p_value = 0.007092`
  - `lambda_sign_consistency = 1.0`
  - `strata_positive_frac = 1.0`

Decision under fixed policy:

- Passes Criterion 1 (theoretical detection).
- Does not pass Criterion 2 (engineering impact at 3% gain).

## Reproduction commands

Calibrated full-data run:

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

Criterion-1 pass-check run:

```bash
python clean_experiments/experiment_M_cosmo_flow.py \
  --input data/processed/wpwp_era5_2017_2019_experiment_M_input.nc \
  --out clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated_threshold \
  --residual-mode physical_zscore \
  --coherence-mode offdiag_fro \
  --n-modes-per-var 6 \
  --window 18 \
  --cov-shrinkage 0.1 \
  --ridge-alpha 1e-6 \
  --min-mae-gain 0.003 \
  --min-positive-strata-frac 0.67 \
  --n-perm 140
```

## Artifact pointers

- `clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated/experiment_M_summary.csv`
- `clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated_threshold/experiment_M_summary.csv`
- `clean_experiments/results/experiment_M_calibration/search_results_screening.csv`
- `clean_experiments/results/experiment_M_calibration_full/calibration_full_compare.csv`
- `clean_experiments/results/experiment_M_calibration_local/local_sweep_summary.csv`
