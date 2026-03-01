# Experiment M: Curated Residual-Closure Evidence for Lambda (March 2026)

## 1) Scope

This note keeps only Experiment M blocks that are both:

1. physically interpretable (`Lambda` is defined from multiscale atmospheric structure), and
2. validated by time-aware residual-closure tests (blocked CV and/or out-of-time test).

The goal is narrow: show where `Lambda` is visible through ML residual closure, and discard noisy or non-physical variants from the core claim.

## 2) Data and variables used in the matrices

Canonical atmospheric input:

- `data/processed/wpwp_era5_2017_2019_experiment_M_input.nc`
- `data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc` (for vertical-channel diagnostics)

Time/grid:

- 2017-01-01 to 2019-12-31 (6-hourly), `n_time=4380`
- WPWP domain grid `ny=81`, `nx=161`

Fields entering multiscale modal matrices:

- `iwv`
- `ivt_u`, `ivt_v`
- `p_minus_e = precip - evap`
- `vorticity = d(v)/dx - d(u)/dy`

Vertical extension fields (diagnostic branch):

- `temp_pl`, `q_pl`, `u_pl`, `v_pl`, `w_pl`

## 3) Formal definition of Lambda in this implementation

Code references:

- [experiment_M_cosmo_flow.py:615](/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_cosmo_flow.py:615)
- [experiment_M_cosmo_flow.py:755](/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_cosmo_flow.py:755)

### 3.1 Scale bands and modal vectors

Spatial Fourier masks define scale bands over wavelength edges (default):

- `25,50,100,200,400,800,1600` km (6 bands)

For each band `b` and time `t`, the script selects top-power Fourier cells per field (default `n_modes_per_var=6`) and concatenates them into a complex coefficient vector `a_{b,t}`.

### 3.2 Density matrix per scale

Within rolling window `W`:

- `C_{b,t} = (A_{b,t}^H A_{b,t}) / W`
- optional shrinkage: `C <- (1-alpha) C + alpha diag(C)`
- ridge stabilization: `C <- C + ridge*I`
- normalization: `rho_{b,t} = C_{b,t} / Tr(C_{b,t})`

Interpretation:

- diagonal of `rho`: normalized modal power partition,
- off-diagonal of `rho`: cross-mode coupling/coherence.

### 3.3 Coherence term

Per-band coherence proxy:

- `coh_{b,t} = ||offdiag(rho_{b,t})||_F` (`offdiag_fro`, used in accepted runs)
- optional relative normalization (`relative_offdiag_fro`)

Macro scaling branch:

- `coh_eff = max(coh + floor, 0)^power`
- blended with entropy-based diagonal-mixing proxy:
  `signal = blend * coh_eff + (1-blend) * diag_mix`

### 3.4 Interscale curvature proxy and Lambda

For neighbor bands `b` and `b+1`, define forward/reverse interscale maps from windowed modal matrices and build:

- `G_fwd = M_fwd M_fwd^H`
- `G_back = M_rev^H M_rev`
- commutator `Comm = G_fwd G_back - G_back G_fwd`
- Hermitian physical projector `F_phys = Hermitian(i * Comm)`

Per-band scalar:

- `lambda_mu[b,t] = Re Tr(F_phys * rho_{b,t})`

Aggregation:

- structural weights `w_{b,t}` from number of band-limited vorticity structures (`n_struct`)
- final `Lambda_struct(t) = sum_b w_{b,t} * signal_{b,t} * lambda_mu[b,t]`

## 4) Baseline physics (`ctrl`) and target residual

Code references:

- [experiment_M_cosmo_flow.py:540](/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_cosmo_flow.py:540)
- [experiment_M_cosmo_flow.py:1284](/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_cosmo_flow.py:1284)

Target residual series:

- `r(t) = <dIWV/dt + div(IVT) + (P-E)>_{x,y}`

Current accepted runs use:

- `y(t) = zscore(r(t))` (`residual_mode=physical_zscore`)

Baseline control:

- if `density` is present: domain-mean density,
- else ideal-gas proxy: `n(t) = <p/(k_B T)>_{x,y}`
- then `ctrl(t) = zscore(log(n(t)))`

This defines the "existing physics" baseline model.

## 5) ML closure test and gain definition

Code reference:

- [experiment_M_cosmo_flow.py:1399](/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_cosmo_flow.py:1399)

Base model:

- `y ~ ctrl`

Full model (core accepted branch):

- `y ~ ctrl + Lambda_struct`

Validation:

- blocked time CV (`folds=6`)
- ridge regression
- permutation test with block shuffle of non-ctrl features

Gain metric:

- `gain = (MAE_base - MAE_full) / MAE_base`

## 6) Curated experiment set (kept for physical claims)

### 6.1 Core detectability runs

1. `experiment_M_cosmo_flow_v3_calibrated`
   - `gain=0.003379`, `perm_p=0.007092`
2. `experiment_M_cosmo_flow_v4_macro_calibrated`
   - `gain=0.003412`, `perm_p=0.007092`

Interpretation: stable, small but significant residual-closure effect.

### 6.2 Horizontal/vertical consistency check

`experiment_M_horizontal_vertical_compare`

- `corr(Lambda_h, Lambda_v): r=0.99631, R^2=0.99263`
- single-feature models: near-identical gains
- combined model is worse (`0.002566`), consistent with redundant signal + multicollinearity

### 6.3 Extreme-regime diagnostics

`experiment_M_extremes_amplitude`

- pure P90 extreme-only slices are noisy and often negative in linear `ctrl+Lambda` form,
- non-extreme slice remains positive (`~0.0069`),
- amplitude grows in extremes (`|residual|` rises), but linear response changes regime.

This block is retained as diagnostic evidence of regime shift, not as standalone detection proof.

### 6.4 Anti-overfit calibrated regime model

`experiment_M_extremes_calibration`

- train: 2017-2018, test: 2019
- one-SE selection and strong ridge (`alpha=0.1`)
- accepted regime models:
  - `horizontal_regime`: test gain `0.140295` (CI95 `[0.064032, 0.206147]`)
  - `vertical_regime`: test gain `0.139717` (CI95 `[0.060624, 0.205027]`)

Ablation (`regime_no_lambda` vs `regime_with_lambda_*`):

- incremental lambda gain is small but positive with paired block-bootstrap CI above zero:
  - `+0.00696` (`lambda_v`)
  - `+0.00752` (`lambda_h`)

### 6.5 Quarterly rolling-origin robustness

`experiment_M_extremes_quarterly`

- each 2019 quarter tested using only prior history for training,
- regime models stay positive each quarter (`mean ~0.144`),
- global linear models remain near zero (`mean ~0.0032`).

## 7) Excluded from the core physical claim

1. `experiment_M_cosmo_flow_v2`
   - negative/non-significant.
2. `experiment_M_cosmo_flow_v4_vertical_entropy` (`feature_set=lambda_entropy_vertical` raw)
   - negative/non-significant in current formulation.
3. extreme-only linear subsets (`union_p90`, `intersection_p90`) as standalone claim
   - too noisy and regime-unstable without explicit regime calibration.

## 8) Reproducible command set (curated)

Horizontal/vertical comparison:

```bash
.venv/bin/python clean_experiments/experiment_M_horizontal_vertical_compare.py
```

Extreme amplitude diagnostics:

```bash
.venv/bin/python clean_experiments/experiment_M_extremes_amplitude.py
```

Anti-overfit calibration:

```bash
.venv/bin/python clean_experiments/experiment_M_extremes_calibration.py
```

Quarterly rolling-origin:

```bash
.venv/bin/python clean_experiments/experiment_M_extremes_quarterly.py
```

## 9) Artifact map

- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex`
- `clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated/`
- `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/`
- `clean_experiments/results/experiment_M_horizontal_vertical_compare/`
- `clean_experiments/results/experiment_M_extremes_amplitude/`
- `clean_experiments/results/experiment_M_extremes_calibration/`
- `clean_experiments/results/experiment_M_extremes_quarterly/`
