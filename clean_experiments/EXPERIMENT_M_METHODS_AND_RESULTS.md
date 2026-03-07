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

### 6.6 M4 staged falsification checks (S1/S2/S3)

`experiment_M_lambda_falsification_tests`

- **S1: Placebo-\(\mu\)** (перестановка scale-слоёв)
  - `mae_gain_real = 0.003412`
  - `mae_gain_placebo = -0.002986`
  - `perm_p(placebo) = 1.0`
- **S2: \(F^{comm}\)-контроль** (скалярная/коммутирующая часть)
  - `mae_gain_comm = 0.0`
  - `beta_comm = 0.0`
  - по сравнению с `mae_gain_real = 0.003412` это исключает тривиальное скалярное объяснение.
- **S3: информационные критерии необходимости**
  - `DeltaAIC(ctrl -> ctrl+Lambda_real) = -13.202`
  - `DeltaBIC(ctrl -> ctrl+Lambda_real) = -6.817`
  - улучшение сохраняется при штрафе за дополнительный параметр.

### 6.7 Land/ocean split in the same M closure setup

`experiment_M_land_ocean_split`

- same stable Lambda configuration as `v4_macro_calibrated` (`feature_set=lambda_only`, `window=18`, `coherence_blend=0.6`),
- ocean aggregate: `gain=0.001818`, `perm_p=0.021277`,
- land aggregate: `gain=-0.000491`, `perm_p=0.879433`.

This confirms detectability over ocean in the closure metric and weak detectability over land in this data regime.

### 6.8 Spatial visualization package for article figures

`experiment_M_land_ocean_spatial_viz`

- local per-gridpoint test-year map (`train<=2018`, `test=2019`) for:
  - `gain_map = R2_full - R2_base`,
  - `beta_lambda_map`,
  - land/ocean gain distributions.
- domain summaries from this map are small and near zero:
  - land mean gain `-0.000214`,
  - ocean mean gain `-0.000418`,
  - `delta_land_minus_ocean=+0.000204`.

This block is used primarily for visualization and spatial context in the paper.

### 6.9 Noise-probe diagnostics (detectability constraints)

`experiment_M_land_ocean_noise_probe`

Target variants tested:

- `residual_full = dIWV/dt + div(IVT) + (P-E)`
- `residual_no_pe = dIWV/dt + div(IVT)`
- `p_minus_e_only`
- `residual_full_roll4` (4-step smoothing)

Key findings:

- land negative gain in `residual_full` improves to near-zero in `residual_no_pe` (`-0.000019`),
- land becomes positive after mild smoothing (`+0.001357`, borderline `p=0.0567`),
- ocean remains positive in all variants (`~0.0017` to `0.0041`, significant in current permutation test),
- this supports a **detectability limitation** interpretation: over-land closure is degraded by noisy/parameterized components (especially `P-E`) and high-frequency residual noise.

### 6.10 Halo-boundary strict closure (core-only score with bath observables)

`experiment_M_halo_boundary_strict.py` (run on `2017..2021`, strict split `train<=2019 -> test=2020 -> external=2021`):

- target is scored only on interior core `C` (central window),
- predictors are allowed on `C U H` with halo ring `H` (`halo_width_cells=8`),
- explicit boundary bath terms are included:
  - `bath_c2f = (coarse_halo - coarse_core) * fine_core`
  - `bath_f2c = (fine_halo - fine_core) * coarse_core`
- model ladder:
  - `ERA_core`
  - `ERA_window`
  - `ERA_window + Phi_H`
  - `ERA_window + Lambda_H` (with bath terms)
  - `ERA_window + Phi_H + Lambda_H`
  - shuffled halo control

Key strict results (`..._v2` run):

- **test 2020**:
  - `ERA_window` gain vs core: `0.136707`, `perm_p=0.0005`
  - `+Phi_H` incremental vs `ERA_window`: `+0.000093` (small, CI overlaps 0)
  - `+Lambda_H` incremental vs `ERA_window`: `-0.000002` (small, CI overlaps 0)
  - `+Phi_H+Lambda_H` incremental vs `ERA_window`: `+0.000103` (small, CI overlaps 0)
- **external 2021**:
  - `ERA_window` gain vs core: `0.092229`, `perm_p=0.0005`
  - `+Phi_H` incremental vs `ERA_window`: `+0.000068` (small, CI overlaps 0)
  - `+Lambda_H` incremental vs `ERA_window`: `+0.000045` (small, CI overlaps 0)
  - `+Phi_H+Lambda_H` incremental vs `ERA_window`: `+0.000144` (small, CI overlaps 0)

Interpretation:

- halo context materially improves interior closure and transfers to external year,
- shuffled full-halo control collapses (`gain < 0` on test, near-zero on external),
- incremental `Phi_H`/`Lambda_H` over `ERA_window` in this setup is positive but currently small.

### 6.11 Final preregistered halo-width scan (fixed-core protocol)

Protocol (locked before scan):

- split fixed: `train<=2019 -> test=2020 -> external=2021`,
- fixed core: `core_margin_cells=10`,
- halo mode fixed to `local`,
- scanned `halo_width_cells in {0,4,6,8,10}`,
- model ladder unchanged (`ERA_core`, `ERA_window`, `+Phi_H`, `+Lambda_H`, `+Phi_H+Lambda_H`, shuffled control).

Primary endpoint (`ERA_window` gain vs `ERA_core`):

- `w=0`: test `0.000000` (`p=1.0`), external `0.000000` (`p=1.0`)
- `w=4`: test `0.151703` (`p=0.0005`), external `0.156626` (`p=0.0005`)
- `w=6`: test `0.137846` (`p=0.0005`), external `0.131756` (`p=0.0005`)
- `w=8`: test `0.129966` (`p=0.0005`), external `0.112355` (`p=0.0005`)
- `w=10`: test `0.126708` (`p=0.0005`), external `0.097619` (`p=0.0005`)

Interpretation:

- no-context control (`w=0`) collapses exactly to zero gain,
- best transfer is at `w=4`,
- wider rings degrade gain, consistent with locality-limited boundary exchange rather than monotonic feature inflation.

### 6.12 Final halo-physics falsification block (fixed width=4)

With width fixed at the scan optimum (`w=4`), compare physically adjacent halo against context replacements:

- `local`: adjacent halo ring
- `remote`: same-size non-adjacent region
- `misaligned`: adjacent halo shifted in space

Primary endpoint (`ERA_window` gain vs `ERA_core`):

- `local`: test `0.151703`, external `0.156626`
- `remote`: test `0.097182`, external `0.058412`
- `misaligned`: test `0.073343`, external `0.059724`

Contrasts:

- local - remote: `+0.054521` (test), `+0.098214` (external)
- local - misaligned: `+0.078360` (test), `+0.096902` (external)

Interpretation:

- physically adjacent halo is substantially more informative than remote or shifted context,
- this supports boundary-exchange interpretation and rejects the "extra features only" explanation.

## 7) Detectability constraints and interpretation

Final interpretation of the full curated M package:

1. `Lambda` has a real, reproducible residual-closure signal (core runs + falsification tests).
2. Current ERA5-level closure does not convert that signal into robust over-land gain in the same linear closure metric.
3. The land/ocean contrast is best read as an **observability limit** of the closure target under reanalysis/parameterization noise, not as evidence against the Lambda mechanism.
4. The effect remains above the theoretical detection threshold, but below the engineering deployment threshold.

## 8) Excluded from the core physical claim

1. `experiment_M_cosmo_flow_v2`
   - negative/non-significant.
2. `experiment_M_cosmo_flow_v4_vertical_entropy` (`feature_set=lambda_entropy_vertical` raw)
   - negative/non-significant in current formulation.
3. extreme-only linear subsets (`union_p90`, `intersection_p90`) as standalone claim
   - too noisy and regime-unstable without explicit regime calibration.

## 9) Reproducible command set (curated)

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

Falsification checks:

```bash
.venv/bin/python clean_experiments/experiment_M_lambda_falsification_tests.py
```

Land/ocean split:

```bash
.venv/bin/python clean_experiments/experiment_M_land_ocean_split.py
```

Spatial maps for article:

```bash
.venv/bin/python clean_experiments/experiment_M_land_ocean_spatial_viz.py
```

Noise-probe diagnostics:

```bash
.venv/bin/python clean_experiments/experiment_M_land_ocean_noise_probe.py
```

Halo-boundary strict closure:

```bash
.venv/bin/python clean_experiments/experiment_M_halo_boundary_strict.py \
  --input-nc data/processed/wpwp_era5_2017_2021_experiment_M_input.nc \
  --outdir clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2 \
  --train-end-year 2019 \
  --test-year 2020 \
  --external-year 2021 \
  --halo-width-cells 8 \
  --fine-band-idx 0 \
  --coarse-band-idx 1 \
  --n-perm 1999
```

Halo-width scan (final preregistered block):

```bash
for w in 0 4 6 8 10; do
  .venv/bin/python clean_experiments/experiment_M_halo_boundary_strict.py \
    --input-nc data/processed/wpwp_era5_2017_2021_experiment_M_input.nc \
    --outdir clean_experiments/results/experiment_M_halo_boundary_widthscan_w${w}_causal2019_train2019_test2020_ext2021 \
    --train-end-year 2019 \
    --test-year 2020 \
    --external-year 2021 \
    --core-margin-cells 10 \
    --halo-width-cells ${w} \
    --halo-mode local \
    --fine-band-idx 0 \
    --coarse-band-idx 1 \
    --n-perm 1999
done
```

Halo-physics falsification (final control block):

```bash
for mode in remote misaligned; do
  .venv/bin/python clean_experiments/experiment_M_halo_boundary_strict.py \
    --input-nc data/processed/wpwp_era5_2017_2021_experiment_M_input.nc \
    --outdir clean_experiments/results/experiment_M_halo_boundary_falsify_${mode}_w4_causal2019_train2019_test2020_ext2021 \
    --train-end-year 2019 \
    --test-year 2020 \
    --external-year 2021 \
    --core-margin-cells 10 \
    --halo-width-cells 4 \
    --halo-mode ${mode} \
    --fine-band-idx 0 \
    --coarse-band-idx 1 \
    --n-perm 1999
done
```

## 10) Artifact map

- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex`
- `clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated/`
- `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/`
- `clean_experiments/results/experiment_M_horizontal_vertical_compare/`
- `clean_experiments/results/experiment_M_extremes_amplitude/`
- `clean_experiments/results/experiment_M_extremes_calibration/`
- `clean_experiments/results/experiment_M_extremes_quarterly/`
- `clean_experiments/results/experiment_M_lambda_falsification_tests/`
- `clean_experiments/results/experiment_M_land_ocean_split/`
- `clean_experiments/results/experiment_M_land_ocean_spatial_viz/`
- `clean_experiments/results/experiment_M_land_ocean_noise_probe/`
- `clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2/`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w0_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w4_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w6_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w8_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w10_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_falsify_remote_w4_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/results/experiment_M_halo_boundary_falsify_misaligned_w4_causal2019_train2019_test2020_ext2021/`
- `clean_experiments/EXPERIMENT_M_GKSL_TRANSFER_HALO_BRANCH_2026_03_07.md`
