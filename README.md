# Shroedinger — Clean Experiments and Documentation

This repository keeps the canonical `clean_experiments` codebase and documentation.

## Scope

- Included:
  - `clean_experiments/` scripts and experiment docs (`*.py`, `*.md`, `*.tex`)
  - root docs (`README.md`, `research_programm_summary.csv`)
  - markdown reports in `clean_experiments/results/**`
- Excluded from Git:
  - legacy `experiments/`
  - local `manuscript/`
  - heavy generated artifacts (`CSV/PNG/NPZ/JSON`)

## Program Landscape (Alphanumeric)

Source of truth: `research_programm_summary.csv`.

Legacy `1-20` numbering is deprecated. The canonical map is now split by data domain:

- `TOY_MODEL` block: `T01 ... T19`
- `ATMOSPHERE_DATA` block: `A01 ... A10`

### TOY_MODEL block

- `T01` A — gauge invariance and noncommutativity (`experiment_A.py`)
- `T02` B / wave-1 — `Lambda_matter` sinusoidal/state checks (`experiment_wave1_user.py`)
- `T03` D — balance closure on `(t,x,mu)` (`experiment_D.py`)
- `T04` E — coherence-driven rate diagnostics (`experiment_E.py`)
- `T05` F — sinusoidal-law robustness (`experiment_F.py`)
- `T06` G — fixed-phase profile scan (`experiment_G.py`)
- `T07` G2 (toy-chain) — Clausius regression vs epsilon (`experiment_G2_toy_chain.py`)
- `T08` G2 (single-qubit) — Clausius regression vs profile (`experiment_G2_single_qubit.py`)
- `T09` H1 — holographic truncation with layer growth (`experiment_H_holographic.py`)
- `T10` H2 — continuum conservation extrapolation (`experiment_I_continuum_conservation.py`)
- `T11` H3 — Berry-phase refinement (`experiment_J_berry_refinement.py`)
- `T12` H4 — `Lambda_matter` bridge test (`experiment_K_lambda_bridge.py`)
- `T13` H4b — theory-space curvature vs RG noncommutativity (`experiment_K2_theory_space_curvature.py`)
- `T14` H5 — matter fields (`fermion+gauge`) (`experiment_L_matter_fields.py`)
- `T15` F1 — fractal emergence at epsilon balance (`experiment_F1_fractal_emergence.py`)
- `T16` F2 — scale covariance at fixed point (`experiment_F2_scale_covariance.py`)
- `T17` F3 — Lambda/fractal bridge (`experiment_F3_lambda_fractal_bridge.py`)
- `T18` F4/F4b — holonomy encoding and independent ablation (`experiment_F4_holonomy_fractal_encoder.py`, `experiment_F4b_independent_holonomy_ablation.py`)
- `T19` F6 — toy SOC avalanche scaling (`experiment_F6_soc_avalanches.py`)

### ATMOSPHERE_DATA block

- `A01` M1 — macro detectability in ERA5 moisture closure (`experiment_M_cosmo_flow.py`)
- `A02` M2 — horizontal/vertical consistency + placebo noise controls (`experiment_M_horizontal_vertical_compare.py`)
- `A03` M3 — land/ocean detectability and noise probe (`experiment_M_land_ocean_split.py`, `experiment_M_land_ocean_noise_probe.py`)
- `A04` O1 — thermodynamic test: Clausius baseline vs `+Lambda` (`experiment_O_entropy_equilibrium.py`)
- `A05` O2 — spatial macro-signal diagnostics (`experiment_O_spatial_variance.py`, `experiment_O_lambda_spatial_viz.py`, `experiment_O_spatial_active_west.py`)
- `A06` M4 — staged falsification of Lambda necessity (`experiment_M_lambda_falsification_tests.py`)
- `A07` F5 — structural Lambda and multiscale/fractional surrogates in ERA5 (`experiment_F5_lambda_struct_fractal_era5.py`, `experiment_F5_spatial_fractal_maps.py`)
- `A08` F6b — strict heavy-tail test of `|Lambda_struct|` (`experiment_F6b_era5_heavy_tails.py`, `experiment_F6b_era5_heavy_tails_panel.py`)
- `A09` F6c — clustered subspace heavy-tail fits (`experiment_F6c_clustered_subspace_tails.py`)
- `A10` F6c-spatial — patch-wise spatial tail maps (`experiment_F6c_spatial_panel_viz.py`)

### Atmosphere extensions (outside canonical A01-A10 block)

- N moisture-budget follow-up branch:
  - `experiment_N_navier_stokes_budget.py`
  - `experiment_N_followup_dual.py`
- Consolidated March 2026 F-series report:
  - `clean_experiments/results/experiment_F_series_2026_03_06/report.md`

## Reproducibility

Run scripts locally and write outputs to `clean_experiments/results/...`.
Heavy artifacts are intentionally not versioned in GitHub.

## Notes

- `manuscript/` is maintained outside this repository.
- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex` is a standalone technical note.
