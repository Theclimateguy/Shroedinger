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
  - A05 scale-space continuation (P1/P2): `experiment_P1_spatial_occupancy_cascade.py`, `experiment_P2_noncommuting_coarse_graining.py`, `experiment_P2_theory_bridge_ablation.py`, `run_p2_calibrated_dense_ingest.py`, `experiment_P2_l8_diagnostic_block.py`, `experiment_P2_memory.py`
- `A06` M4 — staged falsification of Lambda necessity (`experiment_M_lambda_falsification_tests.py`)
- `A07` F5 — structural Lambda and multiscale/fractional surrogates in ERA5 (`experiment_F5_lambda_struct_fractal_era5.py`, `experiment_F5_spatial_fractal_maps.py`)
- `A08` F6b — strict heavy-tail test of `|Lambda_struct|` (`experiment_F6b_era5_heavy_tails.py`, `experiment_F6b_era5_heavy_tails_panel.py`)
- `A09` F6c — clustered subspace heavy-tail fits (`experiment_F6c_clustered_subspace_tails.py`)
- `A10` F6c-spatial — patch-wise spatial tail maps (`experiment_F6c_spatial_panel_viz.py`)

### A05 Scale-Space Continuation (P1/P2)

- `A05.R1_p1_spatial_occupancy_cascade` — P1-lite occupancy cascade on sparse panel.
- `A05.R2_p2_theory_bridge_c009` — P2 density-matrix bridge + C009 calibration on sparse panel.
- `A05.R3_p2_dense_c009` — calibrated dense ingest transfer test (`240x16` panel).
- `A05.R4_p2_l8_resolution` — narrow `l=8` diagnostic block; matched-event/operator/resolution/regime checks, with memory-motivated interpretation.
- `A05.R5_p2_memory` — retarded density-matrix closure; recovers dense `l=8` pass under full operators and standard threshold.

Program-level finalization artifact:

- `clean_experiments/EXPERIMENT_A_ATMOSPHERE_PIPELINE.md`
- `clean_experiments/EXPERIMENT_P2_MEMORY.md`
- `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`
- `clean_experiments/results/experiment_P2_memory/report.md`
- `clean_experiments/results/experiment_P2_memory_x_viz/report.md`
- `clean_experiments/results/experiment_P2_memory_geo_viz/report.md`

### Atmosphere extensions (outside canonical A01-A10 block)

- N moisture-budget follow-up branch:
  - `experiment_N_navier_stokes_budget.py`
  - `experiment_N_followup_dual.py`
- granular ingest branch (high-resolution MRMS + GOES pilot):
  - `clean_experiments/download_mrms.py`
  - `clean_experiments/download_goes.py`
  - `clean_experiments/build_mrms_goes_aligned_catalog.py`
  - `clean_experiments/download_matched_windows.py`
  - `clean_experiments/run_ultralight_mrms_goes_pilot.py`
  - `clean_experiments/pilot_events_template.csv`
  - `clean_experiments/EXPERIMENT_GRANULAR_MRMS_GOES_INGEST.md`
- Consolidated March 2026 F-series report:
  - `clean_experiments/results/experiment_F_series_2026_03_06/report.md`

Numbering policy:

- Canonical codes are fixed at `T01..T19` and `A01..A10`.
- Continuation runs are logged via run-level IDs in `ветка` (for example `A07.R1_*`), not by adding `A11+`.

### M-realpilot prereg (frozen v1)

- frozen script: `clean_experiments/experiment_M_realpilot_v1_frozen.py`
- SHA256: `aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2`
- policy: expand only event list, rerun unchanged script, compare headline metrics only.
- expanded event list seed: `clean_experiments/pilot_events_real_2024_us_convective_expanded_v1.csv`
- locked positive reference: `clean_experiments/results/experiment_M_realpilot_v1_expanded_positive/`
- independent seasonal extension list: `clean_experiments/pilot_events_realpilot_v1_independent_seasonal_2024.csv`
- independent geographic extension list: `clean_experiments/pilot_events_realpilot_v1_independent_geographic_southwest_2024.csv`
- independent extension compare: `clean_experiments/results/experiment_M_realpilot_v1_independent_extension_compare/report.md`
- ABI-only vs ABI+GLM stability: `clean_experiments/experiment_M_realpilot_satellite_component_stability.py`
- applicability-map builder: `clean_experiments/summarize_m_realpilot_applicability.py`
- applicability map: `clean_experiments/results/experiment_M_realpilot_applicability_map/report.md`
- perimeter + regime-detection package builder: `clean_experiments/build_m_realpilot_regime_detection_package.py`
- perimeter + regime-detection package: `clean_experiments/results/experiment_M_realpilot_regime_detection_package/report.md`

## Reproducibility

Run scripts locally and write outputs to `clean_experiments/results/...`.
Heavy artifacts are intentionally not versioned in GitHub.

## Notes

- `manuscript/` is maintained outside this repository.
- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex` is a standalone technical note.
