# Experiment Landscape Manifest

This file defines the canonical alphanumeric landscape used in `research_programm_summary.csv`.

Legacy numeric `1-20` indexing is deprecated.

Canonical blocks:

- `TOY_MODEL`: `T01 ... T19`
- `ATMOSPHERE_DATA`: `A01 ... A10`

Note: output folders listed below are canonical runtime locations. Heavy generated artifacts are local-only by default; this repository keeps code and lightweight documentation.

## TOY_MODEL block

| Code | Branch label | Primary script(s) | Canonical output folder(s) |
|---|---|---|---|
| T01 | A | `clean_experiments/experiment_A.py` | `out/experiment_A`, `clean_experiments/results/experiment_A_robust` |
| T02 | B_wave1 | `clean_experiments/experiment_wave1_user.py` | `out/experiment_B_wave1`, `clean_experiments/results/experiment_B_wave1_robust` |
| T03 | D | `clean_experiments/experiment_D.py` | `out/experiment_D`, `clean_experiments/results/experiment_D_robust` |
| T04 | E | `clean_experiments/experiment_E.py` | `out/experiment_E`, `clean_experiments/results/experiment_E_robust` |
| T05 | F | `clean_experiments/experiment_F.py` | `out/experiment_F`, `clean_experiments/results/experiment_F_robust` |
| T06 | G | `clean_experiments/experiment_G.py` | `out/experiment_G`, `clean_experiments/results/experiment_G_robust` |
| T07 | G2_toy_chain | `clean_experiments/experiment_G2_toy_chain.py` | `out/experiment_G2_toy_chain`, `clean_experiments/results/experiment_G2_toy_chain_robust` |
| T08 | G2_single_qubit | `clean_experiments/experiment_G2_single_qubit.py` | `out/experiment_G2_single_qubit`, `clean_experiments/results/experiment_G2_single_qubit_robust` |
| T09 | H1 | `clean_experiments/experiment_H_holographic.py` | `out/experiment_H_holographic`, `clean_experiments/results/experiment_H_holographic_robust` |
| T10 | H2 | `clean_experiments/experiment_I_continuum_conservation.py` | `out/experiment_I_continuum_conservation`, `clean_experiments/results/experiment_I_continuum_conservation_robust` |
| T11 | H3 | `clean_experiments/experiment_J_berry_refinement.py` | `out/experiment_J_berry_refinement`, `clean_experiments/results/experiment_J_berry_refinement_robust` |
| T12 | H4 | `clean_experiments/experiment_K_lambda_bridge.py` | `out/experiment_K_lambda_bridge`, `clean_experiments/results/experiment_K_lambda_bridge_robust` |
| T13 | H4b | `clean_experiments/experiment_K2_theory_space_curvature.py` | `clean_experiments/results/experiment_K2_theory_space_curvature`, `clean_experiments/results/experiment_K2_theory_space_curvature_robust` |
| T14 | H5 | `clean_experiments/experiment_L_matter_fields.py` | `clean_experiments/results/experiment_L_matter_fields`, `clean_experiments/results/experiment_L_matter_fields_robust` |
| T15 | F1 | `clean_experiments/experiment_F1_fractal_emergence.py` | `clean_experiments/results/experiment_F1_fractal_emergence` |
| T16 | F2 | `clean_experiments/experiment_F2_scale_covariance.py` | `clean_experiments/results/experiment_F2_scale_covariance` |
| T17 | F3 | `clean_experiments/experiment_F3_lambda_fractal_bridge.py` | `clean_experiments/results/experiment_F3_lambda_fractal_bridge` |
| T18 | F4_F4b | `clean_experiments/experiment_F4_holonomy_fractal_encoder.py`, `clean_experiments/experiment_F4b_independent_holonomy_ablation.py` | `clean_experiments/results/experiment_F4_holonomy_fractal_encoder`, `clean_experiments/results/experiment_F4b_independent_holonomy_ablation` |
| T19 | F6 | `clean_experiments/experiment_F6_soc_avalanches.py` | `clean_experiments/results/experiment_F6_soc_avalanches`, `clean_experiments/results/experiment_F6_soc_robustness` |

## ATMOSPHERE_DATA block

| Code | Branch label | Primary script(s) | Canonical output folder(s) |
|---|---|---|---|
| A01 | M1 | `clean_experiments/experiment_M_cosmo_flow.py` | `clean_experiments/results/experiment_M_cosmo_flow`, `clean_experiments/results/experiment_M_cosmo_flow_robust`, `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated` |
| A02 | M2 | `clean_experiments/experiment_M_horizontal_vertical_compare.py` | `clean_experiments/results/experiment_M_horizontal_vertical_compare` |
| A03 | M3 | `clean_experiments/experiment_M_land_ocean_split.py`, `clean_experiments/experiment_M_land_ocean_noise_probe.py` | `clean_experiments/results/experiment_M_land_ocean_split`, `clean_experiments/results/experiment_M_land_ocean_noise_probe` |
| A04 | O1 | `clean_experiments/experiment_O_entropy_equilibrium.py` | `clean_experiments/results/experiment_O_entropy_equilibrium` |
| A05 | O2 | `clean_experiments/experiment_O_spatial_variance.py`, `clean_experiments/experiment_O_lambda_spatial_viz.py`, `clean_experiments/experiment_O_spatial_active_west.py` | `clean_experiments/results/experiment_O_spatial_variance`, `clean_experiments/results/experiment_O_lambda_spatial_viz`, `clean_experiments/results/experiment_O_spatial_active_west` |
| A06 | M4 | `clean_experiments/experiment_M_lambda_falsification_tests.py` | `clean_experiments/results/experiment_M_lambda_falsification_tests` |
| A07 | F5 | `clean_experiments/experiment_F5_lambda_struct_fractal_era5.py`, `clean_experiments/experiment_F5_spatial_fractal_maps.py` | `clean_experiments/results/experiment_F5_lambda_struct_fractal_era5`, `clean_experiments/results/experiment_F5_spatial_fractal_maps` |
| A08 | F6b | `clean_experiments/experiment_F6b_era5_heavy_tails.py`, `clean_experiments/experiment_F6b_era5_heavy_tails_panel.py` | `clean_experiments/results/experiment_F6b_era5_heavy_tails`, `clean_experiments/results/experiment_F6b_era5_heavy_tails_panel` |
| A09 | F6c | `clean_experiments/experiment_F6c_clustered_subspace_tails.py` | `clean_experiments/results/experiment_F6c_clustered_subspace_tails` |
| A10 | F6c_spatial | `clean_experiments/experiment_F6c_spatial_panel_viz.py` | `clean_experiments/results/experiment_F6c_spatial_panel_viz` |

## Atmosphere extensions (outside canonical A01-A10)

- N moisture-budget branch:
  - `clean_experiments/experiment_N_navier_stokes_budget.py`
  - `clean_experiments/experiment_N_followup_dual.py`
  - `clean_experiments/EXPERIMENT_N_DATA_MANIFEST.md`

## Consolidated summaries

- `clean_experiments/results/experiment_F_series_2026_03_06/report.md`
- `research_programm_summary.csv`
