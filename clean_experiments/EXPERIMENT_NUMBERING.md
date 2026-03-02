# Experiment Numbering Manifest

This file defines the program numbering used in `research_programm_summary.csv`.

Note: listed output folders are canonical runtime locations. Heavy generated artifacts are local-only by default; this repository keeps code and lightweight documentation.

| Program ID | Block | Primary script(s) | Canonical output folder(s) |
|---|---|---|---|
| 1 | A | `clean_experiments/experiment_A.py` | `out/experiment_A`, `clean_experiments/results/experiment_A_robust` |
| 2 | B (wave-1) | `clean_experiments/experiment_wave1_user.py` | `out/experiment_B_wave1`, `clean_experiments/results/experiment_B_wave1_robust` |
| 3 | D | `clean_experiments/experiment_D.py` | `out/experiment_D`, `clean_experiments/results/experiment_D_robust` |
| 4 | E | `clean_experiments/experiment_E.py` | `out/experiment_E`, `clean_experiments/results/experiment_E_robust` |
| 5 | F | `clean_experiments/experiment_F.py` | `out/experiment_F`, `clean_experiments/results/experiment_F_robust` |
| 6 | G | `clean_experiments/experiment_G.py` | `out/experiment_G`, `clean_experiments/results/experiment_G_robust` |
| 7 | G2 (toy-chain) | `clean_experiments/experiment_G2_toy_chain.py` | `out/experiment_G2_toy_chain`, `clean_experiments/results/experiment_G2_toy_chain_robust` |
| 8 | G2 (single-qubit) | `clean_experiments/experiment_G2_single_qubit.py` | `out/experiment_G2_single_qubit`, `clean_experiments/results/experiment_G2_single_qubit_robust` |
| 9 | H1 | `clean_experiments/experiment_H_holographic.py` | `out/experiment_H_holographic`, `clean_experiments/results/experiment_H_holographic_robust` |
| 10 | H2 | `clean_experiments/experiment_I_continuum_conservation.py` | `out/experiment_I_continuum_conservation`, `clean_experiments/results/experiment_I_continuum_conservation_robust` |
| 11 | H3 | `clean_experiments/experiment_J_berry_refinement.py` | `out/experiment_J_berry_refinement`, `clean_experiments/results/experiment_J_berry_refinement_robust` |
| 12 | H4 | `clean_experiments/experiment_K_lambda_bridge.py` | `out/experiment_K_lambda_bridge`, `clean_experiments/results/experiment_K_lambda_bridge_robust` |
| 13 | H4b | `clean_experiments/experiment_K2_theory_space_curvature.py` | `clean_experiments/results/experiment_K2_theory_space_curvature`, `clean_experiments/results/experiment_K2_theory_space_curvature_robust` |
| 14 | H5 | `clean_experiments/experiment_L_matter_fields.py` | `clean_experiments/results/experiment_L_matter_fields`, `clean_experiments/results/experiment_L_matter_fields_robust` |
| 15 | M1 | `clean_experiments/experiment_M_cosmo_flow.py` | `clean_experiments/results/experiment_M_cosmo_flow`, `clean_experiments/results/experiment_M_cosmo_flow_robust` |
| 16 | M2 | `clean_experiments/experiment_M_horizontal_vertical_compare.py` | `clean_experiments/results/experiment_M_horizontal_vertical_compare` |
| 17 | M3 | `clean_experiments/experiment_M_land_ocean_split.py`, `clean_experiments/experiment_M_land_ocean_noise_probe.py` | `clean_experiments/results/experiment_M_land_ocean_split`, `clean_experiments/results/experiment_M_land_ocean_noise_probe` |
| 18 | O1 | `clean_experiments/experiment_O_entropy_equilibrium.py` | `clean_experiments/results/experiment_O_entropy_equilibrium` |
| 19 | O2 | `clean_experiments/experiment_O_spatial_variance.py`, `clean_experiments/experiment_O_lambda_spatial_viz.py`, `clean_experiments/experiment_O_spatial_active_west.py` | `clean_experiments/results/experiment_O_spatial_variance`, `clean_experiments/results/experiment_O_lambda_spatial_viz`, `clean_experiments/results/experiment_O_spatial_active_west` |
| 20 | M4 | `clean_experiments/experiment_M_lambda_falsification_tests.py` | `clean_experiments/results/experiment_M_lambda_falsification_tests` |

## Auxiliary blocks (outside 1-20)

- N moisture-budget branch:
  - `clean_experiments/experiment_N_navier_stokes_budget.py`
  - `clean_experiments/experiment_N_followup_dual.py`
  - `clean_experiments/EXPERIMENT_N_DATA_MANIFEST.md`
