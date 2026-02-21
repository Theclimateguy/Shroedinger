# Experiment Numbering Manifest

This file defines the continuous experiment numbering used in the article draft and in `clean_experiments`.

| Continuous ID | Legacy ID | Script | Main output folder | Robust folder |
|---|---|---|---|---|
| Experiment 1 | A | `clean_experiments/experiment_A.py` | `out/experiment_A` | `clean_experiments/results/experiment_A_robust` |
| Experiment 2 | B (wave-1) | `clean_experiments/experiment_wave1_user.py` | `out/experiment_B_wave1` | `clean_experiments/results/experiment_B_wave1_robust` |
| Experiment 3 | D | `clean_experiments/experiment_D.py` | `out/experiment_D` | `clean_experiments/results/experiment_D_robust` |
| Experiment 4 | E | `clean_experiments/experiment_E.py` | `out/experiment_E` | `clean_experiments/results/experiment_E_robust` |
| Experiment 5 | F | `clean_experiments/experiment_F.py` | `out/experiment_F` | `clean_experiments/results/experiment_F_robust` |
| Experiment 6 | G | `clean_experiments/experiment_G.py` | `out/experiment_G` | `clean_experiments/results/experiment_G_robust` |
| Experiment 7 | G2 (toy-chain) | `clean_experiments/experiment_G2_toy_chain.py` | `out/experiment_G2_toy_chain` | `clean_experiments/results/experiment_G2_toy_chain_robust` |
| Experiment 8 | G2 (single-qubit) | `clean_experiments/experiment_G2_single_qubit.py` | `out/experiment_G2_single_qubit` | `clean_experiments/results/experiment_G2_single_qubit_robust` |
| Experiment 9 | H (holographic scaling) | `clean_experiments/experiment_H_holographic.py` | `out/experiment_H_holographic` | `clean_experiments/results/experiment_H_holographic_robust` |
| Experiment 10 | I (continuum conservation) | `clean_experiments/experiment_I_continuum_conservation.py` | `out/experiment_I_continuum_conservation` | `clean_experiments/results/experiment_I_continuum_conservation_robust` |
| Experiment 11 | J (Berry refinement) | `clean_experiments/experiment_J_berry_refinement.py` | `out/experiment_J_berry_refinement` | `clean_experiments/results/experiment_J_berry_refinement_robust` |
| Experiment 12 | K (Lambda bridge) | `clean_experiments/experiment_K_lambda_bridge.py` | `out/experiment_K_lambda_bridge` | `clean_experiments/results/experiment_K_lambda_bridge_robust` |
| Experiment 13 | K2 (theory-space curvature) | `clean_experiments/experiment_K2_theory_space_curvature.py` | `clean_experiments/results/experiment_K2_theory_space_curvature` | `clean_experiments/results/experiment_K2_theory_space_curvature_robust` |
| Experiment 14 | L (matter fields) | `clean_experiments/experiment_L_matter_fields.py` | `clean_experiments/results/experiment_L_matter_fields` | `clean_experiments/results/experiment_L_matter_fields_robust` |

## Notes

- The integrated wave-1 code is now the canonical source for **Experiment 2**.
- File names remain stable (legacy letters) to avoid breaking existing references and scripts.
- Stage-2 expansion beyond the published block starts at **Experiment 9**.
- `clean_experiments/checklist_validations.py` is a non-numbered cross-experiment
  validation block for structural checklist claims; its artifacts are stored in
  `clean_experiments/results/checklist_validations`.
