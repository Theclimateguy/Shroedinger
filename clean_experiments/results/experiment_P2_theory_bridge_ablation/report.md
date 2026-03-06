# P2 Theory Bridge Ablation

- panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- scales: `8, 16, 32`
- screening configs: `75`
- final top-k: `4`
- final permutations per config: `49`

## Selected config
- config_id: `C009`
- decoherence_alpha: `0.5`
- lambda_scale_power: `0.5`
- weights [occ,sq,log,grad]: `[1.5, 1.0, 1.0, 1.0]`
- overall_mae_gain: `7.427353e-06`
- overall_r2_gain: `3.391652e-03`
- PASS_all_scales: `True`

## Program status

- C009 is promoted as canonical Group A baseline for the A05 scale-space continuation.
- Dense-transfer and `l=8` diagnostics are documented in:
  - `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/report.md`
  - `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`

## Artifacts
- `ablation_screening_configs.csv`
- `ablation_screening_per_scale.csv`
- `ablation_final_configs.csv`
- `ablation_final_per_scale.csv`
- `selected_config.csv`
- `selected_run_command.sh`
- `selected_config_tile_dataset.csv`
