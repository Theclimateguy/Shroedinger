# P2 Theory Bridge Ablation

- panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- scales: `8, 16, 32`
- screening configs: `75`
- final top-k: `2`
- final permutations per config: `3`

## Selected config
- config_id: `C004`
- decoherence_alpha: `0.5`
- lambda_scale_power: `0.0`
- weights [occ,sq,log,grad]: `[1.5, 1.0, 1.0, 1.0]`
- overall_mae_gain: `-4.344310e-06`
- overall_r2_gain: `-3.813571e-04`
- PASS_all_scales: `False`

## Artifacts
- `ablation_screening_configs.csv`
- `ablation_screening_per_scale.csv`
- `ablation_final_configs.csv`
- `ablation_final_per_scale.csv`
- `selected_config.csv`
- `selected_run_command.sh`
- `selected_config_tile_dataset.csv`
