# Experiment P1-lite: Spatial Occupancy Cascade

## Setup
- panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- target: `target_density_coarse`
- scales (l cells): `8, 16, 32`
- MRMS downsample: `16`
- MRMS active threshold: `3.0`
- permutations per scale: `49`

## Per-scale metrics
- l=8: mae_gain=0.000056, r2_gain=0.023421, perm_p=0.020000, comm_mean=0.000280, PASS_ALL=True
- l=16: mae_gain=0.000034, r2_gain=0.009254, perm_p=0.020000, comm_mean=0.000137, PASS_ALL=True
- l=32: mae_gain=0.000016, r2_gain=0.004186, perm_p=0.020000, comm_mean=0.000074, PASS_ALL=False

## Scale Sensitivity
- best scale by MAE gain: l=8, mae_gain=0.000056, perm_p=0.020000
- use this row as first candidate when lambda detectability is weak.

## Program status
- This run is logged as `A05.R1_p1_spatial_occupancy_cascade` in Group A.
- P2 continuation (C009 bridge calibration and dense transfer):
  - `clean_experiments/results/experiment_P2_theory_bridge_ablation/report.md`
  - `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated/report.md`

## Data guidance
- P1-lite runs on existing aligned MRMS files from realpilot panel.
- P2 should add dense intra-event ABI/GLM sequences (not only one matched frame per hour).
- P3 should add LES/CRM-like data (<1 km) for direct micro-structure thermodynamics.

## Artifacts
- `summary_metrics.csv`
- `spatial_tile_dataset.csv`
- `spatial_tile_scan.csv`
- `oof_predictions.csv`
- `fold_metrics.csv`
- `permutation_metrics.csv`
- `lambda_local_map_scale_*.png`
- `comm_defect_map_scale_*.png`
- `occupancy_gain_map_scale_*.png`
