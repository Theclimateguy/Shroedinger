# Experiment P2: Noncommuting Coarse-Graining

## Setup
- panel: `clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv`
- target: `target_density_coarse`
- scales (l cells): `8, 16, 32`
- MRMS downsample: `16`
- active threshold: `3.0`
- generator weights [occ,sq,log,grad]: `[1.5, 1.0, 1.0, 1.0]`
- generator scale power: `0.5`
- decoherence alpha: `0.5`
- permutations per scale: `49`

## Formal object
- Delta_comm(x,l,t) = ||Pi_{l->2l} Phi - Phi Pi_{l->2l}||
- Phi set: threshold, square, log1p, gradient-magnitude
- rho_occ from occupancy populations on scales {l,2l}
- lambda_local = Re Tr(F_comm rho_occ), with F_comm = i[A,B]

## Per-scale closure metrics
- l=8: mae_gain=-0.000000, r2_gain=0.002218, perm_p=0.800000, comm_mean=0.308588, PASS_ALL=False
- l=16: mae_gain=0.000001, r2_gain=0.000922, perm_p=0.020000, comm_mean=0.123204, PASS_ALL=True
- l=32: mae_gain=0.000004, r2_gain=0.001930, perm_p=0.020000, comm_mean=0.047571, PASS_ALL=True

## Operator defect summary
- l=8: mean(delta_occ)=0.007503, mean(delta_sq)=0.421483, mean(delta_log)=0.015409, mean(delta_grad)=0.110486, mean(norm_theory)=0.308588, q90(norm_theory)=0.042335, mean(rho_purity)=0.950805
- l=16: mean(delta_occ)=0.008745, mean(delta_sq)=0.535988, mean(delta_log)=0.024463, mean(delta_grad)=0.123147, mean(norm_theory)=0.123204, q90(norm_theory)=0.061874, mean(rho_purity)=0.923332
- l=32: mean(delta_occ)=0.009483, mean(delta_sq)=0.647791, mean(delta_log)=0.035000, mean(delta_grad)=0.161633, mean(norm_theory)=0.047571, q90(norm_theory)=0.079406, mean(rho_purity)=0.873886

## Scale priority
- best scale by MAE gain: l=32, mae_gain=0.000004, perm_p=0.020000

## Follow-up finalization

- Narrow `l=8` diagnostics are documented in:
  - `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`
- Group A policy after diagnostics:
  - keep calibrated C009 as canonical baseline;
  - keep `l=8` retune variants as diagnostic/theory evidence;
  - next theory-close extension is P2-memory (retarded density matrix).

## Artifacts
- `summary_metrics.csv`
- `comm_operator_summary.csv`
- `p2_tile_dataset.csv`
- `spatial_tile_scan.csv`
- `oof_predictions.csv`
- `fold_metrics.csv`
- `permutation_metrics.csv`
- `lambda_local_map_scale_*.png`
- `comm_defect_map_scale_*.png`
- `comm_gain_map_scale_*.png`
