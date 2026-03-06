# Experiment P2: Noncommuting Coarse-Graining

## Setup
- panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
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
- l=8: mae_gain=0.000008, r2_gain=0.003521, perm_p=0.020000, comm_mean=0.344454, PASS_ALL=True
- l=16: mae_gain=0.000006, r2_gain=0.002182, perm_p=0.020000, comm_mean=0.135041, PASS_ALL=True
- l=32: mae_gain=0.000008, r2_gain=0.002181, perm_p=0.020000, comm_mean=0.047724, PASS_ALL=True

## Operator defect summary
- l=8: mean(delta_occ)=0.007399, mean(delta_sq)=0.451751, mean(delta_log)=0.015556, mean(delta_grad)=0.108996, mean(norm_theory)=0.344454, q90(norm_theory)=0.042289, mean(rho_purity)=0.951260
- l=16: mean(delta_occ)=0.008501, mean(delta_sq)=0.573902, mean(delta_log)=0.024786, mean(delta_grad)=0.122826, mean(norm_theory)=0.135041, q90(norm_theory)=0.065411, mean(rho_purity)=0.923170
- l=32: mean(delta_occ)=0.009084, mean(delta_sq)=0.689038, mean(delta_log)=0.035491, mean(delta_grad)=0.152982, mean(norm_theory)=0.047724, q90(norm_theory)=0.079249, mean(rho_purity)=0.873504

## Scale priority
- best scale by MAE gain: l=32, mae_gain=0.000008, perm_p=0.020000

## Program status
- This calibrated C009 run is the locked baseline for Group A continuation `A05.R2`.
- Dense-transfer follow-up is in:
  - `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/report.md`
  - `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`

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
