# Experiment P2: Noncommuting Coarse-Graining

## Setup
- panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- target: `target_density_coarse`
- scales (l cells): `8, 16, 32`
- MRMS downsample: `16`
- active threshold: `3.0`
- generator weights [occ,sq,log,grad]: `[1.0, 1.0, 1.0, 1.0]`
- generator scale power: `0.0`
- decoherence alpha: `4.0`
- permutations per scale: `49`

## Formal object
- Delta_comm(x,l,t) = ||Pi_{l->2l} Phi - Phi Pi_{l->2l}||
- Phi set: threshold, square, log1p, gradient-magnitude
- rho_occ from occupancy populations on scales {l,2l}
- lambda_local = Re Tr(F_comm rho_occ), with F_comm = i[A,B]

## Per-scale closure metrics
- l=8: mae_gain=0.000008, r2_gain=0.003494, perm_p=0.020000, comm_mean=2.739038, PASS_ALL=True
- l=16: mae_gain=0.000006, r2_gain=0.002171, perm_p=0.020000, comm_mean=2.145826, PASS_ALL=True
- l=32: mae_gain=0.000008, r2_gain=0.002174, perm_p=0.020000, comm_mean=1.516783, PASS_ALL=True

## Operator defect summary
- l=8: mean(delta_occ)=0.007399, mean(delta_sq)=0.451751, mean(delta_log)=0.015556, mean(delta_grad)=0.108996, mean(norm_theory)=2.739038, q90(norm_theory)=0.333616, mean(rho_purity)=0.895357
- l=16: mean(delta_occ)=0.008501, mean(delta_sq)=0.573902, mean(delta_log)=0.024786, mean(delta_grad)=0.122826, mean(norm_theory)=2.145826, q90(norm_theory)=1.037603, mean(rho_purity)=0.838195
- l=32: mean(delta_occ)=0.009084, mean(delta_sq)=0.689038, mean(delta_log)=0.035491, mean(delta_grad)=0.152982, mean(norm_theory)=1.516783, q90(norm_theory)=2.522596, mean(rho_purity)=0.733265

## Scale priority
- best scale by MAE gain: l=32, mae_gain=0.000008, perm_p=0.020000

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
