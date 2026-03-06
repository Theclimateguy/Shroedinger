# Experiment P2-memory

- Group A run ID: `A05.R5_p2_memory`
- Status: completed final experiment in the A05 scale-space continuation

## Setup
- base tile csv: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- panel fallback: `clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv`
- target: `target_density_coarse`
- memory source: `occupancy`
- lookback: `2`
- memory scales: `[8]`
- locked baseline: C009 weights=[1.5, 1.0, 1.0, 1.0], scale_power=0.5, decoherence_alpha=0.5

## Best l=8 config
- config_id: `M001`
- memory_eta: `0.800`
- memory_tau: `0.750`
- persistence_power: `0.000`
- l=8 mae_gain: `1.224306e-06`
- l=8 r2_gain: `1.343877e-04`
- l=8 perm_p: `0.020000`
- l=8 event_positive_frac: `0.9375`
- l=8 PASS_ALL: `True`

## Best all-scale summary
- ALL mae_gain: `1.372379e-06`
- ALL r2_gain: `2.177551e-04`
- ALL perm_p(max scale): `0.020000`
- ALL event_positive_frac: `0.8333`
- ALL PASS_ALL: `True`

## Comparison to dense C009 baseline
- baseline l=8 mae_gain: `-1.330761e-07`; perm_p=`0.800000`
- memory   l=8 mae_gain: `1.224306e-06`; perm_p=`0.020000`
- baseline ALL pass: `False`; memory ALL pass: `True`

## Interpretation
- Memory heals the dense-panel `l=8` failure without dropping `sq` and without lowering the active threshold.
- The best bridge is short-memory and occupancy-led: strong mixing (`eta=0.8`), fast decay (`tau=0.75`), no extra persistence gating.
- This closes the A05 theory-close sequence with a retarded density-matrix surrogate rather than an ad-hoc diagnostic retune.

## Artifacts
- `memory_feature_dataset.csv`
- `memory_screening_l8.csv`
- `memory_final_l8.csv`
- `memory_tile_dataset_best.csv`
- `summary_metrics.csv`
- `oof_predictions.csv`
- `fold_metrics.csv`
- `permutation_metrics.csv`
- `baseline_vs_memory.csv`

## Program docs
- `clean_experiments/EXPERIMENT_P2_MEMORY.md`
- `clean_experiments/EXPERIMENT_A_ATMOSPHERE_PIPELINE.md`
