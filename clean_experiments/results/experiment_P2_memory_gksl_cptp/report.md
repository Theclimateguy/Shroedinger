# Experiment P2-memory GKSL/CPTP

## Setup
- base tile csv: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- panel fallback: `clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv`
- target: `target_density_coarse`
- memory scales: `[8]`
- state propagation: `rho_t = Reset o GAD o Dephase o U (rho_{t-1})`
- locked baseline: C009 weights=[1.5, 1.0, 1.0, 1.0], scale_power=0.5, decoherence_alpha=0.5

## Best l=8 config
- config_id: `G001`
- gksl_dephase_base: `0.800`
- gksl_dephase_comm_scale: `0.400`
- gksl_relax_base: `0.800`
- gksl_relax_comm_scale: `0.000`
- gksl_measurement_rate: `1.600`
- gksl_hamiltonian_scale: `0.200`
- l=8 mae_gain: `2.270824e-06`
- l=8 r2_gain: `2.134425e-04`
- l=8 perm_p: `0.020000`
- l=8 event_positive_frac: `0.9375`
- l=8 PASS_ALL: `True`

## Best all-scale summary
- ALL mae_gain: `2.191610e-06`
- ALL r2_gain: `2.891841e-04`
- ALL perm_p(max scale): `0.020000`
- ALL event_positive_frac: `0.8333`
- ALL PASS_ALL: `True`

## CPTP diagnostics
- max cptp violation proxy (trace+psd): `8.882e-16`
- mean gamma_dephase (applied rows): `0.0010`
- mean gamma_relax (applied rows): `0.0008`
- mean reset_kappa (applied rows): `0.0016`

## Comparison
- baseline l=8 mae_gain: `-1.330761e-07`; perm_p=`0.800000`
- gksl     l=8 mae_gain: `2.270824e-06`; perm_p=`0.020000`
- baseline ALL pass: `False`; gksl ALL pass: `True`

## Artifacts
- `gksl_screening_l8.csv`
- `gksl_final_l8.csv`
- `gksl_tile_dataset_best.csv`
- `summary_metrics.csv`
- `oof_predictions.csv`
- `fold_metrics.csv`
- `permutation_metrics.csv`
- `baseline_vs_memory_models.csv`
