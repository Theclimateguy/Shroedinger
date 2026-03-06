# Experiment M-realpilot (Local Structural Proxy)

- input panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_independent_seasonal_2024.csv`
- target: `target_next_mrms_p95`
- baseline features: `base_prev_p95, base_prev_area_gt5, hour_sin, hour_cos`
- structural features: `abi_cold_frac_235, abi_grad_mean, glm_flash_count_log, convective_coupling_index`

## Main metrics
- n_events: 12
- n_model_samples: 24
- MAE baseline: 0.092897
- MAE full: 0.102607
- mean MAE gain: -0.009710
- min event gain: -0.070946
- event positive frac: 0.417

## Placebo tests
- time-shuffle p-value: 0.708000
- event-shuffle p-value: 0.160000

## Active vs calm
- active quantile: 0.67
- mean gain active: -0.005714
- mean gain calm: -0.013706
- active - calm: 0.007992

## Hypotheses
- H1-local (baseline+structural beats baseline): False
- H2-local (placebo does not reproduce gain): False
- H3-local (effect stronger in active windows): True
- PASS_ALL: False

## Notes
- Fixed feature set and fixed ridge alpha were used to avoid pilot-time retuning.
- Event-level CV is leave-one-event-out (strict out-of-event validation).