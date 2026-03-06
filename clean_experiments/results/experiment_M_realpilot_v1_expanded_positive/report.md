# Experiment M-realpilot (Local Structural Proxy)

- input panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- target: `target_next_mrms_p95`
- baseline features: `base_prev_p95, base_prev_area_gt5, hour_sin, hour_cos`
- structural features: `abi_cold_frac_235, abi_grad_mean, glm_flash_count_log, convective_coupling_index`

## Main metrics
- n_events: 24
- n_model_samples: 48
- MAE baseline: 0.060540
- MAE full: 0.058964
- mean MAE gain: 0.001576
- min event gain: -0.023961
- event positive frac: 0.667

## Placebo tests
- time-shuffle p-value: 0.010000
- event-shuffle p-value: 0.008000

## Active vs calm
- active quantile: 0.67
- mean gain active: 0.003150
- mean gain calm: -0.000283
- active - calm: 0.003433

## Hypotheses
- H1-local (baseline+structural beats baseline): True
- H2-local (placebo does not reproduce gain): True
- H3-local (effect stronger in active windows): True
- PASS_ALL: True

## Notes
- Fixed feature set and fixed ridge alpha were used to avoid pilot-time retuning.
- Event-level CV is leave-one-event-out (strict out-of-event validation).