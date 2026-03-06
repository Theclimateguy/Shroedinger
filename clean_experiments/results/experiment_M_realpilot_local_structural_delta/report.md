# Experiment M-realpilot (Local Structural Proxy)

- input panel: `clean_experiments/results/realpilot_2024_dataset_panel_v1.csv`
- target: `target_delta_mrms_p95`
- baseline features: `base_prev_p95, base_prev_area_gt5, hour_sin, hour_cos`
- structural features: `abi_cold_frac_235, abi_grad_mean, glm_flash_count_log, convective_coupling_index`

## Main metrics
- n_events: 5
- n_model_samples: 10
- MAE baseline: 0.035850
- MAE full: 0.039055
- mean MAE gain: -0.003205
- min event gain: -0.008548
- event positive frac: 0.200

## Placebo tests
- time-shuffle p-value: 0.160000
- event-shuffle p-value: 0.190000

## Active vs calm
- active quantile: 0.67
- mean gain active: -0.003501
- mean gain calm: -0.003008
- active - calm: -0.000494

## Hypotheses
- H1-local (baseline+structural beats baseline): False
- H2-local (placebo does not reproduce gain): False
- H3-local (effect stronger in active windows): False
- PASS_ALL: False

## Notes
- Fixed feature set and fixed ridge alpha were used to avoid pilot-time retuning.
- Event-level CV is leave-one-event-out (strict out-of-event validation).