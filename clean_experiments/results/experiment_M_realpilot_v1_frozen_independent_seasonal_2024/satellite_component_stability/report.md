# Satellite Component Stability (Frozen M-realpilot)

- modeling dataset: `clean_experiments/results/experiment_M_realpilot_v1_frozen_independent_seasonal_2024/modeling_dataset.csv`
- target: `target_next_mrms_p95`
- ridge_alpha: 10.0
- permutations per variant: 499

## Headline
- ABI-only mean gain: -0.000609
- ABI+GLM mean gain: -0.009710
- delta (ABI+GLM - ABI-only): -0.009101

## Notes
- Feature extraction and thresholds are inherited from frozen v1.
- Comparison changes only the structural column subset inside the same CV/permutation protocol.