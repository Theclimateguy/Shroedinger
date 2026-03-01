# Quarterly Rolling-Origin Experiment (2019)

Protocol:
- For each quarter Q in 2019, train on all history before Q.
- Thresholds for extreme mask are estimated on train only.
- Evaluate gain on quarter holdout.

## Summary by model
- horizontal_regime: mean=0.144528, std=0.121088, ext_mean=0.180600, non_ext_mean=0.150172, min=0.024342, max=0.269509
- vertical_regime: mean=0.143942, std=0.121737, ext_mean=0.180552, non_ext_mean=0.149223, min=0.022513, max=0.269557
- vertical_global: mean=0.003224, std=0.009802, ext_mean=-0.003139, non_ext_mean=0.007340, min=-0.002427, max=0.017892
- horizontal_global: mean=0.003173, std=0.010145, ext_mean=-0.003490, non_ext_mean=0.007456, min=-0.002686, max=0.018363
