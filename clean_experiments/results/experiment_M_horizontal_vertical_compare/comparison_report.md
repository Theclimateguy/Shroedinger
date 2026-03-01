# Experiment M Horizontal vs Vertical Comparison

## Test 1: Lambda correlation
- Pearson r: 0.996310
- R^2: 0.992633
- p-value: 0

## Test 2: Combined model vs single models
- gain(ctrl+lambda_h): 0.003379
- gain(ctrl+lambda_v): 0.003412
- gain(ctrl+lambda_h+lambda_v): 0.002566
- combined minus best single: -0.000846

## Diagnosis
- verdict: likely_detectability_limit_with_redundant_physical_signal
- mean |gain_h - gain_v| across proxy quartiles: 0.000215

## Context (existing runs)
- horizontal_v3_calibrated: gain=0.003379, p=0.0070922, strata_pos=1.000, pass_all=False
- vertical_v4_entropy_raw: gain=-0.003670, p=0.70922, strata_pos=0.333, pass_all=False
- vertical_v4_macro_calibrated: gain=0.003412, p=0.0070922, strata_pos=1.000, pass_all=True
