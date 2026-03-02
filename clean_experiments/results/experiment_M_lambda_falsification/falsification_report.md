# Experiment M Falsification Checks

## Test 1: Nonlinearity check
- lambda_h: gain_vs_poly=0.003507, verdict=lambda_independent
- lambda_v: gain_vs_poly=0.003533, verdict=lambda_independent

## Test 2: Placebo synthetic Lambda
- lambda_h: p_placebo=0.000000, verdict=specific_information
- lambda_v: p_placebo=0.000000, verdict=specific_information

## Test 3: Spatial specificity
- convective_mean=0.000122, stable_mean=-0.001224, ratio=inf, verdict=no_strong_spatial_specificity

## Note
- NE/SE longitudes are clipped by dataset upper bound if the file ends before 170E.
