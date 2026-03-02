# Experiment N: Navier-Stokes Moisture Budget Closure

## Setup
- Input: `data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc`
- Lambda source: `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv`
- Pressure layer: 300-850 hPa
- Spatial stride: lat=1, lon=1
- Train years: <= 2018, test year: 2019

## Selected config (one-SE)
- model: `global_a` (R + a*(lambda*C1))
- q_extreme: 0.80
- ridge_alpha: 1.0e+00

## Test (out-of-time)
- gain_all: -0.000000
- gain_extreme: -0.000000
- gain_non_extreme: -0.000000
- CI95 gain_all: [-0.000000, 0.000000]
- permutation p-value: 0.787234

## Quarterly rolling-origin 2019
- mean gain_all: 0.000000
- min gain_all: -0.000000
- positive-quarter fraction: 0.500
