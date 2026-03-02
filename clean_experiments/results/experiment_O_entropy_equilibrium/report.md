# Experiment O: Clausius-Consistent Thermodynamic Test

## Setup
- Input: `data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc`
- Lambda source: `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv`
- Taxa: `BL:1000-850, FT:850-300, UT:300-100`
- Train years <= 2018, test year = 2019
- Baseline equation: dS_hor_proxy ~ (1/T_eff)*dQ_in_proxy + b
- Full equation: dS_hor_proxy ~ (1/T_eff)*dQ_in_proxy + c*Lambda + b

## Primary Test (FT all-domain)
- Clausius slope 1/T_eff (baseline): -1.157253
- Clausius T_eff (baseline): -0.864115
- Lambda coefficient (full): 0.000288
- R2 baseline: 0.932165
- R2 full (Clausius + Lambda): 0.932161
- Gain (R2_full - R2_base): -0.000004
- CI95 R2_full: [0.906545, 0.947670]
- CI95 gain: [-0.000015, 0.000008]
- permutation p-value: 0.404255

## Spatial Coherence (FT)
- all: gain=-0.000004
- west: gain=0.000054
- east: gain=-0.000005

## Quarterly Rolling-Origin (2019)
- mean gain: -0.000013, min gain: -0.000072, positive quarter frac: 0.250
