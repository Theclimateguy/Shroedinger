# Experiment O Spatial: Non-Averaged Diagnostics

## Setup
- Input: `data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc`
- Taxon: `FT` (850.0-300.0 hPa)
- Train <= 2018, test = 2019
- Lambda components from: `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv`
- Convective climatology source: `convective_precip`

## Primary criteria
- spatial corr(gain_map, convective_clim) = 0.031721 (threshold > 0.300)
- max(gain_map) = 0.000920 at lat=-10.000, lon=154.250 (threshold > 0.050)

## Additional diagnostics
- west mean gain = -0.000051
- east mean gain = -0.000037
- land mean gain = -0.000028
- ocean mean gain = -0.000046
- corr(gain,conv) over land = 0.001057
- corr(gain,conv) over ocean = 0.024876
- panel (coords,time,scale + land criterion) gain = 0.000003 (R2 base=0.910453, R2 full=0.910457)
- panel beta lambda ocean / land-delta = -0.000000 / -0.000000
- positive gain fraction = 0.384
- median gain = -0.000010
- corr(domainmean lambda_local, global lambda_struct) = 0.969622
