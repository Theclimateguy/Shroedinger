# F5 Spatial Fractal Visualization

- input: `data/processed/wpwp_era5_2017_2019_experiment_M_vertical_input.nc`
- high-|Lambda| threshold (q=0.80): 3.326077e-05, n=600
- low-|Lambda| threshold (q=0.20): 4.350283e-06, n=600
- patch geometry: size=15, stride=6, grid=12x25

## Consistency
- corr(delta_beta, delta_variogram) = 0.300261
- sign agreement fraction = 0.550

## Mean Spatial Deltas (high-|Lambda| minus low-|Lambda|)
- mean delta PSD beta = -0.026560
- mean delta variogram slope = 0.000399
- mean composite delta = -0.000000

## Main figures
- `plot_F5_fractal_maps_panel.png`
- `plot_F5_fractal_composite_delta.png`
- `plot_F5_psd_effect_map.png`
- `plot_F5_variogram_effect_map.png`
- `plot_F5_estimator_sign_agreement.png`
