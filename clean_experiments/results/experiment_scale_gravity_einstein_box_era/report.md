# Einstein in the Atmospheric Column (ERA5 Box)

## Goal
- Test local Einstein-type scale-space closure on real atmosphere data:
  `E[Lambda | Pi]` in forcing/inertial/dissipation ranges.

## Input
- input: `data/processed/wpwp_era5_2017_2019_experiment_M_input.nc`
- field_set: `wind`
- vector vars: `u`, `v`
- shape after slicing: `time=360, lat=81, lon=161`
- median dx,dy (km): `27.80`, `27.80`
- time range: `2017-01-01 00:00:00` -> `2017-03-31 18:00:00`

## Scale and model setup
- scale_edges_km: `50,100,200,400,800,1600,3200`
- regimes: dissipation `< 100.0` km, inertial `[100.0, 1000.0]` km, forcing `> 1000.0` km
- n_modes_per_var requested/effective(min across bands): `6/5`
- rolling window W: `20`
- ridge: `1.00e-06`, cov_shrinkage: `0.050`
- lambda sign requested/effective: `0/-1`

## Density-matrix checks
- max `|Tr(rho)-1|`: `4.441e-16`
- min eigenvalue(rho): `2.302e-04`

## Bandwise results
- band 0 (dissipation), center=70.7 km: R2_binned=0.371, slope_binned=4.476e-14, p_slope=0.0160
- band 1 (inertial), center=141.4 km: R2_binned=0.020, slope_binned=-5.364e-04, p_slope=1.0000
- band 2 (inertial), center=282.8 km: R2_binned=0.017, slope_binned=1.799e-04, p_slope=0.3340
- band 3 (inertial), center=565.7 km: R2_binned=0.252, slope_binned=1.669e-04, p_slope=0.0490
- band 4 (forcing), center=1131.4 km: R2_binned=0.301, slope_binned=9.538e-05, p_slope=0.0230
- band 5 (forcing), center=2262.7 km: R2_binned=0.748, slope_binned=-2.001e-05, p_slope=1.0000

## Regime aggregates
- forcing: n=682, R2_binned=0.179, slope_binned=-7.723e-06, p_slope=1.0000
- inertial: n=1023, R2_binned=0.520, slope_binned=1.367e-04, p_slope=0.0080
- dissipation: n=341, R2_binned=0.371, slope_binned=4.476e-14, p_slope=0.0180

## Pass criteria (ERA)
- inertial R2_binned >= 0.40: `0.520`
- inertial slope_binned > 0: `1.367e-04`
- inertial slope significance p <= 0.05: `0.0080`
- forcing/dissipation breakdown: checked via lower R2 vs inertial or slope sign inversion.
- PASS_ALL: `True`

## Note
- Atmospheric noise is expected to reduce raw fit quality.
- Binned inertial relation is the primary criterion for this box-level closure test.
