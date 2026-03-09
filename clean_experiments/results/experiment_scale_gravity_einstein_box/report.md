# Einstein in a Box (Synthetic 2D Cascade)

## Goal
- Empirical local check of Einstein-type closure in scale space: `Lambda ~ Pi`.
- `Lambda_b(t) = Re Tr(F_b rho_b)` from noncommuting interscale transport.
- `Pi_b(t)` from spectral energy-flow proxy within each scale band.

## Run setup
- seed: `20260309`
- grid: `96 x 96`
- timesteps: `540`
- scale edges: `4,8,16,32,48`
- n_modes_per_var requested/effective: `6/6`
- rolling window W: `20`
- ridge: `1.00e-06`
- cov_shrinkage: `0.050`
- lambda sign requested/effective: `0/-1`

## Data quality checks
- max trace error `|Tr(rho)-1|`: `4.441e-16`
- min eigenvalue over all rho: `1.006e-04`

## Bandwise linearity
- band 0 (dissipation): mu=3.085, center=5.657, R2_raw=0.003, R2_binned=0.229, slope_binned=1.697e+03
- band 1 (inertial): mu=2.085, center=11.314, R2_raw=0.003, R2_binned=0.099, slope_binned=-5.675e+02
- band 2 (inertial): mu=1.085, center=22.627, R2_raw=0.009, R2_binned=0.525, slope_binned=2.354e+02
- band 3 (forcing): mu=0.292, center=39.192, R2_raw=0.009, R2_binned=0.444, slope_binned=-4.332e+01

## Regime aggregates
- forcing: n=521, R2_raw=0.009, R2_binned=0.444, slope_binned=-4.332e+01
- inertial: n=1042, R2_raw=0.015, R2_binned=0.727, slope_binned=2.935e+02
- dissipation: n=521, R2_raw=0.003, R2_binned=0.229, slope_binned=1.697e+03

## Success criterion
- criterion: inertial `R2_binned >= 0.70` and positive slope.
- inertial R2_binned: `0.727`
- inertial slope_binned: `2.935e+02`
- PASS: `True`

## Interpretation
- This run validates the local synthetic option-C protocol only.
- It does not replace atmospheric/JHTDB validation for external physical claims.
