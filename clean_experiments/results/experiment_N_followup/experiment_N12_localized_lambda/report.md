# experiment_N12_localized_lambda: Base source proxy + localized multiscale lambda corrections

## Selected config (one-SE)
- model: `n12_ms_regime` (base + localized c1 terms + extreme interactions)
- q_extreme: 0.80
- ridge_alpha: 1.0e-06

## Out-of-time test (2019)
- gain_all: 0.000854
- gain_extreme: 0.001514
- gain_non_extreme: 0.000184
- CI95 gain_all: [-0.000325, 0.002298]
- permutation p-value: 0.035461

## Quarterly rolling-origin (2019)
- mean gain_all: 0.000760
- min gain_all: -0.000155
- positive-quarter fraction: 0.750
