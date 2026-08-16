# Phase 9: transfer asymmetry A vs ensemble forecast error growth

Protocol: `docs/PROTOCOL_PHASE9_PREDICTABILITY.md` (frozen 2026-08-15, deviations logged there).
Verdict: **PIPELINE_FAULT**

## Positive control
- error grows in 94 % of region-windows; spread monotone in 10 % (pass = False)

## H9a: between-region association (n = 12, one-sided)

| statistic | Spearman rho | p (perm, one-sided) | p (two-sided) | n |
|---|---|---|---|---|
| A vs lambda (self-analysis truth) | -0.252 | 0.800 | 0.430 | 12 |
| A vs lambda (ERA5 truth) | -0.315 | 0.853 | 0.319 | 12 |
| A vs mu (ensemble spread, truth-free) | +0.601 | 0.019 | 0.039 | 12 |
| A vs -T50 (predictability horizon) | +0.510 | 0.039 | 0.090 | 12 |
| A vs lambda, log-linear estimator | +0.469 | 0.059 | 0.124 | 12 |
| A vs mu, log-linear estimator | +0.685 | 0.006 | 0.014 | 12 |

## H9b: after removing controls

Controls: abs_lat_centre, land_frac, log_band_var, spectral_slope, r_init

| statistic | Spearman rho | p (perm) | p (two-sided) | n |
|---|---|---|---|---|
| residual A vs residual lambda | +0.385 | 0.114 | 0.217 | 12 |
| residual A vs residual mu | +0.622 | 0.016 | 0.031 | 12 |

### Confound diagnostics (raw Spearman)

| control | rho with A | rho with lambda |
|---|---|---|
| abs_lat_centre | -0.575 | +0.855 |
| land_frac | -0.566 | -0.196 |
| log_band_var | -0.650 | +0.811 |
| spectral_slope | -0.566 | -0.350 |
| r_init | +0.476 | -0.755 |

## H9d: within-region (region fixed effects, n = 48 windows)

| metric | slope | Spearman rho | p (perm) | pass |
|---|---|---|---|---|
| lambda | +0.049 | +0.307 | 0.084 | False |
| mu | +0.092 | +0.337 | 0.046 | True |

## C9-3: placebo descriptors vs lambda (two-sided)

| descriptor | Spearman rho | p (perm) |
|---|---|---|
| P_rho5_vs_lambda | +0.112 | 0.739 |
| log_band_var_vs_lambda | +0.811 | 0.005 |
| spectral_slope_vs_lambda | -0.350 | 0.261 |
| A (for comparison, |rho|) | 0.252 | - |

A beats the spectral placebo: **False**

## C9-2 / C9-4 controls
- null calibration: KS p(uniform) = 0.013, fraction p<0.05 = 0.040
- sign agreement across the three metrics: False ({'H9a_lambda': -1.0, 'H9a_lambda_era5': -1.0, 'H9a_mu': 1.0})

## Region table (primary sample, medians over 4 windows)

| region | A | lambda_gdas (1/day) | lambda_era5 | mu | T50 (h) | r(12 h) |
|---|---|---|---|---|---|---|
| R10_INDO | 2.40 | 0.118 | -0.099 | 0.459 | 13 | 0.557 |
| R1_WPWP | 2.24 | 0.118 | -0.030 | 0.225 | 19 | 0.453 |
| R5_SPCZ | 2.23 | 0.166 | 0.017 | 0.442 | 15 | 0.506 |
| R8_AUS | 1.86 | 0.207 | 0.157 | 0.239 | 17 | 0.438 |
| R9_NPAC | 1.72 | 0.224 | 0.182 | 0.216 | 24 | 0.397 |
| R2_NATL | 1.70 | 0.231 | 0.178 | 0.235 | 28 | 0.379 |
| R7_CONGO | 1.64 | 0.155 | 0.095 | 0.224 | 18 | 0.329 |
| R6_SATL | 1.60 | 0.213 | 0.188 | 0.183 | 21 | 0.400 |
| R12_SAM | 1.56 | 0.151 | 0.039 | 0.205 | 21 | 0.463 |
| R3_AMAZ | 1.54 | 0.077 | -0.276 | 0.143 | 12 | 0.608 |
| R11_EURO | 1.38 | 0.239 | 0.168 | 0.241 | 87 | 0.271 |
| R4_CASIA | 1.37 | 0.202 | 0.174 | 0.216 | 69 | 0.295 |
