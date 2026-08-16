# Phase 10: A and analysis-time mesoscale irrecoverability

Protocol: `docs/PROTOCOL_PHASE10_ANALYSIS_IRRECOVERABILITY.md` (frozen 2026-08-15).
Verdict: **NEGATIVE**  |  27 new domains, fixed by rule, unused by any earlier phase.

## Sanity (C10-1)

- analysis times per domain-window: min 182 (threshold 150)
- D range 0.409-0.828; r12 range 0.300-0.557
- pass = True

## Raw association (H10a, H10b) — positive

| test | rho | p (perm, one-sided) | n |
|---|---|---|---|
| A vs D | +0.340 | 0.042 | 27 |
| A vs r12 | +0.403 | 0.021 | 27 |
| A vs D, pooled 39 domains | +0.364 | 0.015 | 39 |
| A vs r12, pooled 39 domains | +0.457 | 0.001 | 39 |

## The association does not survive (H10c, H10e, H10f)

Controls: abs_lat_centre, land_frac, log_band_var, spectral_slope

| test | rho | p (perm) |
|---|---|---|
| residual A vs residual D | -0.191 | 0.814 |
| residual A vs residual r12 | -0.005 | 0.521 |
| within-domain, D (n=78) | -0.227 | 0.939 |
| within-domain, r12 (n=78) | -0.211 | 0.846 |

### Placebo (H10f): the spectral control wins outright

| descriptor | rho | p (perm, two-sided) |
|---|---|---|
| logvar_vs_D | -0.866 | 0.001 |
| logvar_vs_r12 | -0.660 | 0.001 |
| slope_vs_D | -0.088 | 0.645 |
| slope_vs_r12 | -0.517 | 0.005 |
| A vs D (for comparison) | +0.340 | - |
| A vs r12 (for comparison) | +0.403 | - |

A beats the placebos: **False**

## Why the raw association appeared

| pair | rho |
|---|---|
| log band variance vs raw absolute inter-analysis difference | +0.911 |
| log band variance vs relative D | -0.866 |
| A vs log band variance | -0.459 |
| A vs raw absolute inter-analysis difference | -0.379 |

The absolute disagreement of two assimilation systems tracks the amplitude of
the field almost deterministically (rho = +0.91). Dividing by that amplitude
therefore makes the relative measure very nearly the inverse of band variance
(rho = -0.87). `A` is itself anti-correlated with band variance (rho = -0.46),
so the positive `A`-vs-`D` correlation is that shadow and nothing more. In
absolute terms the two analyses disagree *less* where `A` is large.

## Domain table

| domain | A | log band var | D | r12 |
|---|---|---|---|---|
| G01_55N100E | 1.33 | -22.25 | 0.477 | 0.347 |
| G02_55N140E | 1.57 | -22.23 | 0.471 | 0.409 |
| G03_55N140W | 1.48 | -22.12 | 0.409 | 0.300 |
| G04_55N100W | 1.42 | -22.57 | 0.500 | 0.419 |
| G05_35N20E | 1.50 | -22.42 | 0.521 | 0.331 |
| G06_35N60E | 1.68 | -22.67 | 0.572 | 0.407 |
| G07_35N100E | 1.76 | -22.71 | 0.493 | 0.374 |
| G08_35N140E | 2.22 | -23.77 | 0.656 | 0.483 |
| G09_35N180W | 1.95 | -23.35 | 0.612 | 0.455 |
| G10_35N140W | 1.70 | -23.68 | 0.582 | 0.385 |
| G11_35N100W | 1.98 | -23.31 | 0.558 | 0.451 |
| G12_35N60W | 1.95 | -23.74 | 0.634 | 0.474 |
| G13_35N20W | 1.60 | -22.64 | 0.622 | 0.383 |
| G14_15N100E | 2.16 | -23.42 | 0.654 | 0.375 |
| G15_15N140W | 1.80 | -23.96 | 0.755 | 0.464 |
| G16_15N100W | 1.56 | -22.92 | 0.584 | 0.354 |
| G17_15N20W | 1.63 | -23.99 | 0.828 | 0.535 |
| G18_5S140E | 1.64 | -23.56 | 0.706 | 0.478 |
| G19_5S140W | 1.94 | -23.87 | 0.702 | 0.557 |
| G20_5S100W | 1.66 | -23.66 | 0.600 | 0.314 |
| G21_5S20W | 1.53 | -24.56 | 0.756 | 0.488 |
| G22_25S20E | 1.94 | -22.91 | 0.570 | 0.468 |
| G23_25S60E | 2.03 | -23.14 | 0.602 | 0.452 |
| G24_25S140E | 1.88 | -22.75 | 0.567 | 0.449 |
| G25_25S180W | 1.71 | -22.92 | 0.562 | 0.440 |
| G26_25S140W | 1.66 | -22.80 | 0.559 | 0.467 |
| G27_25S20W | 1.99 | -23.10 | 0.571 | 0.434 |
