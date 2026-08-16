# Phase 11: A and the mesoscale deficit of learned forecast models

Protocol: `docs/PROTOCOL_PHASE11_LEARNED_REFINEMENT.md` (frozen 2026-08-15).
Verdict: **NEGATIVE**  |  39 domains, 20 initialisations, JFM+JAS 2020, lead 120 h.

## The mechanism ladder works exactly as predicted (H11d: PASS)

| system | mechanism | median deficit | rho with A |
|---|---|---|---|
| graphcast | deterministic, L2 | +0.687 | +0.594 |
| pangu | deterministic, L2 | +0.552 | +0.404 |
| gencast_mean | conditional mean of a generative ensemble | +0.845 | +0.429 |
| gencast_member | generative sample | +0.022 | -0.133 |
| hres | physical integration | +0.024 | -0.012 |

Separation (deterministic minus negative controls): **+0.571**; neither control passes the H11a criterion.

The blurring is real, large and strictly confined to systems that compute a
conditional mean. HRES loses essentially no mesoscale variance and a single
GenCast member loses none; the GenCast *ensemble mean* blurs most of all.

## But A is not what predicts it (H11c, H11f: FAIL)

| test | rho | p (perm) |
|---|---|---|
| A vs GraphCast deficit (raw) | +0.594 | 0.001 |
| A vs Pangu deficit (raw) | +0.404 | 0.013 |
| residual A vs residual deficit, GraphCast | -0.077 | 0.666 |
| residual A vs residual deficit, Pangu | +0.041 | 0.392 |
| **placebo** log band variance vs GraphCast | **-0.685** | 0.001 |
| **placebo** log band variance vs Pangu | **-0.807** | 0.001 |
| within-domain seasonal arm, GraphCast (n=54) | +0.153 | 0.285 |

rho(A, log band variance) over the 39 domains = **-0.559**

Plain band variance predicts the blurring better than `A` does, for both
deterministic models, and `A` carries nothing once it is removed. Third
target, third time the same spectral placebo wins.

## Domain table

| domain | A | log band var | GraphCast | Pangu | GenCast mean | GenCast member | HRES |
|---|---|---|---|---|---|---|---|
| G01_55N100E | 1.33 | -22.25 | +0.378 | +0.493 | +0.623 | +0.014 | +0.128 |
| G02_55N140E | 1.57 | -22.23 | +0.602 | +0.329 | +0.940 | +0.079 | -0.158 |
| G03_55N140W | 1.48 | -22.12 | +0.465 | +0.453 | +0.507 | -0.002 | +0.052 |
| G04_55N100W | 1.42 | -22.57 | +0.654 | +0.634 | +0.897 | +0.336 | +0.142 |
| G05_35N20E | 1.50 | -22.42 | +0.346 | +0.468 | +0.626 | +0.061 | +0.043 |
| G06_35N60E | 1.68 | -22.67 | +0.536 | +0.511 | +0.630 | -0.037 | -0.178 |
| G07_35N100E | 1.76 | -22.71 | +0.612 | +0.526 | +0.682 | +0.057 | +0.083 |
| G08_35N140E | 2.22 | -23.77 | +0.836 | +0.631 | +0.929 | -0.478 | -0.044 |
| G09_35N180W | 1.95 | -23.35 | +0.741 | +0.667 | +0.840 | +0.065 | -0.216 |
| G10_35N140W | 1.70 | -23.68 | +0.745 | +0.658 | +0.826 | +0.022 | +0.095 |
| G11_35N100W | 1.98 | -23.31 | +0.696 | +0.588 | +0.829 | -0.278 | +0.252 |
| G12_35N60W | 1.95 | -23.74 | +0.833 | +0.678 | +0.933 | +0.073 | -0.474 |
| G13_35N20W | 1.60 | -22.64 | +0.484 | +0.535 | +0.731 | +0.111 | +0.004 |
| G14_15N100E | 2.16 | -23.42 | +0.621 | +0.649 | +0.679 | -0.101 | +0.174 |
| G15_15N140W | 1.80 | -23.96 | +0.822 | +0.703 | +0.909 | -0.006 | +0.269 |
| G16_15N100W | 1.56 | -22.92 | +0.416 | +0.454 | +0.448 | +0.030 | -0.042 |
| G17_15N20W | 1.63 | -23.99 | +0.730 | +0.806 | +0.777 | +0.169 | +0.170 |
| G18_5S140E | 1.64 | -23.56 | +0.818 | +0.541 | +0.902 | -0.053 | -0.069 |
| G19_5S140W | 1.94 | -23.87 | +0.853 | +0.697 | +0.933 | -0.158 | -0.166 |
| G20_5S100W | 1.66 | -23.66 | +0.508 | +0.485 | +0.501 | +0.070 | -0.102 |
| G21_5S20W | 1.53 | -24.56 | +0.822 | +0.754 | +0.897 | -0.050 | -0.240 |
| G22_25S20E | 1.94 | -22.91 | +0.687 | +0.556 | +0.845 | +0.036 | -0.185 |
| G23_25S60E | 2.03 | -23.14 | +0.752 | +0.474 | +0.930 | +0.123 | -0.277 |
| G24_25S140E | 1.88 | -22.75 | +0.712 | +0.552 | +0.875 | +0.097 | +0.024 |
| G25_25S180W | 1.71 | -22.92 | +0.810 | +0.538 | +0.961 | -0.061 | -0.110 |
| G26_25S140W | 1.66 | -22.80 | +0.781 | +0.458 | +0.939 | +0.069 | -0.236 |
| G27_25S20W | 1.99 | -23.10 | +0.706 | +0.615 | +0.912 | -0.082 | -0.091 |
| R10_INDO | 2.40 | -24.55 | +0.850 | +0.832 | +0.905 | +0.256 | +0.404 |
| R11_EURO | 1.38 | -21.60 | +0.435 | +0.433 | +0.561 | +0.129 | +0.192 |
| R12_SAM | 1.56 | -22.73 | +0.577 | +0.613 | +0.697 | -0.041 | -0.042 |
| R1_WPWP | 2.24 | -23.89 | +0.606 | +0.489 | +0.621 | +0.016 | +0.256 |
| R2_NATL | 1.70 | -22.41 | +0.637 | +0.434 | +0.946 | +0.204 | -0.040 |
| R3_AMAZ | 1.54 | -24.15 | +0.731 | +0.799 | +0.809 | -0.151 | +0.297 |
| R4_CASIA | 1.37 | -21.87 | +0.278 | +0.358 | +0.460 | -0.034 | +0.046 |
| R5_SPCZ | 2.23 | -24.25 | +0.899 | +0.743 | +0.966 | +0.024 | +0.192 |
| R6_SATL | 1.60 | -22.58 | +0.608 | +0.421 | +0.915 | -0.044 | -0.437 |
| R7_CONGO | 1.64 | -23.26 | +0.422 | +0.612 | +0.476 | +0.089 | +0.076 |
| R8_AUS | 1.86 | -22.92 | +0.598 | +0.615 | +0.868 | -0.225 | +0.031 |
| R9_NPAC | 1.72 | -22.53 | +0.743 | +0.456 | +0.953 | +0.005 | +0.038 |
