# Experiment B21: theory-candidate tests, round 1

Protocol: `docs/PROTOCOL_PHASE21_QT_TESTS.md` (frozen 2026-08-18 with the
downloads launched at freeze time). Code
`clean_experiments/experiment_B21_qt_tests.py`. Seed 20260818.

## PHASE21_VERDICT: (P5_SUPPORTED, P1_UNDERPOWERED, P4_SUPPORTED)

### Arm P5 — sampling clock: SUPPORTED

318 window x size pairs (80 rw x factors 0.35-1.0): pooled Spearman
rho(ln tau, ln T_cross) = **+0.356, p = 0.001** (within-region
permutation null; the frozen "region-block" wording is implemented as
within-region permutation and logged — region-label rotation would
preserve the size structure and have no power). Within-window
consistency: **79/79 windows** show positive tau-vs-size scaling.
Log-log slope 0.32, CI [0.12, 0.62]: sub-linear — tau grows with the
observation window but slower than pure advective crossing (slope 1);
the estimator clock is a mixture of pattern residence and pattern
lifetime. QT-3/QT-P5's core claim — the clock belongs to the sampling
window, not to the band scale — stands (band flatness was already
established in B19).

### Arm P1 — forced drift: UNDERPOWERED (right sign, not significant)

144 tiles, 46 years. Spectrum-fixed P trend pooled over the top
CAPE-trend tercile: D_obs = **-3.9e-5 / yr** (right sign, ~5x the
cross-sectionally predicted D_pred = -0.74e-5 / yr), but p = 0.081 and
the cluster-bootstrap CI [-1.04e-4, +0.29e-4] covers both D_pred and
zero -> UNDERPOWERED by the frozen rule. Not a falsification and not a
confirmation; the record length, not the effect, is the limit. The
protocol's power route forward: more seasons per year (the current
yearly medians discard the seasonal contrast) or the global carrier.

### Arm P4 — the tropical excess has a name: SUPPORTED (IPO)

Named-before-consultation indices (PDO, AMO, DMI, IPO-TPI), max-statistic
over 8 index x smoothing combinations, joint circular-shift null:

| Region | T = max rho | best index | p | significant |
|---|---|---|---|---|
| R1_WPWP | 0.311 | pdo_5yr | 0.155 | no |
| R3_AMAZ | 0.453 | **ipo_tpi_annual** | 0.001 | yes |
| R5_SPCZ | 0.602 | **ipo_tpi_annual** | 0.001 | yes |

Two of three excess regions significant WITH THE SAME index -> the
frozen coherence rule fires: the carrier-robust tropical interannual
variance excess (E10) is associated with the **Interdecadal Pacific
Oscillation**. ENSO (ONI) was dead at 30x power while IPO is not — the
excess is a genuinely decadal, not interannual-ENSO, tier-2 mode.
R1_WPWP (the weakest excess, F=1.75) stays unattributed.

## Candidate bookkeeping (per its standing rules)

- QT-P5: prediction upgraded to SUPPORTED (E11).
- QT-P3: already PASS (Phase 20 Arm B).
- QT-P4: SUPPORTED in its named-predictor form; E10 anomaly ->
  attributed row (IPO), R1 residual noted.
- QT-P1: OPEN (underpowered; right sign). Not entered as pass or fail.
- QT-P2: untested (deferred; needs model fields).

## Figure

- `fig1_qt_tests.png` — tau vs T_cross scaling; D_obs vs D_pred;
  per-region max-statistic vs null.
