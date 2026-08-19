# AUDIT-2 — is P an intermittency measure? Results

Protocol: `docs/PROTOCOL_AUDIT2_INTERMITTENCY_HEADTOHEAD.md` (frozen
before computation). Code:
`clean_experiments/experiment_A2_intermittency_headtohead.py`. Sample:
936 tiles x 2 seasons (JFM/JAS 2023), 902 kept; same mask, band ladder
and the same 99 phase-surrogate realisations as Phase 20 Arm B and
AUDIT-1.

## Control

- **C-A2-0 PASS.** Recomputed anchored P reproduces the committed Arm-B
  map at Spearman = 1.000, max raw difference 3.0e-8.

## A2a (primary) — VERDICT: CASCADE_DISTINCT

Spearman(P, P_cascade) = **-0.417** (bars: 0.90 identity, 0.60
relatedness). P is not the Kolmogorov-Obukhov log-normal cascade
correlation.

The mechanism behind the verdict is visible in the unanchored numbers
(medians over 902 tiles):

| quantity | real | phase surrogate | anchored |
|---|---|---|---|
| P (envelope-envelope correlation) | 0.770 | 0.512 | **0.256** |
| P_cascade (KO62 prediction from the log-envelope variances) | 0.666 | 0.650 | **0.010** |

The cascade prediction is reproduced by the phase-randomized field: it
is a spectrum-level quantity with essentially no beyond-spectrum
component. The observed envelope coupling exceeds the cascade
prediction (0.77 vs 0.67) and the excess is exactly what the anchoring
keeps. Even the raw geographies differ: Spearman(P_real, P_cascade_real)
= +0.142.

## A2b (reported) — the intermittency parameter is a substantial relative

| pair | Spearman |
|---|---|
| P vs INT_sig2 (variance of the log-envelopes, the KO62 intermittency parameter) | **+0.692** |
| P vs INT_flat (log10 flatness of the band fields) | -0.223 |
| P vs INT_mu (intermittency exponent, flatness vs scale) | +0.100 |

This is the honest cost of the audit: the anchored intermittency
parameter shares ~48% of P's rank variance. It was not the frozen
primary and is not scored, but it must be reported in the papers. P is
not the cascade correlation and not the intermittency exponent; it is
substantially, though not exhaustively, related to the local
log-envelope variance.

## A2c (reported) — attribution transfer

| target | LOSO R^2 | p (rotation) | cross-season reliability | share of own ceiling |
|---|---|---|---|---|
| P | +0.536 | 0.001 | 0.766 | 0.91 |
| P_cascade | +0.255 | 1.000 | 0.601 | 0.71 |
| INT_sig2 | +0.296 | 0.001 | 0.746 | 0.53 |
| INT_flat | +0.383 | 0.001 | 0.768 | 0.65 |
| INT_mu | +0.359 | 0.001 | 0.788 | 0.58 |

`P_cascade` fails the rotation null (p = 1.000) because its geography is
almost purely zonal (abs_lat loading -0.63), and the longitude-rotation
null preserves zonal structure by construction. The intermittency
statistics load on orography/land (INT_flat: orog_std +0.52, land_frac
+0.49, coast_var +0.50) — the loading pattern of the SPECTRAL SLOPE
target of Phase 20, not of P (abs_lat +0.66, eke_syn +0.66, cape -0.52).
The two families are attributed to different physics.

## A2d (scored secondary) — increment over the frozen covariates

- R^2 (8 Phase-20 covariates) = 0.536
- R^2 (covariates + 4 intermittency statistics) = **0.761**,
  increment **+0.225**, p = 0.001 (rotation null q95 = 0.003)
- Frozen bar: R^2 >= 0.9 x 0.603 -> **map declared jointly explained**.

**Post-hoc decontamination (reported, not scored; see the protocol's
deviation log).** The frozen bar is contaminated: P and the
intermittency statistics are computed from the SAME tile-season sample,
so they share sampling noise, and the ceiling 0.603 was estimated from
cross-season agreement, which does not contain that shared noise.
Predicting one season's P from the OTHER season's intermittency
statistics removes the shared noise:

| target | covariates | + intermittency of the other season | + intermittency of the same season |
|---|---|---|---|
| P (JAS) | +0.450 | +0.531 (increment **+0.081**) | +0.715 (increment +0.265) |
| P (JFM) | +0.491 | +0.568 (increment **+0.077**) | +0.748 (increment +0.257) |

About 70% of the scored increment is shared estimation noise. The
signal-level increment is **+0.08**, real and reproducible in both
directions, but far below the "jointly explained" reading that the
contaminated bar produced. The papers must quote +0.08, with the
contaminated +0.225 shown as the naive number and this control beside
it.

## Consequences for the papers

1. P is not the log-normal cascade correlation (A2a) and not the
   intermittency exponent (A2b). The cascade prediction has no
   beyond-spectrum component at all.
2. The local intermittency parameter (log-envelope variance) is a
   substantial relative (rho = +0.69) and must be cited and reported,
   not hidden.
3. Intermittency statistics add +0.08 LOSO R^2 to P's attribution after
   noise decontamination; the geography of P is not reducible to them.
4. The intermittency family loads like the spectral slope
   (orography/land), P loads like the storm-track/convection pair.
   Reporting this side by side is the cleanest way to show that P is a
   different physical target.
