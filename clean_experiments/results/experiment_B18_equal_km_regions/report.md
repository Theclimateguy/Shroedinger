# Experiment B18: equal-km regionalization (design control for box geometry)

Protocol: `docs/PROTOCOL_PHASE18_EQUAL_KM_REGIONS.md` (frozen 2026-08-17,
seven pre-computation deviations logged there, incl. renumbering 16->18 and
zonal extent 4000->2800 km to fit on-disk data). Code
`clean_experiments/experiment_B18_equal_km_regions.py`; seeds: permutations
20260817, surrogates inherit the Phase-2/3 per-tag scheme (199 primary /
held-out, 49 aux arms). Samples: 48 primary region-windows (R1-R12 x
W9-W12, 2023-2024, data/b3), 32 held-out (R5-R12 x W5-W8, 2021-2022,
data/b2b). Degree-box references recomputed from the stored Phase-3 JSONs
on the identical windows.

## VERDICT: CONFIRMED_PHYSICAL

The conditional verdict on P's beyond-spectrum component is lifted UPWARD:
with the latitude-dependent km geometry removed from the sampling design
(not regressed out), every scored signature survives. The beyond-spectrum
component of anchored P is physical, not cartographic.

| Test | Result | Bar | Pass |
|---|---|---|---|
| C18-1 sanity | coverage min 1.000; sigma_b log-ratio median 0.01..-0.07 (largest band -7%, expected from narrower zonal window) | coverage >= 0.95 | YES |
| H18a A negative control | regional A deg-vs-km Spearman rho = **0.972**; km signature diff +0.508 p=0.001 | rho >= 0.9 AND p < 0.05 | YES |
| H18b anchored-P clustering (km, 48 rw) | diff **+1.256, p=0.001** (degree same windows: +1.327, p=0.001) | p < 0.05 | YES |
| H18c geography preservation | pooled rho **0.926**; per region 0.7 (R2, R12) to 1.0 | descriptive | — |
| H18d held-out 2021-22 (km, 32 rw) | diff **+1.564, p=0.001**, sign agrees | sign + p < 0.05 | YES |
| H18e Congo-Amazon anchored rho_5 | contrast deg +0.242 -> km +0.136 (**56% retained**, same sign) | sign + >= 50% | YES |
| C18-2 null calibration | KS p = 1.0 — as implemented tautological (perm ranks vs own distribution); no evidence of miscalibration, weight zero | logged | — |
| C18-3 spectral placebo | spec clustering diff +2.538 p=0.001 — **stronger than anchored P**; see note | logged | — |
| C18-4 extent sensitivity 0.9x/1.1x | min per-region rho 0.928 / 0.930, all >= 0.9 | logged | — |
| P18-3 dynamic-scale arm | anchored-P clustering diff +1.598 p=0.001 (Ld-scaled bands, descriptive) | not scored | — |

## The decisive secondary: beyond-spectrum residual clustering

The geometry question was raised by the statistical control (2026-08-13):
adding [abs lat, dx, domain width] collapsed P's beyond-spectrum residual
clustering (+0.454 p=0.003 -> +0.216 p=0.100 on raw P). On the same 48
windows, LOO-residualized on the 7 spectral features:

| Descriptor | boxes | diff | p |
|---|---|---|---|
| anchored P residuals | degree | +0.201 | 0.080 |
| anchored P residuals | **km** | **+0.218** | **0.033** |
| raw P residuals | **km** | **+0.347** | **0.007** |

Under the design control the beyond-spectrum clustering is significant,
where the degree-box version was marginal. The earlier "collapse" under
covariate regression is consistent with the covariates absorbing real
between-region signal correlated with latitude, not only geometry: when the
geometry is equalized by construction, the residual signal is still there.

## C18-3 note (reported as frozen)

Plain spectral features cluster more strongly (+2.538) than anchored P
(+1.256) on km boxes — as they always have on degree boxes. P's claim has
never been "strongest clustering"; it is information beyond the spectrum,
which is exactly what the residual test above isolates and confirms. Both
numbers are reported per the frozen control.

## Descriptive notes

- Weakest geography preservation: R2_NATL and R12_SAM (rho 0.7) — the two
  regions where the km re-boxing changes the sampled area most (midlatitude
  shrink / the 3851-km-wide degree box at 30S). Profile shapes (fig2) move
  within the window spread.
- Amazon anchored rho_5 rises on km boxes (0.12 -> 0.20) while Congo is
  stable (0.36 -> 0.33); the flagship contrast narrows but keeps sign and
  >half its magnitude on the primary windows.
- Band-variance shift (fig6, right): medians ~0 in bands < 400 km, -3%/-7%
  in the two coarsest bands — the expected pure-sampling effect of the
  narrower zonal window; no descriptor-side anomaly.
- Dynamic-scale arm agrees with the fixed-km reading (both p=0.001), which
  under P16-3/P18-3 is the strongest admissible robustness statement.
- A caveat (Phase 12) unchanged: A is scored here only as a geometry-robust
  negative control; no irreversibility interpretation is implied.

## Figures

- `fig1_region_map.png` — degree vs equal-km boxes, all 12 regions.
- `fig2_anchored_profiles.png` — anchored P, degree vs km, per region.
- `fig3_clustering.png` — H18b/H18d permutation nulls + signature strengths.
- `fig4_A_control.png` — H18a regional A, degree vs km (rho=0.972).
- `fig5_congo_amazon.png` — H18e contrast, anchored and raw.
- `fig6_sensitivity.png` — C18-4 extent sensitivity; C18-1 variance shift.
