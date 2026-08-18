# Experiment B20 (Arm A): the geography of P — tile-level attribution

Protocol: `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md` (frozen 2026-08-18).
Carrier change: P as a map of 144 tiles (3x4 per equal-km box, ~667x700
km, anchored fine-P over 50-400 km with 99 phase surrogates per
tile-window), against the on-disk covariates [orog_std, land_frac,
coast_var, cape_mean, eke_syn]. All spatial statistics blocked at the
region level.

## VERDICT: DRIVERS_IDENTIFIED

| Test | Result |
|---|---|
| C20-1 sanity | box-medians of tile P reproduce the Phase-18 box signature, rho = **0.874** (bar 0.7) |
| H20a primary | LOO-region R^2 = **0.404**, block-perm p = 0.001 (perm q95 = -0.007) |
| H20b drivers | **cape_mean rho = -0.61 (p=0.002)**, **eke_syn rho = +0.59 (p=0.002)**; orog_std, land_frac, coast_var n.s. |
| H20c within-region | cape_mean 10/12 negative (p=0.001); eke_syn 11/12 positive (p=0.001) — both survive full removal of between-region confounds |
| C20-2 placebo | slope target attributable too (R^2=0.31, p=0.004) BUT with different loadings (orog +0.82, land +0.67, CAPE/EKE ~0) -> the frozen "spectrum-mediated" clause is NOT triggered |

## The two-layer reading (mediation, post-hoc descriptive, labeled in summary)

Residualizing the anchored tile-P on the tile spectra (4 log band
variances + slope) before attribution:

- pooled attribution disappears (R^2 = -0.06, p = 0.15);
- the **negative CAPE association survives within regions (10/12,
  mean rho = -0.34)**;
- the EKE association does not (8/12, mean rho = +0.03).

So: **synoptic (storm-track) activity raises fine-band envelope coupling
largely through the same variance structure the spectrum sees; deep
convection lowers it beyond what the spectrum sees.** The intermittent,
locally generated fine-scale vorticity of convective regimes decouples
adjacent scales; organized baroclinic cascades couple them.

## First answers to the "what shapes the P regions" list

- Centers of action / storm tracks: YES — the strongest positive driver
  (eke_syn), spectrum-shared. Visible directly in fig1: R9_NPAC,
  R2_NATL, R6_SATL tiles are the global maxima.
- Convective regime: YES — the strongest beyond-spectrum driver, with a
  NEGATIVE sign (Amazon and western Congo cores are P minima).
- Orography: no individual effect at tile scale (it drives the spectral
  slope instead — C20-2 loadings); the R4_CASIA northeast tile (P
  minimum of the whole map) is a suggestive single case, not a result.
- Land surface / coastlines: nothing at this scale.
- Anthropogenic / biological: not on disk; pre-declared as Arm-B
  negative controls.

## Caveats

- n = 12 regions is still the blocking unit; the block permutation is
  honest but the covariate ranges are region-confounded for covariates
  with little within-region variance (orography especially).
- cape_mean is a 2021-2024 climatology; window-matched CAPE would
  sharpen the within-region test (Arm-B refinement).
- The verdict word DRIVERS_IDENTIFIED refers to the frozen H20a/H20b
  rule; the mediation split above is the physically meaningful summary
  and the manuscript should carry both layers.

## Figures

- `fig1_tile_maps.png` — the internal geography of anchored fine-P,
  12 boxes x 12 tiles, coastlines overlaid.
- `fig2_attribution.png` — covariate scatters with block-null p values;
  within-region correlation heatmap.
