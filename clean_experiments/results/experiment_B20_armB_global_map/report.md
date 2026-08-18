# Experiment B20 Arm B: the global tile map of anchored fine-P

Frozen spec: `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md`, "Arm B execution
spec" (2026-08-18; one implementation deviation logged there). 936
global tiles (6-deg rows x 700 km), 60S-60N, JFM + JAS 2023, anchored
fine-P as in Arm A (99 surrogates per tile-season); 34 tiles excluded
(orography > 1200 m).

## ARM_B_VERDICT: GLOBAL_MAP_ATTRIBUTED_DIM1

| Test | Result | Pass |
|---|---|---|
| H-B0 map consistency vs Arm A | rho = **0.840** over 133 overlapping tiles (bar 0.7) | YES |
| H-B1 primary attribution | LOSO R^2 = **0.536**, rotation-null p = 0.001 (null q95 = 0.455) | YES |
| H-B2 tier separation | tier-1 (boundary) R^2 = 0.482 (p = 0.002); tier-2 increment +0.053 | — |
| H-B3 effective dimension (QT-P3) | forward selection: **eke_syn alone R^2 = 0.456** = 85% of full; k80 = **1** (theory bar <= 3) | YES |
| H-B4 spectral placebo | slope target R^2 = 0.57 but with DIFFERENT loadings (orography/land vs eke/lat/sst/cape) — not spectrum-mediated | — |
| H-B5 LAI negative control | gain = -0.004, p = 0.832 — clean | YES |

Season contrast: median |JFM - JAS| = 0.016 (q90 = 0.048) — the map is
season-stable at tile scale.

## Reading

- The global geography of the scale-coupling texture is dominated by a
  single co-emergent coordinate: synoptic (storm-track) activity. The
  planet's P maxima are the Southern Ocean storm-track ring and the
  N Pacific / N Atlantic storm tracks; the minima are the deep
  convective cores (Amazon, Congo, Maritime Continent monsoon land).
  Loadings on anchored P: eke +0.66, abs_lat +0.66, sst_grad +0.55,
  cape -0.52; on the slope target instead: orog_std +0.74, land +0.67 —
  the same two-channel split as Arm A, now global.
- The rotation null retains all zonal structure (abs_lat is
  rotation-invariant), so H-B1's p = 0.001 certifies specifically the
  NON-ZONAL part of the attribution — the longitudinal anchoring of the
  map to where storm tracks and convective zones actually sit.
- Tier-1 boundary fields alone reach R^2 = 0.482: consistent with
  QT-1/QT-4 — the co-emergent tier-2 organization is itself pinned by
  the boundary conditions, so either tier reads the map; the tiers are
  facets, not competitors.
- QT-P3 is the first candidate prediction put to a scored test:
  PASS with k80 = 1 (bar was <= 3). The quenched field theta is, at
  this resolution, effectively LOW-DIMENSIONAL.

## Caveats

- Single year (2023) x two seasons; the season stability and the
  Arm-A consistency (which spans 2021-2024 windows) mitigate but do not
  replace a multi-year map.
- Tiles adjacent to the high-orography exclusion zone (Andes lee,
  Tibet rim) show isolated extreme values; treat single-tile features
  there as unreliable.
- eke_syn is measured from the same wind field as P (different
  functional, same data); the tier-1-only R^2 = 0.482 (independent
  fields) bounds any same-data inflation.

## Figures

- `fig1_global_map.png` — THE map: anchored fine-P, 60S-60N, with the
  12 programme boxes overlaid.
- `fig2_seasons_attribution.png` — per-season maps, season contrast,
  loadings and forward-selection curve.
