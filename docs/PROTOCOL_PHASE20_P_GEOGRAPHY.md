# Protocol Phase 20: the geography of P — tile-level attribution (Arm A)

Status: FROZEN before any Phase-20 computation.
Date frozen: 2026-08-18. Deviations logged with timestamp at the bottom.

## 0. Lineage

Phase 18 established P's regional signature as physical (design control);
Phase 19 established that its dynamics is fast, regime-blind and
scale-undifferentiated on both geometries. The open question moves from
"does P move" to "what sets its static regional value". The existing
attribution attempt (B6b: [abs lat, shear, cape_mean, land_frac,
orog_std] -> rho_5 at n=12 regions, LOO R^2 = -0.62, p = 0.65) failed at
a sample size that cannot honestly host any further covariate family.
Phase 20 changes the data carrier: P as a TILE MAP (n ~ 144 spatial
units), not 12 box scalars. Arm A runs entirely on on-disk data inside
the validated equal-km boxes; Arm B (global sliding-window P map with
external covariate fields: storm-track EKE climatology, SST gradients,
LAI, nightlights) is DECLARED here and frozen separately after Arm A
reports.

## 1. Questions and frozen predictions

- H20a (primary): is the tile-level anchored fine-P attributable to
  physical surface/regime covariates beyond what tile spectra explain?
- Frozen covariate families and mechanisms:
  1. dynamical regime — synoptic EKE (storm-track activity), CAPE
     climatology (convective regime);
  2. orography — fine-scale vorticity injection (orog_std);
  3. surface type — land fraction, coastline presence.
  Anthropogenic and biological covariates are NOT available on disk;
  they are Arm-B territory and are pre-declared there as NEGATIVE
  CONTROLS (mechanistically implausible at 850 hPa, 50-400 km).
- P20-1: no directional prediction is frozen for which family wins;
  the phase is exploratory-confirmatory at the family level with one
  primary scalar and blocked nulls (see 4). A null result
  (NO_TILE_ATTRIBUTION) is publishable and final for the on-disk
  covariate set.

## 2. Data (fixed, all on disk)

- Wind: the 80 km-cropped region-windows of Phase 18/19 (data/b3 48 +
  data/b2b 32), equal-km 1.0x boxes, native grid.
- Covariates: data/b4cov/era5_invariants_global.nc (z -> orography,
  lsm -> land fraction); data/b4cov/era5_cape_monthly_<region>.nc
  (gridded monthly CAPE 2021-2024); synoptic EKE computed from the wind
  files themselves.

## 3. Quantities (fixed)

- Tiles: each equal-km box is split 3 x 4 (rows x cols) into 12 equal
  index-mask tiles (~667 x 700 km). 144 distinct tiles; the analysis
  unit is the TILE, with values pooled as medians over that tile's
  available windows.
- Fine-P per tile-window: the frozen envelope machinery restricted to
  ELLS_FINE = [50, 100, 200, 400] km (3 band steps), interior mask at
  400 km (frozen cap rule). Tile P_fine = median over time of the mean
  of the 3 step correlations.
- Anchoring: per tile-window, 99 phase-surrogates (frozen Phase-2/3
  seed scheme, tag salted with the tile id); anchored tile P_fine =
  real - surrogate median. PRIMARY TARGET: tile median (over windows)
  of anchored P_fine.
- Tile spectral features: fine-range log band variances (4) + isotropic
  slope fitted over 50-400 km; used for the spectral placebo and the
  residualized secondary target.
- Covariates per tile: orog_std (std of z/g over tile), land_frac
  (mean lsm), coast_var (std of lsm over tile), cape_mean (2021-2024
  climatological tile mean), eke_syn (tile-interior mean variance of
  u', v' about a 5-day running mean, median over the tile's windows).
  All z-scored over the 144 tiles.

## 4. Criteria (fixed)

- C20-1 (sanity, computed first): the box-median of tile anchored
  P_fine must reproduce the box-level anchored fine-P signature:
  Spearman rho >= 0.7 across the 12 regions between box-median tile
  values and the Phase-18 box-level anchored means (bands 1-3
  component). Failure -> TILE_SIGNAL_ABSENT; attribution not scored;
  Arm B must redesign the carrier.
- H20a (primary, ONE scalar): leave-ONE-REGION-out R^2 of OLS
  predicting tile anchored P_fine from the 5 covariates (all 12 tiles
  of the held-out region predicted per fold). Null: 999 region-block
  permutations (covariate blocks rotated between regions, within-region
  structure preserved), p < 0.05 AND R^2 > 0.
- H20b (attribution detail, secondary): per-covariate ablation drop in
  LOO-region R^2; single-covariate Spearman with the same block null.
  Reported with family labels; no family-level pass/fail.
- H20c (within-region, strongest confound control): per region, the
  Spearman correlation of tile anchored P_fine with each covariate
  ACROSS the region's 12 tiles; scored by cross-region sign consistency
  (>= 10/12 same sign) plus the region-permutation combined p. This
  test cannot be produced by any between-region confound.
- C20-2 (spectral placebo): H20a repeated with (a) target = tile
  spectral slope, (b) target = raw (un-anchored) tile P_fine
  spectrum-residualized per tile. Reported; if the anchored-P
  attribution pattern is reproduced by the slope target with similar
  R^2 and loadings, the phase reports spectrum-mediated attribution,
  not beyond-spectrum attribution.
- Multiplicity: H20a is the single scored primary. H20b/H20c/C20-2 are
  reported in full whatever they show.

## 5. Verdict rule (fixed)

- TILE_SIGNAL_ABSENT: C20-1 fails.
- DRIVERS_IDENTIFIED: H20a passes AND at least one covariate ablation
  is individually significant under the block null.
- WEAK_ATTRIBUTION: H20a passes, no single covariate survives ablation.
- NO_TILE_ATTRIBUTION: H20a fails. Final for the on-disk covariate set;
  Arm B remains the only continuation.

## 6. Compute plan

- Code: clean_experiments/experiment_B20_p_geography.py
  (stages: tiles / tests), results under
  clean_experiments/results/experiment_B20_p_geography/.
- Seed 20260818 for Phase-20 randomness (permutations); surrogate seeds
  inherit the Phase-2/3 per-tag scheme salted with tile ids.
- Figures: visualize_B20_p_geography.py (tile maps of anchored P over
  each region with covariate underlays — the first pictures of P's
  internal geography).

## 7. Explicitly out of scope

- Arm B (global sliding-window map, external covariates incl.
  anthropogenic/biological negative controls) — separate frozen
  protocol after Arm A.
- Any dynamical language (Phases 14-19 closed it).
- Coarse bands (800-1600 km) at tile scale — geometrically impossible
  on 700-km tiles; the phase speaks only for fine-band P.

## Deviations

- (none yet)
