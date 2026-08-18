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

## Arm B execution spec (frozen 2026-08-18, after the Arm-A verdict and
## the theory note, BEFORE any Arm-B computation; data/b20global complete)

### Carrier

Global tile grid, 60S-60N: latitude rows of 6 deg (~667 km); per row,
longitude tiles of 700 km at the row centre, n_lon = floor(360/dlon),
dlon = 360/n_lon (exact ring tiling). ~900 tiles. Two seasons computed
independently (JFM 2023 = 202301-03, JAS 2023 = 202307-09, 6-hourly);
the scored value per tile is the two-season mean; the seasonal contrast
is descriptive. Tiles with tile-mean surface elevation > 1200 m are
excluded from scoring (850 hPa below ground); count logged.
Per tile-season: anchored fine-P exactly as Arm A (ELLS_FINE
50-400 km, interior mask, 99 phase surrogates, seed scheme salted
"glob_<tile>_<season>"), fine spectral features, eke_syn (variance about
the 5-day running mean, tile interior).

### Covariates, two tiers (the causality structure fixed by the author)

- Tier-1 exogenous boundary fields: orog_mean, orog_std, land_frac,
  coast_var, abs_lat, sst_grad (mean |grad SST| of the 2021-2024
  climatology over ocean pixels; 0 where ocean fraction < 0.3).
  Directional ("conditions") language permitted.
- Tier-2 co-emergent state: cape_mean (2021-2024 global monthly
  climatology), eke_syn. Organization language only; no direction.
- Declared negative control: lai_mean (lai_lv + lai_hv climatology,
  biological): expected to add nothing beyond [land_frac, cape, eke];
  a significant increment flags a confound, not a discovery.
  Anthropogenic control (nightlights): UNAVAILABLE offline — logged as
  a deviation; deferred.

### Statistics

- Null model for everything: circular longitude rotation of the full
  covariate map relative to the P map by a uniform random offset
  >= 30 deg (999 draws, seed 20260818) — preserves all spatial
  autocorrelation of both maps.
- Cross-validation: leave-one-longitude-sector-out (6 sectors of
  60 deg), R^2 pooled over held-out tiles.
- H-B0 (sanity): within the 12 programme boxes, global-map tile values
  must correlate with the Arm-A tile values at Spearman rho >= 0.7
  (pooled over the overlapping tiles). Failure -> MAP_INCONSISTENT,
  no scoring.
- H-B1 (primary, ONE scalar): LOSO R^2 of tier-1 + tier-2 (8
  covariates) predicting tile anchored fine-P; rotation-null p < 0.05
  AND R^2 > 0.
- H-B2 (tier separation): R^2(tier-1 only) and increment from tier-2;
  both against the rotation null; descriptive.
- H-B3 (effective dimension, theory P-3): greedy forward-selection
  R^2(k) curve, k = 1..8; the theory expects <= 3 covariates to reach
  >= 80% of the full R^2; scored as stated.
- H-B4 (spectral placebo): H-B1 with the fine spectral slope as
  target; report loadings comparison as in Arm A.
- H-B5 (negative control): LAI increment over [land_frac, cape_mean,
  eke_syn] via LOSO R^2 gain + rotation null; expected null.

### Verdict

MAP_INCONSISTENT (H-B0 fails) / GLOBAL_MAP_ATTRIBUTED (H-B1 passes;
H-B3 outcome appended as _DIM<k80>) / GLOBAL_MAP_UNATTRIBUTED (H-B1
fails). H-B5 failure appends _CONFOUND_FLAG whatever the rest says.

### Compute

clean_experiments/experiment_B20_armB_global_map.py
(stages: tiles / tests), results under
clean_experiments/results/experiment_B20_armB_global_map/.
Seed 20260818. Figures: visualize_B20_armB_global.py (THE map).
