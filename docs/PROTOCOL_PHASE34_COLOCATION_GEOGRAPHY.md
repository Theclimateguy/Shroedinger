# PROTOCOL — Phase 34: what places the activity of adjacent scales together (B34)

## Status: FROZEN 2026-10-15, before any quantity on bands >= 200 km was computed and before the 2022 fields were downloaded

Purpose: the computations behind article 2 in its new form, "what makes the activity
of adjacent scale levels co-locate": a stationary template (fixed by the surface and
by the climatological circulation) versus a moving part, and a regionalisation of the
globe by the co-location profile. Language follows article 1 after review: the
measures are symmetric and record co-location of activity, not influence or
direction.

Why a new phase. Phase 31 (boundary template, PARTIAL(H31-1, H31-2)) used bands
50-100, 100-200, 200-400 km on 700-km tiles. Since 2026-10-14 the programme uses only
bands not finer than 200 km (below that every available reanalysis is under its
effective resolution; `docs/RECONCILIATION.md`, addendum 2026-10-15). Nothing in
Phase 31 carries over automatically. Expectation recorded here: at >= 200 km the
stationary part of the envelope maps will contain the climatological circulation
(jet and storm-track position, ITCZ, monsoon troughs) as well as the surface; its
tie to relief may weaken.

## Data

- ERA5 850 hPa u, v, 6-hourly, global, 0.25 deg, JFM and JAS of 2023 and 2024
  (`data/b20global/`, on disk; used in Phase 31 only on bands < 400 km).
- **Fresh confirmatory year:** the same for JFM and JAS 2022, downloaded after this
  protocol is committed (`clean_experiments/download_b34_wind2022.py`).
- Free-running ECMWF-IFS-HR, highresSST-present, 2014 JFM and JAS, 6-hourly, 0.5 deg,
  850 hPa (`data/b13hrmip/`, on disk; below-ground points are missing values).
- ERA5 surface geopotential (`data/b32/era5_orography.nc`, V12 mask), sub-grid
  orography `sdor` (`data/xp_physics/era5_static_subgrid_orography.nc`), land mask
  from the `global-land-mask` package.

## Definitions (fixed)

**Field.** Relative vorticity on the sphere, V12: zeta = (1/(a cos phi)) dv/dlambda
- (1/a) du/dphi + u tan(phi)/a, centred differences, periodic in longitude; grid
points where the ERA5 surface lies above the 850-hPa standard height (1457 m) are
excluded (normalised convolution) — for the model, its missing values.

**Bands.** Gaussian bars at ell = 200, 400, 800, 1600 km (sigma = ell/sqrt(12)),
applied globally, kilometre-based in both directions, periodic in longitude,
normalised by the valid mask. Bands B1 = 200-400, B2 = 400-800, B3 = 800-1600 km;
log-envelope E_k = log(bar_{ell_{k+1}}(|B_k|) + 1e-12), as in the V12 estimator.
Computation on 70S-70N; all statistics on 60S-60N.

**Windows (the unit).** 66 fixed windows, about 2200 x 3000 km, close to the
2000 x 2800-km boxes of article 1: latitude belts 60-40, 40-20, 20-0 deg on each
hemisphere; longitude width 40 deg in the 40-60 belts (9 windows), 30 deg in the
others (12 windows); western edges at 0 deg E. Pixel weights cos(phi) x valid.

**Per window, season (JFM, JAS) and year**, envelopes de-meaned within the window at
each time step (weighted):
- C_tot(i, j) = time mean of the weighted spatial covariance of E_i, E_j;
- C_tpl(i, j) = weighted spatial covariance of the time-mean maps, cross-validated
  between two interleaved sets of 5-day blocks (symmetrised);
- C_mov = C_tot - C_tpl;
- template share s = mean over the three bands of C_tpl(i, i) / C_tot(i, i);
- r_tot, r_mov, r_tpl(i, j) = C(i, j)/sqrt(C(i, i) C(j, j)); r_tpl defined where
  C_tpl(i, i) > 0.02 C_tot(i, i) for both bands;
- co-location profile P4 = median over time of the weighted Pearson correlation of
  E_1 and E_2, P5 likewise for E_2 and E_3 (steps 4 and 5 of article 1; the Pearson
  form, rho 0.99 with the rank form by AUDIT-1, allows area weights).
Year value = mean of the two seasons.

**Window covariates.** Land fraction (land >= 0.5, ocean <= 0.1), mean sdor.

**Null for the compensation test.** Per window, every 4th time step: the window
plus a 1100-km margin, resampled to an isotropic grid (zonal step = meridional step
in km), vorticity mirror-extended in both axes, Fourier phases randomised, the
original quadrant kept, envelopes computed in the region and statistics in the
window. One surrogate per step. r_tot_null = the B1|B3 total correlation of the
surrogates; r_tot_reg = the same pipeline on the real fields; anchored total
a_tot = r_tot_reg - r_tot_null.

## Hypotheses and bars

- **H34-1 (template share is a territorial invariant, PRIMARY).** Spearman(s_2023,
  s_2024) over the 66 windows >= 0.70; null: cyclic shift of the 2024 values within
  each latitude belt (999), p < 0.01. Fresh-year check: Spearman(s_2022,
  mean(s_2023, s_2024)) >= 0.70.
- **H34-2 (tie to the surface survives at >= 200 km).** Over the mean of 2023-24:
  Spearman(s, sdor) >= +0.30 and median s (land windows) >= 1.5 x median s (ocean
  windows).
- **H34-3 (compensation, anchored).** Pair B1|B3, s from 2023, couplings from 2024:
  (a) |Spearman(r_tot, s)| <= 0.20; (b) Spearman(r_mov, s) <= -0.30;
  (c) median a_tot >= 0.05 and |Spearman(a_tot, s)| <= 0.20. PASS = (a), (b), (c).
- **H34-4 (one template for all three bands).** Median r_tpl over windows where
  defined >= 0.70 for B1|B2 and B2|B3.
- **H34-5 (free-running model).** Spearman(s_IFS-HR 2014, mean s 2023-24) >= 0.60
  and Spearman(P45_IFS-HR, P45_ERA5) >= 0.60, P45 = (P4 + P5)/2.
- **H34-6 (regionalisation, PRIMARY for article 2).** Features z = [P4, P5, s,
  r_mov(B1|B3)], standardised over windows. Ward clustering, k = 4 (k = 3, 5
  reported), separately for 2023 and 2024; adjusted Rand index (ARI) >= 0.40 and,
  against permutation of the 2024 labels within latitude belts (999), p < 0.05.
  Fresh-year confirmation: 2022 windows assigned to the nearest centroid of the
  pooled 2023-24 clusters; ARI with the pooled labels >= 0.40, within-belt
  permutation p < 0.05.

Reported, not scored: JFM vs JAS agreement of s and of the labels; Spearman of s with
the Phase-31 fine-band share averaged into the windows; maps of s, P45, r_mov and the
clusters; land/ocean and latitude composition of each cluster.

## Verdict ladder

- **REGIONALISATION_CONFIRMED:** H34-1 and H34-6 (both years and the fresh year).
- **TEMPLATE_INVARIANT_ONLY:** H34-1 passes, H34-6 fails.
- **NEGATIVE:** H34-1 fails.
H34-2 decides the word: "boundary template" if it passes, "stationary template"
(surface plus climatological circulation) if it fails. H34-3, H34-4, H34-5 qualify
the text.

## Priors (recorded before computation)

H34-1 0.8; H34-2 0.4; H34-3 0.3; H34-4 0.6; H34-5 0.6; H34-6 0.5 (two years),
0.45 (fresh year).

## Code and outputs

`clean_experiments/experiment_B34_colocation_geography.py`
(stages: `windows --source era5 --year Y --season S`, `windows --source ifshr`,
`null --year 2024`, `tests`) -> `clean_experiments/results/experiment_B34_colocation_geography/`.
Seed 20261015.

## Deviations

- (none yet)
