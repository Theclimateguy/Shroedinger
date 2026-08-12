# Phase 2b Protocol (preregistered): Held-out test of the scale-irreversibility profile as a regional invariant beyond the spectrum

Status: FROZEN before any Phase-2b data download or computation.
Date frozen: 2026-08-11. Any deviation must be logged in Deviations with a
timestamp and reason.

## Context

Phase 2 (docs/PROTOCOL_PHASE2_SCALE_IRREVERSIBILITY.md) returned a formally
NEGATIVE verdict with two post-hoc metric pathologies (saturated C2 baseline,
degenerate C3 similarity). Exploratory analysis on the Phase-2 data suggested:
the envelope-coupling profile P is a season-stable regional signature with
physically sensible ocean/land-convective contrast. That dataset is
exploration-contaminated and is NOT reused here. Phase 2b freezes corrected
metrics and tests on held-out regions AND held-out years.

## Claim under test

The scale-irreversibility profile P (cross-scale envelope coupling along the
cascade, in the bundle formalism's operational vocabulary) is:
(H1) a stable regional signature on data never seen before, and
(H2) not reducible to the power spectrum.

## Data (fixed; none seen before this freeze)

- ERA5 850 hPa u,v; 0.25 deg; 6-hourly.
- Held-out windows (4): W5 2021-01-01..03-31, W6 2021-07-01..09-30,
  W7 2022-01-01..03-31, W8 2022-07-01..09-30.
- Held-out regions (8), 20 deg lat x 40 deg lon, area = [N, W, S, E]:
  - R5_SPCZ  [0, -180, -20, -140]   tropical ocean, SPCZ sector
  - R6_SATL  [-35, -50, -55, -10]   Southern-Hemisphere storm track
  - R7_CONGO [5, 5, -15, 45]        tropical land, deep convection
  - R8_AUS   [-15, 115, -35, 155]   subtropical continent
  - R9_NPAC  [55, -180, 35, -140]   Northern-Hemisphere storm track
  - R10_INDO [10, 60, -10, 100]     tropical ocean, warm pool
  - R11_EURO [55, -10, 35, 30]      midlatitude mixed land
  - R12_SAM  [-20, -75, -40, -35]   subtropical South America
- 32 region-windows. None may be added, dropped, or re-cut after results.

## Quantities (unchanged from Phase 2 as-clarified)

Vorticity omega; Gaussian ladder ell in {50,100,200,400,800,1600} km
(sigma = ell/sqrt(12), grouped-row metric filtering); six octave envelopes
including the sub-50 km residual; profile P = [median_t rho_1..rho_5] with
rho_i = spatial rank correlation of adjacent log envelopes over the
ell=1600 interior mask; entropy profile Q and dQ from the A15 band machinery
(6 modes/var, W=20, ridge 1e-6, shrink 0.05, first 19 steps dropped);
F_spec = {6 log octave variances, isotropic spectral slope over 100..1600 km};
199 shared-phase surrogates per region-window,
seeds SeedSequence(20260811 + crc32(tag) mod 1e5).

## Criteria (fixed)

- C1 (positive control): rho_i real > surrogate 95th percentile in >= 3 of 5
  steps, in >= 26 of 32 region-windows. Failure halts the phase (pipeline or
  data problem), no verdict.
- H1 (held-out regional signature): z-score each P component across the 32
  region-windows; Euclidean distances between all pairs;
  PASS iff median(between-region) - median(within-region) > 0 with
  permutation p < 0.05 (999 shuffles of region labels).
- H2 (beyond spectrum): leave-one-out across the 32 region-windows: for each
  held-out rw, OLS-fit each P component on F_spec (intercept + 7 features)
  using the other 31, form the residual profile res = P_obs - P_pred;
  z-score residual components across 32; PASS iff
  median(between-region) - median(within-region) of residual distances > 0
  with permutation p < 0.05 (999 shuffles).

Verdict: Phase 2b POSITIVE iff H1 AND H2 pass (C1 satisfied).
If H1 fails, the regional-invariant reading retires. If H1 passes and H2
fails, the profile is a regional signature but spectrum-reducible — the
"beyond standard spectral description" claim retires.

Descriptive (reported, not criteria): region classification accuracies
(spec / P / residual-P), per-region mean profiles, ocean vs land contrast,
corr(P, -dQ).

## Phase 3 gate (declared)

A POSITIVE Phase 2b permits a novelty-positioning study ONLY against
established non-Gaussian scale statistics (scattering transform / wavelet
phase harmonics) on further held-out data.

## Compute plan

- Downloader: `clean_experiments/download_b2b_era5_wind.py` -> data/b2b/.
- Experiment: `clean_experiments/experiment_B2b_heldout_invariants.py`
  (reuses the Phase-2 pipeline functions unchanged) -> results under
  `clean_experiments/results/experiment_B2b_heldout_invariants/`.

## Deviations

- (none yet)
