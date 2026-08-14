# Phase 3 Protocol (preregistered): Positioning the irreversibility profile against scattering-type statistics

Status: FROZEN before any Phase-3 data download or computation.
Date frozen: 2026-08-12. Any deviation must be logged in Deviations with a
timestamp and reason.

## Question

Phase 2b established (held-out, preregistered): the profile P is a stable
regional signature with a spectrum-exceeding component. Established
non-Gaussian scale statistics (scattering transform: Mallat 2012, Bruna &
Mallat 2013; scattering-covariance applications in turbulence/cosmology)
measure closely related cross-scale modulus structure. Phase 3 asks: is P
(a) informative beyond scattering statistics (NOVEL), (b) equivalent to them
(EQUIVALENT — value is compactness/interpretability only), or (c) a lossy
subset (SUBSET)?

## Data (fixed; new years, none used in Phases 1-2b)

- ERA5 850 hPa u,v; 0.25 deg; 6-hourly.
- All 12 program regions R1..R12 (areas as frozen in Phases 1-2b protocols).
- Held-out windows (4): W9 2023-01-01..03-31, W10 2023-07-01..09-30,
  W11 2024-01-01..03-31, W12 2024-07-01..09-30.
- 48 region-windows. None may be added, dropped, or re-cut after results.

## Quantities

- Profile P and C1 surrogate control: exactly the Phase-2b pipeline
  (vorticity, 6 octave envelopes, rho_1..rho_5, 199 shared-phase surrogates,
  seeds SeedSequence(20260811 + crc32(tag) mod 1e5)).
- Scattering-type features F_scat per region-window (isotropic, built on the
  same Gaussian octave ladder; simplified scattering covariance):
  - S1_j = log(spatial-temporal mean of E_j), j = 0..5 (6 features);
  - S2_{j1,j2} = log( mean(|band_{j2}(E_{j1})|) / mean(E_{j1}) ) for
    j1 = 0..4, j2 = j1+1..5, where band_j(f) = bar_{ell_j}(f) -
    bar_{ell_{j+1}}(f) on the same ladder (15 features);
  - means over the ell=1600 interior mask and over time (per snapshot,
    then median over time).
  - Total 21 features.

## Criteria (fixed)

- C1 (positive control): as Phase 2b; >= 38 of 48 region-windows with
  >= 3/5 steps beyond the surrogate 95th percentile. Failure halts.
- H3a (P beyond scattering): leave-one-out across 48 region-windows:
  within each fold, z-score F_scat on the training set, fit PCA on training,
  keep 10 components; OLS-fit each P component on [1, PC1..PC10]; residual
  of the held-out rw. Then z-score residual components across 48; Euclidean
  pair distances; PASS iff median(between-region) - median(within-region) > 0
  with permutation p < 0.05 (999 region-label shuffles).
- H3b (scattering beyond P): symmetric: PCA-5 of z-scored F_scat as target
  (5 components, matching P's dimension), OLS on [1, P (5)] per fold,
  residual clustering test, same pass rule.

Verdict (fixed mapping):
- NOVEL:      C1 ok, H3a pass (H3b any).
- EQUIVALENT: C1 ok, H3a fail, H3b fail.
- SUBSET:     C1 ok, H3a fail, H3b pass.

Descriptive (reported, not criteria): H1-style signature strength
(between-within diff, p) for P (5 features) vs PCA-5 of F_scat; region
classification accuracies (P / scat-PCA5 / scat-full); per-region mean
profiles for the new years vs Phase-2b means (replication check).

## Interpretation notes (declared before results)

- NOVEL licenses a methods paper positioning P as a new compact cross-scale
  diagnostic; it does NOT by itself validate any bundle/gravity reading.
- EQUIVALENT is a respectable outcome: P = 5-number interpretable
  parameterization of scattering-level structure; framing must say so.
- SUBSET means scattering statistics supersede P; the program should adopt
  them and keep only the bundle vocabulary if it still earns its place.

## Compute plan

- Downloader: `clean_experiments/download_b3_era5_wind.py` -> data/b3/.
- Experiment: `clean_experiments/experiment_B3_scattering_benchmark.py`
  (reuses Phase-2 pipeline; resume-capable) -> results under
  `clean_experiments/results/experiment_B3_scattering_benchmark/`.
- Seeds fixed: 20260811.

## Deviations

- 2026-08-12 (declared exploratory addendum E1, before any Phase-3
  region-window was computed; criteria unchanged): motivated by Krenke,
  Puzachenko & Puzachenko 2019 (Izv. RAN Ser. Geogr. 3:116-130, semi-fractal
  spectral residuals as signatures of hierarchical levels), an additional
  NON-CRITERION feature set F_frac will be computed per region-window:
  power-law fit of the isotropic spectrum (slope, intercept), best
  single-breakpoint semi-fractal segmentation (break scale, two slopes),
  and the top-3 residual peak scales and amplitudes after the power-law fit.
  E1 reports, descriptively only: (i) whether P is explained by F_frac
  (same LOO residualization machinery as H3a), (ii) whether residual peak
  scales coincide with the profile step at which rho drops most per region.
  E1 cannot affect the NOVEL/EQUIVALENT/SUBSET verdict. Adaptive
  (data-discovered) scale ladders are deferred to a Phase-4 protocol.
