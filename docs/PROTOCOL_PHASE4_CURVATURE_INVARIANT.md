# Phase 4 Protocol (preregistered): Curvature strength as a physical regional invariant

Status: FROZEN before any Phase-4 data download or computation.
Date frozen: 2026-08-12. Deviations must be logged with timestamp and reason.

## Background (declared exploration)

E2 (openly exploratory, on the exploration-contaminated Phase-3 dataset)
found: the band-resolved curvature-strength profile Fnorm_b = median_t
||F_b(t)||_F from the A15 machinery is a strong regional signature carrying
information beyond both the power spectrum and the envelope profile P; the
SIGNED scalar Lam_b = median_t Tr(F_b rho_b) carries none (p=0.7). |Lam|
is informative but dominated by Fnorm. Phase 4 confirms or retires the
curvature-strength claim on fresh data and tests physical explicability.
The signed Lambda remains retired regardless of Phase-4 outcomes.

## Data (fixed; curvature-naive)

- Confirmation set: the existing on-disk region-windows whose YEARS were
  never touched by the E2 curvature exploration (E2 ran only on 2023-2024):
  data/b1 (R1..R4 x W1-W4, 2017-2019, 16 region-windows) and data/b2b
  (R5..R12 x W5-W8, 2021-2022, 32 region-windows) -> 48 region-windows,
  12 regions, no new wind downloads. The Phase-3 2023-2024 dataset is
  EXCLUDED from H4a/H4b (exploration-contaminated for curvature).
- Physical covariates (per region, 12 values each):
  - orog_std: standard deviation of ERA5 surface geopotential / g over the
    region (invariant field);
  - land_frac: mean ERA5 land-sea mask over the region (invariant);
  - cape_mean: mean of ERA5 monthly-averaged CAPE over the region,
    2021-2024 all months.

## Quantities (fixed)

Fnorm and LamAbs profiles (6 bands) exactly as in E2: A15 machinery,
6 modes/var, W=20, ridge 1e-6, shrinkage 0.05, warmup 19 steps, band medians.
P profile and F_spec as in Phases 2b/3 (for the beyond-tests).

## Criteria (fixed)

- H4a (held-out signature): Fnorm profile within/between clustering on the
  48 curvature-naive region-windows; PASS iff diff > 0, permutation
  p < 0.05 (999).
- H4b (beyond spectrum and beyond P, held-out): LOO PCA-residualization of
  Fnorm on F_spec (6 PCs) and separately on P (5 columns, OLS); each residual
  set must cluster by region (diff > 0, p < 0.05). PASS iff both do.
- H4c (physical explicability): region-median fine-band curvature
  (mean of Fnorm bands 0 and 1) across the 12 regions, using ALL available
  region-windows (Phase-3 + Phase-4 fresh): leave-one-region-out OLS on
  [orog_std, land_frac, cape_mean]; PASS iff LOO R^2 > 0 and permutation
  p < 0.05 (999 shuffles of region labels).

Verdict:
- CONFIRMED_PHYSICAL: H4a, H4b, H4c all pass.
- CONFIRMED_DESCRIPTIVE: H4a and H4b pass, H4c fails (curvature is a real
  invariant but its physical driver remains unidentified).
- NEGATIVE: H4a or H4b fails (E2 was an artifact of the explored dataset).

Descriptive (non-criteria): LamAbs same tests; per-band covariate
correlations; Congo/Amazon curvature contrast on fresh data; cross-year
profile replication.

## Compute plan

- Downloader: `clean_experiments/download_b4_covariates.py` (invariant
  orography + land-sea mask + monthly CAPE 2021-2024 -> data/b4cov/).
  No wind downloads needed.
- Experiment: `clean_experiments/experiment_B4_curvature_invariant.py`
  -> results under `clean_experiments/results/experiment_B4_curvature_invariant/`.
- Seeds fixed: 20260811.

## Deviations

- 2026-08-12 (pre-computation revision, before any Phase-4 computation):
  confirmation set changed from newly-downloaded 2025 windows to the
  existing curvature-naive on-disk data (2017-2019 + 2021-2022, 48
  region-windows) at the author's suggestion; the 2023-2024 set stays
  excluded as exploration-contaminated. Criteria unchanged.
