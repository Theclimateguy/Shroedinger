# Phase 2 Protocol (preregistered): Is the scale-irreversibility profile a geographic invariant beyond the power spectrum?

Status: FROZEN before any Phase-2 computation. Date frozen: 2026-08-11.
Any deviation must be logged in the Deviations section with a timestamp and reason.

## Claim under test

The theory's surviving core claim (author's formulation): information recorded
at inter-scale transitions in the Hilbert bundle is not recoverable back, and
this irreversibility is a substantive object for geography.

Honest framing of what is and is not at stake:

- That coarse-graining loses information is a theorem (data-processing
  inequality), not a discovery. NOT under test.
- For Gaussian fields, every information functional is a function of the power
  spectrum. So an irreversibility profile is substantive ONLY if it carries
  structure beyond the spectrum.
- Cross-scale envelope dependence beyond the spectrum in the atmosphere is
  expected from known multifractality (Lovejoy-Schertzer) and is detectable by
  established tools (scattering transforms, wavelet phase harmonics). Its mere
  existence is therefore a POSITIVE CONTROL here, not a finding.
- What IS under test: whether the per-scale irreversibility profile is (a) a
  stable regional signature and (b) adds regime-discriminating information on
  top of cheap spectral features. If not, "recorded in the bundle and not
  recoverable" reduces to standard spectral description for geographic
  purposes, and the claim retires in its geographic reading.

## Data (fixed)

- ERA5 850 hPa u,v; 0.25 deg; 6-hourly; same four windows as Phase 1
  (W1 2017-01..03, W2 2017-07..09, W3 2018-01..03, W4 2019-07..09).
- Four regions, 20 deg lat x 40 deg lon:
  - R1_WPWP: 10N..10S, 130E..170E (tropical ocean; Phase-1 data reused)
  - R2_NATL: 55N..35N, 60W..20W (midlatitude ocean storm track; reused)
  - R3_AMAZ: 5N..15S, 75W..35W (tropical land, deep convection)
  - R4_CASIA: 55N..35N, 60E..100E (midlatitude continental interior)
- 16 region-windows total. None may be added, dropped, or re-cut after
  results are seen.
- Analysis field: relative vorticity omega = dv/dx - du/dy (metric-correct
  gradients as validated in Phase 1).

## Quantities

### Scale ladder and band envelopes

- Gaussian filters (sigma = ell/sqrt(12), per-row metric spacing) at
  ell in {50, 100, 200, 400, 800, 1600} km.
- Band fields D_i = bar(omega)_{ell_i} - bar(omega)_{ell_{i+1}}, i = 1..5
  (fine to coarse).
- Envelopes E_i = |D_i| smoothed with the ell_{i+1} filter.
- Interior mask: the strictest (ell = 1600 km) Phase-1 mask (ring 2*ell capped
  at 25% of extent per side) applied to ALL levels for comparability.

### Primary irreversibility profile (classical, per region-window)

rho_i(t) = spatial Spearman correlation of log(E_i + eps) vs log(E_{i+1} + eps)
over the interior, eps = 1e-12; profile P = [median_t rho_1..rho_5].
Interpretation: cross-scale envelope coupling = how strongly fine-scale
activity is organized by coarser scales; the profile over i is the
irreversibility structure along the cascade.

### Secondary bundle-native profile

Band density matrices rho_b exactly as in the A15 machinery
(6 modes/var, W=20, ridge 1e-6, shrinkage 0.05; first 19 steps dropped).
S_b = von Neumann entropy of rho_b normalized by log(dim);
profile Q = [median_t S_1..S_6] and steps dQ_b = S_{b+1} - S_b.
Role: check that the bundle-formalism entropy sees the same structure
(Spearman correlation between profile P and -dQ reported descriptively);
Q participates in C2 features.

### Surrogates

199 spectrum-preserving surrogates per region-window: per snapshot, one random
phase field applied to u and v jointly (Phase-1 procedure), omega recomputed,
full envelope and entropy pipelines rerun. Row-batched filtering (rows grouped
by pixel-sigma rounded to 0.01) is a compute optimization with identical math.

## Criteria (fixed)

- C1 (positive control, expected to pass): per region-window, rho_i real
  exceeds the surrogate 95th percentile in >= 3 of 5 band steps; satisfied in
  >= 12 of 16 region-windows. Failure means pipeline or data problem — halt
  and diagnose, do not proceed to verdict.
- C2 (beyond-spectrum geographic information): features per region-window:
  F_spec = {log band variances of omega (6), OLS spectral slope (1)};
  F_irr = {P (5), dQ (5)}. Classifier: nearest-centroid on z-scored features,
  leave-one-region-window-out (16 folds), target = region identity (4 classes).
  PASS iff acc(F_spec + F_irr) - acc(F_spec) >= 2/16 AND permutation
  p < 0.05 (999 permutations of F_irr rows across region-windows,
  recomputing the accuracy gain).
- C3 (stable regional signature): Spearman similarity of P across all
  region-window pairs; PASS iff median(within-region pairs, n=24) >
  median(between-region pairs, n=96) AND permutation p < 0.05
  (999 shuffles of region labels).

Verdict: Phase 2 POSITIVE iff C2 AND C3 pass (with C1 satisfied).
If C1 passes but C2 or C3 fails, the geographic reading of the
irreversibility claim retires; exploratory findings may be reported but
cannot overturn the verdict.

## Compute plan

- Downloader: `clean_experiments/download_b2_era5_wind.py` (R3, R4 files
  into data/b1/ naming scheme era5_wind850_{region}__{window}.nc).
- Experiment: `clean_experiments/experiment_B2_scale_irreversibility.py`
  writes per-region-window JSON and a consolidated report under
  `clean_experiments/results/experiment_B2_scale_irreversibility/`.
- Seeds fixed: 20260811.

## Relation to prior art (declared before results)

Cross-scale envelope statistics are established non-Gaussianity tools
(scattering transform, wavelet phase harmonics; multifractal cascade
literature). Phase 2 does not claim novelty for detecting non-Gaussian
cross-scale coupling; the tested contribution is the irreversibility PROFILE
as a stable, spectrum-exceeding geographic signature, in the bundle
formalism's operational vocabulary. If Phase 2 is positive, Phase 3 must
benchmark against scattering-transform features before any novelty claim.

## Deviations

- 2026-08-11 (spec clarification, before any computation): the frozen text
  defines 5 band envelopes but a 5-element profile of ADJACENT pairs, which is
  internally inconsistent (5 envelopes give 4 pairs). Fixed by adding the
  sub-50 km residual envelope E_0 = |omega - bar(omega)_50| smoothed at 50 km;
  the profile is rho_i = corr(E_{i-1}, E_i), i = 1..5, over six octave
  envelopes. The six log band variances in F_spec use the same six octaves.
- 2026-08-11 (spec clarification, before any computation): surrogate seeds
  derived per region-window as SeedSequence(20260811 + crc32(tag) mod 1e5).
- 2026-08-11 (post-hoc finding, AFTER results; verdict unchanged): two frozen
  metrics proved pathological. C2's spectral baseline saturates (LOO accuracy
  1.0 on 4-region identity), making the gain criterion unpassable in
  principle; C3's 5-point Spearman similarity is degenerate because all
  observed profiles are monotone (all pairwise similarities ~1.0). The frozen
  verdict NEGATIVE stands. Any re-test (Phase 2b) must freeze corrected
  metrics and use held-out data (new regions/years), as this dataset has been
  used for exploration.
