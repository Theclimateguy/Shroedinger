# PROTOCOL AUDIT-2 (FROZEN 2026-08-19): is P an intermittency measure?

Frozen before computation. Companion to AUDIT-1
(`docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md`), which retired the
wall-turbulence amplitude-modulation reading. This audit tests the next
prior-art claimant: the intermittent-cascade literature.

## 1. The claim being tested

In turbulence the correlation between the log-amplitudes (envelopes) of
neighbouring scales is the defining property of a multiplicative
cascade. Under the Kolmogorov-Obukhov 1962 log-normal cascade with
independent multipliers, the correlation between the log-envelopes at
scales l_fine < l_coarse is fixed by the variances alone:

    corr(log e_fine, log e_coarse) = sigma(l_coarse) / sigma(l_fine).

If our P is that quantity, then P was known in 1962 and the programme
has measured the local intermittency parameter of the atmosphere under a
new name. That is a live, specific, falsifiable claim and it is scored
here.

## 2. Sample and machinery

Identical to Phase 20 Arm B and AUDIT-1: 936 tiles x 2 seasons
(JFM/JAS 2023), ERA5 850 hPa vorticity, ELLS_FINE = [50,100,200,400] km,
identical interior mask, identical 99 phase-surrogate realisations
(same seed scheme), identical orography exclusion, 60-deg sector LOSO
folds and longitude-rotation null.

## 3. Statistics (per tile-season, spatial statistic per time step,
median over time, then anchored on the same 99 surrogates)

- `P` — the frozen estimator (recomputed for the identity control).
- `P_cascade` (PRIMARY comparator) — the log-normal cascade prediction
  of the envelope-envelope correlation,
  `sqrt(sigma2_coarse / sigma2_fine)` per band step, mean over the 3
  steps, where `sigma2_b` is the spatial variance of the band's
  log-envelope.
- `INT_sig2` — mean over the 4 bands of `sigma2_b`, the
  Kolmogorov-Obukhov intermittency parameter itself.
- `INT_flat` — mean over the 4 bands of `log10` flatness (kurtosis) of
  the band-pass field.
- `INT_mu` — the intermittency exponent: OLS slope of `log flatness`
  against `log scale` over the 4 bands (flatness ~ l^-mu).

## 4. Controls

- C-A2-0 (pipeline identity, computed first): the recomputed anchored
  `P` must reproduce the committed Arm-B map at Spearman >= 0.99 and
  max raw per-step difference <= 1e-6. Failure aborts the audit.

## 5. Scored questions and bars (frozen)

- A2a (SINGLE SCORED PRIMARY): Spearman rho between anchored `P` and
  anchored `P_cascade` across kept tiles (season mean).
  - |rho| >= 0.90 -> P_IS_CASCADE_CORRELATION: the papers state that P
    is the local log-normal-cascade envelope correlation, cite KO62 and
    the multifractal literature, and the novelty claim reduces to the
    geography alone.
  - 0.60 <= |rho| < 0.90 -> CASCADE_RELATED: report the shared share,
    keep the distinct-object claim, cite the cascade literature.
  - |rho| < 0.60 -> CASCADE_DISTINCT.
- A2b (reported): the same rho for `INT_sig2`, `INT_flat`, `INT_mu`.
- A2c (reported): attribution transfer — LOSO-sector R^2 over the 8
  frozen covariates with each intermittency statistic as target, with
  its rotation-null p, its cross-season reliability, and its share of
  its own reliability ceiling (the AUDIT-1 reporting format).
- A2d (scored secondary): incremental LOSO R^2 of the four
  intermittency statistics ADDED to the frozen Phase-20 covariate set,
  target anchored `P`, against the 999-rotation null in which the added
  columns are rotated.
  - If R^2(covariates + intermittency) >= 0.9 x 0.603 (the map's
    cross-season variance ceiling) the map is declared jointly
    explained and the papers say so.
  - Otherwise the increment and its p are reported as they come.

## 6. Multiplicity

A2a is the scored primary; A2d is the single scored secondary. A2b/A2c
are reported. No statistic is redefined after seeing any result.

## 7. Execution

`clean_experiments/experiment_A2_intermittency_headtohead.py`, stages
`tiles` (resumable, atomic per-tile shards), `collect`, `tests`.
Results in `clean_experiments/results/experiment_A2_intermittency/`.

## Deviations

- (none yet)

## Deviations (logged after computation)

1. **A2d's bar was contaminated, and the contamination is reported
   rather than repaired retroactively.** The bar compared
   R^2(covariates + intermittency) against 0.9 x 0.603, where 0.603 is
   the map's CROSS-SEASON variance ceiling. P and the intermittency
   statistics are computed from the same tile-season sample and
   therefore share sampling noise, which the cross-season ceiling
   excludes by construction. The scored result stands as computed
   (increment +0.225, p = 0.001, bar passed); a post-hoc control
   predicting one season's P from the other season's intermittency
   statistics gives the signal-level increment +0.077 / +0.081. Both
   numbers are reported; the papers quote the decontaminated one. No
   bar was changed after seeing the result.
2. `P_cascade` was computed without clipping the variance ratio, exactly
   as specified in Sect. 3 (an early implementation clipped it at 1.0;
   the clip was removed before any production run).
