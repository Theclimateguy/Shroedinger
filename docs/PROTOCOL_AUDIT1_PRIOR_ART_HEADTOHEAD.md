# PROTOCOL AUDIT-1 (FROZEN 2026-08-19): prior-art head-to-head for P

Frozen before computation. Motivation and prior-art table:
`docs/PRIOR_ART_AUDIT_P.md`. This audit does NOT test the theory
candidate; it tests whether the ESTIMATOR is a re-labelling of two
published statistics. Phase numbering is deliberately not used: this is
a methods audit for the three papers in preparation, not a programme
phase.

## 1. Sample and machinery

- Identical to Phase 20 Arm B: 936 global tiles (6-deg rows, 700-km
  width, 60S-60N), seasons JFM and JAS 2023, ERA5 850 hPa u,v,
  6-hourly, `data/b20global`.
- Identical band ladder ELLS_FINE = [50, 100, 200, 400] km, identical
  interior mask, identical vorticity operator, identical 99 phase
  surrogates with the SAME seed scheme (`SEED_SUR + crc32("glob_{tid}_
  {season}")`), so every statistic is anchored on the same surrogate
  realisations. Paired by construction.
- Tile exclusion (orog_mean > 1200 m), season averaging, 60-deg sector
  LOSO folds and the longitude-rotation null are taken unchanged from
  Arm B.

## 2. Statistics compared (all on the same envelopes, per band step,
median over time, then mean over the 3 steps, then anchored)

- `P` — Spearman rank correlation between the log-envelopes of
  adjacent bands. The programme's estimator.
- `P_lin` — Pearson correlation between the same log-envelopes. The
  amplitude-amplitude-coupling (envelope-envelope) form used in
  cross-frequency analysis.
- `R_AM_cyc` (PRIMARY prior-art comparator) — Pearson correlation
  between the SIGNED coarser band-pass field, taken in the cyclonic
  convention (multiplied by sign of the tile centre latitude), and the
  log-envelope of the adjacent finer band. The Mathis-Hutchins-Marusic
  amplitude-modulation coefficient transported to a 2-D field.
- `R_AM_raw` (secondary) — the same without the cyclonic sign
  convention.

## 3. Controls

- C-A1-0 (pipeline identity, computed first): the recomputed anchored
  `P` must reproduce the committed Arm-B map at Spearman >= 0.99 and
  max |difference| <= 1e-6 on the raw per-step values. Failure aborts
  the audit: the comparison would not be like-for-like.

## 4. Scored questions and bars (frozen)

- A1a (PRIMARY): Spearman rho between anchored `P` and anchored
  `R_AM_cyc` across kept tiles (season mean, the Arm-B target
  construction).
  - rho >= 0.90 -> ESTIMATOR_IS_AM: the papers withdraw any novelty
    claim for the statistic and cite Mathis et al. (2009) as its
    source.
  - 0.60 <= rho < 0.90 -> ESTIMATOR_RELATED: cite as a close relative,
    report the shared share, keep the distinct-object claim.
  - rho < 0.60 -> ESTIMATOR_DISTINCT: cite as related work, no
    withdrawal.
- A1b (secondary, attribution transfer): rerun the Arm-B primary
  (LOSO-sector R^2 over the 8 frozen covariates, 999-rotation null)
  with `R_AM_cyc` and `P_lin` as targets; report R^2, p, and the
  Spearman loadings against the Arm-B values (R^2 = 0.536, and the
  published loading pattern). If the geography reproduces with the
  prior-art statistic, the physical claim is strengthened and the
  methodological one is dropped; this outcome is reported as such.
- A1c (secondary, copula ingredient): Spearman rho between anchored
  `P` and anchored `P_lin`. rho >= 0.95 -> the rank/copula language is
  dropped from all three papers as decoration.
- A1d (reported, not scored): the same three rho values computed
  per season separately, and the cross-season reliability of each
  statistic (the Arm-B ceiling for P is r = 0.776).

## 5. Multiplicity

A1a is the single scored primary. A1b/A1c/A1d are reported. No
re-definition of `R_AM_cyc` after seeing A1a.

## 6. Execution

`clean_experiments/experiment_A1_prior_art_headtohead.py`, stages
`tiles` (resumable, one JSON shard per tile-season, atomic writes) and
`tests`. Results in
`clean_experiments/results/experiment_A1_prior_art/`.

## Deviations

- (none yet)
