# PROTOCOL — Phase 28: the inheritance law and its depth L
## Status: FROZEN 2026-09-22, before any Phase-28 statistic was computed

Disclosure of prior exposure. Phase 27 (2026-09-21) stored anchored
log-envelope correlations and covariances for ALL band pairs. Before this
freeze the author's assistant had seen: pooled (48-unit) medians of raw and
surrogate rho for every pair; the 12-region medians of anchored rho by
separation; the pooled tile medians for separations 2-3. NOT seen: any
covariance-based quantity by unit, the region-level or tile-level
geography of any non-adjacent statistic, window-to-window or half-to-half
reproducibility, relations to covariates or to P. The law's FORM (below)
is taken from Arneodo et al. 1998, not from the data; the tests below are
the ones the peeked numbers do not answer.

## Object

Anchored linear covariance of log-envelopes C_ij = Cov(E_i, E_j) - median
surrogate, stored per unit/tile by Phase 27 (full, H1, H2). Bands
a_0..a_5 = 50..1600 km (units), a_0..a_3 = 50..400 km (tiles).
Only NON-ADJACENT pairs (j >= i+2) enter any fit: adjacent pairs carry
filter-overlap covariance that the surrogate median removes only on
average.

## The law (Arneodo common-ancestor form)

Under a scale-local multiplicative cascade with log-multiplier variance
lambda2 per e-fold and outer scale L, the covariance of log-magnitudes at
the same position and scales a_i < a_j depends on the COARSER scale only:
    C_ij = lambda2 * (ln L - ln a_j),   j >= i+2.                    (1)
Two parameters per unit: lambda2 (slope of C_ij against -ln a_j) and the
depth L = exp(b0 / lambda2) where b0 is the intercept. L is the scale at
which inheritance is extrapolated to vanish.

## Carriers

UNITS: 48 Phase-27 unit shards (12 regions x 4 windows), 10 non-adjacent
pairs each. Unit of inference = region (n = 12). SMALL BOXES: the same 48
units recomputed with km_crop factor 0.7 (1400 x 1960 km), identical
machinery, into results/experiment_B27_mechanism/units_x0.7 (the only new
computation of this phase). TILES: 902 kept Phase-27 tiles, non-adjacent
pairs (0,2), (0,3), (1,3) -> two distinct a_j; L_tile from the two-point
fit, used as a replicate only. Covariates and eke_syn as in Phase 20.

## Hypotheses and decision rules (all thresholds fixed here)

H28-1 (PRIMARY) — form of the law: the coarser scale, not the separation,
  organises C_ij. Per unit, two one-way categorical models on the 10
  non-adjacent pairs: M_j (factor = j, 4 levels) and M_sep (factor = j-i,
  4 levels); equal parameter counts. Statistic: dR2 = R2(M_j) - R2(M_sep),
  averaged over the 4 windows -> 12 region values. PASS: region median
  dR2 > 0.10 and one-sided Wilcoxon p < 0.05. Reported: median R2(M_j),
  the linear fit R2 of (1), and the sign of lambda2 per unit (a negative
  slope in any region is reported and that region's L is undefined).

H28-2 (PRIMARY) — L is a regional invariant. Region L = median of the 4
  window values. (a) Window reproducibility: Spearman rho between region-L
  of the two JFM windows vs the two JAS windows (mean of the 4 cross-season
  window pairs) >= 0.60 with permutation p < 0.05 (n = 12, 9999 perms).
  (b) Between/within: ratio of between-region variance to within-region
  (window) variance of ln L, F-test p < 0.05. PASS: both (a) and (b).
  Reported: the same for lambda2; H1/H2 split-half rho of ln L per unit.

H28-3 (PRIMARY, the anti-artefact test) — L is not the box. Ratio
  q = median over regions of L(0.7 box) / L(1.0 box).
  PASS: q >= 0.85 (L set by the flow, not the window). FAIL: q < 0.85 ->
  L is domain-limited; every L below is then reported as a LOWER BOUND and
  the word "invariant" is not attached to it. Reported: lambda2 ratio the
  same way; anchored C_ij themselves for both extents.

H28-4 (PRIMARY) — P obeys the law: adjacent anchored P (stored, A3/Arm B)
  is carried by (lambda2, L). Units: LOSO-region R2 of region-P on
  (lambda2, ln L) with n = 12 is underpowered; scored on TILES instead:
  LOSO-sector R2 of tile-P on z(lambda2_tile), z(ln L_tile) with the
  Phase-20 rotation null (999 rotations, min 30 deg, seed 20260922).
  PASS: LOSO R2 >= 0.30 and p_rot < 0.05. Reported: the increment of
  (lambda2, ln L) over the 8 Phase-20 covariates (same ladder as Phase 25:
  +0.03/p<0.05, +0.05/p<=0.01, reported, not scored), and which of the two
  carries the geography (single-variable LOSO R2 each).

H28-5 (reported, not scored) — geography of L: Spearman of region-L and
  tile-L with cape_mean and eke_syn (rotation null on tiles); unit-vs-tile
  co-location Spearman of L over the 12 regions; map of tile ln L saved.

## Controls

C28-1: surrogate floor — median over units of (q95 - q05) of the surrogate
  C_ij for non-adjacent pairs, against the median anchored C_ij; ratio
  reported. If the anchored C_ij of separation >= 3 is below its floor in
  more than half of the units, the fit is restricted to the pairs above
  floor and this is recorded as a deviation.
C28-2: the 0.7-box adjacent P reproduces the 1.0-box regional ranking
  (Spearman >= 0.8 over 12 regions), otherwise the small-box arm is void.

## Verdict ladder

LAW_CONFIRMED: H28-1, -2, -3, -4 all pass.
LAW_DOMAIN_LIMITED: 1, 2, 4 pass, 3 fails (L is a lower bound; the paper
  reports the form and lambda2 as invariants, L as a bound).
PARTIAL(<list>) otherwise; NOT_SUPPORTED if H28-1 fails (then the
  separation-based description is kept and no L is reported).

## Author's priors (recorded before computing)

H28-1 pass 0.6 (the peeked rank profile mixes both dependences).
H28-2 pass 0.7. H28-3 pass 0.4 (the 2-3 thousand km extrapolation is
near the box size). H28-4 pass 0.5.

## Scope fence

No new bands, extents, covariates or thresholds after the first read.
Anything else computed is exploratory and labelled so.

## Deviations
(none at freeze)
