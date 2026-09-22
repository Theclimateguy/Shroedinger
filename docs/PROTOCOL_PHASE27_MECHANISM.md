# PROTOCOL — Phase 27: mechanism of the inter-level coupling
## Status: FROZEN 2026-09-21, before any Phase-27 statistic was computed
## (no downloads; every input is already on disk; SHA-256 of this file is
## written into the results summary as freeze evidence)

Motivation. Article 3 is being rebuilt around a mechanism-level frame
(`docs/LIT_REVIEW_ARTICLE3_MECHANISM.md`): the activity envelope of level
i+1 is the sum of an INHERITED part (deformation / cascade from coarser
levels: a multiplier shared from above) and an AUTONOMOUS part (local
diabatic sources whose position is set by column thermodynamics, not by
the synoptic envelope); anchored P measures the inherited share. The
frame is so far an interpretation of the Phase-20 signs. This phase
derives three consequences that were NOT used to build the frame and
scores them on existing data. P itself, the band ladder, the envelope
definition, the phase-surrogate anchoring, the tile grid, the orography
exclusion, the covariates and the rotation null are all inherited
unchanged from Phases 18/20 and AUDIT-2b/3.

Prior art for the statistic of Arm A (declared): the space-scale
correlation function of wavelet log-magnitudes, Arneodo, Bacry,
Manneville & Muzy 1998 (PRL 80:708). Adjacent-band P is its slice
dx = 0, a2 = 2 a1; Arm A evaluates the other scale pairs.

## Carriers

- UNITS: 12 equal-km Phase-18 boxes (2000 x 2800 km) x 4 windows
  (W9_2023JFM, W10_2023JAS, W11_2024JFM, W12_2024JAS) = 48 units,
  `data/b3`, full ladder ELLS_KM = 50..1600 km -> 6 log-envelopes
  E0..E5. Unit of inference = REGION (n = 12; mean over its 4 windows).
- TILES: Phase-20 Arm-B global grid, JFM + JAS 2023, `data/b20global`,
  fine ladder ELLS_FINE = 50..400 km -> 4 log-envelopes E0..E3; the 902
  tiles kept by the frozen orography rule; tile value = mean of the two
  seasons. Covariates: the 8 frozen Phase-20 covariates (eke_syn read
  from the Arm-B tile files).

Envelopes, mask, vorticity, surrogates (N_SUR = 99, shared u/v phase,
SEED_SUR = 20260811 with the carriers' original tag scheme so that the
adjacent-step values reproduce the stored ones) exactly as in
`envelope_rho_profile_ells` / A3 / Arm B. Every statistic below is
computed per time step on ranks over interior cells, summarised by the
median over time steps, and ANCHORED: real minus the median of the 99
surrogate values. Each statistic is additionally stored for the first
and second half of the time steps (H1, H2) separately.

## Statistics

S1. Pair matrix: Spearman rho(E_i, E_j) for ALL pairs (15 on units, 6 on
    tiles). "Separation" s = j - i.
S2. Lag-2 partial: r(E_i, E_{i+2} | E_{i+1}) from the per-step rank
    correlation matrix (4 triplets on units, 2 on tiles).
S3. Deformation index. Coarse wind = u, v smoothed with the programme's
    Gaussian bar at ell_c (UNITS: 800 km; TILES: 400 km). From it:
    strain magnitude S = sqrt((u_x - v_y)^2 + (v_x + u_y)^2) and
    |vorticity| Z, both pointwise, log-transformed. For fine envelopes
    F in {E0, E1, E2}:  D_F = partial r(F, ln S | ln Z),
    V_F = partial r(F, ln Z | ln S). D = mean over the three F.
    Lagged placebo D_lag: same with S, Z taken 20 time steps (5 days)
    later (circular shift), anchored by the same surrogate median.
S4 (reported, not scored). Anchored LINEAR covariance of log-envelopes
    Cov(E_i, E_j); per unit the slope lambda2_hat of Cov_anch against
    -ln(a_coarse), and its cross-unit Spearman correlation with the
    stored AUDIT-3 INT_sig2 (open item of AUDIT-3).

## Hypotheses and decision rules

H27-A1 (PRIMARY) — inheritance reaches beyond the adjacent level.
  Statistic: per region, mean anchored rho over all separation-2 pairs.
  PASS: region median > 0 and one-sided Wilcoxon signed-rank p < 0.05
  (n = 12). Reported alongside: separation 3, 4, 5 and the decay profile.
  FAIL meaning: coupling is a strictly adjacent-band property; "multiplier
  shared from above" language is dropped from Article 3.

H27-A2 (classification, not pass/fail) — locality of the inheritance.
  Statistic: per region, mean anchored lag-2 partial (S2).
  LOCAL_CASCADE if the 95% bootstrap CI (regions, 9999 resamples) of the
  median lies inside [-0.05, +0.05]; NONLOCAL if median > 0 and one-sided
  Wilcoxon p < 0.05 and not LOCAL; SCREENED if median < 0 with p < 0.05
  and not LOCAL; else INDETERMINATE. Interpretation fixed now:
  LOCAL -> scale-local (Markov-in-scale) cascade, the Arneodo tree;
  NONLOCAL -> a coarse level modulates all finer levels directly (shared
  variance multiplier / nonlocal straining, as in the enstrophy range).
  Replicated on tiles (block bootstrap, see below); tile result reported.

H27-B1 (PRIMARY) — the autonomous source sits at the fine end.
  Per-step anchored P_k, k = 1..3 (steps E0-E1, E1-E2, E2-E3) on tiles,
  in raw correlation units; OLS on the 8 z-scored covariates.
  Delta_CAPE = beta_CAPE(k=3) - beta_CAPE(k=1). Prediction: beta_CAPE < 0
  and strongest at the finest step, i.e. Delta_CAPE > 0.
  PASS: 95% block-bootstrap CI of Delta_CAPE excludes 0 on the positive
  side AND beta_CAPE(k=1) < 0 (CI excludes 0). Reported: the same for
  eke_syn (expected Delta_EKE > 0), and the full 3-step profile; a
  non-monotone profile peaking at k = 2 is reported as "MCS-scale
  injection", not as a pass.
  Blocks: 30-deg longitude sector x 30-deg latitude band (48 blocks),
  1999 resamples, seed 20260921.

H27-B2 (PRIMARY) — route (a) measured directly: fine-scale activity sits
  where the coarse flow STRAINS, beyond where it merely rotates.
  (i)  TILES: median anchored D > 0, block-bootstrap 95% CI excludes 0.
       UNITS: region median D > 0, one-sided Wilcoxon p < 0.05.
  (ii) GEOGRAPHY (the scored part): split-half Spearman
       rho_x = 0.5 [rho(D_H1, P_H2) + rho(D_H2, P_H1)] over the 902
       tiles (P = mean anchored adjacent-step P; cross-half so that D and
       P never share time steps). Null: 999 within-row longitude
       rotations of D (min 30 deg), seed 20260921.
       PASS: rho_x >= +0.20 and p_rot < 0.05.
  (iii) reported: beta_CAPE and beta_EKE of tile D_anch on the 8
       covariates with the rotation null; lagged placebo.
  FAIL meaning: the deformation/frontogenesis wording for the inherited
  route is unsupported by our own data; Article 3 keeps the statistical
  inheritance language only and cites the dynamical route as literature.

## Controls

C27-1 reproduction: tile mean anchored adjacent P vs the stored Arm-B
  `P_fine_anch_mean`, and unit value vs stored A3 `P_anch_mean`
  (W9, W12): Spearman >= 0.99 each, else the phase is void.
C27-2 placebo: |median D_lag anchored| <= 0.25 x median D anchored on
  tiles; otherwise D is a static-geography artefact and B2 is void.
C27-3 same-machinery negative: on the surrogates themselves all anchored
  statistics are zero by construction; the spread of the surrogate
  values (q05-q95) is reported for every statistic as its noise floor.

## Verdict ladder

MECHANISM_SUPPORTED: A1, B1, B2(ii) all pass (with controls).
PARTIAL(<list>): one or two pass. NOT_SUPPORTED: none passes.
A2 contributes the class label only. No multiplicity correction is
needed for the conjunction; for PARTIAL the three primary p-values are
also reported Holm-adjusted.

## Author's prior (recorded before computing)

A1 pass (0.85). A2: NONLOCAL (0.5) / LOCAL (0.3) / other (0.2).
B1 pass (0.5) — the finest bands are below ERA5's effective resolution,
which may flatten the profile. B2(ii) pass (0.55); B2(i) pass (0.75).

## Scope fence

No new covariates, no new regions, no re-tuning of bands, thresholds or
the strain scale after the first read of any Phase-27 number. Anything
beyond the statistics listed above is exploratory and labelled so.
Known limitations declared in advance: (1) bands below ~200 km are under
ERA5's effective resolution; (2) 400-km smoothing inside a 700-km tile
is edge-affected — surrogates share the identical edge treatment, and
the UNITS arm repeats B2(i) at 800 km in 2000 x 2800 km boxes;
(3) equal-time statistics carry no direction: "inherited" is an
interpretation licensed only jointly with B2.

## Deviations

(none at freeze)
