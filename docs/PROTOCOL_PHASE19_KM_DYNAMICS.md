# Protocol Phase 19: dynamics re-asked on the equal-km territory

Status: FROZEN before any Phase-19 computation.
Date frozen: 2026-08-18. Deviations logged with timestamp at the bottom.

## 0. Lineage

Phase 18 validated the equal-km regionalization as a design control
(CONFIRMED_PHYSICAL) and established the 2000 x 2800 km boxes as the
program's geometry-clean sampling design. The dynamical negatives of the
program divide into two classes:

- ESTIMATOR-CLASS deaths — geometry-immune, NOT re-askable: B15's
  persistence gate failed on a synthetic generator that has no boxes at
  all, and RMS(x_t - x_{t-d}) = sd * sqrt(2(1-rho(d))) is an identity;
  half of B6a's relaxation was estimator-window memory. Nothing about a
  km grid touches these. They stay closed.
- ATMOSPHERE-ANSWER deaths — territory-dependent, re-askable once: B14's
  FORM_REJECTED rests on a regional sign split of the CAPE coefficient
  (R7_CONGO +4/4, R5_SPCZ -4/4, pooled p=0.13) obtained on 20x40 deg
  boxes whose tropical members are ~1.6x wider in km than the equal-km
  boxes. The territory composition changed in Phase 18; the question is
  re-asked ONCE on the km territory, with the anti-selection safeguards
  below.

## 1. Questions and frozen predictions

- ARM A (confirmatory re-ask of B14): do the B14 negatives survive the
  equal-km territory? Both outcomes registered now:
  (a) the sign split and/or the null/transfer failures persist -> the
      dynamical negative is TERRITORY-ROBUST and final for this data
      class; no further re-asks on any future grid;
  (b) regime coupling passes ALL THREE legs frozen in Phase 14 (per-rw
      coherent-null passes with n>=12 AND sign-consistency n>=12 AND
      pooled null beat) AND held-out transfer (H14d) AND the amplitude
      placebo (H14e) -> regime coupling is REOPENED, attributed to the
      degree-box territory mixing. A bare p<0.05 on any single leg is
      NOT a reopening (Phase-16 lesson).
- ARM B (new dynamical law): does the relaxation time of the band
  coupling series scale with band scale, tau_b ~ ell_b^alpha?
  Frozen alternatives: alpha ~ 2/3 (eddy-turnover / inertial-range),
  alpha ~ 1 (sweeping / Taylor advection), alpha ~ 0 (no scale
  hierarchy). Any positive alpha with a clean gate is a dynamical
  structure statement the level- and derivative-based phases could not
  make; discrimination between 2/3 and 1 is reported by CI, not scored.
- ARM C (deferred): km-crop rerun of the Phase-17 long-record analysis
  (ENSO dilution check + the tropical F~2 variance excess) when
  data/b17daily completes. Declared here, not computed in this phase.

## 2. Data (fixed, all on disk)

- Arm A: the 32 region-windows R5-R12 x W5-W8 (data/b2b wind,
  data/b6cape CAPE, data/b5precip precip), km-cropped with the frozen
  Phase-18 region table (2000 x 2800 km, same centres, index masks).
- Arm B: those 32 plus the 48 primary region-windows R1-R12 x W9-W12
  (data/b3), same km crop. 80 region-windows total.
- Degree-box reference series for C19-1: the stored Phase-14
  series_*.npz (identical windows, degree boxes).

## 3. Machinery (fixed)

- Arm A runs the FROZEN Phase-14 code verbatim: series construction
  (instantaneous P from envelope_rho_profile, composite bands 3,4;
  V band amplitude; E = log1p CAPE interior mean; R = log precip),
  deseasonalization, alpha/gamma Euler fit, circular-shift nulls
  (N=500, 20-70 whole days, both signs), pooled fit, H14b-H14f criteria
  and the B14 verdict ladder — the only change is the km crop of the
  input fields. The H14a synthetic estimator gate is NOT rerun: it is
  geography-free and transfers as-is (its cached result is copied in and
  cited). Phase-14 SEED (20260816) is baked into the frozen machinery
  and is retained inside it.
- Arm B, per region-window and band step i (i = 0..4, step scale
  ell_i = sqrt(ELLS[i] * ELLS[i+1]) = 70.7, 141.4, 282.8, 565.7,
  1131.4 km): deseasonalized rho_i(t); r1 = lag-1 (6 h) autocorrelation;
  tau_i = -6h / ln(r1), valid if 0 < r1 < 0.99; alpha = OLS slope of
  ln tau_i vs ln ell_i over valid bands, requiring >= 4 valid bands.
- Arm B estimator gate (computed FIRST): synthetic single-timescale
  cubes — x_{t+1} = a x_t + sqrt(1-a^2) xi_t with a = exp(-6h/12h),
  xi_t independent spatial phase-randomizations of real km-cropped
  snapshots (spatial spectrum preserved, one global timescale by
  construction) — pushed through the identical ladder. 5 realizations x
  4 regions (R1, R2, R7, R12: two tropical, one midlatitude, one
  subtropical). Gate: median |alpha_synth| <= 0.15. Failure -> Arm B
  verdict ESTIMATOR_INVALID, alpha on real data reported descriptively
  only.

## 4. Criteria (fixed)

- C19-1 (sanity): per-rw Pearson correlation of the km vs degree
  deseasonalized composite P series (32 rw). Median must lie in
  [0.5, 0.995]. Above 0.995: the territory change is immaterial and
  Arm A is declared UNINFORMATIVE (logged, not a failure). Below 0.5:
  measurement instability; Arm A aborts.
- H19A: the frozen B14 ladder on km series. Phase-level mapping:
  REOPENED iff final ladder verdict is P_DYNAMICS_ESTABLISHED or
  P_DYNAMICS_PARTIAL; otherwise NEGATIVE_TERRITORY_ROBUST (component
  detail reported; the sign split R7 vs R5 is tracked explicitly).
- H19B-gate: as in section 3. H19B-1 (existence): median alpha over
  valid rw > 0 AND the 95% cluster-bootstrap CI (resampling the 12
  regions, 9999 draws, seed 20260818) excludes 0. H19B-2
  (classification, not scored): CI vs 2/3 and 1 -> TURNOVER-like /
  SWEEPING-like / PRESENT_UNCLASSIFIED. H19B-3 (universality,
  descriptive): between- vs within-region alpha spread, region-label
  permutation (999, seed 20260818); reported whatever it shows.
- Arm B verdict: ESTIMATOR_INVALID / NO_SCALING (H19B-1 fails) /
  TIMESCALE_LAW_<class> (H19B-1 passes).

## 5. Verdict rule (fixed)

The phase verdict is the ordered pair (Arm A, Arm B). No aggregation.
Arm A outcome (a) closes the B14 question finally; outcome (b) mandates
a dedicated confirmatory phase on fresh years before any manuscript
claim (the reopening itself is not a positive result). Arm B
TIMESCALE_LAW_* licenses exactly one sentence class in the manuscript:
"the fluctuation timescale of the coupling profile increases with scale
as ell^alpha" — no irreversibility or causality language.

## 6. Compute plan

- Code: clean_experiments/experiment_B19_km_dynamics.py
  (stages: series / gate-b / tests), results under
  clean_experiments/results/experiment_B19_km_dynamics/.
- Phase seed 20260818 for all Phase-19-specific randomness (bootstrap,
  permutation, gate realizations); frozen Phase-14 internals keep their
  own baked-in seed.
- Figures: visualize_B19_km_dynamics.py.

## 7. Explicitly out of scope

- Any rerun of B15/B6a estimator-class questions (closed by identity).
- Arm C computation (waits for the b17daily download; separate run).
- Signed/oriented transfer events (next protocol if Arm B finds
  structure).
- Any change to band edges, estimator windows, deseasonalization, or
  the B14 hypothesis set.

## Deviations

- (none yet)

## Arm C execution spec (frozen 2026-08-18, before any Arm-C computation;
## data/b17daily complete at 552/552 at freeze time)

- Series: the 552 region-year daily files km-cropped with the Phase-18
  region table; daily P = envelope_rho_profile bands (3,4) composite,
  daily V as in B17; monthly medians with the B17 MIN_DAYS=20 rule.
  Additionally a monthly synoptic-activity proxy E_syn: interior mean of
  the across-days variance of u and v within the month (daily 00Z
  sampling; this is a proxy, not the 6-h EKE of Phase 20).
- H-C1 (confirmatory): the B17 pooled ENSO statistic S on km monthly
  anomalies, identical circular-shift null (999), CONF_YEARS 1979-2016,
  two-sided. Registered both ways: (a) stays null -> the degree-box
  ENSO negative is not a dilution artefact; (b) becomes significant ->
  dilution found, ENSO reopens on the km carrier.
- H-C2 (confirmatory): the B17 C17-0 interannual variance F-ratio per
  region on km anomalies. Question: do the three degree-box excess
  regions (R1_WPWP F=1.75, R3_AMAZ 2.24, R5_SPCZ 2.07) retain
  F > null q95? (a) >=2 of 3 retained -> the tropical excess is
  carrier-robust (physical, still unexplained); (b) <=1 retained ->
  reported as substantially a degree-box geometry artefact.
- Secondary (descriptive, the emergence-order probe): pooled lead-lag
  Spearman S(lag) between monthly P anomalies and E_syn anomalies,
  lags -6..+6 months, circular-shift null band; reported whatever it
  shows, no scoring, no causal vocabulary beyond lead/lag asymmetry.
- Seeds: 20260818. Verdicts: ARM_C = (ENSO leg a/b, excess leg a/b).
