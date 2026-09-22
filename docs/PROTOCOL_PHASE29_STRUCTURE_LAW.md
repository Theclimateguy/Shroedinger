# PROTOCOL — Phase 29: the structure law of inter-level coupling —
# replication, instrument independence, and amplitude vs intermittency
## Status: FROZEN 2026-09-22, before any Phase-29 statistic was computed

Disclosure. Phase 28 (same day) established on the 48 ERA5 units that the
anchored covariance of log-envelopes C_ij depends on the coarser band a_j
only (R2 0.996 vs 0.271 for separation), decays ~ a_j^-1.5, and that its
amplitude C_02 carries P's geography (rho 0.82, regions). Post-hoc, the
"depth L" was withdrawn (ladder artefact). Seen before this freeze: unit
covariances at both extents, pooled tile covariance medians for the three
non-adjacent tile pairs, the tile-level rho of lambda2 with P/CAPE/EKE.
NOT seen: any tile-level structure statistic, any model-run covariance,
any head-to-head of C_02 with the AUDIT-2 intermittency block, C_02's
split-half reliability, its increments over covariates.

## Objects

- Structure statistic (units, full ladder): dR2 = R2(factor a_j) -
  R2(factor j-i) over the 10 non-adjacent pairs, as in Phase 28.
- Structure statistic (tiles, fine ladder, 3 non-adjacent pairs): the law
  predicts C_03 = C_13 (same coarse band) and C_02 != C_13 (same
  separation). d_tile = |C_02 - C_13| - |C_03 - C_13|.
- Amplitude A = anchored C_02 (bands 50 and 200 km: non-overlapping,
  surrogate ~0, defined identically on both ladders and both grids).
- Decay exponent kappa: OLS slope of ln C_0j vs ln a_j over j = 2..5
  (units only; requires C_0j > 0 for all four; otherwise undefined).
- AUDIT-2 intermittency block: (P_cascade, INT_sig2, INT_flat, INT_mu)
  anchored, per tile, from results/experiment_A2_intermittency/tiles
  (season mean), as used by AUDIT-4.

## Carriers

UNITS: Phase-27 shards (12 regions x 4 windows; x1.0 and x0.7).
TILES: 902 Phase-27 tiles, season mean; halves H1/H2 within season.
MODEL: ECMWF-IFS-HR highresSST-present free run, 0.5 deg, 850 hPa,
  W15_2014JFM and W16_2014JAS (data/b13hrmip), the 12 Phase-18 regions
  cut with the same km_crop (factor 1.0), the Phase-27 machinery unchanged
  (full ladder, 99 phase surrogates, seed tag "b29_model_<tag>"). Model
  interior counts are ~4x smaller than ERA5's; that is accepted as is.
  This is the only new envelope computation of the phase.

## Hypotheses and decision rules

H29-1 (PRIMARY) — the structure law holds on tiles.
  Tile median d_tile > 0 with 95 % block-bootstrap CI (30x30-deg blocks,
  1999 resamples, seed 20260922) excluding 0, AND median ratio
  |C_03 - C_13| / |C_02 - C_13| <= 0.5. Reported: the per-tile sign
  frequency, the ratio's CI, and its correlation with CAPE/EKE.

H29-2 (PRIMARY) — the structure law holds in the free-running model.
  Region-level dR2 (mean of the two windows): median > 0.10 and one-sided
  Wilcoxon p < 0.05 (n = 12). Reported: model region A and kappa; Spearman
  of model region A with ERA5 region A (12 regions; reported, threshold
  0.6 as in B13 for the ranking replicate, not scored).

H29-3 (PRIMARY, the same-data test) — A is not a re-expression of the
  intermittency block. On tiles, LOSO-sector R2 with the Phase-20
  rotation null (999 rotations, min 30 deg):
    base  = 8 covariates + 4-column intermittency block;
    inc_A = R2(base + A) - R2(base).
  Null: rotate the A column only (as Phase 25 did for candidates).
  PASS: inc_A >= 0.03 and p_rot < 0.05 (ladder of Phase 25). Reported
  symmetrically: inc_block = R2(covariates + A + block) - R2(covariates + A);
  R2 of A alone, of the block alone; rho(A, INT_sig2).
  Interpretation fixed now: PASS -> A carries geography the intermittency
  parameters do not; FAIL -> A and the block are one axis, and Article 3
  presents the amplitude as the cascade-intermittency axis, not as a new one.

H29-4 (PRIMARY) — A is at least as reliable an invariant as P.
  Within-season split-half reliability of the tile map (H1 vs H2 steps,
  Spearman-Brown corrected, each season, then averaged) for A and for P
  (P halves from the same Phase-27 shards, adjacent anchored rho). PASS:
  r_SB(A) >= r_SB(P) - 0.05. Reported: cross-season rho of A vs of P;
  A units-vs-tiles co-location over 12 regions; A(x0.7)/A(x1.0) ratio.

H29-5 (reported, not scored) — decay exponent kappa: region values, window
  reproducibility (cross-season rho, perm p), F between/within, rho with
  A, with P, with CAPE and EKE (regions, n = 12). Any statement about
  kappa's geography stays exploratory unless a later protocol scores it.

## Controls

C29-1 model reproduction: model region adjacent anchored P (mean of the
  five steps) vs the stored B13 model P (mean of steps 4-5) Spearman >= 0.8
  over the 12 regions, else the model arm is void.
C29-2 tile floor: median |C_03 - C_13| against the median surrogate
  (q05-q95) width of C_03; if the difference is below floor in > 50 % of
  tiles, H29-1's ratio test is reported as floor-limited.

## Verdict ladder

STRUCTURE_LAW_ROBUST: H29-1 and H29-2 pass. AMPLITUDE_NEW_AXIS if in
addition H29-3 passes; AMPLITUDE_IS_INTERMITTENCY if H29-3 fails.
H29-4 qualifies the estimator choice (A vs P) for Article 3 only.
PARTIAL(<list>) / NOT_SUPPORTED as in earlier phases.

## Author's priors (before computing)

H29-1 0.8. H29-2 0.7. H29-3 0.35 (rho(lambda2, INT_sig2) = 0.94 in
Phase 27 S4 argues for absorption). H29-4 0.6.

## Scope fence

No thresholds, bands, extents, covariates or carriers added after the
first read. Everything else is exploratory and labelled so.

## Deviations
(none at freeze)

D1 (2026-09-22, after the first read): C29-1 as frozen compared the
  Phase-29 model value "mean of the five ANCHORED adjacent steps" with the
  stored B13 model P, which is the RAW mean of steps 4-5 on B13's own
  degree-box cut — two different objects. Frozen statistic: rho = 0.643
  (fails 0.8). Like-for-like: raw steps 4-5 rho = 0.972; anchored steps
  4-5 rho = 0.944 — the model arm reproduces B13. H29-2 is therefore
  reported as PASS-with-deviation; the frozen verdict string stays
  PARTIAL(H29_1,H29_3) in summary.json and the corrected reading is
  recorded here and in report.md. The mismatch is a design error in the
  control, not a property of the data.
