Status: FROZEN before any Phase-16 computation.
Date frozen: 2026-08-17. Deviations logged with timestamp at the bottom.

## 0. Lineage (frozen)

The red-team audit recorded that fixed 20x40 deg boxes carry a
latitude-dependent km geometry (dx varies ~40% within midlatitude boxes;
domain width differs ~1.6x across regions; band masks use a single median
dx), and that part of the between-region separation may be a geometric
fingerprint (docs/RECONCILIATION.md, "Where the data may be
non-representative", item 3). The executed control was statistical:
adding [abs lat, dx, domain width] to the residualization collapsed P's
beyond-spectrum clustering (+0.454 p=0.003 -> +0.216 p=0.100), leaving
P's beyond-spectrum claim CONDITIONAL, while ||F|| survived unchanged
(+0.767 -> +0.754, p=0.001).

Phase 16 replaces the statistical control with a DESIGN control: regions
are redefined in kilometres, removing the confound from the sampling
design instead of subtracting it in regression. This phase is a
prerequisite for Phase 14 (flux-derivative dynamics), whose K_b and K_R
are defined in km bands and km boundary integrals and are therefore not
comparable across regions under degree boxes.

Relation to Phase 15 (long-term trends / ENSO): none. Phase 16 changes
only the sampling geometry; it consumes the same ERA5 region-window files
re-extracted at new boundaries.

## 1. Question and frozen predictions (fixed before any data are touched)

Question: do the validated regional signatures — the anchored
envelope-coupling profile P (primary) and the transfer-asymmetry index A
(negative control) — survive equal-km regionalization?

Directional predictions, fixed here:

- P16-1 (A, negative control): A's between-region signature strength and
  its regional geography change negligibly (Spearman rho of the regional
  A values, degree boxes vs km boxes, >= 0.9). Rationale: A survived the
  statistical geometry control; a design control should not move it.
- P16-2 (P, the open question): UNRESOLVED BY DESIGN. Two outcomes are
  both publishable and neither is renegotiated post-hoc:
  (a) anchored-excess P signature survives (clustering statistic p<0.05)
      -> the conditional verdict is lifted upward; P's beyond-spectrum
      component is physical, not cartographic;
  (b) it collapses (p>0.05) -> the conditional verdict is confirmed
      downward; the manuscript reports the beyond-spectrum component of P
      as substantially a box-geometry artefact, and P's defensible
      content is its raw (spectrum-consistent) regional profile.
- P16-3 (dynamic-scale robustness arm, declared secondary): bands defined
  in units of the local deformation-radius proxy are run as robustness
  only, never scored. A signature present in BOTH the fixed-km and the
  dynamic-scale readings is the strongest admissible claim.

## 2. Data (fixed)

- ERA5 850 hPa u,v already on disk is reused where the km windows fall
  inside the existing downloads; where they do not, a boundary-extended
  re-extraction is logged per region-window (downloader
  clean_experiments/download_b16_era5_kmwin.py, CDS API, identical
  variables/step/level).
- Region set: the frozen 12 regions (same centres). Two seasons per
  region (JFM, JAS) x the Phase-3 window years 2023-2024 = 48
  region-windows primary; 2021-2022 (W5-W8, R5-R12) held out for the
  out-of-sample arm, unconsulted until primary hypotheses are scored.
- MERRA-2 and the free-running HighResMIP fields are NOT re-run in this
  phase; cross-system replication of the km design is Phase-17 territory
  and is declared out of scope here.

## 3. Quantities (fixed)

### 3.1 Equal-km region table (primary, Variant B)

Regions keep their frozen centres and their nominal extent 2000 x 4000 km.
For each region, degree boundaries are chosen such that the
great-circle width at the region's mean latitude and the meridional
height equal the nominal km extents. No reprojection, no interpolation:
fields stay on the native 0.25 deg lat/lon grid; only the index masks
change. The number of native cells differs across regions (logged);
Spearman-based statistics are insensitive to cell count by construction.

### 3.2 Band masks

_build_band_masks with SCALE_EDGES_KM = [50,100,200,400,800,1600,3200],
but the local dx is now per-direction: zonal and meridional grid spacings
at each latitude row enter the mask construction separately, eliminating
the single-median-dx anisotropy (at 60N the zonal cell step is ~14 km vs
~28 km meridional on the native grid). Interior margin rule unchanged
(same physical width in km as Phase 2-4, recomputed per region).

### 3.3 Descriptors

- P: recomputed with the frozen Phase-2/3 machinery, primary form =
  ANCHORED excess (P - P_phase-surrogate), surrogates regenerated on the
  km windows (the surrogate base must share the new geometry).
- A: recomputed with the frozen Phase-4 machinery (curvature_profiles,
  SEED, WINDOW 20, RIDGE 1e-6, SHRINK 0.05, NMODES 6, WARMUP 19),
  resolved bands b = 2,3.
- Per-region-window values: medians over time, exactly as in Phases 3-4.

### 3.4 Secondary arm (not scored)

Same computation with band edges scaled by the local Rossby
deformation-radius proxy (f-dependent), reported descriptively.

## 4. Hypotheses and criteria (fixed)

- C16-1 (pipeline sanity, computed FIRST). On every region-window: the
  climatological band variances sigma_b on km windows must agree with the
  degree-box values within the known geometric scaling (logged, no
  threshold on the descriptor side); vorticity field interior coverage
  >= 95% of cells in every region-window. Failure aborts Phase 16 as a
  pipeline fault; no hypothesis is scored.

- H16a (negative control, A). Spearman rho between the 12 regional A
  values (degree boxes, existing Phase-4 outputs) and the 12 regional A
  values (km boxes) >= 0.9, AND the between-region signature test
  (region-label permutation, 999 shuffles) passes on km boxes at p<0.05.
  Failure of the rho>=0.9 leg means the design change moves even the
  geometry-robust descriptor: Phase 16 then reports a measurement
  instability and ALL km-box results are demoted to exploratory.

- H16b (primary, P anchored). The anchored-excess between-region
  clustering statistic on km boxes: permutation p<0.05 (999 region-label
  shuffles), effect size reported alongside. Scored on 48 region-windows;
  the 39-domain geometry-control rerun remains out of scope (logged).

- H16c (regional geography preservation). Spearman rho between degree-box
  and km-box regional anchored-P profile vectors (rho_1..rho_5 per
  region, flattened): reported per region and pooled; pooled rho is
  descriptive, no threshold — the scored object is H16b.

- H16d (out-of-sample). H16b recomputed on the 2021-2022 held-out set
  (opposite ENSO): sign of the clustering statistic must agree with the
  primary sample; permutation p<0.05 required for a CONFIRMED verdict.

- H16e (specific contrast). The Congo-Amazon anchored rho_5 contrast
  (0.71 vs 0.46 on independent periods in the degree design) must retain
  its sign and at least 50% of its magnitude on km boxes. This is the
  most confound-resistant existing result (similar observing density);
  its loss specifically implicates cartography in the programme's
  flagship regional finding.

Controls, all reported whatever they show:

- C16-2 (null calibration). H16b p-value distribution over 999 shuffled
  labels uniform (KS p>0.05).
- C16-3 (spectral placebo). The same clustering test with the isotropic
  spectral slope and log band variance as descriptors: reported; if plain
  spectral features cluster as strongly as anchored P on km boxes, Phase
  16 reports that instead.
- C16-4 (boundary-extent sensitivity). One-step sensitivity: all regions
  recomputed at 0.9x and 1.1x of the nominal km extent; descriptor values
  must correlate with the 1.0x values at rho>=0.9 per region. Logged,
  not scored.

## 5. Verdict rule (fixed)

- CONFIRMED_PHYSICAL: C16-1, H16a pass; H16b passes; H16d passes; H16e
  passes. The conditional verdict on P is lifted; the manuscript states
  the anchored profile as geometry-free, and Phase 14 proceeds on km
  regions.
- CONFIRMED_ARTEFACT: C16-1, H16a pass; H16b fails OR H16d fails OR H16e
  fails. The manuscript reports the beyond-spectrum component of P as a
  box-geometry artefact and demotes P's defensible content to the raw
  regional profile; Phase 14 proceeds on km regions with P in its
  demoted role.
- UNSTABLE_MEASUREMENT: H16a fails. The km design destabilises even the
  negative control; Phase 16 is reported as a failed design control and
  the degree-box results stand with their existing caveats unchanged.

No intermediate verdict is available. The outcome (a)/(b) wording of
P16-2 maps exactly onto CONFIRMED_PHYSICAL / CONFIRMED_ARTEFACT.

## 6. Compute plan

- Downloader: clean_experiments/download_b16_era5_kmwin.py ->
  data/b16km/ (only boundary-extended windows; overlap with data/b3 and
  data/b2b reused by index slicing where possible).
- Experiment: clean_experiments/experiment_B16_equal_km_regions.py ->
  clean_experiments/results/experiment_B16_equal_km_regions/.
- Seeds fixed: 20260817. All permutation/surrogate generators inherit
  Phase-3/4 seeds per region-window to keep surrogate bases comparable.

## 7. Explicitly out of scope (frozen)

- Dynamic re-regionalization by zero-divergence flux boundaries
  (candidate Phase-18, after Phase 14).
- Reprojection-based Variant A (equal-area grid interpolation): declared
  robustness-only, not run unless H16a fails, in which case it becomes
  the diagnostic of whether the instability is interpolation-free.
- MERRA-2 / free-running replications of the km design (Phase-17).
- Any change to band edges, estimator windows, or descriptor formulas:
  Phase 16 changes ONLY the region geometry and the per-direction dx in
  the masks.

## Deviations


- 2026-08-17 (pre-computation, logged before any descriptor was read):
  1. RENUMBERED 16 -> 18: the draft numbering collided with executed
     phases (16 = ENSO modulation of P^eq, 17 = long-record download).
     Internal references map: draft-"Phase 14" = executed B15
     (flux-derivative), draft-"Phase 15" = executed B16 (ENSO). Hypothesis
     labels below carry H18*/C18* in code and results.
  2. NOMINAL ZONAL EXTENT 4000 -> 2800 km. 4000 km at the 45-deg-mean-lat
     regions requires a 51-deg lon span; the on-disk degree boxes are
     40 deg wide and the user directed the phase to run on existing data
     (no CDS re-extraction; download_b16_era5_kmwin.py not created).
     2800 km is the largest zonal extent for which the full +-10%
     boundary-extent sensitivity arm (C16-4/C18-4) also fits inside every
     existing file (1.1 x 2800 = 3080 km -> 39.2 deg at 45N). Meridional
     2000 km unchanged.
  3. PER-ROW dx IN FOURIER MASKS: unimplementable without reprojection,
     which section 3.1 forbids. Implemented as per-direction spacing
     (dy_km; dx_km at the region mean latitude) in the A band masks —
     the P ladder already uses per-row dx by frozen Phase-2 construction.
     Within-box cos(lat) variation of dx (~38% at 45 deg) remains and is
     reported, not corrected.
  4. H16a/H18a DEGREE-BOX A REFERENCE: taken from the Phase-3 result files
     (E2_curvature.Fnorm_profile on the identical 48 primary
     region-windows) instead of the Phase-4 outputs, which are computed on
     different years (2017-2022); using them would confound period with
     geometry. Phase-4 machinery itself is unchanged (curvature_profiles
     is imported from experiment_B4_curvature_invariant.py).
  5. SURROGATE COUNTS: 199 for the scored primary and held-out arms
     (Phase-2/3 seed scheme inherited per region-window tag); 49 for the
     unscored 0.9x/1.1x and dynamic-scale arms (compute budget).
  6. DYNAMIC-SCALE ARM: deformation-radius proxy
     Ld = min(c/|f|, sqrt(c/beta)) at the region-centre latitude with
     c = 25 m/s; scale factor s = Ld/median(Ld over 12 regions), clipped
     to [0.5, 2.0]. Band edges (both the P ladder and the A Fourier
     edges) multiplied by s. Descriptive only, as frozen.
  7. A is scored as the NEGATIVE CONTROL exactly as frozen, with the
     Phase-12 caveat restated in reporting: A is a persistence statistic;
     its regional-signature status (B4/B8/B13) is what is being checked,
     not any irreversibility interpretation.
