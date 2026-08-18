# Protocol Phase 22: theory-candidate tests, round 2 (QT-P2; QT-P1 seasonal power)

Status: FROZEN before any Phase-22 computation.
Date frozen: 2026-08-18. Deviations logged with timestamp at the bottom.

## 0. Lineage

Round 1 (Phase 21) left QT-P2 untested and QT-P1 open (right sign,
underpowered), with the protocolized power route "more seasons per
year". Both are computable on on-disk data: the B13 ECMWF-IFS-HR
free-running globals (ua/va 850 hPa, 6-hourly, JFM+JAS 2014, 0.5 deg)
and data/b17daily. No new downloads (author's decision at freeze:
optional extensions deferred).

## 1. Arm P2 — model transfer of the global map

- Claim (QT-P2): a free-running model sharing tier-1 theta reproduces
  the global P map tile-by-tile. Falsified if a model with correct
  tier-1 fields yields a significantly different map.
- Resolution matching: the model grid is 0.5 deg. Three maps are
  computed on the Phase-20 Arm-B tile grid (60S-60N, orography
  exclusion unchanged): (i) MODEL: ECMWF-IFS-HR highresSST-present,
  JFM+JAS 2014, native 0.5 deg; (ii) ERA5_05: the Phase-20 ERA5 2023
  fields subsampled to 0.5 deg (every 2nd point), same seasons as
  Phase 20; (iii) ERA5_025: the existing Phase-20 Arm-B map. Anchored
  fine-P per tile-season exactly as Arm B (ELLS_FINE, 99 surrogates,
  seed scheme salted per carrier); scored value = 2-season mean.
- Ceiling: rho_ceil = Spearman(ERA5_025, ERA5_05) over scored tiles —
  the pure resolution-and-machinery effect on the same atmosphere.
- Criteria:
  - H22-P2a (primary): rho_mod = Spearman(MODEL, ERA5_05) over scored
    tiles. SUPPORTED if rho_mod >= 0.5 AND rho_mod >= 0.6 * rho_ceil.
  - H22-P2b (falsification): rho_mod < 0.3 -> QT-P2 FALSIFIED
    (significantly different map; the model shares tier-1 theta by
    construction).
  - Between the bars -> P2_PARTIAL (reported; neither pass nor kill).
  - Descriptive: sign pattern of the covariate loadings on the MODEL
    map (eke from model wind, tier-1 fields as in Arm B) vs the ERA5
    loadings; the year mismatch (2014 vs 2023) is noted and accepted —
    the map was season-stable and the claim is climatological.
- Verdict: P2_SUPPORTED / P2_PARTIAL / P2_FALSIFIED.

## 2. Arm P1s — forced drift, seasonal carrier (declared power route)

- Same prediction and falsification condition as QT-P1 (unchanged bar);
  carrier upgraded from annual medians to season-type series: per tile
  and per season-type s in {JFM, JAS}, the yearly series of seasonal
  medians of the fine composite (>= 60 days required per season-year),
  seasonal spectral features, seasonal CAPE means (data/b21 monthly).
- Spectrum-fixing, Theil-Sen trends, top-CAPE-trend-tercile selection
  and D_pred exactly as Phase 21 Arm P1, applied per season-type; the
  pooled statistic D_obs pools tile-season-types (blocks = regions;
  within-region shuffle null, 999; cluster bootstrap CI over regions,
  9999; seed 20260818).
- Criteria (unchanged from the frozen P1 bar): H22-P1a sign
  (D_obs < 0, p < 0.05); H22-P1b magnitude (CI vs D_pred; UNDERPOWERED
  iff CI covers both 0 and D_pred; FALSIFIED iff CI excludes D_pred,
  lies above 0.5 * D_pred, and contains 0).
- Verdict: P1_SUPPORTED / P1_FALSIFIED / P1_UNDERPOWERED / P1_MIXED.
  This is the declared second and final test of QT-P1 on this record;
  a second UNDERPOWERED closes the prediction as UNTESTABLE_ON_RECORD
  (candidate keeps it open only for future records or new carriers).

## 3. Verdict rule

PHASE22_VERDICT = (P2, P1s). Results entered into the candidate per its
standing rules (v1.2 -> v1.3).

## 4. Compute plan

- Code: clean_experiments/experiment_B22_qt_round2.py
  (stages: model-tiles / era5-05-tiles / p1s-series / tests).
- Results: clean_experiments/results/experiment_B22_qt_round2/.
- Seed 20260818. Figures: visualize_B22_qt_round2.py.

## 5. Out of scope

- CMCC-CM2-VHR4 second witness and ERA5-2019 map robustness (deferred
  options, author's call).
- Any SPCZ-ENSO scoring.

## Deviations

- 2026-08-18 (pre-computation amendment, before any Phase-22 series were
  run; prompted by the author's observation that Phase-21 D_obs was ~5x
  the cross-sectional D_pred): Arm P1s additionally reports a TEMPORAL
  calibration coefficient beta_T — the pooled within-tile regression
  coefficient of DETRENDED spectrum-fixed seasonal P on DETRENDED
  seasonal CAPE (across years, per tile-season-type; linear detrending
  of both sides so the calibration shares no information with the trend
  statistic). D_pred_T = beta_T x the same CAPE-trend average is
  reported next to the frozen D_pred (beta_CS-based). SCORING IS
  UNCHANGED: H22-P1a/b keep the frozen beta_CS bar. beta_T is entered
  as the calibration prior for the future powered protocol (global
  multi-year carrier), where the space-for-time substitution question
  becomes a scored hypothesis (beta_T vs beta_CS consistency).
