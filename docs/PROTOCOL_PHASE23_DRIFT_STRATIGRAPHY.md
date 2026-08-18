# Protocol Phase 23: drift physicality and the stratigraphic hypothesis

Status: FROZEN before any Phase-23 computation.
Date frozen: 2026-08-18. Deviations logged with timestamp at the bottom.
Goal registered by the author: a fully closed candidate status
(v2.0 = every prediction scored; the P1 magnitude leg resolved or
bounded; the reorganization-detection question given one clean scored
round).

## 0. Lineage and inputs

Candidate v1.3: P2/P3/P4/P5 supported; P1 = DRIFT_DETECTED_MAGNITUDE_OPEN
with a mandatory ERA5-internal caveat (observing-system trends). The
author's synthesis (2026-08-18, external document) adds the
stratigraphic hypothesis: P'_texture as an indicator of atmospheric
reorganization, potentially earlier/sharper than mean-field indicators.
Its monthly-scale lead-lag form is already answered negatively (B19-C,
lag 0); the decadal form is scored here. The synthesis item "the drift
is no longer a hint" is NOT adopted: the caveat stands until H23a.

## 1. Arms and criteria

### H23a (decisive): is the ERA5 drift physical?

- Data: ECMWF-IFS-HR highresSST-present r1i1p1f1, day table ua/va,
  850 hPa slice, years 1979-1986 and 2007-2014 (epoch contrast, 8+8),
  downloaded under this freeze (CEDA node; full files sliced to 850 hPa
  and removed).
- Carrier: the 144 Phase-20 tiles (12 km-boxes x 3x4), seasonal
  (JFM/JAS) medians of the fine composite per year; spectral features
  per tile-season-year; spectrum-fixing regression per tile-season-type
  (as B22 P1s).
- Selection: the SAME top-CAPE-trend-tercile tile-season-types as
  B22 (selection fixed from ERA5; no re-selection in the model).
- Statistic: epoch contrast E = mean(spectrum-fixed P, 2007-2014) -
  mean(1979-1986), pooled over the selection; region-block within
  shuffle null (999), cluster bootstrap CI (regions, 9999).
  Reference: the identical statistic computed on ERA5 (b17daily, same
  tiles, same epochs) = E_era5.
- Outcomes (frozen):
  (a) E_model < 0 with p < 0.05 AND CI overlapping the ERA5 CI ->
      DRIFT_PHYSICAL: the ERA5-internal caveat is lifted; the model
      share bounds the reanalysis-artifact share near zero.
  (b) E_model < 0 with p < 0.05, CI disjoint from ERA5's and smaller in
      magnitude -> DRIFT_PART_PHYSICAL: the physical share is
      E_model / E_era5 (reported with CI); caveat replaced by the
      quantified artifact bound.
  (c) E_model null-consistent (p >= 0.05, CI covering 0) ->
      DRIFT_NOT_CONFIRMED_IN_MODEL: the ERA5 drift remains
      ERA5-internal-suspect; QT-P1 stays open and the candidate carries
      the wound explicitly.
- Escalation rule (frozen): members r2i1p1f1, r3i1p1f1 are downloaded
  and added ONLY under outcomes (a) or (b) — H23b separates forced
  from internal via the 3-member mean and spread with the same
  statistic. Under (c) the phase stops at one member (no
  member-shopping after a null).

### H23c: the stratigraphic test at the 1998/99 IPO transition

- Data: on disk (b17daily seasonal tile series from B22; IPO-TPI index).
- Named transition (fixed): the 1998-2000 IPO phase change; window
  1979-2016 (consultation history: these series met IPO only through
  the B21 excess regions; the transition-shape question is new).
- Quantities per tile-season-type, for X in {spectrum-fixed P, CAPE,
  E_syn proxy, spectral slope}: the step statistic
  S_X = |mean(X, 1999-2008) - mean(X, 1989-1998)| / SD_X where SD_X is
  the residual SD around the two-epoch means. Tile family: the B21
  excess regions' tiles (R1, R3, R5 boxes = 36 tiles x 2 season-types).
- Criterion H23c-1 (scored): paired comparison across the family —
  fraction of tile-season-types where S_P > S_CAPE and S_P > S_Esyn
  and S_P > S_slope; sign-permutation null (999, flipping which member
  of each pair is "P"); scored SUPPORTED if the fraction exceeds the
  null 95th percentile for ALL THREE comparisons; FAILED if for none;
  MIXED otherwise.
- This is deliberately a hard bar: the stratigraphic claim earns
  support only by beating every standard indicator on its home ground.

### H23d: trend-sensitivity comparison (synthesis item 24)

- Data: on disk (B22 p1s series + b21 CAPE + E_syn from Arm-C series).
- Statistic per tile-season-type and X in {spectrum-fixed P, CAPE,
  E_syn, slope}: SNR_X = |Theil-Sen slope| / cluster-bootstrap SE.
  Scored: median over ALL tile-season-types of SNR_P vs each SNR_X,
  paired sign-permutation p (999). Reported as the sensitivity ranking;
  SUPPORTED for item 24 iff SNR_P ranks first with p < 0.05 against
  each; otherwise the ranking is simply reported.

## 2. Bookkeeping

Results enter the candidate as v1.4: H23a resolves or bounds the E14
caveat; H23c/H23d score the stratigraphic hypothesis (new prediction
row QT-P6 entered as SCORED with the outcome, never as an unscored
promise). v2.0 is declared only if: P1 leg closed or bounded (H23a a/b),
and QT-P6 scored either way. A (c) outcome on H23a blocks v2.0 and is
said so plainly.

## 3. Compute plan

- Downloader: clean_experiments/download_b23_hrmip_transient.py
  (r1 epochs now; r2/r3 gated by the escalation rule).
- Code: clean_experiments/experiment_B23_drift_stratigraphy.py
  (stages: model-series / tests-ondisk / tests-model).
- Results: clean_experiments/results/experiment_B23_drift_stratigraphy/.
- Seed 20260818. Figures: visualize_B23.py.

## Deviations

- (none yet)
