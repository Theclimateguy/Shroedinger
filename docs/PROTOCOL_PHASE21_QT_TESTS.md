# Protocol Phase 21: theory-candidate tests, round 1 (QT-P5, QT-P1, QT-P4)

Status: FROZEN before any Phase-21 computation.
Date frozen: 2026-08-18. Deviations logged with timestamp at the bottom.

## 0. Lineage

The quenched-texture candidate (docs/THEORY_CANDIDATE_QUENCHED_TEXTURE.md
v1.1) carries five falsifiable predictions. QT-P3 was scored in Phase 20
Arm B (PASS, k80=1). QT-P2 requires new model runs and is deferred. This
phase scores the three predictions testable with on-disk or trivially
downloadable data: QT-P5 (sampling clock), QT-P1 (forced drift), QT-P4
(tropical excess predictors). Each arm names its falsification condition
verbatim from the candidate; a falsified prediction is reported as such
and wounds the candidate per its standing rules — no renegotiation.

## 1. Arm P5 — the sampling clock (existing data only)

- Claim under test (QT-P5): tau of P-estimate fluctuations scales with
  the advective crossing time of the observation window, not with band
  scale.
- Data: the 80 km-cropped region-windows (b3 + b2b). Window sizes:
  factors f in {0.35, 0.5, 0.7, 1.0} of the 2000 x 2800 km box, centred,
  both dimensions scaled by f.
- Quantities per rw x f: fine composite P(t) = mean of the 3 fine band
  steps (ELLS_FINE 50-400 km, computable at all sizes) at 6-h cadence;
  tau = -6h / ln(r1) of the deseasonalized series (B19 machinery, valid
  if 0 < r1 < 0.99); crossing time T_cross = L_zonal(f) / U_mean, with
  U_mean the time-and-interior-mean wind speed of that rw at f = 1.0
  (one U per rw, so T_cross varies only through L).
- Criteria:
  - H21-P5a (primary): pooled Spearman rho(ln tau, ln T_cross) over all
    rw x f, positive, with region-block permutation p < 0.05 (999,
    permuting whole regions' tau sets, seed 20260818).
  - H21-P5b (within-rw consistency): fraction of rw with positive
    within-rw Spearman(tau, T_cross) over the 4 sizes; >= 2/3 required.
  - Descriptive: log-log slope (cluster bootstrap CI over regions);
    band-scale flatness at fixed window is already established (B19)
    and is not re-scored.
- Falsification (verbatim consequence): no positive scaling (H21-P5a
  fails) -> QT-P5 FALSIFIED.
- Verdict: P5_SUPPORTED / P5_FALSIFIED / P5_MIXED (P5a passes, P5b
  fails: scaling present but not per-window-consistent; reported).

## 2. Arm P1 — forced drift of the texture (b17daily + monthly CAPE)

- Claim under test (QT-P1): where CAPE trends positive, tile-P declines
  over 1979-2024 at fixed spectrum.
- Data: data/b17daily (552 region-years, km-cropped, daily 00Z);
  data/b21/era5_cape_monthly_longrecord_<region>.nc (ERA5 monthly CAPE
  1979-2024, downloaded under this freeze).
- Carrier: the Phase-20 3x4 tile grid inside each km box (144 tiles).
  Per tile-year: raw fine-P (median over days of the fine composite);
  tile spectral features (fine log band variances + slope, yearly);
  tile CAPE annual mean.
- Fixed-spectrum reduction: per tile, regress the 46 annual raw fine-P
  values on the 5 yearly spectral features (OLS, all years); the
  residual series is the spectrum-fixed P. Trends: Theil-Sen slope per
  tile for spectrum-fixed P and for CAPE.
- Prediction quantification (matched power, fixed here): the
  cross-sectional Arm-A coefficient beta_CAPE = the OLS coefficient of
  z-scored cape_mean in the Phase-20 Arm-A 5-covariate regression,
  converted to physical units (dP per J/kg). Predicted per-tile drift =
  beta_CAPE_phys x observed CAPE trend. Pooled over the tercile of
  tiles with the most positive CAPE trends: predicted mean drift
  D_pred < 0.
- Criteria:
  - H21-P1a (primary, sign): pooled mean spectrum-fixed P trend over
    the top-CAPE-trend tercile of tiles, D_obs, with region-block
    permutation of the tile->trend assignment (999): D_obs < 0 with
    p < 0.05.
  - H21-P1b (magnitude): cluster-bootstrap 95% CI of D_obs (regions,
    9999) compared with D_pred: consistent if CI covers D_pred.
  - Falsification (verbatim): matched power and no drift where E8
    predicts it -> the CI excludes D_pred AND excludes all values
    <= 0.5 * D_pred while containing 0 -> QT-P1 FALSIFIED. CI covering
    both 0 and D_pred -> UNDERPOWERED (reported, not scored).
- Verdict: P1_SUPPORTED (P1a passes and CI consistent with D_pred) /
  P1_FALSIFIED / P1_UNDERPOWERED / P1_MIXED.

## 3. Arm P4 — the tropical excess predictors (named before consultation)

- Claim under test (QT-P4): the carrier-robust tropical interannual
  excess (R1_WPWP, R3_AMAZ, R5_SPCZ; F = 1.5-1.8) reflects a slow
  tier-2 tropical mode outside ONI.
- NAMED PREDICTORS (fixed now, none consulted against P): PDO, AMO
  (unsmoothed), DMI (IOD), IPO-TPI (ERSSTv5) — monthly index series
  from NOAA PSL, downloaded under this freeze into data/b21/.
- Data: the Phase-19 Arm-C km monthly P^eq anomalies (already computed;
  their consultation history: ONI only).
- Quantities: annual-mean P anomalies per excess region (1979-2016
  confirmatory years, as in B17); annual-mean index values, and the
  same with a 5-yr running mean (decadal reading); both tested.
- Criteria:
  - H21-P4 (per region, multiplicity-correct): T_r = max over the 4
    named indices x 2 smoothings of |Spearman(P_r, index)|; null: 999
    circular shifts (20-70 months, both signs, applied to the index
    family jointly), max-statistic preserved; p < 0.05.
  - Success (QT-P4 supported): >= 2 of 3 excess regions significant
    WITH the same best index (a coherent mode).
  - Falsification (verbatim): no named slow mode accounts for the
    excess at matched power -> all three regions null -> QT-P4
    FALSIFIED in its named-predictor form; the excess stands as an
    anomaly against QT-1 stationarity (reported in the candidate).
  - Intermediate (1 region, or 2 with different indices):
    P4_INCONCLUSIVE.
- Verdict: P4_SUPPORTED / P4_FALSIFIED / P4_INCONCLUSIVE.

## 4. Verdict rule (fixed)

PHASE21_VERDICT is the ordered triple (P5, P1, P4). Falsified
predictions are entered into THEORY_CANDIDATE_QUENCHED_TEXTURE.md
verbatim by appending to its evidence table (the candidate's standing
rules apply); supported predictions upgrade the corresponding row. No
aggregation across arms.

## 5. Compute plan

- Code: clean_experiments/experiment_B21_qt_tests.py
  (stages: p5-series / p1-series / tests), results under
  clean_experiments/results/experiment_B21_qt_tests/.
- Seed 20260818. Downloader clean_experiments/download_b21_cape_indices.py
  (launched at freeze time; indices + 12 small monthly-CAPE files).
- Figures: visualize_B21_qt_tests.py.

## 6. Explicitly out of scope

- QT-P2 (model transfer): requires HighResMIP global fields; deferred.
- Any SPCZ-ENSO scoring (the single preregisterable ENSO target needs
  fresh data or a declared-consultation protocol; not this phase).
- Any change to the candidate's falsification conditions.

## Deviations

- (none yet)
