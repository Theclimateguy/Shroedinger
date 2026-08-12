# Phase 6 Protocol (preregistered): Source-sink dynamics of curvature and the driver of synoptic decoupling

Status: FROZEN before any Phase-6 computation beyond the already-reported E6
exploration. Date frozen: 2026-08-12. Deviations logged with timestamp.

## 6A. Dynamic source-sink equation for curvature

Hypothesis (from E6): tile-level fine-band curvature C behaves as a charged
quantity: charging by convective potential, discharge by precipitation,
relaxation otherwise.

Model (fixed):
  dC_t = a + alpha*E_t - beta*R_t - gamma*C_t + eps
where, per tile: C_t = log(fine-band ||F|| + 1e-12) (tile machinery as in 5a),
dC_t = C_{t+1} - C_t; E_t = log(1 + CAPE) tile mean (6-hourly, new download);
R_t = log(precip + 1e-6) tile mean; tile-demeaned within each region-window
(fixed effects); warmup 19 steps dropped.

Data: b2b region-windows (R5-R12 x W5-W8; wind + precip on disk, CAPE new).
Fit set: R5_SPCZ, R6_SATL, R7_CONGO, R8_AUS (16 rws).
Validation set: R9_NPAC, R10_INDO, R11_EURO, R12_SAM (16 rws).

Criteria:
- C6a1 (discharge): beta > 0 in >= 12 of 16 fit rws.
- C6a2 (charge): alpha > 0 in >= 12 of 16 fit rws.
- C6a3 (held-out dynamics): on validation rws, one-step-ahead prediction of
  C using fit-set-pooled (alpha, beta, gamma) beats the AR(1)-only model
  (alpha=beta=0, gamma refit per rw) in MSE for >= 12 of 16 rws.
- Report (non-criterion): relaxation time 1/gamma distribution, pooled
  coefficients, variance explained.
PASS = all three; then the source-sink equation is declared established at
tile level and becomes the Phase-7 forecasting backbone.

## 6B. Driver of profile mode 1 (rho_5, synoptic decoupling)

Target: region-median rho_5 across the 12 regions (all available windows).
Primary frozen hypothesis: rho_5 is set by [abs(center latitude) (Coriolis)
and bulk shear ||V850 - V500|| (region-median over the two on-disk wind500
windows)].
- C6b: LOO-across-regions OLS of rho_5 on [abs_lat, shear]: R^2_LOO > 0 with
  permutation p < 0.05 (999).
Secondary (descriptive): single-predictor Spearman table over the extended
set [abs_lat, shear, cape_mean, land_frac, orog_std] and best LOO subset.

## Compute plan

- Download: `download_b6_cape6h.py` (32 CAPE files -> data/b6cape/).
- Experiments: `experiment_B6a_source_sink.py`, `experiment_B6b_rho5_driver.py`
  -> results under `clean_experiments/results/experiment_B6_source_sink/`.
- Seeds fixed: 20260811.

## Deviations

- (none yet)
