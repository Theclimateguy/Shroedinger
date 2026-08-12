# Phase 7 Protocol (preregistered): Do the validated invariants carry moisture-budget closure information?

Status: FROZEN before any Phase-7 data download or computation.
Date frozen: 2026-08-12. Deviations logged with timestamp.

## Background

The program's original motivation was the poorly-closing atmospheric moisture
budget. The retired Lambda gave +0.34% MAE on one domain (Phase-1 assessment)
and exactly zero in the strict Navier-Stokes budget branch. Phase 7 asks the
same question with the objects that survived falsification: the curvature
series and the envelope-coupling series.

## Data (fixed)

- 12 program regions x W9_2023JFM, W10_2023JAS (24 region-windows).
- On disk: 850 hPa wind (data/b3). Arriving (N2 branch): IVT east/north.
- New download: total_column_water_vapour, evaporation, total_precipitation
  (6-hourly, same boxes/windows) -> data/b7budget/.

## Quantities (fixed)

Per region-window, domain-mean 6-hourly series over the ell=1600 interior mask:
- Residual r(t) = dW/dt + div(IVT) - (E - P), centered differences for dW/dt,
  metric-correct divergence (Phase-1 gradient code), all in kg m-2 per 6h.
- Invariant features (from 850 hPa wind, A15 machinery, W=20, warmup 19):
  C(t) = log fine-band ||F|| (mean of bands 0-1); R5(t) = rho_5 envelope
  coupling series (spatial rank corr per t).
- Baseline features: r(t-1), |r|(t-1), P(t), |IVT|(t), dW/dt(t), day-of-run.

## Test (fixed)

Per region-window: chronological 60/40 split. Ridge regression (lambda=1e-3,
features z-scored on train) of r(t) on baseline features vs baseline +
invariant features {C(t), R5(t), C(t-1), R5(t-1)}. Score: held-out R^2.
- C7: median incremental R^2 (full minus baseline) > 0 across the 24
  region-windows AND sign test (count of positive increments) p < 0.05.
- Context report (non-criterion): magnitude of median |r| vs the
  Trenberth-style correction scale (Mayer et al. 2021 reference values);
  per-region increments; which invariant dominates.

Verdict: POSITIVE iff C7 passes. NEGATIVE otherwise; either way the result
closes the program's original moisture-closure question with validated
instruments.

## Compute plan

- Download: `download_b7_budget.py` -> data/b7budget/ (chained after the
  Phase-6 queue to respect CDS limits).
- Experiment: `experiment_B7_moisture_closure.py` -> results under
  `clean_experiments/results/experiment_B7_moisture_closure/`.
- Seeds fixed: 20260811.

## Deviations

- 2026-08-12 (spec clarification, before any computation): tp and e retrieved
  at synoptic times are ERA5 1-hour accumulations; they are scaled by 6 as a
  6-hour proxy. This inflates |r| roughly uniformly and cannot affect the
  relative baseline-vs-invariants comparison, which is the criterion.
