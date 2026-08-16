# Phase 9 Protocol (preregistered): Does A predict ensemble forecast error growth?

Status: FROZEN before any Phase-9 forecast data are downloaded or examined.
Date frozen: 2026-08-15. Deviations logged with timestamp at the bottom.

## Question

Section 6.2 of the manuscript states, as an untested hypothesis, that regional
differences in the inter-level transfer asymmetry index `A` and in the
level-coupling profile `P` "should correspond to regional differences in the
rate of error growth in ensemble forecasting systems, with regimes of high
asymmetry and weak synoptic coupling exhibiting a faster loss of
predictability."

Phase 9 tests that statement directly against an operational ensemble
prediction system. The directional prediction is fixed here, before the
forecast data are touched:

**A up  =>  error-growth rate up, predictability horizon down.**

A negative result is a publishable outcome and retires the 6.2 hypothesis
in its present form; it is not to be re-parameterised after the fact.

## Declared power limit (before results)

The unit of the primary test is the region: n = 12. At alpha = 0.05,
one-sided, a Spearman correlation is detectable only for |rho| >= 0.50
(n = 12, one-sided). Phase 9 is therefore powered for **strong effects only**.
This is stated in advance so that a null cannot later be dismissed as
"underpowered" without also conceding that any positive would have had to be
strong. The secondary within-region test (H9d) uses 48 region-windows and is
the higher-power arm.

## Data (fixed)

### Forecasts

- NOAA GEFS v12 operational ensemble, AWS Open Data mirror
  `https://noaa-gefs-pds.s3.amazonaws.com` (anonymous HTTPS, GRIB2 byte-range
  requests against the published `.idx` files).
- Product `atmos/pgrb2ap5`, 0.5 deg global, variables `UGRD` and `VGRD` at
  850 hPa (the same field and level from which `A` is computed).
- Members: `gec00` (control) + `gep01` ... `gep10` = 11 members.
- Initialisations: 00 UTC, every 5th day inside each seasonal window
  (day-of-window 0, 5, 10, ...), giving 18-19 inits per window.
- Lead times: 12, 24, 48, 72, 96, 120, 144, 168 h.
- Only initialisations whose 168 h valid time still falls inside the seasonal
  window are used, so that every init contributes the complete lead profile
  and the design stays balanced across regions and windows.

### Verification truth

- ERA5 850 hPa u, v already on disk from earlier phases
  (`data/b3` for 2023-2024, `data/b2b` for 2021-2022), block-mean coarsened
  from 0.25 deg to the GEFS 0.5 deg grid, matched at valid time.
- The dependence of the verification on the analysis system is a declared
  confound; it is controlled by the co-primary spread-based metric (C9-4),
  which requires no truth field at all.

### Samples

- **Primary (confirmation)**: windows W9_2023JFM, W10_2023JAS, W11_2024JFM,
  W12_2024JAS x all 12 regions = 48 region-windows.
- **Out-of-sample (independent years)**: W5_2021JFM, W6_2021JAS, W7_2022JFM,
  W8_2022JAS x regions R5-R12 = 32 region-windows. These are opposite-ENSO
  years and are not consulted until H9a-H9d are computed and logged.

### Descriptor side

`A` is recomputed with the frozen Phase-4 machinery
(`curvature_profiles`, SEED 20260811, WINDOW 20, RIDGE 1e-6, SHRINK 0.05,
NMODES 6, WARMUP 19) from the ERA5 region-window files, per band.
Region-window value: `A_b = log10 Fnorm_b`. Region value: median over the
window set. Headline `A` = mean of `A_b` over the resolved bands
b = 2 (200-400 km) and b = 3 (400-800 km), matching the manuscript definition.
`P` is taken from the existing Phase-3/Phase-4 outputs unchanged.

## Quantities (fixed)

All spectral band-passing uses `_build_band_masks` with
`SCALE_EDGES_KM = [50, 100, 200, 400, 800, 1600, 3200]` on the regional
0.5 deg grid; only bands b = 2, 3 (200-800 km) enter the confirmatory tests,
b = 4 (800-1600 km) is reported. Relative vorticity is computed on the sphere
with the existing `compute_vorticity`; an interior margin is excluded exactly
as in earlier phases.

For each region-window, band b and lead tau, pooled over the inits of the
window:

- `e_b(tau)` = RMS over inits and interior grid points of
  (band-passed vorticity of the ensemble mean forecast minus band-passed
  ERA5 vorticity at the same valid time).
- `s_b(tau)` = RMS ensemble spread of the band-passed vorticity, with the
  (M/(M-1)) unbiasing factor, M = 11.
- `sigma_b` = RMS of the band-passed ERA5 vorticity over the window
  (climatological band amplitude).
- Relative error `r_b(tau) = e_b(tau) / (sqrt(2) * sigma_b)`;
  relative spread `q_b(tau) = s_b(tau) / (sqrt(2) * sigma_b)`.
  `sqrt(2) * sigma_b` is the saturation value for two uncorrelated fields
  with that variance, so r, q are in [0, ~1].

Predictability metrics per region-window and band:

- `lambda_b` (**primary, error growth**): OLS slope of `ln r_b(tau)` on tau
  (days), over the leads with `r_b(tau) <= 0.6`; if fewer than 3 such leads
  exist, the first 3 leads are used and the case is flagged `early_sat`.
- `mu_b` (**co-primary, spread growth**): the same slope computed on
  `ln q_b(tau)`. Requires no truth field.
- `T50_b` (**secondary**): lead time, linearly interpolated in tau, at which
  `r_b` first reaches 0.5; right-censored at 168 h with a flag.

Region-level metric values are medians over that region's windows.

## Hypotheses and criteria (fixed)

- **C9-1 positive control (pipeline sanity).** `r_b(168) > r_b(12)` in
  100 % of region-windows and `q_b` monotone non-decreasing in tau in
  >= 90 % of region-windows. Failure aborts Phase 9 as a pipeline fault; no
  hypothesis is scored.

- **H9a (primary, between-region).** Spearman rho over the 12 regions between
  `A` and `lambda` (one-sided, positive predicted), permutation p (999
  shuffles of the regional labels) < 0.05. Scored separately for `mu`.
  Reported two-sided as well.

- **H9b (beyond controls).** Leave-one-out linear residualisation of both `A`
  and `lambda` on the control set
  [ |latitude of region centre|, land fraction, log sigma_b, isotropic
  spectral slope, r_b(12 h) ] ; Spearman of the residuals must stay positive
  with permutation p < 0.05. `r_b(12 h)` in the control set is what removes
  the observing-density / initial-error confound.

- **H9c (band specificity).** Across regions, the matched-band correlation
  rho(A_b, lambda_b) must exceed the mean of the mismatched-band correlations
  rho(A_b, lambda_b') for b != b'; permutation over band labels, p < 0.05.
  Declared exploratory if fewer than 3 usable bands remain.

- **H9d (within-region, secondary but higher power).** Over the 48
  region-windows, `lambda ~ A + region fixed effects`: the within-region
  slope must be positive with region-cluster permutation p < 0.05 (window
  labels permuted within region, 999 shuffles). This arm is immune to every
  static regional confound (latitude, land, observing density).

- **H9e (out-of-sample).** The region-level relation fitted on the 2023-2024
  sample is applied unchanged to the 2021-2022 sample (8 regions, opposite
  ENSO phase): Spearman of predicted vs observed `lambda` > 0 with p < 0.05,
  and out-of-sample R^2 > 0.

Falsification / placebo controls, all reported whatever they show:

- **C9-2 null calibration.** With regional `A` labels shuffled, the H9a
  p-value distribution over 999 draws must be uniform (KS p > 0.05).
- **C9-3 descriptor specificity.** The same tests run with (i) the coupling
  profile `P` (rho_5 and the resolved-band mean), and (ii) the plain spectral
  features (log band variance, spectral slope) in place of `A`. A claim that
  `A` carries the signal requires |rho_A| > |rho_spectral|. If a plain
  spectral feature does as well, Phase 9 reports that instead.
- **C9-4 verification-independence.** Sign agreement between the truth-based
  `lambda` and the truth-free `mu`. Disagreement means the apparent signal is
  a property of the verifying analysis, not of forecast error growth.

## Verdict rule (fixed)

- **CONFIRMED**: C9-1 passes; H9a passes for both `lambda` and `mu`; H9b
  passes; H9e passes; C9-3 shows `A` beating the spectral placebo.
- **PARTIAL**: C9-1 passes; H9a passes for at least one metric and H9b
  passes; remaining arms mixed. The manuscript may then state a
  data-supported association with the explicit scope of what failed.
- **NEGATIVE**: H9a fails for both metrics, or C9-4 shows sign disagreement.
  Section 6.2 is then reported as *tested and not supported* at this sample
  size, with the declared power limit attached.

C9-1 failure is not a verdict; it is a bug, to be fixed and rerun.

## Compute plan

- Downloader: `clean_experiments/download_b9_gefs.py` -> `data/b9gefs/`
  (per init/lead regional subsets, all 12 regions extracted from each global
  field before it is discarded; on-disk footprint of order 1 GB).
- Experiment: `clean_experiments/experiment_B9_predictability.py`
  -> `clean_experiments/results/experiment_B9_predictability/`.
- Seeds fixed: 20260815.

## Deviations

- **2026-08-15, before any hypothesis was scored** (pipeline sanity check on
  the first 3 initialisations of W9_2023JFM only, no descriptor involved):
  verifying GEFS against ERA5 gives a relative error of 0.45-0.72 already at
  the 12 h lead in the 200-400 km band. That is cross-system representativeness
  error between GDAS and ERA5 at the edge of both systems' effective
  resolution, not forecast error growth; it leaves almost no dynamic range for
  a growth-rate fit. Remedy, adopted now and fixed for the rest of Phase 9:
  the **GEFS control analysis** (`gec00` f000 on the same 0.5 deg product,
  the system's own verifying analysis) is added as a second verification
  source, and all error-based quantities are computed twice --
  `lambda_era5` (ERA5 truth, as originally frozen) and `lambda_gdas`
  (self-analysis truth). `lambda_gdas` becomes the primary error metric,
  `lambda_era5` is retained and reported as the cross-system control. The
  truth-free spread metric `mu` is unchanged. C9-4 is correspondingly
  strengthened: sign agreement is now required across all three
  (`lambda_gdas`, `lambda_era5`, `mu`); disagreement between the two
  verification sources is itself reported as evidence that the signal
  belongs to the analysis system rather than to the forecast.
  No hypothesis result had been computed at the time of this change.

- **2026-08-15, second pipeline check, still before any hypothesis was
  scored** (same 8 initialisations of W9_2023JFM, four regions, no descriptor
  involved): with self-analysis verification the error curves are clean and
  monotone, but the frozen log-linear growth-rate estimator turns out not to
  be comparable across regions. Its fitting window is chosen by where the
  curve crosses `r = 0.6`, and that crossing happens at different leads in
  different regions (immediately in the tropical bands, after 4-5 days in
  Europe), so the frozen estimator silently measures different things in
  different places and is compressed toward zero wherever the initial error
  is already large. Remedy, adopted now: the primary growth rate becomes the
  **logistic (Dalcher-Kalnay) rate** -- the OLS slope of `logit(r)` on lead
  time over the full, fixed lead set. For a logistic error curve this slope
  is the growth rate with the initial-error level absorbed into the
  intercept, so it is comparable across regions by construction and uses an
  identical fitting window everywhere. The frozen log-linear rate is retained
  and reported for every hypothesis as
  `H9a_robustness_log_linear_estimator`, so that the choice of estimator is
  visible rather than selected on the outcome. `mu` and `T50` are unchanged
  in definition. No hypothesis result had been computed at the time of this
  change.

- **2026-08-15, resource constraint, before any hypothesis was scored.** The
  available link sustains about 1 MB/s against the AWS mirror; the full
  11-member design (about 7 GB) would take some six hours. The ensemble is
  therefore cut to **6 members** (`gec00` + `gep01`...`gep05`). The cut is
  applied identically to every region, window, init and lead, so ensemble
  size is a constant of the design and cannot act as a regional confound; the
  spread estimator already carries the M/(M-1) unbiasing factor, and the
  ensemble-mean error inherits a small, uniform upward bias. Files fetched
  before the change hold 11 members and are sliced to the same first 6 in the
  same fixed order. No hypothesis result had been computed at the time of
  this change.

- **2026-08-15, C9-1 re-specified AFTER the first battery was run and seen.**
  This one is different from the three above and is flagged as such. The run
  printed the sanity gate and the whole hypothesis battery together, so the
  H9a-H9d numbers were visible before C9-1 was rewritten. The verbatim first
  run is kept unaltered as
  `results/experiment_B9_predictability/summary_run1_C9_1_as_frozen.json`
  and `report_run1_C9_1_as_frozen.md`.

  What was wrong: C9-1 as frozen demanded that ensemble spread be
  non-decreasing at *every* one of the 21 lead steps (3 bands x 7 steps) in
  90 % of region-windows. Near saturation the curves are flat and sampling
  noise from 17 initialisations makes single-step decreases routine; only
  10 % of region-windows passed, while the pooled curves are strictly
  monotone in every band. The criterion was unachievable by construction,
  not a symptom of a broken pipeline.

  Replacement: error and spread must each grow from the 12 h to the 168 h
  lead in >= 90 % of region-windows (observed 94 % and 96 %), and the pooled
  curves must be strictly monotone in every band (observed: yes). The count
  of bands already saturated at the 12 h lead is now logged rather than
  silently absorbed (these are mostly the 200-400 km band in the moist
  tropics, where the forecast has no skill at the shortest lead and a growth
  rate is not measurable; they are *not* excluded, since excluding them would
  select on the tested variable).

  No hypothesis threshold, metric definition, sample or verdict rule was
  touched, and the change cannot flatter the outcome: C9-4 (sign agreement
  across the three metrics) already fails, which forces the verdict to
  NEGATIVE regardless of C9-1. The re-run exists to score the battery on the
  normal verdict path instead of aborting at the gate.

- **2026-08-15, exploratory section added post-hoc.** After the battery was
  scored, an explicitly non-preregistered block (`EXPLORATORY_posthoc`) was
  added to record the relation between A, the initial relative error r(12 h)
  and T50. It carries no criterion and no verdict weight; it exists so the
  structure behind the null is visible rather than buried.
