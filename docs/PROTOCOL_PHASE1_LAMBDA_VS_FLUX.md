# Phase 1 Protocol (preregistered): Does Lambda_b carry information about the true cross-scale energy flux beyond cheap baselines?

Status: FROZEN before any Phase-1 data download or computation. Date frozen: 2026-08-11.
Any deviation must be logged in the Deviations section with a timestamp and reason.

## Motivation

The A15 result correlated Lambda_b with a band-weighted kinetic-energy proxy
`Pi_proxy = <|k| E(k)>_b`, which is not a spectral energy flux, on a single
3-month window that was the only passing window in a 6-window scan, using
binned R^2 and an in-sample sign alignment. Phase 1 replaces this with a
falsification-grade test:

1. Target = the real cross-scale kinetic-energy flux Pi_ell(t) computed by
   physical-space coarse-graining (Germano/Eyink/Aluie), not a spectrum proxy.
2. Lambda_b competes against standard cheap surrogates that K41 phenomenology
   already predicts to track the flux.
3. Multiple fixed windows and regions, raw (unbinned) statistics,
   autocorrelation-aware inference, spectrum-preserving surrogate nulls,
   and a fixed sign convention.

If Lambda_b does not beat the baselines under this protocol, the
Lambda-as-flux-diagnostic branch is closed.

## Data (fixed)

- Source: ERA5 pressure-level reanalysis via CDS API.
- Variables: u_component_of_wind, v_component_of_wind at 850 hPa.
- Grid: 0.25 x 0.25 deg, 6-hourly (00/06/12/18 UTC).
- Regions (2), each 20 deg lat x 40 deg lon (81 x 161 grid points):
  - R1 "WPWP": lat 10N..10S, lon 130E..170E (replication of the A15 domain class).
  - R2 "NATL": lat 55N..35N, lon 60W..20W (midlatitude storm track; independent regime).
- Windows (4), applied to both regions (8 region-windows total):
  - W1: 2017-01-01 .. 2017-03-31 (A15 replication window)
  - W2: 2017-07-01 .. 2017-09-30
  - W3: 2018-01-01 .. 2018-03-31
  - W4: 2019-07-01 .. 2019-09-30
- No region or window may be added, dropped, or re-cut after results are seen.

## Quantities

### Lambda_b(t) — unchanged from A15 pipeline

Computed exactly as in `experiment_scale_gravity_einstein_box_era.py`
(`_build_coefficients`, `_compute_rho_and_lambda`) with A15 defaults:
scale edges 50,100,200,400,800,1600,3200 km; n_modes_per_var=6; W=20;
ridge=1e-6; cov_shrinkage=0.05.

Sign convention: FIXED global sign +1 (`lambda_sign=+1`). No in-sample
alignment. All tests two-sided; a negative relation is not counted as success.

Inertial bands under test: band centers ~141, ~283, ~566 km
(the three inertial bands of A15).

### Target Pi_ell(t) — true cross-scale flux

Physical-space coarse-graining (a-priori LES / Aluie method):

- Filter: isotropic Gaussian G_ell, applied per snapshot on the lat-lon grid
  with metric-correct km spacing (dx = R cos(lat) dlon), implemented via
  per-row scaled convolution.
- Subfilter stress: tau_ij = bar(u_i u_j) - bar(u_i) bar(u_j).
- Flux through scale ell: Pi_ell(x,t) = -tau_ij dbar(u_i)/dx_j
  (i,j in {x,y}; S_ij symmetrization equivalent under contraction).
- Spatial aggregation: mean over the interior, excluding a boundary ring of
  width 2*ell on all sides (no periodicity assumption).
- ell in {141, 283, 566} km (matched to the tested Lambda bands).

### Baselines (same filter, same interior mask, per t and ell)

- B1 Smagorinsky proxy: (0.17 * ell)^2 * <|S_bar|^3>, |S_bar| = sqrt(2 S_ij S_ij).
- B2 band kinetic-energy proxy: the original A15 `Pi_proxy = <|k| E(k)>_b`.
- B3 filtered enstrophy: <omega_bar^2>.
- B4 resolved gradient: <|grad u_bar|^2 + |grad v_bar|^2>.

## Statistics (fixed)

Per region-window and band:

1. Raw Pearson R^2 and Spearman rho between each predictor X(t)
   {Lambda_b, B1..B4} and target Pi_ell(t). No binning anywhere.
2. Out-of-sample: OLS fit of Pi on X over the first half of the window,
   R^2_oos evaluated on the second half (chronological split, no shuffling).
3. Uncertainty: moving-block bootstrap on time (block length 20 steps = 5 days),
   999 resamples, two-sided 95% CI for the correlation.
4. Surrogate null for Lambda: 199 surrogates; per snapshot, randomize spatial
   Fourier phases of (u,v) (independent draws per snapshot, same for u and v
   componentwise), preserving each snapshot's power spectrum, then rerun the
   full Lambda pipeline and the correlation with the (original-field) Pi_ell.
   Lambda passes the surrogate test in a band iff its observed raw R^2 exceeds
   the 95th percentile of the surrogate R^2 distribution.

## Success criteria (fixed, evaluated on medians across all 8 region-windows)

Lambda_b is declared an informative flux diagnostic iff ALL of:

- C1: median raw R^2(Lambda -> Pi) > 0.05 with a stable sign
      (same correlation sign in >= 6 of 8 region-windows) in >= 2 of the
      3 inertial bands;
- C2: in those bands, median raw R^2(Lambda) > median raw R^2 of EVERY
      baseline B1..B4;
- C3: surrogate test passed in those bands (majority of region-windows, >= 5 of 8);
- C4: median R^2_oos(Lambda) > 0 in those bands.

If any of C1..C4 fails: Phase 1 verdict is NEGATIVE, and the
Lambda-as-flux-diagnostic claim (A15-style) is retired. Exploratory findings
may be reported but cannot overturn the verdict.

## Compute plan

- Downloader: `clean_experiments/download_b1_era5_wind.py` (CDS API,
  8 region-window NetCDF files under `data/b1/`).
- Experiment: `clean_experiments/experiment_B1_true_flux_baselines.py`
  writes per-region-window JSON + a consolidated report under
  `clean_experiments/results/experiment_B1_true_flux_baselines/`.

## Phase 2 stub (moisture budget) — to be detailed only after Phase 1 reporting

Compare any Lambda-based moisture-budget closure gain against the
mass-consistent Trenberth-corrected budget (Mayer et al. 2021; Copernicus
mass-consistent energy/moisture budget dataset) as the baseline, not against
the raw residual. Success requires beating that baseline out-of-sample on
independent windows. Details frozen in a separate protocol before any run.

## Deviations

- 2026-08-11 (spec clarification, before any data seen): the protocol did not
  fix the Gaussian filter width convention. Fixed as sigma = ell / sqrt(12)
  (Pope's filter-width second-moment convention).
- 2026-08-11 (forced deviation, before any data seen): the frozen interior
  exclusion ring of width 2*ell empties the domain interior for ell = 566 km
  (meridional extent 20 deg ~ 2224 km < 2 * 2 * 566 km). Ring width capped at
  25% of the domain extent per side; the cap binds only for ell = 566 km.
- 2026-08-11 (spec clarification, before any data seen): the first
  window-1 = 19 time steps are dropped from all statistics for all predictors
  (rolling-window warm-up, matching the A15 t0 convention). The protocol did
  not specify warm-up handling.
