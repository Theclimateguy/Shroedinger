# Phase 17 Protocol (preregistered): P^eq over the long record — is there
# ANY interannual variance in the fingerprint, and is it ENSO-shaped?

Status: FROZEN 2026-08-17, before any Phase-17 P value was computed. The
data download (section 2) was launched the same day; no downloaded field
had been opened when this file was frozen. Deviations logged with
timestamp at the bottom.

## 0. Lineage

Phase 14: within windows, P relaxes in ~12 h around P^eq. Phase 16: on
2017-2024 (4 year-points per region-season) the fresh confirmatory arm
showed NO ENSO modulation of P^eq (S = 0.000), while the
hypothesis-generating arm "self-confirmed" — a selection artifact caught
by design. Phase 16's declared power limit: only mean within-cell
|rho| >= 0.35 was detectable. Phase 17 is the properly powered version:
~46 years of monthly P^eq per region.

Multiplicity discipline carried over: everything ever looked at came
from 2017-2024 (and 2021-2022). Therefore the confirmatory period is
**1979-2016** — untouched by any phase — and 2017-2024 is a reported
consistency check only. After the Phase-16 null, no direction is
preregistered: the primary test is **two-sided**.

Two questions, in order:

1. **C17-0 (prerequisite): does P^eq possess interannual variance at all
   beyond sampling noise?** If not, no slow mode of any flavour can
   exist, and the stability statement gets its final quantitative bound.
2. **H17a: is whatever interannual variance exists correlated with ENSO?**

## 1. Declared power (before results)

Per region: 456 confirmatory months; ENSO decorrelation ~9 months gives
~50 effective degrees of freedom, so per-region |rho| >= ~0.28 is
detectable at alpha = 0.05 two-sided; the pooled 12-region statistic
(against the shared-shift null of section 4, which respects both
autocorrelation and cross-region dependence) detects pooled mean
|rho| >= ~0.1. This closes the power gap that limited Phase 16.

## 2. Data (fixed)

- **New download** (launched 2026-08-17, `download_b17_era5_daily.py`):
  ERA5 850 hPa u, v, DAILY 00Z snapshots, 0.25 deg, the 12 frozen
  program boxes, 1979-2024, one NetCDF per region-year ->
  `data/b17daily/era5_wind850daily_{region}_{year}.nc` (552 files,
  ~21 GB). Daily sampling suffices: Phase 14 measured ~12 h
  decorrelation for P fluctuations, so 00Z snapshots are effectively
  independent draws; a monthly median over ~30 of them has sampling SE
  ~ sd/sqrt(30) ~ 0.005 in correlation units.
- **ENSO index**: NOAA PSL monthly ONI file
  https://psl.noaa.gov/data/correlation/oni.data retrieved 2026-08-17,
  stored at `data/oni_monthly_psl.txt` (sha256 prefix e9359c8c6b7f89af).
  Note: the PSL file is the ERSST-v5-based ONI while Phase 16 froze the
  v6 table; the two differ by ~0.1 degC in places. Phase 17 uses the PSL
  file exclusively, frozen here before any P was computed.
- P machinery unchanged (Phase-2 envelopes, vorticity of 850-hPa wind,
  interior mask at 1600 km, composite = mean of rho_3, rho_4).

## 3. Quantities (fixed)

- Daily composite P per region day; monthly value
  `P_m` = median over the month's days (>= 20 valid days required, else
  month dropped).
- Deseasonalized anomaly `P~_m`: subtract the region's month-of-year
  climatology computed over the confirmatory period 1979-2016 only (so
  the check period inherits, not contaminates, the climatology).
- Amplitude placebo series `V_m`: monthly median of the daily composite
  band log-amplitude (log interior-mean D_i^2, bands 3, 4 mean), same
  deseasonalization.
- ONI series aligned to months; ONI anomalies used as published.

## 4. Hypotheses and criteria (fixed)

Nulls, both preserving all autocorrelation:

- **N-shift (primary, pooled)**: circular shift of the ONI series by k
  months, k drawn uniformly from |k| in [60, 396], the SAME k applied
  against every region in a draw (preserves cross-region dependence of
  P). 999 draws.
- **N-permyear (for C17-0)**: within a region, permute whole calendar
  years of `P~_m` blocks... no — C17-0 compares variance levels, not
  alignment: see below.

Criteria:

- **C17-0 (interannual variance prerequisite).** Per region, on
  1979-2016: ratio F = var(annual means of P~_m) / (var(P~_m)/12).
  Under "no interannual structure" (months independent given season) F ~ 1;
  null distribution from 999 shuffles of months across years (within
  month-of-year, preserving seasonality exactly). Region positive if
  F > 95th percentile of its null. Score: N_var = number of regions
  (of 12) with excess interannual variance. **C17-0 outcome is reported
  whatever it is; it gates interpretation, not computation.**
- **H17a (primary, confirmatory period 1979-2016).** Pooled
  S = mean over 12 regions of Spearman(P~_m, ONI_m). Two-sided against
  N-shift: pass if |S| > 97.5th percentile of |S_null| (i.e. p < 0.05
  two-sided).
- **C17-1 (amplitude placebo).** Same S with V_m in place of P~_m.
  Required for CONFIRMED: |S_P| > |S_V|. (The Phase-16 synthetic
  amplitude gate for P passed and carries over; cited, not re-run.)
- **C17-2 (era robustness, observing-system caveat).** S computed
  separately on 1979-1997 and 1998-2016. Required for CONFIRMED: same
  sign in both sub-eras. (ERA5 assimilation stream changes across
  1979-2024 could imprint steps on derived statistics; sign-stability
  across the satellite-era split is the minimal guard. Attribution
  beyond "co-varies" is out of scope regardless.)
- **H17b (known-era consistency, 2017-2024).** S on the check period,
  reported with sign comparison to H17a. Not a criterion (the era is
  data-tainted); noted in the verdict text only.

Reported, descriptive, no criterion weight: per-region rho map;
per-region linear trend of P~_m per decade (and its era-split); the
periodogram of the 12-region-mean P~_m (where any non-ENSO slow mode
would show); lag structure of the pooled cross-correlation P~ x ONI
(+-24 months); same battery for the placebo V_m.

## 5. Verdict rule (fixed)

- **STATIC_CONFIRMED**: C17-0 shows excess interannual variance in at
  most 2 regions AND H17a fails. The fingerprint is static at every
  timescale from 12 h to ~40 years within measurement power; the
  programme's stability statement becomes final with a quantitative
  bound (the 95th-percentile F per region is the bound). This is a
  positive, publishable closure, not a failure.
- **VARIANCE_WITHOUT_ENSO**: C17-0 positive in >= 3 regions but H17a
  fails. Interannual variance exists but is not ENSO-shaped; its
  spectrum is reported; any further attribution (PDO/AMO/trend) is a
  new phase with its own protocol.
- **ENSO_CONFIRMED**: H17a passes AND C17-1 AND C17-2. Wording licensed:
  "P^eq co-varies with ENSO over 1979-2016" (direction as found);
  causal language stays forbidden.
- **ENSO_PARTIAL**: H17a passes, C17-1 or C17-2 fails.

## 6. Compute plan

- `clean_experiments/experiment_B17_penv_long_record.py`
  -> `clean_experiments/results/experiment_B17_penv_long_record/`.
  Stages: `--stage series` (daily P per region-year, cached npz),
  `--stage tests` (monthly assembly, C17-0, H17a/b, C17-1/2,
  descriptives, verdict).
- Seed fixed: 20260817. Draws: 999 everywhere.
- Download integrity check before series: every file opens, has >= 360
  time steps, both variables finite on the interior mask; failures
  re-queued via the resume-safe downloader before any P is computed.

## 7. Explicitly out of scope (frozen)

- Pre-1979 ERA5 (sparse observing network; a separate protocol may
  extend if Phase 17 confirms anything).
- Attribution to PDO/AMO/global-warming trend (descriptive spectrum
  only; any test is Phase 18+).
- Any change to bands, composite, or the P machinery.

## Deviations

- (none yet)
