# PROTOCOL — Phase 26: does the coupling map carry information on the geography of precipitation extremes?
## Status: FROZEN 2026-09-21, before any daily precipitation field was read

Motivation (author's conjecture, 2026-09-21). On the Arm-B map the land
tiles order as: northern inner Eurasia (P ~ 0.28-0.30) > Europe (~0.25) >
Mediterranean (~0.24) > East/South China (~0.17), while CAPE orders the
other way. Conjecture: low coupling marks territories where mesoscale
extremes are convectively self-organised and therefore (i) more intense and
(ii) more sensitive to warming; high coupling marks territories whose
mesoscale variability is slaved to synoptic systems. The reading "high P =
calm" is NOT the hypothesis (the Southern Ocean has the highest P on the
planet); the hypothesis concerns the TYPE of extremes and only
precipitation extremes are tested here.

Prior logged before data (assistant): NEGATIVE for the scored increment -
in Phase 25 the convective fraction and precipitable water were absorbed
by the CAPE channel (loadings on P -0.61 / -0.56), so P is expected to
correlate with extremes but to add little beyond CAPE and moisture.

## Data (to be downloaded after freezing)

ERA5 daily sums of total precipitation (CDS
`derived-era5-single-levels-daily-statistics`, from hourly, UTC days),
60S-60N, native 0.25 deg, two epochs: E1 = 1981-1990, E2 = 2015-2024.
Each month is reduced on download to per-grid-cell annual statistics and
the raw file is deleted (disk constraint): Rx1day (annual maximum daily
sum), PRCPTOT (annual total), wet days (>= 1 mm). Downloader:
`clean_experiments/download_b26_precip_daily.py` -> `data/b26precip/`.
Moisture baseline `pw`: climatological total column water vapour from the
Phase-25 file (`data/b25hydro`, 2021-2024), unchanged.

Feasibility clause. If E1 cannot be obtained in reasonable time, T2 and
H26b are declared NOT RUN (not negative). E2 alone suffices for H26a.

## Targets (tile means over the frozen Arm-B grid, 902 tiles)

- T1 `rx1_clim` = log10 of the E2 mean of Rx1day (mm/day): climatological
  intensity of daily precipitation extremes.
- T2 `rx1_change` = ln(mean Rx1day E2 / mean Rx1day E1): epoch change of
  extreme intensity, the proxy for sensitivity to warming (35-year
  separation of epoch centres).
- T1b (reported only) `conc` = log10(mean Rx1day / mean PRCPTOT), E2:
  share of the annual total delivered by the wettest day.

## Predictors

- Baseline B0 (11): the 8 frozen Phase-20 covariates (orog_mean, orog_std,
  land_frac, coast_var, abs_lat, sst_grad, cape_mean, eke_syn) + `pw` +
  log10 mean PRCPTOT (E2) + mean wet-day count (E2). No interaction terms.
  For T2 the baseline adds T1 (initial intensity).
- Candidate: P = two-season mean anchored fine-P of the committed Arm-B map.

## Scored hypotheses (bars set here, same ladder as Phase 25)

- H26a (primary): LOSO-sector (six 60-deg sectors) R^2 increment of P over
  B0 for T1. Null: 999 longitude rotations (>= 30 deg) of P only.
  - P_ADDS_STRONG: increment >= +0.05 and p <= 0.01;
  - P_ADDS:        increment >= +0.03 and p < 0.05;
  - NEGATIVE otherwise.
- H26b (secondary, scored, conditional on E1): the same for T2.

## Reported (not scored)

- R1: Spearman of P with T1, T2, T1b; rotation-null p; partial Spearman
  controlling cape_mean and pw (rank-residual method).
- R2: the same table for land tiles only (land_frac >= 0.5).
- R3: descriptive box table (land tiles; boxes fixed here): N inner Eurasia
  50-58N 40-140E; East-European plain 50-58N 30-60E; Siberia 50-58N
  60-140E; Europe 44-52N 0-30E; Mediterranean 32-46N 0-40E; East/South
  China 20-34N 100-122E; North America 44-58N 235-290E - mean P, CAPE,
  Rx1day (E2), epoch change.

## Controls

- C26-1 (placebo): spectral slope of the tile in place of P; if the slope
  increment is >= half of the P increment, a positive H26a/b is flagged
  spectrum-shared.
- C26-2 (negative control): LAI in place of P; expected increment ~ 0.
- C26-3: B0 alone must predict T1 with LOSO R^2 > 0 (sanity of the design).

## Interpretation fence

Association only. ERA5 precipitation is a short-range model forecast
constrained by the assimilated state; daily extremes are underestimated
and multi-decadal precipitation changes in reanalyses are affected by
observing-system changes. Whatever the outcome, T2 is labelled
low-confidence, and no statement about observed impacts, damage or
"stability" of any country is licensed by this phase. No other target or
predictor may be added after data are read; any addition demotes the
phase to exploratory.

## Deviations

(to be logged at computation)

## Outcome

(entered after computation)
