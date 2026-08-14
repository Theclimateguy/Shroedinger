# Phase 8 Protocol (preregistered): Cross-reanalysis replication (MERRA-2)

Status: FROZEN before any Phase-8 data download or computation.
Date frozen: 2026-08-13. Deviations logged with timestamp.

## Question

Are the two validated signatures — the cross-scale envelope-coupling profile
P and the transfer-asymmetry index ||F|| — properties of the atmosphere, or
artifacts of the ERA5 model/assimilation system? MERRA-2 (different model,
different DA system, largely shared observations) is the standard first
control. Declared limitation: shared observations mean this does NOT fully
resolve the observing-density confound; it resolves model/DA-system
specificity. Geometry control (2026-08-13, committed) already demoted P's
beyond-spectrum claim (residual clustering p=0.10 after geometry); ||F||
survived (p=0.001) — Phase 8 therefore treats ||F|| as primary and P as
secondary.

## Data (fixed)

- MERRA-2 M2I6NPANA (inst6_3d_ana_Np), U and V at 850 hPa, 6-hourly,
  0.5 x 0.625 deg, subset to the 12 program regions.
- Windows W9_2023JFM, W10_2023JAS, W11_2024JFM, W12_2024JAS
  (48 region-windows; mirrors the Phase-3 ERA5 set).
- Access: NASA Earthdata via earthaccess; lazy HTTPS subsetting.

## Quantities (fixed)

P and Fnorm profiles with the standard machinery on the MERRA-2 grid
(band masks adapt to grid spacing automatically). Given the coarser grid
(dx ~ 55 km xhigher), the sub-200-km octaves are excluded a priori:
P steps 3-5 and Fnorm bands 2-5 ("resolved subset") are the tested objects,
matching the resolved-scales controls already reported for ERA5.

## Criteria (fixed)

- H8a (signature replication): within/between region clustering of the
  MERRA-2 resolved-subset Fnorm profile: diff > 0, permutation p < 0.05
  (999). Same reported (non-criterion) for P.
- H8b (cross-reanalysis geographic consistency): Spearman across the 12
  regions between region-median resolved-band log||F|| in ERA5 (Phase-3 set,
  same windows) and MERRA-2: rho > 0, p < 0.05.

Verdict:
- REPLICATED: H8a and H8b pass -> the ||F|| invariant is not an ERA5
  artifact (observing-density caveat remains, stated).
- NOT_REPLICATED: either fails -> the invariant is reanalysis-specific;
  publication must present it as an ERA5-system diagnostic.

## Compute plan

- Downloader: `clean_experiments/download_b8_merra2.py` -> data/b8merra2/
  (per-region-window NetCDF, ~4 MB each; requires ~/.netrc with Earthdata).
- Experiment: `clean_experiments/experiment_B8_cross_reanalysis.py`
  -> results under `clean_experiments/results/experiment_B8_cross_reanalysis/`.
- Seeds fixed: 20260811.

## Deviations

- (none yet)
