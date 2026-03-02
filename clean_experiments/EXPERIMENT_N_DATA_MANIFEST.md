# Experiment N Data Manifest (N13 Candidate Inputs)

This document defines practical data bundles for the next iteration of Experiment N.

## Scope

- Domain: WPWP box `N=10, W=120, S=-10, E=280` (same as current Experiment M/N setup)
- Period: `2017-01-01` to `2019-12-31`
- Cadence target: `3-hourly`
- Goal: build a moisture-budget dataset with explicit tendency/process terms for stronger closure tests.

## Bundle A (ERA5-first, CDS + MARS)

Use this bundle if you want to stay on ERA5.

### A1. ERA5 model-level state (CDS dataset: `reanalysis-era5-complete`)

- `param=130` temperature (`t`)
- `param=131` zonal wind (`u`)
- `param=132` meridional wind (`v`)
- `param=133` specific humidity (`q`)
- `param=135` vertical velocity (`w`)
- `levtype=ml`, `levelist=1/to/137`
- `stream=oper`
- `type=an`

### A2. ERA5 model-level physics diagnostics (same dataset, MARS-backed)

- `param=235006` mean total moisture tendency (`mqtpm`)
- `param=235009` mean updraught mass flux (`mumf`)
- `param=235010` mean downdraught mass flux (`mdmf`)
- `param=235011` mean updraught detrainment rate (`mudr`)
- `param=235012` mean downdraught detrainment rate (`mddr`)
- `levtype=ml`, `levelist=1/to/137`
- `stream=oper`
- `type=fc`
- typical `time=06/18`, `step=3/6/9/12` for 3-hourly valid times

Notes:
- These diagnostics are listed in ERA5 table 13 and are marked as not available from CDS disks (MARS-backed retrieval path through `reanalysis-era5-complete`).

### A3. ERA5 single-level hydrology/instability (CDS dataset: `reanalysis-era5-single-levels`)

- `total_precipitation`
- `evaporation`
- `convective_available_potential_energy`
- `convective_inhibition`
- optional: `convective_precipitation`, `large_scale_precipitation`

## Bundle B (MERRA-2 process-tendency path)

Use this bundle if you want direct process-decomposed moisture tendencies.

- `M2I3NPASM` (`inst3_3d_asm_Np`) state variables (`Q`, `U`, `V`, `T`, pressure-level context)
- `M2T3NPQDT` (`tavg3_3d_qdt_Np`) moisture tendency decomposition:
  - `DQVDTANA`, `DQVDTDYN`, `DQVDTMST`, `DQVDTTRB`
- `M2T3NPMST` (`tavg3_3d_mst_Np`) moist/convection process diagnostics (including convective mass-flux-related terms)
- `M2T3NPCLD` (`tavg3_3d_cld_Np`) cloud/detrainment diagnostics (`DTRAIN`, etc.)

Notes:
- Earthdata Login is required for programmatic downloads.
- For storage efficiency, subset to WPWP box and required variables as early as possible.

## Storage Planning (rough)

- ERA5 model-level state + diagnostics for 3 years over WPWP at 0.25 deg:
  - compressed GRIB: often tens of GB to low hundreds of GB, depending on chunking and fields kept.
- MERRA-2 raw full-granule pulls:
  - can exceed `100 GB`; use variable/region subsetting immediately after download.

## Scripts Included

- `clean_experiments/download_N_data_era5.py`
  - dry-run planner by default
  - can execute CDS downloads with `--download`
  - writes request JSONs + retrieval manifest CSV
- `clean_experiments/download_N_data_merra2.py`
  - Earthaccess-based search + optional download
  - writes per-granule manifest CSV and URL list

## Recommended Start Order

1. Run ERA5 planner in dry-run mode and inspect generated JSON requests.
2. Run MERRA-2 search in dry-run mode to inspect granule count/size.
3. Choose one bundle for N13 ingestion (or run both and compare).

## Primary References

- ERA5 data documentation, table 13 params and MARS notes:
  - https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
- MERRA-2 dataset and collection references:
  - https://gmao.gsfc.nasa.gov/gmao-products/merra-2/citing-merra-2-data_merra-2/
  - https://data.nasa.gov/dataset/merra-2-tavg3-3d-moist-physics-tendencies-time-average-3-dimensional-3-hourly-3d-
  - https://data.nasa.gov/dataset/merra-2-tavg3-3d-cloud-processes-time-average-3-dimensional-3-hourly-interv-91f95
  - https://data.nasa.gov/dataset/merra-2-inst3-3d-assimilation-state-time-3-hourly-3d-assimilation-assimilat-cbeab
