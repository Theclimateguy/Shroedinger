# Granular Ingest Plan: MRMS + GOES (Stage 1-2)

This document defines an executable pilot ingest plan for high-granularity atmospheric data,
focused on local multi-scale events for downstream `|Lambda_local|`-style analysis,
patch statistics, and strict tail tests.

## Scope

- Stage 1: MRMS pilot ingest (open NOAA S3 `noaa-mrms-pds`)
- Stage 2: GOES pilot ingest (open NOAA S3 buckets; optional `goes2go` mode)
- Stage 3 (alignment utility): nearest-time catalog `MRMS time -> nearest GOES scan`

## Scripts

- `clean_experiments/download_mrms.py`
- `clean_experiments/download_goes.py`
- `clean_experiments/build_mrms_goes_aligned_catalog.py`
- `clean_experiments/download_matched_windows.py`
- `clean_experiments/run_ultralight_mrms_goes_pilot.py`
- `clean_experiments/pilot_events_template.csv`
- shared helper: `clean_experiments/_open_s3_http.py`

## Target Outputs

- `data_raw/mrms/raw/...`
- `data_raw/goes/raw/...`
- `manifests/mrms_manifest.csv`
- `manifests/goes_manifest.csv`
- `manifests/mrms_coverage.csv`
- `manifests/goes_coverage.csv`
- `metadata/download_log.json`
- `quicklooks/mrms/*.png`
- `quicklooks/goes/*.png`
- `aligned_catalog.parquet` (or `aligned_catalog.csv` fallback if parquet engine is unavailable)

## Naming and Folder Convention

- MRMS file layout:
  - `data_raw/mrms/raw/<REGION>/<PRODUCT>/<YYYYMMDD>/<filename>`
- GOES file layout:
  - `data_raw/goes/raw/<SATELLITE>/<PRODUCT>/<YYYY>/<DOY>/<HH>/<filename>`
- Manifests are run-level snapshots and can be regenerated idempotently.

## Idempotence Rule

Both downloaders are idempotent by local target-file existence + size check:

- if a target file already exists with expected size, status is `exists`
- if missing, it is downloaded (`downloaded`)
- in dry-run mode, status is `skipped_dry_run`

## Current Environment Constraints

This repository environment does not require `aws` CLI or `boto3` for these scripts.
Public S3 listing/downloading is done via HTTPS list API.

Spatial crop and lossless reprojection to a single analysis cube are intentionally deferred
until geospatial dependencies (`xarray`, `rasterio`, `pyproj`, GRIB readers) are installed.
Current stage preserves raw granularity and full indexing, which is required before that step.

## Recommended Pilot Run

### 1) MRMS dry-run (index first)

```bash
python clean_experiments/download_mrms.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-03 \
  --region-prefix CONUS \
  --products MultiSensor_QPE_01H_Pass2_00.00 MergedReflectivityQCComposite_00.50 Reflectivity_-10C_00.50 \
  --dry-run
```

### 2) GOES dry-run (same window)

```bash
python clean_experiments/download_goes.py \
  --start-date 2026-01-01 \
  --end-date 2026-01-03 \
  --satellites G19 G18 \
  --products ABI-L2-CMIPF GLM-L2-LCFA \
  --abi-channels C02 C08 C13 \
  --dry-run
```

### 3) Real ingest (remove `--dry-run`)

- rerun both scripts without `--dry-run`
- missing files only will be downloaded

### 4) Build aligned nearest-time catalog

```bash
python clean_experiments/build_mrms_goes_aligned_catalog.py \
  --mrms-manifest manifests/mrms_manifest.csv \
  --goes-manifest manifests/goes_manifest.csv \
  --out aligned_catalog.parquet \
  --tolerance-minutes 10
```

For dry-run manifests, use:

```bash
python clean_experiments/build_mrms_goes_aligned_catalog.py \
  --mrms-manifest manifests/mrms_manifest.csv \
  --goes-manifest manifests/goes_manifest.csv \
  --out aligned_catalog.parquet \
  --tolerance-minutes 10 \
  --allowed-status skipped_dry_run
```

### 5) Download only matched windows

```bash
python clean_experiments/download_matched_windows.py \
  --aligned-catalog aligned_catalog.parquet \
  --mrms-manifest manifests/mrms_manifest.csv \
  --goes-manifest manifests/goes_manifest.csv \
  --report manifests/matched_download_report.csv \
  --summary manifests/matched_download_summary.csv
```

Add `--download` to execute actual selective downloads.

## Ultra-Light M4 Workflow (1 region, 3-5 events)

One-command orchestration for your proposed small pilot:

```bash
python clean_experiments/run_ultralight_mrms_goes_pilot.py \
  --events-csv clean_experiments/pilot_events_template.csv \
  --max-events 5 \
  --stage manifest_only \
  --mrms-product MultiSensor_QPE_01H_Pass2_00.00 \
  --goes-satellite G19 \
  --goes-product ABI-L2-CMIPF \
  --goes-channel C13 \
  --tolerance-minutes 10
```

Then switch to selective download:

```bash
python clean_experiments/run_ultralight_mrms_goes_pilot.py \
  --events-csv clean_experiments/pilot_events_template.csv \
  --max-events 3 \
  --stage download_matched \
  --mrms-product MultiSensor_QPE_01H_Pass2_00.00 \
  --goes-satellite G19 \
  --goes-product ABI-L2-CMIPF \
  --goes-channel C13
```

## Acceptance Criteria

- end-to-end automated pull from public sources (no manual clicks)
- repeat run downloads only missing files
- manifest + coverage + quicklooks are generated
- nearest-time aligned catalog is generated and ready for event segmentation / patch extraction

## Next Step After This Stage

- install geospatial stack and add strict AOI crop + common analysis format (`zarr`/`netCDF`)
- event segmentation and patch extraction on aligned windows
- compute local multiscale surrogates and run strict tail diagnostics per regime
