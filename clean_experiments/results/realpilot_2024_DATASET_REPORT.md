# Real Pilot Dataset (MRMS + GOES ABI/GLM)

Generated: 2026-03-06

## Event windows (UTC)
Source file: `clean_experiments/pilot_events_real_2024_us_convective.csv`

- E240426A: 2024-04-26T21:00:00Z .. 2024-04-26T23:59:59Z
- E240427A: 2024-04-27T00:00:00Z .. 2024-04-27T02:59:59Z
- E240506A: 2024-05-06T21:00:00Z .. 2024-05-06T23:59:59Z
- E240507A: 2024-05-07T00:00:00Z .. 2024-05-07T02:59:59Z
- E240521A: 2024-05-21T21:00:00Z .. 2024-05-21T23:59:59Z

## Stage 1: MRMS + GOES ABI (G16, ABI-L2-CMIPF, C13)
Workdir: `clean_experiments/results/realpilot_2024_stage1_abi_download_g16`

- matched files downloaded: 30
- matched total size: 0.343 GB
- errors: 0
- disk usage (`data_raw`): ~357 MB

Key artifacts:
- `.../manifests/matched_download_summary.csv`
- `.../aligned_catalog.csv`
- `.../manifests/mrms_manifest.csv`
- `.../manifests/goes_manifest.csv`

## Stage 2: MRMS + GOES GLM (G16, GLM-L2-LCFA)
Workdir: `clean_experiments/results/realpilot_2024_stage2_glm_download_g16`

- matched files downloaded: 30
- matched total size: 0.019 GB
- errors: 0
- disk usage (`data_raw`): ~19 MB

Key artifacts:
- `.../manifests/matched_download_summary.csv`
- `.../aligned_catalog.csv`
- `.../manifests/mrms_manifest.csv`
- `.../manifests/goes_manifest.csv`

## Unified panel for next analysis
- `clean_experiments/results/realpilot_2024_dataset_panel_v1.csv`
- `clean_experiments/results/realpilot_2024_dataset_panel_v1_event_stats.csv`

Panel content:
- one row per aligned MRMS hour (5 events x 3 hours = 15 rows)
- columns for MRMS reference + ABI matched file + GLM matched file
- alignment tolerances retained from pipeline (`tolerance_minutes=10`)

## Notes for next step
- This is intentionally ultra-light (no full-scene archive).
- Suitable for first-pass heavy-tail checks, patch statistics, and local |Lambda|-proxy experiments on matched event windows.
