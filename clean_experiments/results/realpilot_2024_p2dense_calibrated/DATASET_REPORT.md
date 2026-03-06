# Real Pilot Dense Dataset (P2-calibrated ingest)

Generated: 2026-03-06 19:15:56Z

## Selection policy
- base events: `clean_experiments/pilot_events_real_2024_us_convective_expanded_v1.csv`
- scored by: `score = P95(|lambda_local|) * P95(|comm_defect_operator|)`
- selected top events: `16`
- context expansion: `+/- 6h`
- selected events file: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/stable_events_dense.csv`
- score table: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/stable_event_scores.csv`

## P2 selected config
- config_id: `C009`
- decoherence_alpha: `0.5`
- lambda_scale_power: `0.5`
- weights [occ,sq,log,grad]: `[1.5, 1.0, 1.0, 1.0]`

## Budget check (manifest-only)
- budget: `50.000 GB`
- planned unique files (ABI+GLM union): `306`
- planned unique size (ABI+GLM union): `2.332 GB`
- ABI summary gb_total (non-unique): `2.295 GB`
- GLM summary gb_total (non-unique): `0.102 GB`

## Download result
- skip_download: `False`
- present unique files (downloaded|exists, ABI+GLM union): `306`
- present unique size (downloaded|exists, ABI+GLM union): `2.332 GB`
- ABI downloaded: `112` exists: `92`
- GLM downloaded: `102` exists: `102`

## Artifacts
- stage1 ABI workdir: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/stage1_abi_download`
- stage2 GLM workdir: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/stage2_glm_download`
- unified panel: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated.csv`
- event stats: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/results/realpilot_2024_p2dense_calibrated/realpilot_2024_dataset_panel_p2dense_calibrated_event_stats.csv`

## Panel snapshot
- rows: `240`
- events: `16`
