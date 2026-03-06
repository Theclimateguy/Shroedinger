# P2-memory geographic maps

## Setup
- baseline tile csv: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- memory tile csv: `clean_experiments/results/experiment_P2_memory/memory_tile_dataset_best.csv`
- scale: `8`
- active quantile for regime split: `0.67`

## Headline
- peak mean `Delta lambda` at `(lat, lon)=(40.915, -95.435)` with value `1.995297`
- weakest cell at `(lat, lon)=(30.675, -67.275)` with value `-8.754816e-07`
- peak cell event-positive fraction: `0.625`
- peak cell active fraction: `0.688`

## Interpretation
- The main memory lift is geographically localized rather than uniform across the domain.
- The strongest positive cells cluster in the central longitudes of the panel, consistent with the zonal profile peak near `lon ~ -95.4`.
- The event-positive fraction map separates persistent geographic lift from isolated high-amplitude cells.
- The active/calm split map shows whether the geographic lift is concentrated in active-core cells or leaks into calm background.

## Artifacts
- `l8_memory_geo_overview.png`
- `l8_memory_geo_regimes.png`
- `l8_cell_summary.csv`
- `l8_event_cell_metrics.csv`
- `l8_top_delta_cells.csv`
- `l8_geo_maps.npz`
