# P2-memory x-modulation visualization

## Setup
- baseline tile csv: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- memory tile csv: `clean_experiments/results/experiment_P2_memory/memory_tile_dataset_best.csv`
- normalized coordinate: `x = 2 * ((tile_ix + 0.5) / n_x) - 1`
- scales visualized: `[8, 16, 32]`

## Headline
- all-scale memory run remains `PASS_ALL=True` with `mae_gain=1.372379e-06` and `perm_p=0.020000`
- on `l=8`, mean `Delta lambda(x)` is positive on `100.0%` of x-bins
- strongest memory lift is at `x=0.000` with `Delta lambda=0.434212`
- the same peak sits at longitude `-95.435`

## Scale modulation peaks
- `l=8` peak `|lambda|` at `x=0.000` with mean `0.051960`
- `l=16` peak `|lambda|` at `x=0.000` with mean `0.069229`
- `l=32` peak `|lambda|` at `x=0.167` with mean `0.015481`

## Geographic note
- `lon_center` is globally aligned across events in this panel, so the longitude-profile is a real zonal geographic section rather than a per-event internal coordinate only.
- Longitudes are on the CONUS MRMS grid; west is more negative.

## l=8 memory interpretation
- the signed `lambda(x)` curve shifts upward relative to dense baseline across the full x-range.
- the strongest lift is concentrated near the central x-bins, not at the panel edges.
- in absolute longitude, the strongest lift sits near the central US longitudes rather than the far Pacific or Atlantic edges of the panel.
- the raw occupancy memory push is much smaller than the lambda lift and changes sign across x, which indicates that the visible effect comes from the nonlinear density-matrix bridge, not from a trivial additive inflation of occupancy.

## Artifacts
- `x_modulation_by_scale.png`
- `lon_modulation_by_scale.png`
- `l8_memory_vs_baseline.png`
- `l8_memory_vs_baseline_lon.png`
- `l8_memory_delta_heatmap.png`
- `l8_memory_delta_heatmap_lon.png`
- `l8_memory_state_profile.png`
- `l8_memory_state_profile_lon.png`
- `x_profile_by_scale.csv`
- `lon_profile_by_scale.csv`
- `l8_memory_profile.csv`
- `l8_memory_profile_eventwise.csv`
- `l8_memory_delta_heatmap.csv`

## Heatmap order
- events are sorted by mean `Delta lambda` from weakest to strongest: `E240506A, E240427C, E240602A, E240506B, E240427B, E240601C, E240522C, E240601B, E240426B, E240522B, E240521B, E240521C, E240522A, E240521A, E240506C, E240507A`
