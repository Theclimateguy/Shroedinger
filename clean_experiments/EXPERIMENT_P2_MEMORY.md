# Experiment Spec: P2-Memory (Retarded Density Matrix)

Program placement:
- canonical Group A atmosphere continuation: `A05.R5_p2_memory`
- this closes the P1/P2 scale-space series after `A05.R4_p2_l8_resolution`

## Goal
Introduce a theory-close memory term into the local scale bridge so that the
state on fine scales is not treated as purely instantaneous. The target is the
dense-panel `l=8` failure observed under locked baseline `C009`.

## Physical motivation
The scale state at `(x, l, t)` should carry information from earlier vertical
transport and dissipation, not only from the current frame. In the Hilbert
bundle language this means the effective local density matrix is retarded:

`rho_bar(t) = (1 - eta) rho_inst(t) + eta * sum_k w_k rho_inst(t - k Delta t)`

with exponentially decaying weights `w_k ~ exp(-k / tau)`.

Operationally this tests whether the dense-panel loss of `l=8` signal was caused
by missing cross-scale memory rather than by a need to manually drop operators
or retune thresholds.

## Minimal implemented form
- script: `clean_experiments/experiment_P2_memory.py`
- base bridge: locked `C009`
  - `lambda_weights = [1.5, 1.0, 1.0, 1.0]`
  - `lambda_scale_power = 0.5`
  - `decoherence_alpha = 0.5`
- target panel: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- standard active threshold remains `3.0`
- full operator block remains active: `occ`, `sq`, `log`, `grad`

Memory-lite implementation:
- lagged tile features on `(event_id, scale_l, tile_iy, tile_ix)`
  - `m_occ_l_t1`, `m_occ_l_t2`
  - `m_occ_2l_t1`, `m_occ_2l_t2`
  - `m_density_l_t1`, `m_density_l_t2`
  - `m_density_2l_t1`, `m_density_2l_t2`
  - `m_pers_l_t1`, `m_pers_l_t2`
- memory is applied only on selected scales, default `l=8`
- state populations and off-diagonal coherence are blended with lagged history
- blocked CV remains by `event_id`

## Confirmed final configuration
- config ID: `M001`
- `memory_source = occupancy`
- `lookback = 2`
- `memory_eta = 0.8`
- `memory_tau = 0.75`
- `persistence_power = 0.0`
- `memory_scales = [8]`

## Confirmed results
Dense C009 baseline:
- `l=8`: `mae_gain = -1.330761e-07`, `perm_p = 0.80`, `event_positive_frac = 0.50`, `PASS_ALL = False`
- `ALL`: `mae_gain = 3.097983e-07`, `PASS_ALL = False`

P2-memory final:
- `l=8`: `mae_gain = 1.224306e-06`, `r2_gain = 1.343877e-04`, `perm_p = 0.02`, `event_positive_frac = 0.9375`, `PASS_ALL = True`
- `16`: `mae_gain = 1.351097e-06`, `perm_p = 0.02`, `PASS_ALL = True`
- `32`: `mae_gain = 4.321956e-06`, `perm_p = 0.02`, `PASS_ALL = True`
- `ALL`: `mae_gain = 1.372379e-06`, `r2_gain = 2.177551e-04`, `perm_p = 0.02`, `event_positive_frac = 0.833333`, `PASS_ALL = True`

## Interpretation
- Memory restores the fine-scale bridge without ad-hoc channel dropping.
- Memory restores the fine-scale bridge without lowering the active threshold.
- The successful regime is short-memory and occupancy-led, which is consistent
  with fast but non-instantaneous decorrelation on the smallest resolved scale.
- Within the current A05 program this is the preferred physical closure of the
  `l=8` problem and the final experiment in the P1/P2 series.

## Canonical outputs
- final report:
  `clean_experiments/results/experiment_P2_memory/report.md`
- output directory:
  `clean_experiments/results/experiment_P2_memory/`

## Supporting visual diagnostics
- zonal/internal x profiles:
  - script: `clean_experiments/visualize_p2_memory_x_profiles.py`
  - report: `clean_experiments/results/experiment_P2_memory_x_viz/report.md`
- geographic `lat-lon` maps for `l=8`:
  - script: `clean_experiments/visualize_p2_memory_geo_maps.py`
  - report: `clean_experiments/results/experiment_P2_memory_geo_viz/report.md`

These diagnostics are not separate canonical runs. They are article/support
visualizations attached to `A05.R5_p2_memory`.

## Article-facing caveat
This is a minimal effective memory bridge, not a full GKSL plus explicit CPTP
history model. In the manuscript it should be described as a theory-motivated
retarded surrogate that empirically closes the dense fine-scale gap.
