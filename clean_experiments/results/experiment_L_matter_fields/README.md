# Experiment L (Matter Fields)

This folder stores artifacts for Experiment 14 (`L`): embedding a coupled
`fermion + gauge` toy system and testing:

- global Ward identity proxy: `||[Q, H]||_F`
- local continuity residual per site
- total-charge drift over time

## Files

- `experiment_L_local_continuity_dataset.csv`: per-step/per-site balance terms.
- `experiment_L_charge_timeseries.csv`: total-charge history.
- `experiment_L_case_summary.csv`: per-initial-state maxima.
- `experiment_L_summary.csv`: headline metrics and pass flags.

## Headline metrics (latest run)

- `||[Q,H]||_F = 0.0`
- max continuity residual: `1.155e-16`
- max total-charge drift: `1.332e-15`
- max link unitarity deviation: `1.110e-16`
- `pass_all = True`
