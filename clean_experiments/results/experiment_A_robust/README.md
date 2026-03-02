# Experiment A Robust Artifacts

## What was validated
- Gauge invariance of `Lambda = Re Tr(F_phys rho)`.
- Noncommutativity of adjacent vertical links.

## Runs
- Grid sweep: 180 runs = `20 seeds x 3 Lx x 3 K`.
- Full-lattice gauge transform stress test: 20 runs.

## Key outcomes
- Grid sweep: all runs passed gauge invariance tolerance.
- Worst `max_abs_delta_lambda_gauge`: `1.110223e-15`.
- Full transform sweep: all runs passed `max_abs_delta_lambda_full_gauge < 1e-12`.
- Worst full-transform delta: `1.054712e-15`.

## Files
- `robustness_grid_results.csv`
- `robustness_grid_summary.csv`
- `robustness_full_gauge_results.csv`
- `robustness_full_gauge_summary.csv`
- `seed_*/...` case-level outputs
