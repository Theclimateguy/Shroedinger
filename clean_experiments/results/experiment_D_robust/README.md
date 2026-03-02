# Experiment D Robust Artifacts

## What was validated
- Balance closure on `(t, x, mu)` in the form
  `d<O>/dt + div_x J + div_mu J_mu - sources = 0`.

## Runs
- Tier 1 broad grid: 270 runs = `10 seeds x 3 K x 3 gamma x 3 eta0`, each with 16 samples.
- Tier 2 stress runs: 5 runs with larger dimensions/samples (`L up to 5`, `K up to 6`).
- Total: 275 runs.

## Key outcomes
- All runs passed `balance_closure_ok_1e-10 = True`.
- Worst `max_abs_balance_residual`: `4.440892e-16`.
- Runs with residual `> 1e-12`: `0`.

## Files
- `tier1_grid_results.csv`
- `tier1_grid_summary.csv`
- `tier2_stress_results.csv`
- `tier2_stress_summary.csv`
- `combined_results.csv`
- `combined_summary.csv`
- `tier1_cases/...` and `tier2_stress_cases/...` per-run outputs
