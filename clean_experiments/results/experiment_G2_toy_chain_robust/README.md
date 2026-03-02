# G2 Toy-Chain Robust Artifacts

## What was validated
- Clausius-type regression behavior across `epsilon` in toy-chain G2.
- Stability of fitted slope `1/Teff` and `R2` across multiple seeds.

## Runs
- 10 seed runs.
- Per seed: full epsilon scan (`0.05, 0.15, 0.30, 0.60`).
- For each epsilon fit: `n_steps=220`, `n_traj=800`.

## Key outcomes
- Positive fitted slope for all epsilon and all seeds (`frac_positive_slope = 1.0`).
- Mean per-epsilon `R2`:
  - `eps=0.05`: `0.019165`
  - `eps=0.15`: `0.160570`
  - `eps=0.30`: `0.439254`
  - `eps=0.60`: `0.766565`
- Global fit across all epsilon:
  - mean slope: `0.644215`
  - mean `R2`: `0.246269`
  - positive slope fraction: `1.0`

## Files
- `robustness_fit_by_eps_all_seeds.csv`
- `robustness_global_fit_all_seeds.csv`
- `robustness_summary_by_eps.csv`
- `robustness_summary_global.csv`
- `worst_15_by_r2_eps_fit.csv`
- `worst_10_by_r2_global_fit.csv`
- `cases/...` per-seed outputs
