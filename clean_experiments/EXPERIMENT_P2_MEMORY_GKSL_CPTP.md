# Experiment Spec: P2-Memory Full GKSL/CPTP

Program placement:
- canonical Group A atmosphere continuation: `A05.R6_p2_memory_gksl_cptp`
- this is the full-model continuation after surrogate step `A05.R5_p2_memory`

## Goal
Replace lag-mixing surrogate memory with explicit CPTP state evolution and test
whether the `l=8` recovery remains under a full effective GKSL formulation.

## Model
For each `(event_id, scale_l, tile_iy, tile_ix)` trajectory, for selected
memory scales (default `l=8`) propagate:

`rho_t = Reset_t o GAD_t o Dephase_t o U_t (rho_{t-1})`

Components:
- `U_t`: Hamiltonian unitary (`sigma_z` phase rotation, data-driven frequency)
- `Dephase_t`: dephasing channel (`gamma_phi`)
- `GAD_t`: generalized amplitude damping (`gamma_relax`, equilibrium from instant state)
- `Reset_t`: measurement/update channel that anchors to instantaneous `rho_inst(t)`

Each block is CPTP; composition is CPTP. After each step, numerical projection to
PSD trace-1 is applied for stability.

## Implementation
- script: `clean_experiments/experiment_P2_memory_gksl_cptp.py`
- baseline bridge: locked `C009`
  - `lambda_weights = [1.5, 1.0, 1.0, 1.0]`
  - `lambda_scale_power = 0.5`
  - `decoherence_alpha = 0.5`
- panel: `clean_experiments/results/experiment_P2_noncommuting_coarse_graining_dense_calibrated/p2_tile_dataset.csv`
- CV: blocked by `event_id`

## Confirmed best configuration (49 permutations)
- config ID: `G001`
- `gksl_dephase_base = 0.8`
- `gksl_dephase_comm_scale = 0.4`
- `gksl_relax_base = 0.8`
- `gksl_relax_comm_scale = 0.0`
- `gksl_measurement_rate = 1.6`
- `gksl_hamiltonian_scale = 0.2`

## Confirmed results
Dense C009 baseline:
- `l=8`: `mae_gain = -1.330761e-07`, `perm_p = 0.80`, `PASS_ALL = False`
- `ALL`: `mae_gain = 3.097983e-07`, `PASS_ALL = False`

Surrogate memory (`A05.R5`) best:
- `l=8`: `mae_gain = 1.224306e-06`, `perm_p = 0.02`, `PASS_ALL = True`
- `ALL`: `mae_gain = 1.372379e-06`, `perm_p = 0.02`, `PASS_ALL = True`

Full GKSL/CPTP (`A05.R6`) best:
- `l=8`: `mae_gain = 2.270824e-06`, `r2_gain = 2.134425e-04`, `perm_p = 0.02`,
  `event_positive_frac = 0.9375`, `PASS_ALL = True`
- `ALL`: `mae_gain = 2.191610e-06`, `r2_gain = 2.891841e-04`, `perm_p = 0.02`,
  `event_positive_frac = 0.833333`, `PASS_ALL = True`

## CPTP validity diagnostics
- max CPTP violation proxy (trace + PSD): `~8.88e-16` (numerical precision)
- mean channel coefficients on applied rows:
  - `gamma_dephase ~ 0.0010`
  - `gamma_relax ~ 0.0008`
  - `reset_kappa ~ 0.0016`

## Interpretation
- Memory benefit on `l=8` is retained under explicit CPTP dynamics.
- The gain is not a surrogate-only artifact: full GKSL/CPTP outperforms
  surrogate in `mae_gain` on both `l=8` and `ALL`.
- This removes the main methodological caveat of `A05.R5` and closes the memory
  branch with a full effective dynamical model.

## Canonical outputs
- run report:
  `clean_experiments/results/experiment_P2_memory_gksl_cptp/report.md`
- output directory:
  `clean_experiments/results/experiment_P2_memory_gksl_cptp/`
