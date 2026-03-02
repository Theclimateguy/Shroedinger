# Shroedinger — Clean Experiments and Documentation

This repository keeps the canonical `clean_experiments` codebase and documentation.

## Scope

- Included:
  - `clean_experiments/` scripts and experiment docs (`*.py`, `*.md`, `*.tex`)
  - root docs (`README.md`, `research_programm_summary.csv`)
  - markdown reports in `clean_experiments/results/**`
- Excluded from Git:
  - legacy `experiments/`
  - local `manuscript/`
  - heavy generated artifacts (`CSV/PNG/NPZ/JSON`)

## Program Numbering (1-20)

Source of truth: `research_programm_summary.csv`.

1. A — gauge invariance and noncommutativity (`experiment_A.py`)
2. B / wave-1 — `Lambda_matter` sinusoidal/state checks (`experiment_wave1_user.py`)
3. D — balance closure on `(t,x,mu)` (`experiment_D.py`)
4. E — coherence-driven rate diagnostics (`experiment_E.py`)
5. F — sinusoidal-law robustness (`experiment_F.py`)
6. G — fixed-phase profile scan (`experiment_G.py`)
7. G2 (toy-chain) — Clausius regression vs epsilon (`experiment_G2_toy_chain.py`)
8. G2 (single-qubit) — Clausius regression vs profile (`experiment_G2_single_qubit.py`)
9. H1 — holographic truncation with layer growth (`experiment_H_holographic.py`)
10. H2 — continuum conservation extrapolation (`experiment_I_continuum_conservation.py`)
11. H3 — Berry-phase refinement (`experiment_J_berry_refinement.py`)
12. H4 — `Lambda_matter` bridge test (`experiment_K_lambda_bridge.py`)
13. H4b — theory-space curvature vs RG noncommutativity (`experiment_K2_theory_space_curvature.py`)
14. H5 — matter fields (`fermion+gauge`) (`experiment_L_matter_fields.py`)
15. M1 — macro detectability in ERA5 moisture closure (`experiment_M_cosmo_flow.py`)
16. M2 — horizontal/vertical consistency + placebo noise controls (`experiment_M_horizontal_vertical_compare.py`)
17. M3 — land/ocean detectability and noise probe (`experiment_M_land_ocean_split.py`, `experiment_M_land_ocean_noise_probe.py`)
18. O1 — thermodynamic (Clausius baseline vs `+Lambda`) test (`experiment_O_entropy_equilibrium.py`)
19. O2 — spatial macro-signal diagnostics (`experiment_O_spatial_variance.py`, `experiment_O_lambda_spatial_viz.py`, `experiment_O_spatial_active_west.py`)
20. M4 — staged falsification of Lambda necessity (S1/S2/S3) (`experiment_M_lambda_falsification_tests.py`)

## Auxiliary (not in 1-20 program numbering)

- N moisture-budget follow-up branch:
  - `experiment_N_navier_stokes_budget.py`
  - `experiment_N_followup_dual.py`

## Reproducibility

Run scripts locally and write outputs to `clean_experiments/results/...`.
Heavy artifacts are intentionally not versioned in GitHub.

## Notes

- `manuscript/` is maintained outside this repository.
- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex` is a standalone technical note.
