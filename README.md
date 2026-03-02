# Shroedinger — Clean Experiments and Documentation

This repository intentionally keeps only the canonical `clean_experiments` codebase and documentation.

## Scope

- Included:
  - `clean_experiments/` scripts and experiment docs (`*.py`, `*.md`, `*.tex`)
  - root documentation (`README.md`, `research_programm_summary.csv`)
  - generated markdown reports under `clean_experiments/results/**` (documentation only)
- Excluded from version control:
  - legacy `experiments/`
  - local `manuscript/`
  - raw/intermediate data and heavy generated artifacts (CSV/PNG/NPZ/JSON)

## Experiment set

Canonical experiment mapping is documented in:

- `clean_experiments/EXPERIMENT_NUMBERING.md`
- `clean_experiments/HYPOTHESIS_ROADMAP.md`
- `research_programm_summary.csv`

Current sequence is A, B (wave-1), D, E, F, G, G2, H, I, J, K, K2, L, M, N, O,
including the M4 staged falsification block (S1/S2/S3) in Experiment M.

## Reproducibility note

Run scripts to generate local outputs into `clean_experiments/results/...`.
These outputs are not versioned by design; keep them local or publish separately when needed.

## Quick start

```bash
python clean_experiments/experiment_A.py --out out/experiment_A
python clean_experiments/experiment_wave1_user.py --out out/experiment_B_wave1
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow
python clean_experiments/experiment_M_lambda_falsification_tests.py
python clean_experiments/experiment_N_navier_stokes_budget.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_N_navier_stokes_budget
python clean_experiments/experiment_O_entropy_equilibrium.py --input-nc /path/to/wpwp_vertical_data.nc --lambda-csv /path/to/experiment_M_timeseries.csv --outdir clean_experiments/results/experiment_O_entropy_equilibrium
```

## Notes

- `manuscript/` is maintained outside this repository.
- `clean_experiments/EXPERIMENT_M_LAMBDA_RESIDUAL_CLOSURE_STANDALONE.tex` is kept as a standalone technical note.
