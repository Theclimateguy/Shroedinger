# Shroedinger v2.0 — Full Experiments Release

This release keeps the full cleaned research program and its markdown result reports.

Canonical experiment blocks:
- `TOY_MODEL`: `T01 ... T20`
- `ATMOSPHERE_DATA`: `A01 ... A15`

Canonical numbering/history source:
- `clean_experiments/EXPERIMENT_NUMBERING.md`

## Repository scope (v2.0)
Included:
- Full `clean_experiments/` codebase (`*.py`) and experiment documentation (`*.md`, `*.tex`)
- Canonical numbering/history and program manifests
- Markdown reports under `clean_experiments/results/**`
- Root program index: `research_programm_summary.csv`

Not versioned in Git:
- Local manuscript workspace (`manuscript/`)
- Heavy generated artifacts (`csv/png/npz/json`) except curated markdown reports

## Environment
Recommended:
- Python `>=3.9`

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy matplotlib xarray netCDF4 global-land-mask
```

## Reproducibility
1. Use canonical mapping from `clean_experiments/EXPERIMENT_NUMBERING.md`.
2. Run scripts with explicit `--outdir`.
3. For atmospheric experiments (`A*`), provide local ERA5/NetCDF-like inputs.

Examples:
```bash
python clean_experiments/experiment_A.py --outdir out/experiment_A
python clean_experiments/experiment_scale_gravity_einstein_box.py --outdir out/experiment_scale_gravity_einstein_box
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/data.nc --outdir out/experiment_M_cosmo_flow
python clean_experiments/experiment_scale_gravity_einstein_box_era.py --input /path/to/era_patch.nc --outdir out/experiment_scale_gravity_einstein_box_era
```

Use `--help` per script for full CLI parameters.

## Research program structure and results
- `T01 ... T20`: toy-model ladder from gauge/noncommutativity checks to fractal/holonomy bridges and synthetic Einstein-in-a-box closure (`T20`).
- `A01 ... A15`: atmospheric ladder from closure detectability and falsification blocks to strict transfer/halo tests and ERA Einstein-in-a-box closure (`A15`).
- Run-level atmospheric extensions (`A05.R*`, `A07.R*`, `A11.E*`) and consolidated markdown reports are included in `clean_experiments/results/` and linked manifests.

This v2.0 release is prepared as the full-code, full-report archival line from `main_gen2`.
