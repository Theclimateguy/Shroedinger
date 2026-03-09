# Shroedinger v1.0 — Core Experiments Release

This release keeps only the canonical core experiment program:
- `TOY_MODEL`: `T01 ... T20`
- `ATMOSPHERE_DATA`: `A01 ... A15`

Canonical numbering/history source:
- `clean_experiments/EXPERIMENT_NUMBERING.md`

## Repository scope (v1.0)
Included:
- Core scripts for `T01 ... T20` and `A01 ... A15`
- `clean_experiments/EXPERIMENT_NUMBERING.md`
- This `README.md`

Runtime support modules retained for reproducibility of core scripts:
- `clean_experiments/common.py`
- `clean_experiments/experiment_M_gksl_hybrid_bridge.py`

Everything else was intentionally removed for a minimal archival release.

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
1. Use the canonical mapping in `clean_experiments/EXPERIMENT_NUMBERING.md`.
2. Run a target script with explicit output folder.
3. For atmospheric experiments (`A*`), provide a local NetCDF/ERA-like input dataset.

Examples:
```bash
python clean_experiments/experiment_A.py --outdir out/experiment_A
python clean_experiments/experiment_scale_gravity_einstein_box.py --outdir out/experiment_scale_gravity_einstein_box
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/data.nc --outdir out/experiment_M_cosmo_flow
python clean_experiments/experiment_scale_gravity_einstein_box_era.py --input /path/to/era_patch.nc --outdir out/experiment_scale_gravity_einstein_box_era
```

Use `--help` on each script to reproduce exact CLI parameters.

## Research program structure and results
- `T01 ... T20`: toy-model verification ladder from gauge/noncommutativity checks to scale/fractal bridges and synthetic Einstein-in-a-box closure (`T20`).
- `A01 ... A15`: atmospheric validation ladder from closure detectability and falsification blocks to strict halo-transfer tests and ERA Einstein-in-a-box closure (`A15`).

This v1.0 archive is prepared as a compact, DOI-ready code release for Zenodo publication.
