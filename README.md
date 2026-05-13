# Shroedinger v2.0

This repository accompanies the manuscript *Scale Geometry of Complex Systems: Formalism and Empirical Verification of Local Lambda_b--Pi_b Closure in Atmospheric Data*.

The manuscript was originally written in Russian and then translated into English. Both PDF versions are included in this repository. The GitHub repository is the active research/development tree; immutable archival records for both the manuscript and the repository snapshot are published on Zenodo.

## Archival Links

- Manuscript record on Zenodo: [Scale Geometry of Complex Systems](https://zenodo.org/records/19565805)
- Repository archive on Zenodo: [Shroedinger repository snapshot](https://zenodo.org/records/19565770)
- Active GitHub repository: [Theclimateguy/Shroedinger](https://github.com/Theclimateguy/Shroedinger)
- Citation metadata in this repository: [`CITATION.cff`](./CITATION.cff)

## Paper Summary

### Goal
The paper tests the hypothesis that scale in complex systems can be treated as a bona fide coordinate of description, with interscale transfer represented as geometrically organized dynamics rather than simple averaging.

### Method
The work introduces a gauge-invariant functional Lambda_matter, computed from the curvature of the connection and the state, and studies its band-resolved counterpart Lambda_b in the operational closure relation Lambda_b ~ Pi_b, where Pi_b is the spectral energy flux in a given scale band.

The empirical program combines:
- toy-model and internal validation blocks `T01 ... T20`
- atmospheric and process-resolved blocks `A01 ... A15`
- reproducible code in `clean_experiments/`
- consolidated experiment metadata in `research_programm_summary.csv`

### Main Results
- Synthetic local-cell test `T20`: stable quasi-linear Lambda_b ~ Pi_b relation in the inertial range with `R^2_binned = 0.727`
- ERA5 atmospheric local-cell test `A15`: positive inertial-range relation with `R^2_binned = 0.520` and `p = 0.008`
- Process-resolved branch `A05`: memory is required at the finest scale, and the stronger GKSL/CPTP branch confirms that the restoration is not a numerical artifact

## Repository Contents

Included:
- `README.md`
- `LICENSE`
- `.gitignore`
- `CITATION.cff`
- `main.pdf`
- `main_ru.pdf`
- `clean_experiments/`
- `research_programm_summary.csv`

Main code and documentation:
- `clean_experiments/EXPERIMENT_NUMBERING.md`: canonical numbering and experiment history
- `clean_experiments/results/**/*.md`: curated experiment reports
- `research_programm_summary.csv`: top-level program index
- `docs/REPOSITORY_CATALOG.md`: branch audit and repository consolidation plan

## Manuscript Artifact

The current manuscript PDFs are available directly in this repository and are also archived in the Zenodo manuscript record:

- [`main.pdf`](./main.pdf)
- [`main_ru.pdf`](./main_ru.pdf)
- [Zenodo manuscript record](https://zenodo.org/records/19565805)

Local editable manuscript sources are kept in `manuscript/` as a local-only workspace and are not part of the tracked repository scope.
Local runtime/build parking lives in `legacy/`, which is also local-only and ignored by Git.

## Citation and Versioning

For citation metadata, use [`CITATION.cff`](./CITATION.cff).

- Use the Zenodo manuscript record when you want a stable public reference to the paper text.
- Use the Zenodo repository record when you want a stable public reference to the archived code snapshot.
- Use the GitHub repository when you want the current working tree and ongoing repository history.

## Environment

Recommended:
- Python `>=3.9`

Minimal setup:

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

Use `--help` for script-specific CLI options.

## License

This repository is distributed under the MIT License. See [`LICENSE`](./LICENSE).
