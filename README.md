# Shroedinger — gen3

Falsification-grade research program on cross-scale organization of the
atmospheric circulation (ERA5 / MERRA-2). This branch supersedes the gen1/gen2
program and **retracts the central empirical claim of the v1 manuscript**
(the A15 "Lambda_b ~ Pi_b closure"): see `docs/RECONCILIATION.md`.

## What stands (preregistered, held-out, audited)

- **Transfer-asymmetry index ||F||** — a regional invariant of the 850 hPa
  circulation (beyond spectrum, beyond envelope coupling, robust to geometry
  controls), **replicated across reanalyses** (ERA5 vs MERRA-2, regional
  geography Spearman rho = 0.93).
- **Cross-scale envelope-coupling profile P** — a season-stable regional
  signature (its "beyond-spectrum" component is conditional after geometry
  control).
- A complete catalogue of negative results (flux closure, universal laws,
  local sources, level invariance, moisture-budget information content).

## Repository map

- `docs/RESEARCH_PROGRAM_FULL.csv` — all 73 experiments (gen1-2 + gen3) with
  final reconciliation statuses.
- `docs/PROTOCOL_PHASE1..8_*.md` — frozen preregistered protocols with
  deviation logs.
- `docs/RECONCILIATION.md` — authoritative final scoreboard, audit findings,
  retractions. `docs/PROGRAM_MAP.md` — narrative program map.
- `clean_experiments/experiment_B*.py` — gen3 experiments (B1-B8);
  `download_b*.py` — data downloaders (CDS API / NASA Earthdata).
- `clean_experiments/results/` — per-region-window JSON results, reports,
  figures (`results/figures/`).
- `run_phase67_pipeline.sh` — resumable download+experiment pipeline.
- Legacy gen1-2 scripts (T/A series) are retained for provenance;
  their statuses are in the CSV.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib xarray netCDF4 h5netcdf earthaccess cdsapi
```

ERA5 requires `~/.cdsapirc` (CDS API); MERRA-2 requires NASA Earthdata
credentials. All experiments are resumable via per-region-window caches.

## Provenance

- The earlier manuscript of this project is preserved for the record under
  `legacy_manuscript_v1/` (see the note there) and archived separately as
  [Zenodo 19565805](https://zenodo.org/records/19565805). Its central
  empirical claim was re-tested here and not confirmed.
- Citation metadata: `CITATION.cff`; Zenodo record metadata: `.zenodo.json`.
