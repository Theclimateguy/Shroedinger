# Shroedinger — gen4

Falsification-grade research program on cross-scale organization of the
atmospheric circulation (ERA5 / MERRA-2). This branch supersedes gen1-gen3 and **retracts the central empirical claim of the v1 manuscript**
(the A15 "Lambda_b ~ Pi_b closure"): see `docs/RECONCILIATION.md`.

## What stands (preregistered, held-out, audited)

- **Cross-scale envelope-coupling profile P** — a regional invariant of the
  850 hPa circulation: a season-, year- and ENSO-stable signature (p = 0.001
  on held-out samples), beyond the semi-fractal level features and beyond
  symmetric cross-scale statistics. Its "beyond full power spectrum"
  component is conditional after the geometry control (p = 0.10).
- **The observing-network confound is closed** (Phase 13). The regional
  geography of P replicates in a free-running CMIP6 HighResMIP integration
  that assimilates no atmospheric observations (rho = 0.71 against an
  ERA5-to-ERA5 across-period ceiling of 0.93). The property is atmospheric,
  not an artefact of the observing system.
- **P measures what it is said to measure** (Phase 12). It agrees with an
  independent rank mutual-information measure of addressability (rho = -0.93)
  while being nearly independent of band amplitude (+0.20) and of temporal
  persistence (+0.02).
- A complete catalogue of negative results (flux closure, universal laws,
  local sources, level invariance, moisture-budget information content,
  and the three applied predictions of Phases 9-11).

## What has been retracted

- The **v1 manuscript's central claim** (A15 "Lambda_b ~ Pi_b closure"):
  see `docs/RECONCILIATION.md`.
- The **interpretation of the transfer-asymmetry index A** (Phase 12). The
  index is a reproducible regional statistic — replicated in MERRA-2
  (rho = 0.93) and in the free-running model (rho = 0.90) — but it does not
  measure the irreversibility of generalisation. On synthetic fields where
  irrecoverability is known and swept it does not respond in either the
  spatial or the modal reading, while it tracks temporal persistence
  (Spearman 1.0 synthetic, +0.53 on real domains). A is a persistence
  statistic, and the cause is methodological: the transfer operators are
  fitted over a 20-step sliding window.
- The **Sect. 6.2 predictability hypothesis** (Phase 9) and its level-based
  reformulation (Phase 10). In all three applied tests, including the
  mesoscale deficit of AI weather models (Phase 11), plain band variance was
  a stronger predictor than the descriptors.

## Repository map

- `docs/RESEARCH_PROGRAM_FULL.csv` — all 73 experiments (gen1-2 + gen3) with
  final reconciliation statuses.
- `docs/PROTOCOL_PHASE1..13_*.md` — frozen preregistered protocols with
  deviation logs.
- `docs/RECONCILIATION.md` — authoritative final scoreboard, audit findings,
  retractions. `docs/PROGRAM_MAP.md` — narrative program map.
- `clean_experiments/experiment_B*.py` — gen3 experiments (B1-B13);
  `download_b*.py` — data downloaders (CDS API, NASA Earthdata, NOAA GEFS on
  AWS Open Data, WeatherBench 2, ESGF).
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

## Branches

- **`gen4`** — current line of work (this branch).
- `gen3` — **frozen** at release v4.0; kept for the record. It carries a
  parallel, squashed import of the B1-B8 programme and does not contain
  `manuscript_v2/` or Phases 9-13.
- `gen2`, `gen1` — superseded.

## Provenance

- Current manuscript: `manuscript_v2/article_v3.tex`.
- v1 manuscript (superseded; central claim retracted by Phase B1):
  [Zenodo 19565805](https://zenodo.org/records/19565805) — `main.pdf`,
  `main_ru.pdf`, `manuscript/` are kept for the record.
- Release v4.0 (gen3): [Zenodo 21940262](https://zenodo.org/records/21940262).
- Citation metadata: `CITATION.cff`; Zenodo metadata: `.zenodo.json`.
