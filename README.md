# Shroedinger — gen4

Falsification-grade research program on cross-scale organization of the
atmospheric circulation (ERA5 / MERRA-2). This branch supersedes gen1-gen3 and **retracts the central empirical claim of the v1 manuscript**
(the A15 "Lambda_b ~ Pi_b closure"): see `docs/RECONCILIATION.md`.

## What stands (preregistered, held-out, audited)

- **Cross-scale envelope-coupling profile P** — a regional invariant of the
  850 hPa circulation: a season-, year- and ENSO-stable signature (p = 0.001
  on held-out samples), beyond the semi-fractal level features and beyond
  symmetric cross-scale statistics. Its beyond-spectrum component is
  **physical, not cartographic**: the equal-km design control (Phase 18)
  removed the box-geometry confound by construction and the beyond-spectrum
  residual clustering survived (p = 0.033 anchored / 0.007 raw), lifting
  the former conditional verdict.
- **The estimator is not a re-labelling of the nearest prior art**
  (AUDIT-1, `docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md`). On the same
  902 tiles, anchored P vs the Mathis-Hutchins-Marusic
  amplitude-modulation coefficient: rho = +0.14 (ESTIMATOR_DISTINCT);
  the AM form carries almost no reproducible tile-level signal
  (cross-season reliability 0.26 vs P's 0.77) and does not reproduce the
  map (LOSO R^2 = 0.06). But P vs its Pearson twin: rho = +0.99 — the
  rank/copula construction is decoration and is dropped from the papers;
  the nearest prior art for the FORM is the amplitude-amplitude coupling
  of cross-frequency analysis, cited as such. The attribution is
  invariant to the estimator convention (R^2 0.536 vs 0.546, identical
  loadings). AUDIT-2 (`docs/PROTOCOL_AUDIT2_INTERMITTENCY_HEADTOHEAD.md`)
  retires the second claimant: P vs the Kolmogorov-Obukhov cascade
  prediction rho = -0.42 (CASCADE_DISTINCT) — the cascade prediction has
  no beyond-spectrum component at all (anchored 0.010 vs P's 0.256) —
  while the local intermittency parameter is a substantial relative
  (rho = +0.69) that adds +0.21 LOSO R^2 (AUDIT-2b, split-half
  decontaminated). AUDIT-2b also corrects the map's reliability ceiling:
  within-season split-half gives R^2 = 0.914, so the Phase-20
  attribution explains 59% of the reproducible variance, not 89% — the
  global map is NOT nearly saturated. AUDIT-3
  (`docs/PROTOCOL_AUDIT3_SCALE_YEAR_ROBUSTNESS.md`, 12 equal-km boxes x
  2 windows, bands to 1600 km) reproduces both verdicts off the tile
  grid (AM +0.39, cascade +0.04, P_lin +0.98) and carries one open item:
  the tie to the intermittency parameter strengthens with scale
  (rho = +0.92 at box scale vs +0.69 at tile scale).

- **P has no accessible dynamics of its own** (Phases 14, 15, 19). Fast
  (~10 h), memoryless relaxation to a static regional value; no regime
  coupling on either grid (the R7/R5 sign split reproduces on equal-km
  territory); no scale hierarchy of fluctuation timescales (tau flat
  9-11 h over 71-1131 km); the derivative-estimator class is invalid by
  identity. No ENSO modulation at 30x power over 1979-2024, on both
  carriers (Phases 16, 17, 19C). One open anomaly: a tropical interannual
  variance excess (F = 1.5-1.8 in R1/R3/R5), carrier-robust, non-ENSO.
- **P's geography is attributable** (Phase 20, Arm A). As a 144-tile map,
  out-of-region R^2 = 0.40 (p = 0.001): convective regime lowers P beyond
  the spectrum (10/12 regions within-region), storm-track activity raises
  it through the spectrum (11/12); orography acts on the spectral slope,
  not on P. P and synoptic activity covary at lag 0 with no lead either
  way (co-emergence, Phase 19C).
- **The global map** (Phase 20, Arm B). 936 tiles, 60S-60N: maxima on the
  Southern Ocean ring and the NH storm tracks, minima on the deep
  convective cores (Amazon, Congo, Maritime Continent); season-stable,
  consistent with the box-level results (rho = 0.84), attributed at
  LOSO R^2 = 0.54 (p = 0.001, non-zonal component certified), and
  effectively one-dimensional (synoptic activity alone = 85% of skill).
- **Theory candidate** (v1.4): `docs/THEORY_CANDIDATE_QUENCHED_TEXTURE.md` —
  quenched-texture organization (P as a functional of the local invariant
  measure over a quenched constraint field), postulates QT-1..5, evidence
  map E1-E15, predictions QT-P1..P6. Score after three rounds (Phases
  21-23): **P2 SUPPORTED (free-running model reproduces the global map at
  97% of the resolution ceiling), P3 PASS (k80 = 1), P4 SUPPORTED (the
  tropical excess is coherent with the IPO), P5 SUPPORTED (sampling
  clock)**; P1 open — the ERA5 drift is detected (p = 0.025, CI excludes
  zero) but one transient free-running member cannot confirm it
  (right sign, ~27% magnitude, null-consistent), so the reanalysis-trend
  caveat stands; **P6 (stratigraphic/early-warning form) FAILED** — P is
  an architecture registrar, not a sensitive reorganization indicator
  (trend SNR ranks last, behind CAPE by 3.6x).
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
- `docs/PROTOCOL_PHASE1..23_*.md` — frozen preregistered protocols with
  deviation logs. `docs/PRIOR_ART_AUDIT_P.md` +
  `docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md` — the estimator's
  prior-art audit and its frozen head-to-head test.
- `docs/RECONCILIATION.md` — authoritative final scoreboard, audit findings,
  retractions. `docs/PROGRAM_MAP.md` — narrative program map (through
  Phase 23). `research_programm_summary.csv` — canonical experiment table
  (TOY_MODEL / ATMOSPHERE_* / CLEAN_PHASES blocks).
- `docs/THEORY_CANDIDATE_QUENCHED_TEXTURE.md` — theory candidate v1.4
  (seed note kept as `docs/THEORY_NOTE_QUENCHED_TEXTURE.md`).
- `clean_experiments/experiment_B*.py` — gen3/gen4 experiments (B1-B23);
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
