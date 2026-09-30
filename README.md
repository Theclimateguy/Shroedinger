# Shroedinger — gen4

Falsification-grade research program on cross-scale organization of the
atmospheric circulation (ERA5 / MERRA-2). This branch supersedes gen1-gen3 and **retracts the central empirical claim of the v1 manuscript**
(the A15 "Lambda_b ~ Pi_b closure"): see `docs/RECONCILIATION.md`.

## Status (2026-09-30): an estimator artefact and what replaced it

An exploratory audit of the physics of P
(`docs/EXPLORATION_PHYSICS_2026-09-29.md`) found that the latitude trend of
the anchored tile map is produced by the estimator, not by the atmosphere.
The zonal step of the lat-lon grid shrinks poleward; together with edge
leakage in the FFT phase surrogates this lowers the surrogate level of the
<50|50-100 km band pair with latitude. Three independent controls agree:

- isotropic regridding of each tile: the ocean midlatitude-minus-tropics
  contrast drops 0.086 -> 0.018;
- artificial zonal oversampling of tropical tiles: P rises 0.236 -> 0.320
  with the field unchanged;
- mirror-extension surrogates remove the trend on the native grid.

The sub-50 km band lies below the ERA5 spectral truncation (62.6 km).

**What this does to the sections below.**

- *Affected — 700-km tile maps (papers 2 and 3).* Zonal share of the map
  variance 0.51 -> 0.05; covariate attribution LOSO R^2 0.54 -> 0.11
  (synoptic activity +0.66 -> +0.14, CAPE -0.52 -> -0.06); cross-season
  reliability 0.71 -> 0.33-0.41; model agreement "97% of the ceiling" ->
  non-zonal part 0.62 against a 0.82 ceiling. Theory-candidate evidence
  rows E8, E12, E13 rest on the affected map. Beyond-spectrum values on
  tiles also depend on how the surrogate is built (FFT vs mirror: rank
  agreement 0.54); a boundary-free null is not implemented yet.
- *Little affected — 2000 x 2800 km boxes (paper 1).* Only the first band
  pair carries the artefact; pairs 2-5 are unchanged (region-mean rank
  agreement 0.97-0.99), a region is identified from its profile without the
  first pair at 0.21 against 0.08 by chance on both grids, and Congo stays
  above Amazon in every pair. Not upheld: "storm tracks lead on the fine
  bands", and the claimed control of the grid geometry (it controlled the
  size of the regions, not the grid step).
- *Not affected.* The temporal results (no dynamics of its own, no ENSO
  modulation) and the form of the structure law. A synthetic test shows,
  however, that any common amplitude modulator reproduces that law
  (R^2 0.99 vs 0.2), so the law is not evidence of a cascade.

**Phase 31 — the boundary template** (frozen protocol, independent year
2024, no surrogates; `docs/PROTOCOL_PHASE31_BOUNDARY_TEMPLATE.md`, report in
`clean_experiments/results/experiment_B31_boundary_template/`). The spatial
covariance of band log-envelopes splits exactly into a stationary template
and a moving part. Verdict PARTIAL(H31-1, H31-2):

- the template share is a territorial invariant: 2023 vs 2024 Spearman 0.97
  over land, 0.84 over all tiles; ocean 0.008, flat land below 0.03, up to
  0.75 at mountainous coasts and islands. Exploratory: seasons 0.80,
  vorticity vs temperature 0.90, a free-running model of another year 0.90;
- compensation: total coupling over land does not depend on the template
  share (0.363 / 0.367 / 0.371) while the moving part falls
  (0.362 / 0.356 / 0.307);
- one template for all levels (0.91, 0.86 adjacent; 0.65 non-adjacent), and
  it is not the below-ground extrapolation (5.9 x the ocean value on land
  tiles with no below-ground point);
- failed: "no geography without a boundary" — over open ocean the coupling
  map repeats between years within a season (0.31-0.39).

Papers 2 and 3 as released (v6.3, v6.4) predate this audit.

## Results of Phases 1-30 (read with the status note above)

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
  (rho = +0.92 at box scale vs +0.69 at tile scale). AUDIT-4
  characterises what is left: after the covariates (59% of the
  reproducible variance) and the intermittency block (83%), the
  remainder is itself reproducible (split-half r_full = 0.64), not
  spectral, and geographically organised on monsoon margins, orography
  flanks and deep-convective cores. Its drivers are named future work;
  the analysis programme for the papers closes there.

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

- `docs/RESEARCH_PROGRAM_FULL.csv` — all 73 experiments of gen1-gen3 with
  final reconciliation statuses (the gen1-2 scripts themselves were removed
  from this branch on 2026-09-30; they remain in branches `gen1`-`gen3` and
  in tags up to `v6.4`).
- `docs/PROTOCOL_PHASE*.md`, `docs/PROTOCOL_AUDIT*.md` — frozen
  preregistered protocols with deviation logs (Phases 1-31, AUDIT-1..4).
- `docs/EXPLORATION_PHYSICS_2026-09-29.md` — exploratory note (not a
  protocol): hypotheses registered before computation, the grid-anisotropy
  artefact, the boundary-template idea.
- `docs/RECONCILIATION.md` — authoritative scoreboard, audit findings,
  retractions. `docs/PROGRAM_MAP.md` — narrative program map.
- `docs/THEORY_CANDIDATE_QUENCHED_TEXTURE.md` — theory candidate v1.4 with a
  status addendum (seed note: `docs/THEORY_NOTE_QUENCHED_TEXTURE.md`).
- `clean_experiments/experiment_B*.py`, `experiment_A*.py` — experiments of
  the programme (B1-B31, AUDIT-1..4); `explore_physics_*.py` — exploratory
  scripts of 2026-09-29/30; `download_*.py` — data downloaders (CDS API,
  NASA Earthdata, NOAA GEFS on AWS Open Data, WeatherBench 2, ESGF).
- `clean_experiments/results/` — JSON results, reports, figures.
- `manuscript_v2/` — current manuscripts of the three papers.
- `reproducibility/` — per-paper reproduction routes.
- `run_phase67_pipeline.sh` — resumable download+experiment pipeline for
  Phases 6-7.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib xarray netCDF4 h5netcdf earthaccess cdsapi
```

ERA5 requires `~/.cdsapirc` (CDS API); MERRA-2 requires NASA Earthdata
credentials. All experiments are resumable via per-region-window caches.

**Per-paper reproduction.** Each paper of the series has a dedicated route
(data -> computations -> artifact check) that names the exact 8-12 files
behind its tables and figures, so a reviewer never has to guess which parts
of this repository a given paper uses:

- Paper 1 (regional invariant): [`reproducibility/article1_README.md`](reproducibility/article1_README.md)
  — one-command orchestrator `reproducibility/reproduce_article1.py`
  (`--stage download|compute|check`), pinned environment
  `reproducibility/requirements_article1.txt`, SHA-256 of every expected
  artifact, and a smoke test that verifies the computational chain on a
  single region-window without bulk downloads.
- Paper 2 (planetary map of coupling, its controls and a regionalisation):
  [`reproducibility/article2_README.md`](reproducibility/article2_README.md)
  — **single entry point** `reproducibility/reproduce_article2.py`
  (`--stage download|compute|figures|manuscript|check|all`). One script goes
  from the primary fields to the manuscript files: 9 resumable downloaders;
  20 computation steps in dependency order (Phase-20 arms A/B, AUDIT-2/2b/4,
  Phases 17, 21-23, 25, descriptive readouts); the descriptive layer of the
  paper (zonal profiles, typology and regionalisation map, map of the
  unexplained part, per-cell table and a gazetteer that ties every place
  name in the text to a cell id); Russian-label 300-dpi journal figures; and
  the journal-format manuscript build (author-year citations, GOST list +
  References, typography checks, character count; author rules of the target
  journal are kept in `docs/journal/`). `--stage check` verifies 16 artifacts
  against the SHA-256 list; `check` and `figures` need no downloads because
  all derived tables are committed. Shared pinned environment:
  `reproducibility/requirements.txt`.
- Paper 3 (the law of inheritance between scale levels):
  [`reproducibility/article3_README.md`](reproducibility/article3_README.md)
  — single entry point `reproducibility/reproduce_article3.py`
  (`--stage download|compute|figures|manuscript|check|all`): 4 downloaders,
  9 computation steps (Phases 27-30: all-pair envelope statistics on 48
  regional units, 902 global tiles and the free-running IFS-HR model; law
  form and depth; tile replicate and amplitude vs intermittency;
  unification P <-> law), Russian-label 300-dpi figures 1-5, the
  journal-format manuscript build, and a check of 6 artifacts against
  SHA-256. Uses three committed artifacts of papers 1-2 as inputs.

Raw ERA5 / MERRA-2 / HighResMIP fields are deliberately **not** stored in
the repository (large externally licensed products); the exact download
scripts and all derived tables behind the papers' statistics are.

## Branches

- **`gen4`** — current line of work (this branch).
- `gen3` — **frozen** at release v4.0; kept for the record. It carries a
  parallel, squashed import of the B1-B8 programme and does not contain
  `manuscript_v2/` or Phases 9-13.
- `gen2`, `gen1` — superseded.

## Provenance

- Manuscripts: `manuscript_v2/article1/` (paper 1, submitted to Izvestiya
  RAN Ser. Geogr., source `article1_v7_src.tex`), `manuscript_v2/article2/`
  (paper 2, `article2_v4_src.tex`), `manuscript_v2/article3/` (paper 3,
  `article3_v2_src.tex`); shared bibliography
  `manuscript_v2/references_v2.bib`. See `manuscript_v2/README.md`.
- v1 manuscript (superseded; central claim retracted by Phase B1):
  [Zenodo 19565805](https://zenodo.org/records/19565805). Its files
  (`main.pdf`, `main_ru.pdf`, `manuscript/`), earlier drafts of the papers
  and the gen1-2 experiment layer were removed from this branch on
  2026-09-30; they remain in the git history, in tags up to `v6.4` and in
  branches `gen1`-`gen3`.
- Release v4.0 (gen3): [Zenodo 21940262](https://zenodo.org/records/21940262).
- Citation metadata: `CITATION.cff`; Zenodo metadata: `.zenodo.json`.
