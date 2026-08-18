# Experiment B22: theory-candidate tests, round 2

Protocol: `docs/PROTOCOL_PHASE22_QT_ROUND2.md` (frozen 2026-08-18; the
beta_T calibration amendment logged pre-computation). No new downloads;
carriers: B13 ECMWF-IFS-HR free-running globals (JFM+JAS 2014, 0.5 deg),
Phase-20 ERA5 globals (2023), data/b17daily, data/b21 monthly CAPE.

## PHASE22_VERDICT: (P2_SUPPORTED, P1_MIXED)

### Arm P2 — the model draws the same map: SUPPORTED

902 scored tiles, resolution-matched at 0.5 deg:

| Comparison | Spearman rho |
|---|---|
| ceiling: ERA5@0.25 vs ERA5@0.5 (same atmosphere, machinery cost) | 0.897 |
| **model (free-running, 2014) vs ERA5@0.5 (2023)** | **0.869** |
| model vs ERA5@0.25 | 0.810 |

The free-running integration — different year, no assimilated
observations — reproduces the global scale-coupling texture at 97% of
the resolution ceiling, far above both frozen bars (abs 0.5;
0.6 x ceiling = 0.54). Model-map loadings mirror ERA5: eke +0.68,
abs_lat +0.70, sst_grad +0.51, cape -0.50, orography ~0. QT-P2 is the
candidate's strongest confirmation to date: the map is a property of
the physics given tier-1 boundary conditions, on both the box carrier
(B13) and now tile-by-tile globally.

### Arm P1s — the drift exists; its magnitude is not the frozen calibration: MIXED

Seasonal carrier (288 tile-season-types, >= 30 usable years each):

- H22-P1a: D_obs = **-8.5e-5 / yr**, p = 0.025 — negative,
  significant, and the cluster-bootstrap CI **[-1.53e-4, -1.15e-5]
  excludes zero**: a declining spectrum-fixed P trend in the
  top-CAPE-trend tiles is now an observed fact (first detection; the
  annual carrier of Phase 21 saw the same sign at p = 0.081).
- H22-P1b: the CI excludes BOTH predictions — beta_CS-based
  (-1.14e-5/yr; misses the CI upper edge marginally) and beta_T-based
  (-0.29e-5/yr) -> by the frozen ladder: P1_MIXED (sign and existence
  supported; magnitude unexplained by the linear CAPE calibration).
- Calibration answer to the author's question: beta_T / beta_CS =
  **0.25** — the interannual (detrended) response is FOUR TIMES WEAKER
  than the cross-sectional coefficient, while the multidecadal drift is
  ~7x the cross-sectional and ~30x the interannual prediction. The
  response is timescale-dependent: slow theta movement carries the
  texture further than either fast covariation or spatial contrast
  implies. Space-for-time under-, not over-, states the decadal
  response — the opposite arm of the author's calibration concern, now
  quantified.

### Mandatory caveat (trend claims in reanalysis)

The 1979-2024 record spans major observing-system changes. B13 closed
the observing-network confound for the GEOGRAPHY of P, not for its
TRENDS; a spurious reanalysis drift component cannot be excluded by any
test in this phase (spectrum-fixing removes variance-level artifacts
only). The drift fact should be treated as ERA5-internal until a
free-running historical integration (model with transient forcing) is
scored against it — the natural QT-P1 completion and a named candidate
for the next protocol.

### Candidate bookkeeping

- QT-P2 -> SUPPORTED (new E13).
- QT-P1 -> status DRIFT_DETECTED_MAGNITUDE_OPEN: sign leg supported at
  p=0.025 with CI excluding zero; magnitude leg open (linear CAPE
  calibration insufficient; timescale-dependence quantified,
  beta_T/beta_CS = 0.25). The prediction's falsification condition was
  not met and its support condition was half-met; the candidate is
  updated to v1.3 with this exact wording.

## Figure

- `fig1_round2.png` — model-vs-ERA5 map scatter; drift CI vs both
  predictions.
