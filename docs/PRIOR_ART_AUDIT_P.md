# PRIOR-ART AUDIT OF P (v0.1, 2026-08-19)

Question audited: is P a new object, or a re-labelling of an existing
statistic? Asked now because three papers are in preparation; the
attractor-reducibility programme (PROTOCOL_PHASE24_REDUCIBILITY_DRAFT)
is DEFERRED to a later stage and is not needed to answer this.

## 1. What P is, as a formula

Rank (Spearman) dependence between the spatial envelopes of adjacent
scale bands of one field (850 hPa relative vorticity), bands
50-100-200-400 km, median over time, minus the median of 99 phase
surrogates, mapped per tile.

Three separable ingredients: (a) envelope-envelope dependence between
scale bands; (b) rank/copula rather than linear; (c) phase-surrogate
anchoring; and then (d) the geography of the result.

## 2. Nearest prior art

| Field | Statistic | Relation to P | Verdict |
|---|---|---|---|
| Wall turbulence | Amplitude-modulation coefficient R (Mathis, Hutchins & Marusic, JFM 2009): correlation of the large-scale signal with the filtered envelope of small scales | Same construction, one side signal instead of envelope; 1-D time series at a point; no anchoring | **Form is prior art** |
| Wall turbulence, recent | Critique of R as partly a spectral/phase quantity, transport-based alternative proposed (arXiv:2408.15944, JFM) | Identifies the exact defect that our phase-surrogate anchoring removes | Supports our anchoring, and must be cited |
| Neurophysiology | Amplitude-amplitude coupling / amplitude envelope correlation (cross-frequency coupling family; Bruns et al. 2000 onward) | Envelope-envelope correlation between frequency bands of one signal - the closest formal twin | **Form is prior art** |
| Geophysics | Wavelet cross-correlation, multiscale coupling (Casagrande et al., JGR 2015) | Scale-by-scale correlation between TWO variables, not between bands of one field | Different object |
| Statistics / ML | Wavelet scattering spectra, scale dependencies (Morel & Mallat, arXiv:2204.10177) | The general framework for non-Gaussian scale dependence | Already benchmarked against: Phase 3 verdict NOVEL (P not reproduced by scattering moments) |
| Econophysics | Multifractal detrended cross-correlation rho_q(n) | Scale-dependent cross-correlation between two series | Different object |
| Neuroimaging, 2026 | Spatial Neighboring Scattering Transform, "cross-channel amplitude coupling" (arXiv:2607.08855) | Independent reinvention of the same family in another domain | Evidence the FORM is generic |

## 3. Verdict

- **The estimator's form is NOT new.** Envelope-envelope band coupling
  has been invented at least twice (wall turbulence, neurophysiology).
  Any claim of a new statistic will be refuted by a referee from either
  field. This must be conceded in print, with citations, in all three
  papers.
- **No prior art was found for the object.** Specifically not found:
  the 2-D spatial band-envelope dependence of a geophysical field
  computed tile-locally, surrogate-anchored to its beyond-spectrum
  component, mapped globally, and shown to be (i) season/year/ENSO
  stable, (ii) instrument-independent (MERRA-2, free-running model),
  (iii) reproduced by a free-running model at 97% of the resolution
  ceiling, (iv) attributable to two named coordinates, (v) without
  accessible dynamics of its own.
- Therefore: **claim novelty at the level of the object and its
  geography, never at the level of the statistic.**

Draft sentence for Paper 3: "The envelope coupling between adjacent
scale bands is a known construction - the amplitude-modulation
coefficient of wall turbulence and the amplitude-amplitude coupling of
cross-frequency analysis are its closest relatives. What is new here is
neither the estimator nor its name, but the demonstration that its
surrogate-anchored, beyond-spectrum component is a stable regional
invariant of the atmospheric circulation, mappable globally,
reproducible in a model that assimilates no observations, and
attributable to convective and storm-track activity."

## 4. Recommended audit test (AUDIT-1), before the papers go out

Cheap head-to-head on data already on disk (data/b20global, 936 tiles
x 2 seasons). Compute per tile, alongside anchored P:

1. R_AM - the Mathis-style coefficient: correlation between the
   coarse-band signal and the envelope of the adjacent finer band.
2. P_lin - the same envelope-envelope dependence with Pearson instead
   of Spearman (isolates the copula ingredient).
3. Anchored versions of both, same 99-surrogate scheme.

Scored questions (bars set before computing):

- A1a: rho(P, R_AM_anchored) across tiles. If >= 0.9, P IS the
  amplitude-modulation coefficient in disguise; the estimator claim is
  withdrawn and the papers cite Mathis et al. as the source of the
  statistic.
- A1b: does the Phase-20 attribution (LOSO R^2, loadings) reproduce
  with R_AM_anchored as target? If yes, the geography is a property of
  band-envelope coupling generally, not of this particular estimator -
  which STRENGTHENS the physical claim while weakening the
  methodological one. This outcome is good for the programme and must
  be reported as such.
- A1c: rho(P, P_lin_anchored). If >= 0.95, drop the copula language
  from the papers; it is decoration.

Cost: hours, no downloads, no new theory. Outcome enters the papers as
a methods paragraph either way.

## 5. Deferred

The reducibility of P to invariant-measure observables (local
dimension, extremal index, transfer-operator spectrum, entropy rate) -
PROTOCOL_PHASE24_REDUCIBILITY_DRAFT.md - is DEFERRED. Scoping numbers
in that draft remain valid: the pilot found P vs local dimension
rho = +0.48 (48 region-windows), so the question is real, but it is a
doctoral-scale programme, not a prerequisite for the three papers.

## 6. AUDIT-1 outcome (2026-08-19, computed under the frozen protocol)

`clean_experiments/results/experiment_A1_prior_art/` — full report in
`report.md`, machine record in `summary.json`.

- C-A1-0 PASS (rho = 1.000 vs the committed Arm-B map, max raw
  difference 3.0e-8).
- **A1a: ESTIMATOR_DISTINCT.** Spearman(P, R_AM_cyc) = +0.143 (bar for
  identity 0.90, for relatedness 0.60). P is NOT the
  amplitude-modulation coefficient. The wall-turbulence prior art is
  cited as related work; nothing is withdrawn on its account.
- **A1c: the copula ingredient is decoration.** Spearman(P, P_lin)
  = +0.992 -> per the frozen bar, rank/copula language is dropped from
  all three papers. The true formal relative of P is the
  amplitude-amplitude coupling (amplitude envelope correlation) of
  cross-frequency analysis, and that is the citation to make.
- **A1b:** P LOSO R^2 = 0.536, P_lin 0.546 with identical loadings;
  R_AM_cyc 0.062 (p = 0.049), R_AM_raw 0.025 (p = 0.81). The
  amplitude-modulation form reaches 95% of its own reliability ceiling
  but has almost no reproducible tile-level signal (cross-season
  reliability 0.26 vs P's 0.77). The reproducible geography is a
  property of the envelope-envelope form specifically.

Net: the novelty verdict of Sect. 3 stands, with one correction — the
nearest prior art is the cross-frequency amplitude-amplitude coupling,
not the turbulence amplitude-modulation coefficient, and the rank
construction must no longer be presented as an ingredient.

## 7. AUDIT-2 outcome (2026-08-19, frozen protocol
`docs/PROTOCOL_AUDIT2_INTERMITTENCY_HEADTOHEAD.md`)

Second prior-art claimant tested: the intermittent-cascade literature
(KO62 log-normal cascade, multifractal formalism). Results in
`clean_experiments/results/experiment_A2_intermittency/`.

- C-A2-0 PASS (rho = 1.000, max raw difference 3.0e-8).
- **A2a: CASCADE_DISTINCT.** Spearman(P, P_cascade) = -0.417 against a
  0.90 identity bar. Decisive diagnostic: the cascade prediction is a
  spectrum-level quantity — real 0.666 vs phase-surrogate 0.650, i.e.
  anchored 0.010 — while P is real 0.770 vs surrogate 0.512, anchored
  0.256. The observed envelope coupling EXCEEDS the log-normal cascade
  prediction, and the excess is the programme's observable.
- **A2b (reported): the intermittency parameter is a substantial
  relative.** rho(P, INT_sig2) = +0.692 (log-envelope variance);
  rho(P, INT_flat) = -0.223; rho(P, INT_mu) = +0.100. This must be
  conceded and cited in the papers.
- A2c: the intermittency family loads like the Phase-20 SPECTRAL SLOPE
  target (orography/land), P loads like storm tracks + convection —
  different physics, reported side by side.
- **A2d: superseded by AUDIT-2b — the number of record is +0.21.** The
  scored increment was +0.225 (p = 0.001) but both it and its bar were
  contaminated (shared sampling noise; a cross-season ceiling). See
  Sect. 8.

Net after two audits: the estimator is neither the amplitude-modulation
coefficient (AUDIT-1) nor a cascade/intermittency statistic (AUDIT-2);
its rank construction is decoration (AUDIT-1); its nearest live relative
is the local log-envelope variance (rho = +0.69), which is cited, and
which does not reproduce either its geography or its attribution.

## 8. AUDIT-2b outcome (2026-08-19, frozen protocol
`docs/PROTOCOL_AUDIT2B_SPLITHALF_DECONTAMINATION.md`)

Split-half decontamination: P and the comparators computed on disjoint
time-parity halves of the same tile-season. Results in
`clean_experiments/results/experiment_A2b_splithalf/report_halves.md`.

- **B1 — the ceiling was wrong, and it was wrong in the direction that
  flattered the programme's competitors.** Within-season split-half
  reliability of the anchored P map is 0.916, i.e. R^2 = **0.914** after
  the Spearman-Brown correction. The 0.603 used before was the
  cross-season figure, which charges real seasonal change to noise.
  Restated: Phase-20's attribution explains **59%** of the reproducible
  variance of the map, not 89%; covariates plus intermittency reach 83%.
  About 40% of the map's reproducible variance is still unexplained.
- **B2 (scored): INTERMITTENCY_ADDS, +0.207** (four combinations,
  +0.203 to +0.215, all p = 0.001).
- **B3:** the same-half increment is +0.261, so the shared-noise share
  is 21%, not the ~70% the quick cross-season control suggested; that
  control was biased low because the intermittency statistics are
  themselves season-dependent.
- **B4:** every earlier verdict survives a noise-immune recomputation —
  P vs R_AM_cyc +0.145, P vs P_cascade -0.287, P vs P_lin +0.893
  (~0.98 disattenuated), P vs INT_sig2 +0.600 (~0.65 disattenuated).
- **Correction to AUDIT-1:** R_AM_cyc has within-season reliability 0.90
  — it is not "nearly signal-free", it is season-dependent. P's map is
  season-stable; the amplitude-modulation map is not. That is the
  sharper claim and the one the papers carry.
