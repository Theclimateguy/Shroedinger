# Program reconciliation (2026-08-13)

Final audit-informed reconciliation of the B1-B7 program, incorporating the
independent red-team audit, the literature positioning study, and the
post-audit remediation computations (5c retest, B7, N2).

## Final scoreboard

| Phase | Claim | Final status |
|---|---|---|
| B1 | Lambda_b ~ Pi_b flux closure | FALSIFIED (stands) |
| B2/B2b | envelope-coupling profile P = regional signature | CONFIRMED, with reinterpretation (see below) |
| B3 | P beyond scattering statistics | CONFIRMED but marginal (p=0.026, exchangeability caveat); anchored-excess version is the defensible form |
| B4 | transfer-asymmetry index ||F|| = regional signature beyond spectrum and P | CONFIRMED (p=0.001, survives coarse-bands-only control) |
| B4 H4c | curvature physically predicted by CAPE/land/orography | DEMOTED to descriptive ocean/land contrast (jackknife sign-flips; CAPE alone negative) |
| B5c | connection integrability | **RETRACTED** — temporal-spectrum-preserving null retest: 7/48 (was 48/48 vs a flawed null) |
| B5a/B5b/B5d | local source / universal law / level invariance | NEGATIVE (stand) |
| B6a | charge-discharge dynamics | NOT ESTABLISHED — discharge term sign-consistent (13/16) but ~half of relaxation is estimator-window memory; zero held-out predictive content (7/16) |
| B6b | rho_5 driver | NOT FOUND |
| N1/N4/N2 | normalization collapse | NO COLLAPSE: characteristic-scale rescaling removes ~23% of signature; zonal cleaning and moisture-transport (IVT) fields SHARPEN it (0.91 vs 0.73) |
| B7 | invariants inform moisture-budget residual | NEGATIVE (median increment -0.018, sign test p=0.92) |

## Where the task was misformulated

1. **Flux reading of Lambda** (first paper): the operational proxy was not a
   flux; the correlation was near-circular. Corrected by B1.
2. **"Irreversibility profile"**: the statistic is a symmetric cross-scale
   envelope correlation; nothing directional or entropic is measured.
   Correct name: cross-scale envelope-coupling profile (family: amplitude
   modulation coefficient, Mathis et al. 2009; scattering covariance).
3. **"Curvature / connection / gauge structure"**: ||F|| is a
   finite-sample commutator norm of ridge-regression transfer maps — a
   directional transfer-asymmetry index. Geometry language is declared
   metaphor (Phase-5 verdict VOCABULARY), and after the 5c retraction no
   result licenses more.
4. **Integrability test design** (own error): the spatial phase-surrogate
   null does not preserve temporal coherence; the test measured
   autocorrelation, not geometry.
5. **Charge-discharge as dynamics**: a regime-level anticorrelation with
   known convective-lifecycle timescales (Masunaga 2012: 1-2 day CAPE
   recovery) was over-read as a dynamical equation; the estimator's 5-day
   window contributes roughly half the apparent relaxation, and the model
   has no held-out predictive content.

## Where the data may be non-representative

1. **ERA5 effective resolution** (~4-7 dx ~ 125-220 km; Bolgiani et al.
   2022): octaves below 200 km sit partly in the reanalysis
   dissipation/parameterization range. Mitigation: both headline signatures
   survive with sub-200-km bands removed (P coarse-only diff +0.61 p=0.002;
   ||F|| bands 2-5 diff +0.87 p=0.001). The demoted H4c lived entirely in
   the unresolved bands — a further reason for its demotion.
2. **Observing-system geography confound (UNRESOLVED)**: regional
   invariants stable across years cannot be distinguished from stable
   regional signatures of assimilation density (Hersbach EDA spread;
   Bonavita-Laloyaux increments). Held-out years do not protect against a
   pipeline-stable artifact. REQUIRED pre-publication control:
   cross-reanalysis replication (MERRA-2 downloader exists in repo) and/or
   correlation against EDA-spread climatology. The Congo-Amazon contrast
   (similar observing density) is the most confound-resistant result.
3. **Box geometry**: fixed 20x40 deg boxes have latitude-dependent km
   geometry (dx varies ~40% within midlatitude boxes; domain width differs
   ~1.6x across regions); band masks use a single median dx. Part of
   between-region separation may be geometric fingerprint. Control: add
   |lat| and box geometry to residualization or use equal-km boxes.
4. **Sampling**: single level (850 hPa — level-specificity demonstrated by
   5d); two seasons only; 6-hourly step undersamples the finest bands;
   interior mask cap makes coarse envelopes boundary-affected in a
   region-dependent way. 2021-22 windows are La Nina, 2023-24 El Nino —
   replication across ENSO states is a genuine strength worth stating.

## Where the mathematics was wrong or fragile

1. 5c null design (fatal, fixed by retest -> retraction).
2. W=20 rolling estimator induces ~3.5-step memory floor -> the "2-day
   relaxation" is at least half mechanical.
3. Octave overlap: ~half of raw envelope coupling is Gaussian filter
   overlap. The anchored excess P - P_surrogate is the physical object;
   it is NON-monotone (interior maximum ~0.36 at step 2-3), unlike the raw
   monotone profile. Anchored profiles must be primary in publication.
4. H4c: n=12 with 3 correlated covariates -> jackknife-unstable.
5. Marginal p-values (0.026-0.029) under region-label permutation are not
   safe against within-region dependence (ENSO, adjacent years); the
   p=0.001 results survive any reasonable effective-sample-size correction.

## What stands (defensible before hostile review)

1. The B1 falsification, with DNS-validated flux code.
2. The cross-scale envelope-coupling profile of ERA5 850-hPa vorticity as a
   two-season-stable, cross-ENSO-replicated regional signature whose
   non-Gaussian (anchored) excess carries the regional information
   (p=0.001-0.002), on resolved scales.
3. The directional transfer-asymmetry index ||F|| as the strongest such
   signature, beyond spectrum and beyond P (p=0.001), resolved-scales
   robust; regional pattern: maxima over tropical-oceanic
   convective regimes, minima over dry continents (descriptive).
4. The Congo-Amazon contrast (rho_5: 0.71 vs 0.46 across independent
   periods) — two similar convective land regions with different
   scale-coupling architecture.
5. The complete negative catalogue: flux closure, signed scalar, universal
   law, local instantaneous source, level invariance, moisture-budget
   information content, dynamical charge-discharge, simple normalization
   collapse (IVT sharpens rather than simplifies).
6. The methodology itself: frozen protocols, deviation logs, per-phase
   held-out data, published negatives, independent red-team audit with
   remediation.

## Applicability of what was achieved

- **Yes**: regional climatological fingerprinting of scale-coupling
  architecture (a compact 5+6-number descriptor family complementary to
  spectra and scattering statistics); a candidate observable proxy for the
  cross-scale error-growth coupling of the predictability literature
  (interpretation, untested against ensemble error-growth data); a
  reviewer-grade methodological template for falsification-first
  climate-diagnostics work.
- **No / not shown**: precipitation forecasting value; moisture-budget
  closure; fundamental scale-space geometry; universal laws; dynamical
  source equations.

## Relation to the first paper (Zenodo 19565805)

The first paper's central empirical claim (A15 Lambda_b ~ Pi_b closure,
R2_binned = 0.520) is retracted by B1: the proxy was not a flux, the
statistic was inflated by binning and in-sample sign alignment, and the
window was the only passing one of six. The bundle formalism survives only
as organizing vocabulary with acknowledged prior art (MERA/cMERA,
Beny-Osborne, Mori-Zwanzig, holographic RG). The successor paper reports
the two validated descriptors, the negative catalogue, and the corrections
explicitly.

## Pre-publication checklist (required)

- [ ] Cross-reanalysis replication (MERRA-2) of P and ||F|| signatures
- [ ] Correlation of signature maps against ERA5 EDA spread / increment
      climatology
- [ ] Box-geometry/latitude control in residualization
- [ ] Anchored (surrogate-excess) profiles as primary objects, incl. Fnorm
- [ ] Remove/reframe all geometry, irreversibility, charge-discharge and
      integrability language per this document

## Addendum (2026-08-13, post-reconciliation controls)

- Box-geometry control EXECUTED: adding [abs lat, dx, domain width] to the
  residualization collapses P's beyond-spectrum residual clustering
  (+0.454 p=0.003 -> +0.216 p=0.100) — P's beyond-spectrum claim is now
  CONDITIONAL (substantially geometry-explained). ||F|| survives unchanged
  (+0.767 -> +0.754, p=0.001) and is the program's primary object.
- Phase 8 (MERRA-2 cross-reanalysis replication) protocol frozen; downloader
  ready; blocked on user's Earthdata credentials (~/.netrc).
