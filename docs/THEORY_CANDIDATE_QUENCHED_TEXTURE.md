# THEORY CANDIDATE: Quenched-Texture Organization of Atmospheric Scale Coupling
## Status: CANDIDATE v1.1 (2026-08-18)

Standing rules: the candidate is falsifiable through QT-P1..QT-P5; each
prediction names its falsification condition. Any edit that weakens a
falsification condition demotes this document to note status. The
retired vocabularies (Lambda as flux, A as irreversibility) are not
part of the candidate.

## 1. Postulates

- QT-1 (two layers). The resolved atmospheric state evolves as a fast
  stochastic process with spatially varying parameters,
      d omega = F_theta(x)(omega) dt + noise,
  mixing time T_fast ~ hours. The constraint field theta(x) —
  exogenous boundary conditions (orography, land-sea geometry,
  insolation, SST distribution) together with the stationary
  circulation organization they pin (storm tracks, convective zones) —
  is quenched on the observation window: T_theta >> 46 yr (measured
  lower bound).
- QT-2 (the observable). Locally the fast layer is ergodic on an
  invariant measure mu_theta(x). The validated observable is
      P(x) = Phi[mu_theta(x)],
  where Phi is the rank-dependence (copula) between adjacent
  scale-band envelopes of 850 hPa vorticity: the component of
  mu_theta not fixed by second moments nor by scattering-moment
  hierarchies (the scale-coupling texture).
- QT-3 (no functional dynamics). A functional of a stationary measure
  has no dynamics; only its sample estimates fluctuate, decorrelating
  on the mixing time of the sampled weather patterns — one clock,
  common to all resolved scales (~10 h at the programme's domain
  sizes).
- QT-4 (two-tier causality). Directional language applies only to
  tier-1 exogenous components of theta. Tier-2 state components
  (synoptic activity, convective regime) are co-emergent with P:
  covariation without direction.
- QT-5 (texture coordinates). Two tier-2 coordinates set the texture:
  convective regime lowers P beyond the spectrum; synoptic
  (storm-track) activity raises P through the spectrum. Tier-1 relief
  acts on the spectral slope, not on P.

## 2. Evidence

All rows trace to frozen preregistered protocols
(docs/PROTOCOL_PHASE*.md) and committed results.

| # | Claim | Phase(s) | Result |
|---|---|---|---|
| E1 | P is physical, not box-geometry | B18 | CONFIRMED_PHYSICAL (design control); beyond-spectrum residual clustering on equal-km boxes p=0.033 (anchored), p=0.007 (raw) |
| E2 | P is beyond spectrum and beyond scattering statistics | B2b, B3 | H2 positive; PHASE3_VERDICT = NOVEL |
| E3 | P is instrument-independent | B8, B13 | REPLICATED (MERRA-2); ATMOSPHERIC (free-running model, no assimilation) |
| E4 | P has no level dynamics | B14, B19-A | FORM_REJECTED; reproduced on km territory incl. exact R7/R5 sign split (territory-robust) |
| E5 | No valid derivative observable exists | B15 | ESTIMATOR_INVALID: RMS(Delta X) = sd * sqrt(2(1-rho)) identity |
| E6 | One clock, no scale hierarchy | B19-B | tau_b = 9-11 h flat over 71-1131 km; alpha = 0.000, CI [-0.08, +0.11] |
| E7 | Long-record stationarity; no ENSO modulation | B16, B17, B19-C | Null at 30x Phase-16 power, on both carriers; era halves null |
| E8 | Texture coordinates (QT-5) | B20-A | LOO-region R^2 = 0.40, p = 0.001; CAPE negative in 10/12 regions within-region (beyond-spectrum); EKE positive in 11/12 (spectrum-shared); orography loads on slope target only |
| E9 | Co-emergence at tier 2 (QT-4) | B19-C | P vs E_syn lead-lag: peak at lag 0 (S = +0.165, null q95 = 0.04), no month-scale lead either way |
| E10 | Known anomaly, open | B17, B19-C | Tropical interannual variance excess F = 1.5-1.8 (R1, R3, R5), carrier-robust, non-ENSO, unexplained |

## 3. Predictions and falsification conditions

- QT-P1 (forced drift of theta). Where tier-1/2 fields trend
  (convective-margin expansion, CAPE trend > 0), tile-P declines over
  1979-2024 at fixed spectrum. Falsified if a preregistered trend
  protocol at matched power finds no drift where the attribution of E8
  predicts it.
- QT-P2 (model transfer). A free-running model sharing tier-1 theta
  reproduces the global P map tile-by-tile. Falsified if a model with
  correct tier-1 fields yields a significantly different map.
- QT-P3 (effective dimension). The global P map is predictable from
  <= 3 independent covariate fields at out-of-block skill >= 80% of
  the full-set skill. Falsified otherwise. (Scored: Phase 20 Arm B,
  H-B3.)
- QT-P4 (tropical excess). E10 reflects a slow tier-2 tropical mode
  outside ONI; candidate predictors must be named in a frozen protocol
  before consultation. Falsified if no named slow mode accounts for it
  at matched power (E10 then stands as an anomaly against QT-1).
- QT-P5 (sampling clock). tau of P-estimate fluctuations scales with
  the advective crossing time of the observation window, not with band
  scale. Falsified if tau varies with band scale at fixed window, or
  fails to scale with window size.

## 4. Scope fence

- No irreversibility, entropy-production, or information-flux claims
  (retired: Phases 1, 12, 15).
- No predictive-skill claims (Phases 9-11 negative).
- No novelty claim for storm tracks or convective regimes per se. The
  contribution is the conjunction: one scalar functional of the local
  measure that is (i) beyond-spectrum, (ii) instrument-independent,
  (iii) globally mappable and attributable, (iv) strictly without
  accessible dynamics of its own.
