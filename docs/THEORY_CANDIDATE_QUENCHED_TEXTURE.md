# THEORY CANDIDATE: Quenched-Texture Organization of Atmospheric Scale Coupling
## Status: CANDIDATE v1.3 (2026-08-18; Phase-22 round-2 results entered)

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
| E10 | Tropical excess attributed to a named decadal mode | B17, B19-C, B21-P4 | F = 1.5-1.8 excess (R1, R3, R5), carrier-robust, non-ENSO; associated with IPO-TPI in R3 (T=0.45, p=0.001) and R5 (T=0.60, p=0.001) under a named-before-consultation max-statistic test; R1 unattributed (p=0.155) |
| E11 | Sampling clock (QT-3) scales with the observation window | B21-P5 | rho(ln tau, ln T_cross) = +0.36, p=0.001; 79/79 windows positive; log-log slope 0.32 CI [0.12, 0.62] (sub-linear) |
| E12 | Global map effectively low-dimensional | B20-B | k80 = 1 (eke_syn alone = 85% of LOSO skill); QT-P3 bar was <= 3 |
| E13 | A free-running model reproduces the global map tile-by-tile | B22-P2 | rho(model, ERA5) = 0.869 at a resolution ceiling of 0.897 (97%); loadings mirror ERA5; different year, no assimilation |
| E14 | A negative spectrum-fixed P drift exists in top-CAPE-trend tiles | B22-P1s | D_obs = -8.5e-5/yr, p = 0.025, CI [-1.53e-4, -1.15e-5] excludes 0; magnitude exceeds both linear calibrations (beta_T/beta_CS = 0.25, timescale-dependent response); ERA5-internal until a transient free-running control |

## 3. Predictions and falsification conditions

- QT-P1 (forced drift of theta). STATUS: DRIFT_DETECTED_MAGNITUDE_OPEN
  (Phase 22 Arm P1s, seasonal carrier): the sign leg is supported
  (D_obs = -8.5e-5/yr, p = 0.025, CI excludes zero — see E14); the
  magnitude leg is open — the drift exceeds the cross-sectional
  calibration ~7x and the interannual calibration ~30x
  (beta_T/beta_CS = 0.25: the response is timescale-dependent).
  Completion requires (i) a transient free-running control for the
  reanalysis-trend caveat and (ii) a calibration model that carries a
  decadal response coefficient; both belong to a future protocol.
- QT-P2 (model transfer). SCORED: SUPPORTED (Phase 22 Arm P2:
  rho = 0.869 vs a 0.897 resolution ceiling, 902 tiles,
  resolution-matched; bars 0.5 abs / 0.6 x ceiling). See E13.
- QT-P3 (effective dimension). SCORED: PASS (Phase 20 Arm B, H-B3:
  k80 = 1 against the bar <= 3). See E12.
- QT-P4 (tropical excess). SCORED: SUPPORTED in the named-predictor
  form (Phase 21 Arm P4: IPO-TPI coherent in R3+R5 at p=0.001, joint
  circular-shift max-statistic null). Residual: R1_WPWP unattributed.
  See E10.
- QT-P5 (sampling clock). SCORED: SUPPORTED (Phase 21 Arm P5: positive
  window-size scaling of tau, 79/79 windows, p=0.001; band-scale
  flatness at fixed window from B19). Sub-linear slope 0.32
  [0.12, 0.62]: the clock mixes pattern residence with pattern
  lifetime. See E11.

## 4. Scope fence

- No irreversibility, entropy-production, or information-flux claims
  (retired: Phases 1, 12, 15).
- No predictive-skill claims (Phases 9-11 negative).
- No novelty claim for storm tracks or convective regimes per se. The
  contribution is the conjunction: one scalar functional of the local
  measure that is (i) beyond-spectrum, (ii) instrument-independent,
  (iii) globally mappable and attributable, (iv) strictly without
  accessible dynamics of its own.
