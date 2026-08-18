# THEORY CANDIDATE: Quenched-Texture Organization of Atmospheric Scale Coupling
## Status: CANDIDATE v1.0 (2026-08-18). Successor of THEORY_NOTE_QUENCHED_TEXTURE.md (v0.1).

Rules of standing: this candidate is falsifiable through predictions
QT-P1..QT-P5 below; each prediction names its falsification condition.
It replaces — it does not rescue — the retired flux/irreversibility
vocabulary (Lambda: Phase 1; A-as-irreversibility: Phase 12). Any future
edit that weakens a falsification condition demotes the document back to
note status. The manuscript is NOT updated from this document yet
(author's decision, 2026-08-18).

---

## 1. Postulates

- QT-1 (two layers). The resolved atmospheric state evolves as a fast
  stochastic process with spatially varying parameters,
  d omega = F_theta(x)(omega) dt + noise, with mixing time T_fast of
  hours, while the constraint field theta(x) — exogenous boundary
  conditions (orography, land-sea geometry, insolation, SST
  distribution) together with the stationary circulation organization
  they pin (storm tracks, convective zones) — is quenched on the
  observation window: T_theta >> decades.
- QT-2 (the object). Locally the fast layer is ergodic on an invariant
  measure mu_theta(x). The programme's validated observable is
      P(x) = Phi[mu_theta(x)],
  where Phi extracts the rank-dependence (copula) between adjacent
  scale-band envelopes of 850 hPa vorticity — the component of mu not
  fixed by second moments (beyond spectrum) nor by scattering-moment
  hierarchies. Short name: the SCALE-COUPLING TEXTURE.
- QT-3 (no functional dynamics). Functionals of a stationary measure
  have no dynamics; only their sample estimates fluctuate, and those
  fluctuations decorrelate on the mixing time of the sampled weather
  patterns — one clock, common to all resolved scales, ~10 h at the
  programme's domain sizes.
- QT-4 (two-tier causal structure). Directional ("conditions",
  "sets") language applies only to tier-1 exogenous components of
  theta. Tier-2 components (synoptic activity, convective regime) are
  co-emergent facets of the same organization as P: covariation
  without direction, per the author's emergence thesis.

## 2. Evidence map (claims traceable to frozen-protocol results)

| Postulate element | Supporting phase(s) | Result |
|---|---|---|
| P physical, geometry-free | B18 (design control) | CONFIRMED_PHYSICAL; beyond-spectrum residual p=0.033/0.007 on km |
| P beyond spectrum / scattering | B2b, B3 | H2 positive; PHASE3 NOVEL |
| Instrument-independent | B8, B13 | REPLICATED (MERRA-2); ATMOSPHERIC (free-running model) |
| No level dynamics | B14, B19 Arm A | FORM_REJECTED; territory-robust (sign split reproduces window-by-window) |
| No derivative observable | B15 | ESTIMATOR_INVALID by identity RMS = amp x persistence |
| One clock, no scale hierarchy | B19 Arm B | tau_b flat 9-11 h over 71-1131 km (gate failed AND alpha ~ 0) |
| Long-record stationarity | B16, B17, B19 Arm C | ENSO null at 30x power, both carriers; era halves null |
| Tier-2 texture coordinates | B20 Arm A | CAPE negative beyond-spectrum (10/12 within-region), EKE positive spectrum-shared (11/12) |
| Co-emergence (no direction) | B19 Arm C lead-lag | lag-0 peak S=+0.165, no month-scale lead either way |
| Known anomaly (open) | B17, B19 Arm C | tropical interannual F~1.5-1.8 excess, carrier-robust, non-ENSO, unexplained |

## 3. Corrections absorbed from the author's seed idea

1. Timescale of theta: measured content is T_theta >> 46 yr only; the
   geological claim is an inference from the identification of theta.
   Anthropogenic forcing moves parts of theta (SST, convective margins)
   on decades — the source of QT-P1.
2. Conditioning: restricted to tier-1 (QT-4); inside the organization
   the data show lag-0 co-emergence.
3. Dimensionality: an empirical parameter, currently small (two
   coordinates carry the tile attribution); measured by QT-P3.

## 4. Falsifiable predictions

- QT-P1 (warming drift). Along convective-margin expansion zones
  (CAPE-trend positive), tile-P declines over 1979-2024 at fixed
  spectrum. FALSIFIED IF a preregistered trend protocol at matched
  power finds no drift where tier-2 attribution predicts it.
- QT-P2 (model transfer). Any free-running model sharing tier-1 theta
  reproduces the GLOBAL P map tile-by-tile (beyond B13's 12-region
  ranking). FALSIFIED IF a model with correct tier-1 fields produces a
  significantly different map.
- QT-P3 (effective dimension). The global P map is predictable from
  <= 3 independent covariate fields at out-of-block skill within 80%
  of the full-set skill. FALSIFIED IF many more fields are required.
  (Scored in Phase 20 Arm B, H-B3.)
- QT-P4 (tropical excess). The F~1.6 excess reflects a slow tier-2
  tropical mode outside ONI; a frozen protocol naming candidate
  predictors (decadal SST patterns) before consultation must find it.
  FALSIFIED IF no named slow mode accounts for it at matched power
  (the excess then stands as an anomaly against QT-1 stationarity).
- QT-P5 (sampling clock). tau of P-estimate fluctuations scales with
  the domain-crossing (advective) time of the OBSERVATION window, not
  with band scale. FALSIFIED IF tau varies with band scale at fixed
  window, or fails to scale with window size. (Testable on existing
  data by varying tile size.)

## 5. Non-claims (scope fence)

- No irreversibility, entropy-production, or information-flux claim is
  made or implied. Those vocabularies are retired (Phases 1, 12, 15).
- No skill claim: P carries no demonstrated predictive content for
  weather or extremes (Phases 9-11 negative).
- No novelty claim about the existence of storm tracks or convective
  regimes; the claim is that ONE scalar texture functional of the
  measure (i) is beyond-spectrum, (ii) is instrument-independent,
  (iii) has a readable, attributable global geography, and (iv) has
  strictly no accessible dynamics of its own — the combination, not
  the ingredients, is the contribution.

## 6. Naturalistic statement

Weather is water; theta is the riverbed. The bed does not move while
you watch the water — twenty phases looked for the bed answering the
water and found silence at every power we could buy. But the bed
decides, at every point, how the water is allowed to swirl: over the
deep organized reaches (storm tracks) eddies of different sizes move
together, braided; over the boiling shallows (deep convection) each
eddy churns alone and the braiding tears. P is the braiding, measured.
Not a flow, not a flux, not a memory — the local texture of the river's
turbulence, and its map is a rubbing of the riverbed. The map can be
read, it is the same in every honest instrument, and it will move only
as fast as the riverbed itself — which is now being recarved on a human
clock, and prediction QT-P1 says the rubbing must show it.
