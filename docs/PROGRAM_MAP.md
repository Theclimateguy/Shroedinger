# Program map: scale-geometry assessment and rebuild (2026-08-11 .. 2026-08-12)

All phases ran under frozen, preregistered protocols with logged deviations.
Chronology and verdicts below; every claim traces to a protocol file and a
results directory.

## Phase 1 — Lambda vs true flux (B1) — NEGATIVE

- Protocol: `PROTOCOL_PHASE1_LAMBDA_VS_FLUX.md`
- Claim tested: A15-style Lambda_b tracks the cross-scale energy flux.
- Method: true Germano/Aluie coarse-graining flux (DNS-validated code),
  4 cheap baselines, 8 region-windows, raw statistics, fixed sign,
  phase-surrogate nulls.
- Result: Lambda raw R^2 0.002-0.007 vs Smagorinsky 0.35-0.47; the A15
  "flux" proxy itself does not track the real flux (R^2 0.01-0.02).
  All criteria failed. **Lambda-as-flux-diagnostic retired.**
- Results: `clean_experiments/results/experiment_B1_true_flux_baselines/`

## Phase 2 — irreversibility profile, first pass (B2) — NEGATIVE (metrics pathological)

- Protocol: `PROTOCOL_PHASE2_SCALE_IRREVERSIBILITY.md`
- Object introduced: profile P = cross-scale envelope-coupling rho_1..rho_5
  (operational irreversibility profile).
- Formal verdict NEGATIVE, but both failed criteria were design flaws
  (saturated classification baseline; degenerate monotone-profile Spearman).
  Exploratory: strong regional signature. Logged post-hoc; led to 2b.
- Results: `clean_experiments/results/experiment_B2_scale_irreversibility/`

## Phase 2b — held-out invariant test (B2b) — POSITIVE

- Protocol: `PROTOCOL_PHASE2B_HELDOUT_INVARIANTS.md`
- 8 NEW regions x 2021-2022 (32 region-windows), corrected metrics.
- C1 32/32; H1 profile signature p=0.001; H2 beyond-spectrum residual
  p=0.029. **First clean preregistered positive: P is a season-stable
  regional invariant with a spectrum-exceeding component.**
- Results: `clean_experiments/results/experiment_B2b_heldout_invariants/`

## Phase 3 — scattering benchmark (B3) — NOVEL

- Protocol: `PROTOCOL_PHASE3_SCATTERING_BENCHMARK.md`
- 12 regions x 2023-2024 (48 region-windows); 21 scattering-type features.
- H3a (P beyond scattering) p=0.026; H3b (scattering beyond P) p=0.001 —
  mutually complementary. Scattering is the stronger classifier (0.79 vs
  0.42); P's value = orthogonal component + compactness.
- E1 addendum (Krenke-Puzachenko semi-fractal features): P not explained by
  them either (p=0.001); vorticity spectral breaks 141-225 km, stable.
- Cross-year replication of profiles; Congo-vs-Amazon contrast confirmed.
- Results: `clean_experiments/results/experiment_B3_scattering_benchmark/`

## Phase 4 — curvature invariant (B4) — CONFIRMED_PHYSICAL

- Protocol: `PROTOCOL_PHASE4_CURVATURE_INVARIANT.md` (motivated by declared
  E2 exploration; confirmation on curvature-naive years 2017-19 + 2021-22).
- Object: curvature-strength profile Fnorm_b = median||F_b||; the SIGNED
  Lambda remains dead (E2 p=0.7) — only curvature magnitude informs.
- H4a signature p=0.001; H4b beyond spectrum AND beyond P p=0.001;
  H4c physical: [orog_std, land_frac, cape_mean] -> fine-band curvature,
  LOO R^2=0.27 p=0.023. Tropical oceans highest, dry continents lowest.
- Results: `clean_experiments/results/experiment_B4_curvature_invariant/`

## Phase 5 — gauge-structure escalation (B5) — VOCABULARY

- Protocol: `PROTOCOL_PHASE5_GAUGE_STRUCTURE.md`
- 5c integrability: **PASS 48/48** — time/scale path consistency beyond
  phase-null; the connection is genuinely integrable structure.
- 5b universality: FAIL — covariates explain curvature level (R^2=0.58),
  profile shape does not collapse; no law on [CAPE, land, orography].
- 5a source localization: FAIL 0/32 — tiled curvature does not track
  precipitation at 6-hourly tile scale; the source is regime-level.
- 5d level invariance: FAIL — 500 hPa geography differs (underpowered but
  frozen). Strong "functional fifth dimension" reading retired; the
  structural reading survives.
- Results: `clean_experiments/results/experiment_B5_gauge_structure/`

## What stands (validated, held-out, preregistered)

1. Profile P: season-stable regional invariant, not reducible to spectrum,
   scattering statistics, or semi-fractal features.
2. Curvature strength ||F||: strongest regional invariant of the program;
   regime-level physically predictable (convective-oceanic maximum).
3. Integrability: the estimated scale connection is closer to a consistent
   geometry than spectrum-matched noise, universally across windows.

## What is retired (falsified or failed escalation)

- Lambda_b ~ Pi_b closure (flux reading), and signed Lambda in any role.
- Universal covariate law for profile shape.
- Instantaneous local precipitation sourcing of curvature.
- Level-invariance ("whole-column fifth dimension") of the curvature field.
- Atmospheric SOC heavy tails (legacy F6 series, pre-assessment).

## Prior-art positioning

Formal ingredients are established structures (MERA/cMERA, Berry/Wilczek-Zee,
Beny-Osborne CPTP-RG, Nakajima-Mori-Zwanzig, holographic RG); the validated
contribution is the pair of compact atmospheric invariants + their
falsification-grade validation, in the Puzachenko hierarchical-organization
tradition (co-cite Krenke et al. 2019).

## Post-audit reconciliation (2026-08-13)

An independent red-team audit plus remediation computations updated the
verdicts; see docs/RECONCILIATION.md for the authoritative final scoreboard.
Headline changes: 5c "integrability" RETRACTED (temporal-null retest 7/48);
H4c demoted to descriptive ocean/land contrast; charge-discharge dynamics
not established (estimator-memory null); B7 moisture closure NEGATIVE;
N2 IVT normalization sharpens rather than collapses the signature;
anchored (surrogate-excess) profiles adopted as primary objects — the
anchored profile is non-monotone with an interior maximum.


## Phases 9-13 (2026-08-16)

The applied promise was tested three times and failed three times: the rate of
ensemble forecast error growth (B9), the level of analysis-time mesoscale
error (B10), and the mesoscale deficit of machine-learning weather models
(B11). In each, plain band variance was a stronger predictor than the
descriptors.

B12 then asked whether the instrument measured what it was said to measure,
and found that it did not: the transfer-asymmetry index does not respond to
controlled irrecoverability at all, and follows temporal persistence instead.
Its interpretation is retracted; the coupling profile P, which does respond,
becomes the primary object.

B13 closed the programme's declared decisive limitation: the regional
geographies replicate in a free-running model that assimilates no atmospheric
observations, so they are properties of the atmosphere and not of the
observing network.

Headline changes: A's interpretation RETRACTED; P promoted to primary and
independently validated; observing-network confound CLOSED; applied content
tested and not found, with the AI-blurring mechanism established as a result
in its own right.

## Phases 14-17 (2026-08-16 .. 2026-08-18) — the dynamics and slow-modulation sweep

- Phase 14 (B14) — P relaxation — **FORM_REJECTED**. tau ~ 12 h,
  memoryless; CAPE regime coupling fails pooled null with an R7/R5 sign
  split. `PROTOCOL_PHASE14_P_RELAXATION.md`
- Phase 15 (B15) — flux-derivative dynamics — **ESTIMATOR_INVALID**.
  RMS(dX/dt) = amplitude x persistence by identity; the derivative
  estimator class is closed. `PROTOCOL_PHASE15_FLUX_DERIVATIVE.md`
- Phase 16 (B16) — ENSO modulation of P^eq — **NEGATIVE** (fresh
  confirmatory arm flat; multiplicity trap caught and documented).
  `PROTOCOL_PHASE16_PENV_SLOW_MODULATION.md`
- Phase 17 (B17) — long record 1979-2024 — **VARIANCE_WITHOUT_ENSO**.
  ENSO dead at 30x power; tropical interannual excess F~2 in R1/R3/R5,
  unexplained. `PROTOCOL_PHASE17_PENV_LONG_RECORD.md`

## Phase 18 (2026-08-18) — equal-km design control (B18) — CONFIRMED_PHYSICAL

Regions re-cut as 2000x2800 km boxes (same centres, index masks, no
reprojection). All scored hypotheses passed; beyond-spectrum residual
clustering significant on km (p=0.033 anchored / 0.007 raw) where the
degree-box statistical control had left it at p=0.10. The box-geometry
confound is closed BY DESIGN; the 2026-08-13 CONDITIONAL flag on P's
beyond-spectrum claim is lifted. `PROTOCOL_PHASE18_EQUAL_KM_REGIONS.md`

## Phase 19 (2026-08-18) — dynamics re-asked on km territory (B19) — NEGATIVE_TERRITORY_ROBUST

Arm A: frozen B14 pipeline verbatim on km-cropped fields: FORM_REJECTED
reproduces, R7 +4/4 / R5 -4/4 sign split survives the territory change —
the territory-composition alternative is excluded; the dynamical
negative is final for this data class. Arm B: tau_b(ell) — gate failed
(spurious |alpha|=0.18) and the real data are flat anyway (~9-11 h at
all scales 71-1131 km). Arm C (long record on km): ENSO stays null
(S x3 but p=0.077, carried by SPCZ rho=+0.32 — the one preregisterable
target); tropical excess retained 3/3; lead-lag P vs E_syn peaks
sharply at lag 0 with no lead either way (co-emergence).
`PROTOCOL_PHASE19_KM_DYNAMICS.md`

## Phase 20 (2026-08-18) — the geography of P (B20) — DRIVERS_IDENTIFIED (Arm A)

Carrier change: anchored fine-P (50-400 km) as a 144-tile map inside the
km boxes. LOO-region R^2 = 0.40 (block-perm p = 0.001). Two drivers pass
the within-region control: convective regime (CAPE, NEGATIVE, 10/12,
beyond-spectrum) and storm-track activity (EKE, POSITIVE, 11/12,
spectrum-shared). Orography moves the spectral slope, not P. Arm B
(global ~936-tile map, two-tier covariates, longitude-rotation nulls) is
frozen and computing at the time of this entry.
`PROTOCOL_PHASE20_P_GEOGRAPHY.md`

## Theory status (2026-08-18)

The results of Phases 1-20 are consolidated into a falsifiable candidate:
**Quenched-Texture Organization** (`THEORY_CANDIDATE_QUENCHED_TEXTURE.md`,
v1.0; seed note v0.1 retained). One sentence: P is a functional of the
local invariant measure of a fast ergodic layer over a quenched
constraint field theta (boundary conditions + pinned circulation
regimes); it has geography and no dynamics, its texture coordinates are
convection (decouples, beyond-spectrum) and storm tracks (couples,
spectrum-shared), and it must drift with anthropogenic movement of theta
(prediction QT-P1). Predictions QT-P1..P5 carry named falsification
conditions. The retired vocabularies (Lambda, A-as-irreversibility) are
not part of the candidate.

## Phase 21 (2026-08-18) — theory-candidate tests, round 1 (B21) — (P5_SUPPORTED, P1_UNDERPOWERED, P4_SUPPORTED)

First scored round of the quenched-texture predictions.
- QT-P5 (sampling clock): SUPPORTED — tau of the P estimate scales with
  the window's advective crossing time (rho=+0.36 p=0.001, 79/79
  windows positive, slope 0.32 [0.12, 0.62]); with B19's band flatness
  the estimator-clock reading of QT-3 is complete.
- QT-P4 (tropical excess): SUPPORTED — among four indices named before
  consultation, IPO-TPI is coherent in R3_AMAZ and R5_SPCZ (both
  p=0.001): the carrier-robust excess is a decadal mode invisible to
  ONI. R1_WPWP unattributed.
- QT-P1 (forced drift): UNDERPOWERED — D_obs = -3.9e-5/yr, right sign,
  ~5x prediction, p=0.081; open at its original bar.
Candidate updated to v1.2; score: P3 PASS, P4+P5 SUPPORTED, P1 open,
P2 untested. `PROTOCOL_PHASE21_QT_TESTS.md`

## Phase 22 (2026-08-18) — theory-candidate tests, round 2 (B22) — (P2_SUPPORTED, P1_MIXED)

QT-P2: the free-running ECMWF-IFS-HR global map matches ERA5 at
rho = 0.869 vs a 0.897 resolution ceiling (97%; 902 tiles, loadings
mirrored) — the strongest confirmation of the candidate to date.
QT-P1 on the seasonal carrier: the negative spectrum-fixed drift in
top-CAPE-trend tiles is now detected (D_obs = -8.5e-5/yr, p = 0.025,
CI excludes zero) but exceeds both linear calibrations
(beta_T/beta_CS = 0.25 — timescale-dependent response); ERA5-internal
until a transient free-running control. Candidate v1.3.
`PROTOCOL_PHASE22_QT_ROUND2.md`
