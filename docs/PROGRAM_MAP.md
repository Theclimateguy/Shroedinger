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
