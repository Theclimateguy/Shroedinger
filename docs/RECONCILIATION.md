# Program reconciliation (2026-08-13; superseded in part on 2026-08-16)

Final audit-informed reconciliation of the B1-B7 program, incorporating the
independent red-team audit, the literature positioning study, and the
post-audit remediation computations (5c retest, B7, N2).

> **Superseded in part, 2026-08-16.** Phases 9-13 changed two verdicts in the
> scoreboard below. The interpretation of the transfer-asymmetry index
> ||F|| (A) is **RETRACTED** (Phase 12), and the shared-observations caveat
> attached to B8 is **CLOSED** (Phase 13). See the addendum at the end of
> this file; where the two disagree, the addendum governs.

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
- Phase 8 EXECUTED (2026-08-14): MERRA-2 cross-reanalysis replication
  VERDICT REPLICATED. H8a Fnorm signature on MERRA-2 diff=+1.119 p=0.001;
  P signature +1.151 p=0.001 (descriptive); H8b cross-reanalysis regional
  geography Spearman rho=+0.930 p<1e-4 (12 regions). The transfer-asymmetry
  invariant is not an ERA5-system artifact. Shared-observations caveat
  remains stated. Pre-publication checklist: MERRA-2 [x], geometry [x];
  EDA-spread correlation and anchored-Fnorm residualization remain optional
  strengthening controls.


## Addendum (2026-08-16): Phases 9-13

Five further preregistered phases, protocols in `docs/PROTOCOL_PHASE9..13_*.md`,
results under `clean_experiments/results/experiment_B9..B13_*/`.

| Phase | Question | Verdict |
|---|---|---|
| B9 | Does A predict the rate of ensemble forecast error growth (Sect. 6.2 hypothesis)? | **NEGATIVE** — sign opposite to prediction (-0.25 self-analysis, -0.32 ERA5); error growth is governed by baroclinicity (+0.86 with abs latitude); band-variance placebo wins (+0.81 vs 0.25) |
| B10 | Does A predict the level of analysis-time mesoscale error? | **NEGATIVE** on 27 new lattice domains — raw +0.34/+0.40 but -0.19/-0.005 after controls; within-domain arm reverses sign; placebo -0.87 |
| B11 | Does A predict the mesoscale deficit of ML weather models? | **NEGATIVE** for A (raw +0.59, -0.08 after controls, placebo -0.69). The mechanism ladder itself is a **positive result**: at 120 h in 200-800 km GraphCast loses 69 %, Pangu 55 %, GenCast ensemble mean 85 %, while HRES and a single GenCast sample lose ~0 % — the loss is confined to systems computing a conditional mean |
| B12 | Is A a valid estimator of cross-level irrecoverability? | **ESTIMATOR_INVALID** — see below |
| B13 | Do the regional geographies survive without data assimilation? | **ATMOSPHERIC** — P 0.71, A 0.90, u 0.77, persistence 0.98 (ERA5-to-ERA5 ceilings 0.93/0.90/0.81/0.98), all p=0.001 |

### Two scoreboard entries above are hereby amended

**B4 (transfer-asymmetry index).** The signature stands as a *statistic*; its
*interpretation* is retracted. On synthetic fields where irrecoverability is
known and swept over nine rungs x 20 realisations, A does not respond in
either the spatial (rho = -0.32) or the modal (rho = -0.35) reading, and its
across-rung spread (0.016) is smaller than the within-rung noise (0.046). It
tracks temporal persistence instead: Spearman 1.0 on synthetic fields, +0.53
on the 39 real domains, against +0.03 with the information measure. The cause
is methodological and is the same defect already recorded for B6a: the
transfer operators are fitted over a 20-step sliding window, so the more
persistent the field, the more systematically the commutator norm is
displaced. **A is a persistence statistic, not a measure of the
irreversibility of generalisation.**

**B8 (cross-reanalysis replication).** The stated caveat — that ERA5 and
MERRA-2 share an observing network, so their agreement cannot separate the
atmosphere from the observing system — is **closed** by Phase 13. The
geographies replicate in a free-running CMIP6 HighResMIP integration
(ECMWF-IFS-HR, `highresSST-present`) that assimilates no atmospheric
observations. The properties belong to the atmosphere.

### What this promotes

**P becomes the primary object of the programme.** Phase 12 validated it
where A failed: on the same frozen synthetic ladder P responds with
rho = -0.883 (direction correct; the reference information measure u gives
+0.917, A gives -0.317), with a signal-to-noise ratio of 0.65 against 0.69
for u and 0.35 for A. On real domains P agrees with u at rho = -0.93 while
being nearly independent of band amplitude (+0.20) and of persistence (+0.02).
Recorded honestly: -0.883 is marginally short of the |0.90| bar the protocol
set for the reference estimator, the shortfall arising from the flattening of
the ladder at its saturated end, which u shows as well.

### Pre-publication checklist, updated

- MERRA-2 replication [x]; geometry control [x]; free-running control [x]
  (Phase 13, supersedes the "optional strengthening" line above).
- Second free-running model (CMCC-CM2-VHR4) — declared in the Phase-13
  protocol as required only on a negative result; the result was positive, so
  it was not run. Available on the same ESGF node if hardening is wanted.
- Geometry control for P re-run on the 39-domain set — **not done**; the
  p = 0.10 figure comes from the 12-region set.
- Scattering and semi-fractal controls re-run on the 39-domain set — **not
  done**; those figures also come from the 12-region set.

## Addendum, 2026-08-16 (bis): Phase 14 — the dynamics of P itself

Protocol `docs/PROTOCOL_PHASE14_P_RELAXATION.md` (frozen before computation);
code `clean_experiments/experiment_B14_p_relaxation.py`; results
`clean_experiments/results/experiment_B14_p_relaxation/`. Tested:

> dP_b/dt = -gamma_b (P_b - P_b^eq(regime)) + eta_b, regime = CAPE state,
> on the 32 region-windows of R5-R12 x W5-W8 (6-hourly, instantaneous P,
> no estimator window anywhere), fit R5-R8, validation R9-R12.

**Verdict: FORM_REJECTED** (with the estimator itself validated). By
component:

- **Estimator gate passed.** The pipeline recovers known gamma on synthetic
  OU ladders driven by the real CAPE series (median |log ratio| 0.25 and
  0.08 at gamma = 0.2, 0.5), detects implanted coupling with 83% power, and
  — the Phase-6a/12 disease control — flags coupling on Fourier
  phase-randomized surrogates of the real P (full spectrum preserved) at
  only 6.1% against a 5% nominal level (3200 surrogates, bound <= 10%). The
  negatives below are therefore about the atmosphere, not the instrument.
- **Relaxation exists but is fast and essentially memoryless.** Median
  ACF of deseasonalized composite P: 0.61 at 6 h, 0.34 at 12 h, 0.15 at
  24 h, ~0 beyond 48 h. Implied relaxation time ~12 h. Step-invariance of
  gamma passes at 6-24 h (median max/min ratio 1.34 <= 2.0), but the
  log-ACF linearity criterion fails by a hair (median R^2 0.8905 vs the
  0.90 bar) — partly genuine multi-timescale structure in about half the
  region-windows, partly the noise floor at lags where ACF has already
  decayed to zero (post-hoc diagnostic `diagnostic_daily_steps.json`,
  no criterion weight: at purely daily lags the exponential form is worse,
  R^2 0.77, because there is no signal left beyond two days, not because a
  slow component appears).
- **No transferable regime coupling.** Per-region-window CAPE coefficient
  exceeds the coherent (whole-day circular-shift) null in only 5/16 fit
  region-windows; the sign is regionally split (R7_CONGO positive in 4/4
  windows, R5_SPCZ negative in 4/4) so the pooled coefficient dies
  (p = 0.13); the frozen pooled model beats a locally-fit AR(1) on held-out
  regions in 4/16; CAPE beats the band-amplitude placebo in 8/16. On raw
  (non-deseasonalized) series: 2/16. The regime-dependent-equilibrium
  hypothesis, in any form that transfers across regions, is rejected.
- **Noise term is clean.** Full-model residuals are white at daily
  subsampling in 32/32 region-windows (Ljung-Box 10 lags, p > 0.01).
- Descriptive only: CAPE-tercile equilibria are monotone-up in 13/32 and
  monotone-down in 3/32 (weak, sign-inconsistent); window-median P is
  higher in the El-Nino-side windows in 6/8 (JFM) and 7/8 (JAS) regions —
  n = 8, season-confounded, not a claim.

**What this settles.** The fluctuations of P around its regional value
carry ~12 h of memory and no detectable regime dependence that survives the
persistence-safe null, held-out transfer, and the amplitude placebo. Together
with Phases 2-13 this closes the picture coherently: **P's information is its
static regional value; its dynamics is fast, regime-blind relaxation around
that value** — which is also why P fluctuations could never have helped the
multi-day predictability targets of Phases 9-11. The honest equation of
motion licensed by the data is dP/dt = -gamma (P - P^eq_region) + eta with
gamma ~ (12 h)^-1, P^eq static per region, eta white at daily scale — a
statement of stability, not of exploitable dynamics. Any future dynamical
claim for the programme must look elsewhere than the time evolution of P
(e.g. slow modulation of P^eq itself across seasons/years, which the
92-day windows cannot resolve).

## Addendum, 2026-08-16 (ter): Phase 15 — flux-derivative dynamics

Author's postulation (corrected per section 9 of the frozen protocol,
`docs/PROTOCOL_PHASE15_FLUX_DERIVATIVE.md`): the dynamical object is not
the level of a scale-coupling descriptor but the rate of change of the
Aluie cross-scale KE transfer itself, K_b(t) = 6-h difference of the net
transfer into the resolved bands, predicted to correlate with the Phase-9
error-growth rate lambda across regions. Code
`clean_experiments/experiment_B15_flux_derivative.py`; series computed for
all 80 region-windows with a pipeline that reproduces the frozen B1
implementation exactly (C15-1: max relative difference 0.0).

**Verdict: ESTIMATOR_INVALID.** The preregistered persistence gate C15-2
— K evaluated on the Phase-12 synthetic generator with irrecoverability
and temporal persistence swept independently — failed decisively before
any forecast hypothesis was scored: Spearman(K, persistence) = -0.81
(bar 0.5) and the persistence sweep moves K about four times more than
the irrecoverability sweep. On the real fields the mandatory ACF report
shows the flux series has the same fast memory as P (0.54 @ 6 h,
0.13 @ 24 h).

The failure is structural, and this is the finding of the phase: for any
series with lag-delta autocorrelation rho(delta),

    RMS(x_t - x_{t-delta}) = sd(x) * sqrt(2 (1 - rho(delta))),

so ANY window-RMS summary of a finite-difference "derivative" is an exact
function of fluctuation amplitude and persistence — precisely the two
quantities the programme has already adjudicated (amplitude is the placebo
that defeated A in Phases 9-11; persistence is the retracted content of
A from Phase 12) — and carries no third degree of freedom. The
flux-derivative family |dX/dt| summarized by magnitude cannot, in
principle, pass the persistence gate.

**The dynamical line of the programme is closed as tested three times and
not supported**: B6a (descriptor charge-discharge: half the relaxation
was estimator-window memory, zero held-out content), B14 (P relaxation:
fast, regime-blind, FORM_REJECTED), B15 (flux derivative:
persistence-inseparable by construction, ESTIMATOR_INVALID). Per the
frozen Phase-15 rule the programme reverts to P as a static regional
fingerprint — its validated role. Anything dynamical that remains lives
either in slow modulation of P^eq across seasons/years (unresolvable in
92-day windows) or in objects that are neither levels, nor lag-difference
magnitudes, of scale statistics (e.g. signed/oriented transfer events,
which would need a new protocol with its own persistence gate).

## Addendum, 2026-08-17: Phase 16 — slow (ENSO) modulation of P^eq

The one dynamical layer left open by Phases 14-15: does the static
regional fingerprint P^eq move across years with ENSO? Hypothesis and
direction came from the Phase-14 descriptive (window-median P higher on
the El-Nino side in 13/16 same-season comparisons, regions R5-R12 only) —
so the frozen design (`docs/PROTOCOL_PHASE16_PENV_SLOW_MODULATION.md`)
made regions R1-R4, which no ENSO-flavoured look had ever touched, the
sole confirmatory arm, with the ONI table (ERSST.v6) frozen into the
protocol before any P was read. Inputs were the stored P values of
Phases 2/2b/3 verbatim; the estimator gate (amplitude specificity of P on
the Phase-12 generator) passed.

**Verdict: NEGATIVE — and maximally instructive.** The fresh confirmatory
arm is exactly flat: S = 0.000 (mean within-region-season Spearman of
P^eq against ONI over 8 cells), p = 0.51. The replication arm — the very
regions the hypothesis was generated on — "confirms" it at S = 0.375,
p = 0.005. A pooled analysis (S = 0.25 over all 24 cells) would have
cleared p < 0.05 and put an ENSO-modulation claim into the manuscript on
the strength of the data that suggested it. The fresh-arm firewall
existed precisely to prevent that publication. Next to a dead-flat fresh
arm, the replication-arm significance is the textbook signature of
hypothesis selection, not of a modulation that four independent regions
mysteriously lack. (Declared power limit: the fresh arm could only see
mean rho >= ~0.35, so this is a strong-effect exclusion, not proof of
strict constancy — but the strong effect is what the descriptive had
suggested.)

**Programme status after Phases 14-16.** The stability statement is now
final at every timescale the data resolve: within windows P relaxes in
~12 h around P^eq (B14); P^eq shows no ENSO-scale movement detectable in
eight years of windows (B16); and no derivative-magnitude object can
carry dynamics at all (B15). P is a static regional fingerprint, full
stop. The remaining open dynamical candidates are exactly two, both
requiring new data or new objects: (i) monthly-resolved P over the full
ERA5 period (1940-present) for a properly powered slow-modulation test;
(ii) signed/oriented transfer events, with their own persistence gate.

## Addendum, 2026-08-18: Phase 17 — P^eq over the long record (1979-2024)

The properly powered version of the Phase-16 question, on new data:
daily 00Z ERA5 850-hPa wind, 12 program boxes, 46 years, 552 monthly
P^eq values per region (protocol frozen before download completed;
confirmatory period 1979-2016, untouched by any phase; two-sided after
the Phase-16 null; shared circular-shift ONI null preserving both
autocorrelation and cross-region dependence).
`docs/PROTOCOL_PHASE17_PENV_LONG_RECORD.md`,
`clean_experiments/experiment_B17_penv_long_record.py`.

**Verdict: VARIANCE_WITHOUT_ENSO.** Two findings:

1. **ENSO modulation of P^eq is dead at scale.** Pooled S = 0.009
   (p = 0.71, two-sided); the record resolves pooled |S| >= ~0.03 —
   thirty times the Phase-16 power — and finds nothing. Era halves both
   null-scale; amplitude placebo null-consistent. The 2017-2024 check
   era alone gives S = +0.14: the anomaly that generated the hypothesis
   in Phase 14 now sits exposed as an 8-year outlier against 38 flat
   years. Phase 16's fresh-arm null and its selection-artifact reading
   are confirmed.
2. **A small, real, unattributed interannual layer exists in the deep
   tropics.** Exactly three regions carry interannual variance above
   sampling noise: R3_AMAZ (F = 2.24), R5_SPCZ (2.07), R1_WPWP (1.75);
   the other nine are flat (F ~= 1, the quantitative stability bound the
   programme sought). The excess is not ENSO-shaped, has no dominant
   spectral peak (best: ~4.7 yr at power 0.60 over a ~0.5 floor), and
   decadal trends are <= 0.009 per decade in correlation units.
   R5_SPCZ is the one region whose ONI correlation (+0.35 over 456
   months) is physically suggestive (SPCZ displacement is ENSO-tied),
   but after a x12 selection correction it is not claimable; it is
   logged as the single candidate for a future preregistered
   one-region test, nothing more.

**Programme status.** The stability statement survives its strongest
test yet, now with numbers: for nine of twelve regions P^eq is constant
to within sampling noise over 46 years; the tropical residual is small,
non-ENSO, and unexplained. The slow-modulation line is closed as tested
(B16 -> B17); what remains of it is exactly one preregisterable
question (SPCZ) and one open descriptive fact (the tropical F ~= 2
excess), neither of which licenses any dynamical language for P.

## Addendum, 2026-08-18: Phase 18 — equal-km regionalization (design control)

Protocol `docs/PROTOCOL_PHASE18_EQUAL_KM_REGIONS.md` (frozen 2026-08-17;
zonal extent 2800 km, renumbering and further deviations logged there);
code `clean_experiments/experiment_B18_equal_km_regions.py`; results and
figures `clean_experiments/results/experiment_B18_equal_km_regions/`.

The box-geometry item of this document ("Where the data may be
non-representative", item 3) is now closed by DESIGN rather than by
regression: the 12 regions were re-cut as 2000 x 2800 km boxes (same
centres, native grid, index masks only) and the frozen Phase-2/3/4
machinery re-run on the 48 primary and 32 held-out region-windows, with
phase surrogates regenerated per km window.

**Verdict: CONFIRMED_PHYSICAL.** All scored hypotheses passed: A negative
control rho(deg,km)=0.972 with km signature p=0.001; anchored-P clustering
on km boxes +1.256 p=0.001 (held-out 2021-22: +1.564 p=0.001); geography
preserved (pooled rho 0.926); Congo-Amazon anchored rho_5 contrast keeps
sign at 56% magnitude; extent sensitivity all-region rho >= 0.93; the
deformation-radius-scaled reading agrees (p=0.001).

The decisive number: LOO beyond-spectrum residual clustering of anchored P
on equal-km boxes is +0.218 p=0.033 (raw P: +0.347 p=0.007), where the
degree-box statistical control had left it at p=0.10. The 2026-08-13
"collapse" under [abs lat, dx, domain width] regression is therefore best
read as covariates absorbing latitude-correlated physical signal, not as
evidence that the beyond-spectrum component was cartographic. P's
beyond-spectrum claim is UNCONDITIONAL under the design control; the
CONDITIONAL flag from the addendum of 2026-08-13 is lifted. The spectral
placebo still clusters more strongly than P on km boxes (+2.538 p=0.001),
as on degree boxes — P's claim remains "beyond spectrum", not "strongest
clustering". A remains a persistence statistic (Phase 12); Phase 18 scores
it only as a geometry-robust negative control.


## Addendum, 2026-08-18 (bis): Phase 19 — dynamics re-asked on the km territory

Protocol `docs/PROTOCOL_PHASE19_KM_DYNAMICS.md` (frozen 2026-08-18); code
`clean_experiments/experiment_B19_km_dynamics.py`; results
`clean_experiments/results/experiment_B19_km_dynamics/`. Motivated by the
author's objection that the dynamical negatives of B14 were obtained on
degree boxes and might reflect territory composition rather than the
atmosphere. Estimator-class deaths (B15, half of B6a) were declared
non-re-askable in the frozen protocol; only the atmosphere-answer
negative of B14 was re-asked, once, on the Phase-18 equal-km territory.

**Verdict: (NEGATIVE_TERRITORY_ROBUST, ESTIMATOR_INVALID).**

- Arm A: the frozen B14 pipeline on km-cropped fields returns
  FORM_REJECTED again, every component at least as negative (per-rw
  coupling 1/16 vs 5/16; pooled p=0.074; held-out 4/16), and the
  R7_CONGO +4/4 / R5_SPCZ -4/4 CAPE-coefficient sign split reproduces
  window-by-window on substantially different territory (sanity: km/deg
  series correlation median 0.801, inside the informative window). The
  territory-composition explanation is excluded; the B14 negative is
  final for this data class.
- Arm B: the new object tau_b(ell) — the scale hierarchy of coupling-
  fluctuation timescales — failed its own preregistered estimator gate
  (single-timescale synthetics yield spurious median |alpha|=0.178 >
  0.15), and the real data show no hierarchy anyway: median alpha
  0.000, CI [-0.08, +0.11], tau_b flat at ~9-11 h from 71 to 1131 km,
  where eddy-turnover would give ~6x and sweeping ~11x spread.

The stability statement of Phases 14-17 is thereby hardened: P's
dynamics is fast, regime-blind, scale-undifferentiated relaxation around
a static regional value, and this is now known on both the degree and
the equal-km territory. Open dynamical candidates remain exactly two:
the deferred km-crop of the Phase-17 long record (Arm C, waits for
b17daily), and signed/oriented transfer events (new protocol, own gate).

## Addendum, 2026-08-18 (ter): Phase 20 Arm A — the geography of P

Protocol `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md` (frozen 2026-08-18); code
`clean_experiments/experiment_B20_p_geography.py`; results and tile maps
`clean_experiments/results/experiment_B20_p_geography/`. After B6b's
n=12 attribution failure, the phase changed the data carrier: anchored
fine-P (50-400 km) as a map of 144 tiles inside the equal-km boxes,
with leave-one-region-out prediction and region-block permutation nulls.

**Verdict: DRIVERS_IDENTIFIED** — the first positive attribution of P in
the program. Tiles reproduce the box signature (rho=0.874); the five
on-disk covariates predict tile anchored P out-of-region (LOO R^2=0.40,
p=0.001). Two drivers survive every control including the within-region
test that no between-region confound can produce: convective regime
(cape_mean, NEGATIVE, 10/12 regions, p=0.001) and storm-track activity
(eke_syn, POSITIVE, 11/12, p=0.001). Orography, land fraction and
coastlines show nothing at tile scale (orography instead drives the
spectral slope — the C20-2 placebo has different loadings, so the
spectrum-mediation clause is not triggered).

Mediation (post-hoc, labeled): the EKE association is shared with the
tile spectrum; the negative CAPE association survives spectrum
residualization within regions. One-line physics: organized baroclinic
cascades couple adjacent scales; intermittent deep convection decouples
them beyond what the spectrum records. This is the program's first
mechanism-level statement about what the P regionalization measures.
Arm B (global sliding-window map; external covariates with anthropogenic
and biological fields as pre-declared negative controls) is the frozen
next step.

## Addendum, 2026-08-18 (quater): Phase 19 Arm C — long record on the km carrier

Spec frozen inside `docs/PROTOCOL_PHASE19_KM_DYNAMICS.md` after b17daily
completed (552/552) and before any computation; code
`clean_experiments/experiment_B19_armC_km_longrecord.py`.

**Verdict: (a_ENSO_STAYS_NULL, a_EXCESS_CARRIER_ROBUST).** The ENSO
negative survives the km carrier (pooled S=0.030, p=0.077, two-sided) —
with the honest nuance that the statistic triples relative to degrees
and is carried almost entirely by R5_SPCZ (rho=+0.32), which stays the
programme's single preregisterable ENSO target (now with 1979-2016 km
SPCZ consulted, so any future SPCZ protocol must use other data or
declare the consultation). The tropical interannual variance excess is
retained 3/3 (R1 1.79, R3 1.50, R5 1.76 vs q95~1.43): carrier-robust,
non-ENSO, unexplained — the geometry alternative for both long-record
findings is closed. The declared lead-lag probe (P vs E_syn, 46 years)
shows a sharp contemporaneous peak (S=+0.165, null q95=0.04) with no
month-scale lead either way: synoptic activity and cross-band coupling
covary as facets of one organization, supporting the co-emergence
framing over any directional "centers of action drive P" reading.

## Addendum, 2026-08-18 (quinquies): Phase 20 Arm B — the global map

Spec frozen in `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md` before computation
(one implementation deviation logged); code
`clean_experiments/experiment_B20_armB_global_map.py`; results and maps
`clean_experiments/results/experiment_B20_armB_global_map/`.

**Verdict: GLOBAL_MAP_ATTRIBUTED_DIM1.** 936 global tiles (60S-60N,
JFM+JAS 2023, 34 excluded for orography): the map is consistent with the
Arm-A boxes (rho=0.840 over 133 overlapping tiles), season-stable
(median seasonal contrast 0.016), and attributed at LOSO R^2=0.536
against a conservative rotation null that certifies the non-zonal
component specifically (p=0.001). The LAI biological negative control is
clean (gain -0.004, p=0.83). The planet's P maxima are the Southern
Ocean storm-track ring and the N Pacific / N Atlantic storm tracks; the
minima are the deep convective cores (Amazon, Congo, Maritime
Continent). Forward selection: eke_syn alone carries 85% of the full
skill (k80=1) — the first scored test of a theory-candidate prediction
(QT-P3, bar <=3) PASSES: the quenched constraint field is effectively
low-dimensional at this resolution. Loadings reproduce the Arm-A
two-channel split globally (P <- eke/lat/sst/cape; spectral slope <-
orography/land). Phase 20 closes with (Arm A, Arm B) =
(DRIVERS_IDENTIFIED, GLOBAL_MAP_ATTRIBUTED_DIM1).

## Addendum, 2026-08-18 (sexies): Phase 21 — theory-candidate tests, round 1

Protocol `docs/PROTOCOL_PHASE21_QT_TESTS.md` (frozen with downloads
launched at freeze time); code
`clean_experiments/experiment_B21_qt_tests.py`; results
`clean_experiments/results/experiment_B21_qt_tests/`.

**Verdict: (P5_SUPPORTED, P1_UNDERPOWERED, P4_SUPPORTED).**

- QT-P5 (sampling clock): SUPPORTED. tau of the P estimate scales with
  the advective crossing time of the observation window (rho=+0.36,
  p=0.001; 79/79 windows individually positive; log-log slope 0.32
  [0.12, 0.62], sub-linear). Together with B19's band flatness this
  completes the estimator-clock picture of QT-3.
- QT-P1 (forced drift): UNDERPOWERED, right sign. Spectrum-fixed tile-P
  trend over the top CAPE-trend tercile D_obs = -3.9e-5/yr (predicted
  -0.74e-5/yr), p=0.081; CI covers prediction and zero. Remains open
  with its original bar.
- QT-P4 (tropical excess): SUPPORTED — the excess has a name. Among the
  four indices named before consultation (PDO, AMO, DMI, IPO-TPI), the
  Interdecadal Pacific Oscillation is coherent in two of the three
  excess regions (R3_AMAZ T=0.45 p=0.001, R5_SPCZ T=0.60 p=0.001;
  max-statistic over 8 index x smoothing combinations, joint
  circular-shift null). ENSO was dead at 30x power while IPO is not:
  the tropical interannual excess is a decadal tier-2 mode. R1_WPWP
  stays unattributed.

Candidate updated to v1.2 (E10 upgraded, E11/E12 added, predictions
P3/P4/P5 marked scored, P1 open). Score to date: P3 PASS, P4 SUPPORTED,
P5 SUPPORTED, P1 open (right sign), P2 untested.

## Addendum, 2026-08-18 (septies): Phase 22 — theory-candidate tests, round 2

Protocol `docs/PROTOCOL_PHASE22_QT_ROUND2.md` (frozen with the beta_T
calibration amendment logged pre-computation); code
`clean_experiments/experiment_B22_qt_round2.py`; results
`clean_experiments/results/experiment_B22_qt_round2/`. No new data.

**Verdict: (P2_SUPPORTED, P1_MIXED).**

- QT-P2: SUPPORTED near the ceiling. On 902 resolution-matched tiles the
  free-running ECMWF-IFS-HR integration (2014, no assimilation)
  reproduces the ERA5 (2023) global texture map at rho = 0.869 against a
  same-atmosphere resolution ceiling of 0.897 — 97% of the attainable
  agreement; covariate loadings mirror ERA5. The map is a property of
  the physics given tier-1 boundary conditions.
- QT-P1 (seasonal carrier): the drift EXISTS — D_obs = -8.5e-5/yr in the
  top-CAPE-trend tercile, p = 0.025, CI excluding zero (Phase 21's
  annual carrier had the same sign at p = 0.081). The magnitude exceeds
  the cross-sectional calibration ~7x and the detrended interannual
  calibration ~30x (beta_T/beta_CS = 0.25): the response is
  timescale-dependent, and space-for-time UNDERSTATES the decadal
  response. By the frozen ladder this is P1_MIXED
  (DRIFT_DETECTED_MAGNITUDE_OPEN in the candidate). Mandatory caveat:
  observing-system changes make any ERA5 trend ERA5-internal until a
  transient free-running control is scored — the named next protocol,
  together with a decadal-response calibration model.

Candidate updated to v1.3 (E13, E14). Prediction scoreboard: P2, P3,
P4, P5 supported/passed; P1 sign-supported with magnitude open. The
candidate has survived its first two adversarial rounds intact.

## Addendum, 2026-08-19: Phase 23 — drift physicality and stratigraphy

Protocol `docs/PROTOCOL_PHASE23_DRIFT_STRATIGRAPHY.md`; code
`clean_experiments/experiment_B23_drift_stratigraphy.py`; results
`clean_experiments/results/experiment_B23_drift_stratigraphy/`.

**Verdict: (DRIFT_NOT_CONFIRMED_IN_MODEL, STRATIGRAPHY_FAILED,
SENSITIVITY_RANKED_ONLY).** One free-running transient member
(ECMWF-IFS-HR highresSST-present r1, epochs 1979-86 vs 2007-14, the
frozen ERA5 tile selection) shows the ERA5 drift's sign at ~27%
magnitude but is null-consistent (p=0.22): the ERA5-internal suspicion
stands, QT-P1 stays open, member escalation is blocked by the frozen
no-member-shopping rule, and candidate v2.0 is NOT declared. The
stratigraphic hypothesis fails both on-disk arms: at the 1998/99 IPO
transition P transitions neither earlier nor sharper than CAPE, E_syn
or the spectral slope (25/47/42% vs the 60% bar), and P ranks LAST in
trend SNR (0.67 vs CAPE 2.39). The reorganization-detection question is
answered NO on this record; P's established value — the only
beyond-spectrum, instrument-independent, model-transferable structural
map — is an architecture registrar, not an early-warning variable.
Candidate updated to v1.4 (E14 annotated, E15 and QT-P6 FAILED entered).
The only path to closing QT-P1: a new preregistered multi-member power
protocol (~3 members resolve 3.5e-5/yr at this variance).
