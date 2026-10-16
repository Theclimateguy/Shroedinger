# PROTOCOL — Phase 35: co-location and predictability (B35)

## Status: FROZEN 2026-10-15, before any Phase-35 field was downloaded or any Phase-35 quantity computed

Purpose: the computations behind article 3 in its new form, "what co-location of
adjacent scales means: predictability". Phase 33 Arm C (CONFIRMED) showed that
forecasts behaving as conditional means co-locate the activity of adjacent bands more
than the atmosphere does, while single realisations do not. The literature review
(`reports/Новизна метрики верности наследования.md`) found that the sign of this
effect follows from elementary probability and that reviewers will ask for: a
dose-response in the number of averaged realisations, a quantitative link to the
predictable fraction, nulls that rule out spurious coupling, a second truth, a lead
curve and estimator variants. This phase supplies them. Language: co-location of
activity, symmetric; "predictable" means "reproduced by the ensemble", no direction
of influence between levels.

## Definitions carried over (fixed)

From Phase 33, Arm C: WeatherBench-2 fields, 850 hPa u, v, 0.25 deg, the 39
programme domains, year 2022; V12 vorticity and mask; anchored profile P̂ = median
over fields of the raw profile minus the median of 99 phase-randomised surrogates;
headline = mean of steps 4-5 (200-400|400-800, 400-800|800-1600 km);
D = P̂_system - P̂_truth on the same valid times; sign-flip test over domains, 9999
flips, one-sided; δ = 0.010. Code basis: `experiment_B33_model_fidelity.py`.

Initialisations, unless stated: the first 6 of each 2022 window of Arm C (stride 5
days from 1 Jan, 1 Apr, 1 Jul, 1 Oct), 24 in total — the Arm-C member set.

## Data (downloaded after this protocol is committed)

| block | system | content | leads |
|---|---|---|---|
| A | IFS ENS, all 50 members | `ifs_ens/2018-2022-1440x721.zarr` | 24, 120 h |
| A | FGN, all 64 members | `fgn/2022-1440x721.zarr` | 120 h |
| D | GraphCast, IFS ENS mean, FGN mean, HRES | as Arm C; `fgn/2022-1440x721_mean.zarr` | 24, 48, 72, 120, 168, 240 h |
| E | IFS analysis as truth | `hres_t0/2016-2022-6h-1440x721.zarr` | valid times of Arm C (75 inits, 24 and 120 h) |
| — | ERA5 truth for the new valid times | as Arm C | — |

Downloader: `clean_experiments/download_b35_wb2.py` → `data/b35_wb2/`.

## Blocks, hypotheses and bars

**A. Dose-response in the number of averaged realisations.** For IFS ENS (M = 50)
and FGN (M = 64), the N-member mean of members 1..N, N ∈ {1, 2, 4, 8, 16, 32, M};
D(N) against ERA5 at 120 h (IFS ENS also at 24 h, reported).
- **H35-1 (PRIMARY).** For each ensemble at 120 h: Spearman(N, median D(N)) ≥ 0.90;
  D(1) not positive (sign-flip p > 0.05); D(M) ≥ δ with p ≤ 0.01.

**B. Quantitative link to the predictable fraction.** For each ensemble, domain and
step s ∈ {4, 5} with fine band f (B1 for step 4, B2 for step 5): predictable fraction
q = (V_mean - V_mem/M) / (V_mem (1 - 1/M)), V_mean = spatial variance of band f of
the M-member mean (over fields), V_mem = mean member variance of band f. Mixture
model for the raw profile: P(N) = (q ρ_p + (1 - q) ρ_u / N) / (q + (1 - q) / N); ρ_p,
ρ_u solved from P(1) and P(M); P(N) predicted for N = 2..32.
- **H35-2.** Pooled over domains and steps: median |P_obs(N) - P_pred(N)| /
  |P(M) - P(1)| ≤ 0.25 for both ensembles. Qualifies "quantitative" in the text;
  not part of the verdict.

**C. Calibration: what the estimator responds to.**
- **H35-3 (independent placement).** On the ERA5 truth fields of Arm C (75 inits,
  120 h valid times): bands B1 and B2 replaced by their own phase-randomised versions
  (each band keeps its spectrum, its placement becomes independent of the coarse
  field), coarse band B3 and the rest kept. Median D_decoupled (vs the untouched
  truth) ≤ -0.05, sign-flip p ≤ 0.01 (lower tail).
- **H35-4 (no coupling from frequency modulation; Hyafil 2015).** Synthetic fields on
  the domain grid (81 x 161, 0.25 deg at 45 deg latitude), 100 fields each:
  (i) Gaussian random field whose dominant wavelength varies smoothly in space
  (between 250 and 900 km over the domain) with spatially uniform amplitude:
  |median P̂ (steps 4-5)| ≤ 0.02; (ii) a log-normal multiplicative cascade (five
  octave levels, intermittency 0.2): median P̂ ≥ 0.10.

**D. Lead curve.** GraphCast, IFS ENS mean, FGN mean, HRES at 24-240 h.
- **H35-6.** For each of the three conditional means: Spearman(lead, median D) ≥ 0.80;
  for HRES: |median D| < δ at every lead.

**E. Second truth.** Arm C recomputed with the IFS analysis (`hres_t0`) as truth.
- **H35-7.** At 120 h: GraphCast, Pangu, IFS ENS mean median D ≥ δ, p ≤ 0.01; HRES and
  the IFS ENS member |median D| < δ.

**F. Estimator variants and the nearest prior form (reported).**
- **H35-8.** On Arm C at 120 h, the class separation (three conditional means D ≥ δ,
  two realisations |D| < δ) holds under (i) the Pearson form of the profile instead of
  the default rank form; (ii) mirror-extended instead of plain FFT surrogates.
- **H35-5 (descriptive).** On the same fields: the adjacent-scale modulus correlation
  without log, ranks or anchoring (Portilla-Simoncelli form) and the scattering ratio
  s21 = S2(j1, j2)/S1(j1) (Brochet et al. 2023) for the same band pairs. Reported: the
  effect size of the class separation (median D over its across-domain IQR) for each
  measure. No bar; the text says what the anchored log-rank form adds or does not add.

## Verdict ladder

- **CO-LOCATION_PREDICTABILITY_CONFIRMED:** H35-1 (both ensembles), H35-3, H35-4,
  H35-6, H35-7.
- **PARTIAL:** H35-1 for at least one ensemble, H35-3 and H35-4.
- **NEGATIVE:** H35-1 fails for both ensembles, or H35-4 fails (the estimator creates
  co-location from frequency modulation alone).
H35-2, H35-5, H35-8 qualify the text.

## Priors (recorded before computation)

H35-1 0.8 (IFS ENS), 0.7 (FGN); H35-2 0.35; H35-3 0.9; H35-4 0.7; H35-6 0.7;
H35-7 0.75; H35-8 0.8.

## Code and outputs

`clean_experiments/experiment_B35_colocation_predictability.py` (stages per block) →
`clean_experiments/results/experiment_B35_colocation_predictability/`. Seed 20261016.

## Deviations

- (none yet)

## Result (2026-10-16)

All data and stages complete (members, decouple, synth, leads, truth2, variants);
results `clean_experiments/results/experiment_B35_colocation_predictability/`. No
deviations.

**Verdict: NEGATIVE** (H35-4 failed; H35-1 failed for both ensembles).

| hypothesis | result | numbers |
|---|---|---|
| H35-1 dose-response (PRIMARY) | FAIL | IFS ENS 120 h: median D for N = 1, 2, 4, 8, 16, 32, 50 = +0.007, -0.006, -0.008, +0.001, +0.005, +0.040, +0.045 (rho 0.61); FGN: +0.005 ... +0.040 (rho 0.89, bar 0.90); D(M) > 0 with p <= 0.001 for both |
| H35-2 mixture model | FAIL | median relative error 0.42 (IFS ENS), 0.33 (FGN), bar 0.25 |
| H35-3 independent placement | PASS | D = -0.189, 39/39 domains negative |
| H35-4 frequency modulation | FAIL | uniform amplitude, dominant wavelength 250 -> 900 km: step 4 -0.39, step 5 +0.35, mean -0.021 (bar abs <= 0.02); cascade +0.22 (pass) |
| H35-6 lead curve | FAIL | GraphCast rho 0.89, FGN mean 0.94, IFS ENS mean 0.66; HRES -0.014 at 120 h |
| H35-7 IFS analysis as truth | PASS | GraphCast +0.054, Pangu +0.034, IFS ENS mean +0.035 (p 1e-4); HRES -0.005, member -0.002 |
| H35-8 estimator variants | FAIL | Pearson form keeps the separation (member +0.012 just above delta); **mirror surrogates remove it: GraphCast +0.008, Pangu +0.011, IFS ENS mean -0.026** |
| H35-5 prior forms | — | Portilla-Simoncelli modulus correlation separates the classes strongly (effect 1.5-3.2, raw, no anchoring); s21 weakly |

**Decomposition of the Phase-33 effect (Arm C, 120 h, steps 4-5, median over domains).**

| system | raw difference | FFT-surrogate difference | anchored (FFT) | mirror-surrogate difference |
|---|---|---|---|---|
| GraphCast | +0.066 | +0.017 | +0.045 | +0.060 |
| Pangu | +0.078 | +0.028 | +0.056 | +0.070 |
| IFS ENS mean | +0.064 | +0.012 | +0.041 | +0.064 |
| HRES | +0.002 | -0.004 | +0.001 | -0.006 |
| IFS ENS member | -0.002 | -0.009 | -0.000 | -0.007 |

Reading. The conditional-mean forecasts do have a higher raw adjacent-band envelope
correlation (+0.06-0.08). But stationary random fields with each forecast's own
spectrum, built without periodic edges (mirror surrogates), reproduce almost all of
it. The raw excess is a property of the spectrum: with overlapping Gaussian bands, a
spectrum that falls steeply inside a band concentrates the band's content at its
coarse edge and raises its envelope correlation with the next band. Periodic FFT
surrogates miss this because their edge discontinuity adds broadband fine-scale power;
the Phase-33 control S1 missed it because it matched band-integrated variances, not
the within-band spectral shape. **The Phase-33 claim that the over-coupling is
non-spectral is retracted.** What stands: the raw statistic separates conditional
means from realisations (Phase 33, all arms; H35-7 with a second truth) — as the
spectrum already does.
