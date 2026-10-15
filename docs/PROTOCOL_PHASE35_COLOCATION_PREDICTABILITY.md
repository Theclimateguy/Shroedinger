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
