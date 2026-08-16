# Phase 15 Protocol (preregistered): Flux-derivative dynamics — does the time
# derivative of the cross-scale / cross-boundary flux carry dynamical content
# where the level descriptors P and A did not?

Status: FROZEN before any Phase-15 computation. Date frozen: 2026-08-16.
Deviations logged with timestamp at the bottom.

Author's draft: `~/Desktop/phase14.txt` (written as "Phase 14"). Corrections
applied to that draft before freezing are listed in section 9 with reasons;
everything not listed there is the author's postulation unchanged.

## 0. Lineage and motivation (frozen)

Phases 9-11 established that the LEVELS of the regional invariants (the
transfer-asymmetry index A, the envelope-coupling profile P) do not predict
forecast error growth, analysis-time mesoscale error, or ML-model mesoscale
deficit better than plain band variance. Phase 12 established WHY: A is a
persistence statistic, not a measure of cross-level irrecoverability; P is
validated as responsive to irrecoverability but is a state descriptor.
Phase 6a established that the charge-discharge equation for the descriptor
ITSELF has no held-out predictive content and that ~half of its apparent
relaxation was estimator-window memory.

**Phase 14 (P-relaxation, closed 2026-08-16, verdict FORM_REJECTED)
sharpened this**: the fluctuations of P around its regional value have
~12 h memory (ACF 0.61 @ 6 h, 0.15 @ 24 h, ~0 beyond 48 h), are white at
daily subsampling, and carry no CAPE-regime coupling that survives the
coherent null, held-out transfer, or the amplitude placebo. P's information
is its static regional value. Two consequences are binding here:

1. P may enter Phase 15 ONLY as a window-level static value (its validated
   role), never as a time series. Same for any modulator.
2. Any lag-difference statistic of a short-memory series is dominated by
   the series' fluctuation amplitude, not by its "rate of change": for lag
   d >> tau, RMS(x_t - x_{t-d}) -> sqrt(2)*sd(x). A "derivative" claim
   therefore requires beating the level-fluctuation-amplitude placebo
   (C15-4 vi below), and the ACF of the flux series must be reported so the
   reader can see whether the differencing step sits inside the memory.

The Phase-15 hypothesis relocates the dynamical claim: the object on the
right-hand side of an evolution equation is not the level of a
scale-coupling descriptor but the RATE OF CHANGE OF THE FLUX ITSELF:

    K_b(R,t) = d/dt [ net cross-scale KE transfer into band b in region R ]

with the regionalized companion

    K_R(t) = d/dt [ net band-KE inflow through the boundary of R's box ].

P is demoted to a regime-level MODULATOR (candidate coefficient), never a
dynamical variable. A is excluded entirely (Phase-12 retraction; its
content — persistence — is already a control in H15b). This demotion is
fixed here and may not be renegotiated post-hoc.

Directional prediction, fixed before any data are touched:

    Larger |K_b| (or |K_R|) in the analysis time series over the window
    =>  faster subsequent ensemble error growth (lambda up), SHORTER
    predictability horizon (T50 down).

A negative result retires the flux-derivative reading and closes the
dynamical line of the programme (descriptor dynamics B6a -> P dynamics
B14 -> flux-derivative dynamics B15, tested three times); it is not to be
re-parameterised after the fact.

## 1. Declared power limit (before results)

Primary unit: region-window. Confirmation sample = the 48 region-windows of
Phase 9 (12 regions x W9-W12, 2023-2024), out-of-sample = the 32
region-windows W5-W8 (8 regions, 2021-2022, opposite ENSO phase: triple
La Nina vs the 2023-24 El Nino development). Forecast metrics for BOTH
epochs are on disk from Phase 9 (verified 2026-08-16: 48 + 32 metric
files). At n = 12 regions (between-region arm) only |rho| >= 0.50 is
detectable at alpha = 0.05 one-sided; the within-region arm (48
region-windows) is the higher-power arm. Stated in advance so a null
cannot be dismissed as underpowered without conceding that any positive
had to be strong.

## 2. Data (fixed)

- ERA5 850 hPa u, v on disk: `data/b3` (12 regions x W9-W12), `data/b2b`
  (8 regions x W5-W8), 6-hourly. No new downloads for the primary battery.
- Forecast-side metrics (lambda, lambda_era5, mu, T50) taken UNCHANGED from
  `clean_experiments/results/experiment_B9_predictability/metrics_*.json`;
  the logistic (Dalcher-Kalnay) rate is the primary lambda, exactly as
  fixed by the Phase-9 deviation log. Re-deriving them is forbidden;
  Phase 15 adds only the descriptor side.
- P per region-window: stored `P_real` from
  `results/experiment_B3_scattering_benchmark/` (W9-W12) and
  `results/experiment_B2b_heldout_invariants/` (W5-W8); composite = mean of
  bands 3, 4 as fixed in Phases 9-12. Not recomputed.
- Moisture arm: IVT is on disk ONLY for W9-W10 (24 region-windows,
  `data/b6ivt`). The moisture arm is therefore scoped to those windows,
  secondary, and cannot be re-scoped later.

## 3. Quantities (fixed)

### 3.1 Cross-scale flux (per region-window)

Aluie/Germano coarse-graining flux Pi_ell exactly as in the DNS-validated
Phase-B1 implementation (`flux_and_gridded_baselines` machinery: metric-
aware Gaussian filter at sigma = ell/sqrt(12), tau-tensor contraction with
resolved strain, interior mask per ell), evaluated at the band edges
ell in {200, 400, 800, 1600} km. Net transfer into band b:

    Pi_band_b(t) = Pi_{ell_coarse(b)}(t) - Pi_{ell_fine(b)}(t)

for b = 2 (200-400 km), b = 3 (400-800), b = 4 (800-1600); b = 2, 3
primary (mean), b = 4 reported — matching the Phase-9 resolved bands.
Interior-mean series at 6-hourly resolution, hour-of-day means removed and
linearly detrended per region-window (the Phase-14 deseasonalization,
frozen there).

### 3.2 The dynamical object

    K_b(t) = Pi_band_b(t) - Pi_band_b(t - 6h)

The 6-h step is primary: Phase 14 measured ~12 h memory for instantaneous
scale statistics, so a 24-h difference would sit entirely outside the
memory and reduce to sqrt(2)*sd (see section 0). The 24-h difference is
reported as robustness, not rescored. The diurnal harmonic is handled by
deseasonalization (3.1), not by the differencing step. K is computed
WITHOUT any sliding regression window. Region-window summary:

    K_b(rw) = RMS of K_b(t) over the full window

(the Phase-9 inits are evenly spaced through each 90-day window, so the
full-window RMS equals the init-pooled 5-day RMS for a stationary series;
fixed here once). Reported alongside, mandatory: lag-1..4 ACF of
Pi_band_b(t), so the memory of the flux itself is on record.

### 3.3 Regionalized companion (secondary)

The programme boxes are geographically scattered; adjacent shared
boundaries do not exist. K_R is therefore defined on each region's own box
boundary: band velocities u_b, v_b (Gaussian band differences at the band
edges), band KE density e_b = (u_b^2 + v_b^2)/2, and

    F_R(t) = net inflow of e_b through the rectangle at the 1600-km
             interior-mask inset (line integral of e_b * (u_b . n_inward)),
    K_R(t) = F_R(t) - F_R(t - 6h),

band b = 3, deseasonalized as in 3.1, summarized as RMS as in 3.2.

### 3.4 Modulator (never a predictor in the primary test)

P enters ONLY in H15c as an interaction coefficient, as the static
window-level composite from stored Phase-2b/3 outputs (section 2). A does
not enter anywhere.

## 4. Hypotheses and criteria (fixed)

- C15-1 (pipeline sanity, computed FIRST). The Phase-15 series code must
  reproduce the frozen Phase-B1 `flux_and_gridded_baselines` Pi_ell on one
  reference region-window to relative agreement < 1e-6, and all series must
  be finite. Failure aborts as a pipeline fault; no hypothesis is scored.
  (The author's draft required a 5% band-energy budget closure; replaced —
  see section 9, correction 4: an 850-hPa single-level ERA5 budget is open
  by construction and cannot close, so that criterion would abort the
  phase regardless of pipeline correctness.)

- C15-2 (persistence gate — THE Phase-12 control, scored BEFORE H15a).
  On synthetic fields from the frozen Phase-12 generator: (a) the
  irrecoverability ladder (indep 0..1, 9 rungs x 12 realisations, fixed
  temporal persistence), (b) a persistence ladder (rho_t 0.5..0.95, 6
  rungs x 12 realisations, fixed indep = 0.5). K (pipeline of 3.1-3.2)
  must show: (i) |Spearman(K, rho_t rung)| <= 0.5 on ladder (b);
  (ii) across-rung spread of K on ladder (b) smaller than on ladder (a).
  Failure means K is another persistence statistic: verdict
  ESTIMATOR_INVALID, no forecast hypothesis is scored.

- H15a (primary, between-region). Spearman over the 12 regions between
  K_b (b = 2, 3 mean; region value = median over its four W9-W12
  region-windows) and lambda (same aggregation, bands 2, 3 mean):
  positive, permutation p < 0.05 (999 pairing shuffles). Scored also for
  mu and T50 (negative predicted for T50).

- H15b (beyond controls). LOO linear residualisation of both K and lambda
  on [ |center lat|, land fraction, log sigma_band (Phase-9
  sigma_band_era5, bands 2-3 mean), spectral slope (stored Phase-2/2b
  F_spec), ACF of Pi_band at 12 h, window-mean Pi_band, sd of Pi_band ]:
  residual Spearman stays positive, p < 0.05. The last two controls are
  what separates the RATE from the LEVEL and from the FLUCTUATION
  AMPLITUDE of the flux.

- H15c (modulation, the P role). On the 48-window arm:
  lambda ~ K + P + K:P + region fixed effects, LOO across regions;
  the K:P interaction must improve held-out R^2 over the K-only model
  (R^2 gain > 0, permutation p < 0.05 by shuffling region-level P, 999).
  This is the ONLY arm where P enters.

- H15d (within-region, higher power). Over the 48 region-windows,
  lambda ~ K + region fixed effects: pooled within-region slope positive,
  permutation p < 0.05 (999; K shuffled across windows within each
  region).

- H15e (out-of-sample). The H15d model (slope + region intercepts) fitted
  on 2023-2024, applied UNCHANGED to the 32 region-windows of 2021-2022:
  Spearman(predicted, observed lambda) > 0 with p < 0.05, and
  out-of-sample R^2 > 0.

- H15f (regionalized companion, secondary). H15a/H15d repeated with K_R.
  Positive here with negative H15a is scored PARTIAL with the horizontal
  reading; it does not rescue a failed H15a.

Placebos and falsifiers, all reported whatever they show (scored on the
H15a arm):

- C15-3 (null calibration). The H15a permutation null must be well
  calibrated: KS test of the null p-value distribution against uniform,
  p > 0.05.
- C15-4 (placebo ladder). The same H15a statistic with, in place of K:
  (i) window-median log band KE (sigma placebo);
  (ii) RMS of the 6-h difference of band KE (the trivial "flux-like"
  placebo);
  (iii) persistence (12-h ACF of band KE);
  (iv) A (stored Phase-9/11 descriptor, region level);
  (v) P (stored composite, region level);
  (vi) sd of Pi_band (flux fluctuation amplitude — the Phase-14-motivated
  decisive placebo).
  A claim that K carries the signal requires |rho_K| > |rho_placebo| for
  ALL six, with (ii), (iii) and (vi) decisive: if any of those matches K,
  Phase 15 reports that instead.
- C15-5 (verification independence). Sign agreement of the H15a rho across
  lambda (GDAS-verified), lambda_era5, mu.

## 5. Verdict rule (fixed)

- CONFIRMED: C15-1, C15-2 pass; H15a passes for lambda AND mu; H15b, H15d,
  H15e pass; C15-4 shows K beating all six placebos; C15-5 holds. Only
  then may the manuscript state the flux-derivative as a dynamical
  predictor and write the section-7 equation as a fitted result.
- PARTIAL: H15a passes for at least one metric and H15b passes; remaining
  arms mixed. Scoped association only.
- NEGATIVE: H15a fails for both lambda and mu, or C15-5 fails, or K loses
  to placebo (ii), (iii) or (vi). The dynamical line of the programme
  (B6a -> B14 -> B15) is then reported as tested three times and not
  supported, and is closed.
- ESTIMATOR_INVALID: C15-2 fails. Same consequence as NEGATIVE for the
  dynamical line, plus an explicit statement that the flux-derivative at
  6-hourly resolution cannot be separated from persistence.

C15-1 failure is not a verdict; it is a bug, to be fixed and rerun.

## 6. Compute plan

- Experiment: `clean_experiments/experiment_B15_flux_derivative.py`,
  reusing the frozen B1 flux machinery and the Phase-14 deseasonalization;
  results under `clean_experiments/results/experiment_B15_flux_derivative/`.
- Stages: `--stage series` (80 region-windows), `--stage sanity` (C15-1),
  `--stage gate` (C15-2), `--stage tests` (everything else + verdict).
- Seed fixed: 20260816. Permutations: 999.
- No new downloads.

## 7. The equation this phase licenses (fixed wording)

If and only if the verdict is CONFIRMED, the manuscript may write, for the
region-mean band kinetic energy E_b of each region R and resolved band b:

    dE_b/dt = [resolved advective tendency] + lambda_b(P; R) * K_b(t) + eps

with lambda_b a regime-modulated coefficient (H15c), K_b the observed 6-h
change of net cross-scale transfer (3.2), and eps tested for whiteness
against the persistence null. Under PARTIAL the equation may appear only
as a stated hypothesis with the failed arms attached. Under
NEGATIVE/ESTIMATOR_INVALID the equation is retired and the programme
reverts to P as a static regional fingerprint (its validated role).

## 8. Explicitly out of scope (frozen)

- Dynamic re-regionalization by zero-divergence boundaries (K_R uses the
  frozen boxes only); flagged as Phase-16 candidate.
- Moisture/IVT flux derivative outside W9-W10.
- Any re-fit of A or P with changed windows/ridge/shrink parameters.

## 9. Corrections applied to the author's draft (before freezing)

1. **Numbering.** The draft was titled Phase 14; Phase 14 is taken and
   closed (P-relaxation, FORM_REJECTED, 2026-08-16). This is Phase 15;
   the experiment file is `experiment_B15_flux_derivative.py`.
2. **Lineage.** Section 0 extended with the Phase-14 result; the
   modulator role of P is restricted to its static window-level value
   (B14: P(t) fluctuations are ~12-h-memory noise with no transferable
   regime coupling).
3. **Differencing step.** Draft: K at 24 h to suppress the diurnal
   harmonic. Corrected: primary step 6 h on hour-of-day-deseasonalized
   series, 24 h as robustness. Reason: B14 measured ~12 h memory for
   instantaneous scale statistics; a 24-h difference of a 12-h-memory
   series is sqrt(2)*sd(level) in disguise, i.e. an amplitude statistic —
   exactly what the phase must distinguish itself from. Deseasonalization
   handles the diurnal harmonic instead. Added placebo (vi) sd(Pi_band)
   and the sd/mean of Pi_band to the H15b controls as the decisive
   rate-vs-amplitude separators; mandatory reporting of the Pi ACF.
4. **Sanity criterion.** Draft: band-energy budget residual < 5% of
   RMS(Pi) in >= 90% of steps. Corrected to a regression test against the
   frozen DNS-validated B1 implementation. Reason: the 850-hPa
   single-level ERA5 KE budget is open (vertical flux, pressure work,
   parameterized sources/sinks are unobserved at one level); a 5% closure
   is unattainable for any correct pipeline, so the criterion as drafted
   aborts the phase unconditionally.
5. **K_R.** Draft: flux through boundaries shared by adjacent regions.
   Corrected to each region's own box-boundary net band-KE inflow: the
   programme's 12 boxes are scattered and share no boundaries.
6. **A as modulator.** Draft: P and A both modulators. A removed: Phase 12
   retracted A's interpretation (persistence statistic); its content is
   already an explicit H15b control and placebo (iii). Keeping it as a
   "modulator" would reintroduce the retracted reading by the back door.
7. **Moisture arm.** IVT verified on disk only for W9-W10 (24
   region-windows); arm scoped accordingly instead of "if on disk".
8. **Out-of-sample arm.** Verified feasible before freezing: Phase-9
   metric files exist for all 32 W5-W8 region-windows.
9. **RMS summary.** Draft: RMS over days -5..0 before each Phase-9 init,
   pooled. Fixed to full-window RMS (equivalent for evenly spaced inits;
   init dates are not stored in the Phase-9 metric files).
10. **Equation wording.** Section 7 restated for the region-mean band
    energy (scalar per region), since K_b is a region scalar and the
    draft's d zeta_b/dt mixed a field with a scalar forcing.

## Deviations

- **2026-08-16, run record.** C15-1 passed (exact agreement with the frozen
  B1 implementation, max relative difference 0.0; all 80 series finite).
  **C15-2 FAILED**: Spearman(K, persistence rung) = -0.81 (bar: |rho| <=
  0.5) and the across-rung spread under the persistence sweep (6.4e-7) is
  ~4x the spread under the irrecoverability sweep (1.6e-7). Verdict
  **ESTIMATOR_INVALID**, recorded before any forecast hypothesis was
  scored, exactly as the gate prescribes. The mandatory Pi-ACF report
  (section 3.2) was still produced: median ACF of the deseasonalized
  Pi_band composite is 0.54 @ 6 h, 0.24 @ 12 h, 0.17 @ 18 h, 0.13 @ 24 h.
- **2026-08-16, closing note (analytic, not a rescoring).** The gate
  failure is structural, not incidental: for any series x with lag-delta
  autocorrelation rho(delta), RMS(x_t - x_{t-delta}) =
  sd(x) * sqrt(2 (1 - rho(delta))). A window-RMS summary of a
  finite-difference derivative is therefore an exact function of the two
  quantities the programme has already adjudicated — fluctuation amplitude
  (the placebo that defeated A in Phases 9-11) and persistence (the
  retracted content of A, Phase 12) — and contains nothing else. No
  re-parameterisation of K within the |dX/dt|-magnitude family can pass
  C15-2; per section 0, the dynamical line is not to be re-parameterised
  after the fact.
- **2026-08-16, moisture arm.** Not run: the estimator gate failed, so no
  hypothesis (primary or secondary) may be scored. Logged as moot, not
  deferred.
