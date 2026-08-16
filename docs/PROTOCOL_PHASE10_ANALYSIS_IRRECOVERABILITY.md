# Phase 10 Protocol (preregistered): A and analysis-time irrecoverability

Status: FROZEN before any Phase-10 data were retrieved and before any
Phase-10 quantity was computed. Date frozen: 2026-08-15. Deviations logged
with timestamp at the bottom.

## Why the question is being reformulated, and why that is not HARKing

Phase 9 falsified the Sect.-6.2 hypothesis: `A` does not predict the *rate*
of ensemble forecast error growth, and the sign is opposite to the one
predicted (`docs/PROTOCOL_PHASE9_PREDICTABILITY.md`, verdict NEGATIVE).

The reformulation below is derived from the definition of `A`, not from the
Phase-9 data, and could have been written before any of it:

`A_b = || G^up_b G^down_b - G^down_b G^up_b ||_F` is a property of the
*mapping between levels of description*. It says how far aggregation and
refinement fail to commute — that is, how much of the placement of mesoscale
variability cannot be recovered from the coarse field even in a statistical
sense. Nothing in that statement is about time evolution.

An operational atmospheric analysis is exactly such a reconstruction: the
mesoscale state is inferred from sparse observations plus the constraint of
the resolved large-scale flow and the model. The direct corollary of the
definition is therefore about the **level** of mesoscale error present at
analysis time, not about the rate at which forecast error subsequently grows.
Sect. 6.2 borrowed a Lorenz-type rate corollary that was never derived from
`A`; Phase 9 tested it and it failed. Phase 10 tests the corollary that does
follow.

Phase 10 is scored on **27 domains that no earlier phase of this programme
has used**, so the reformulated claim is not tested on the sample that
suggested it. The Phase-9 material is reported alongside but carries no
criterion weight.

## Declared power limit (before results)

n = 27 new domains: detectable |rho| >= 0.32 at alpha = 0.05, one-sided.
Pooled with the 12 existing regions, n = 39: |rho| >= 0.27. Better than
Phase 9 (n = 12, |rho| >= 0.50) but still a design for moderate-to-strong
effects only.

## Domains (fixed by rule, see `clean_experiments/regions_b10.py`)

1. Tile the globe with 20 deg x 40 deg boxes: latitude bands with northern
   edges at 55N, 35N, 15N, 5S, 25S; longitude edges every 40 deg starting at
   20E (the offset places a lattice edge on 180 deg so that no domain
   straddles the antimeridian). 45 boxes.
2. Drop any box overlapping an existing programme region by more than 20 %
   of its area.
3. Keep every survivor: 27 domains, named by their own bounds so that no
   regime label enters through the name.

No domain is added, moved or removed for any other reason, and no domain is
dropped after the data are seen.

## Data (fixed)

- **ERA5** 850 hPa u, v, 0.25 deg, 6-hourly, windows W11_2024JFM and
  W12_2024JAS, for the 27 new domains. Source of `A` and of one side of the
  inter-analysis comparison.
- **GEFS v12 control analysis** (`gec00` f000, 0.5 deg) at the matching
  valid times, and **GEFS v12 forecasts** (control + 5 perturbed members) at
  the 12 h lead, initialised 00 UTC every fifth day of each window.
- **ERA5 EDA ensemble spread** of 850 hPa u, v (`ensemble_spread`,
  0.5 deg, 3-hourly) for the same domains and windows: the assimilation
  system's own estimate of where the observing network constrains the flow.
  Used only as a control (H10d).

`A` is computed with the frozen Phase-4 machinery (`curvature_profiles`,
SEED 20260811, WINDOW 20, RIDGE 1e-6, SHRINK 0.05, NMODES 6, WARMUP 19);
region-window value `A_b = log10 Fnorm_b`, headline `A` = mean over the
resolved bands b = 2 (200-400 km) and b = 3 (400-800 km).

## Quantities (fixed)

Band-passing, vorticity, interior mask and the 0.5 deg common grid are
identical to Phase 9. Resolved bands b = 2, 3 enter the confirmatory tests;
b = 4 (800-1600 km) is reported.

- **D_b (inter-analysis mesoscale disagreement, co-primary).** RMS over
  interior points and matched analysis times of
  (band-passed vorticity of the GEFS control analysis minus band-passed
  vorticity of ERA5), divided by `sqrt(2) * sigma_b` with `sigma_b` the RMS
  band amplitude of the ERA5 field over the window. Two independent
  assimilation systems ingesting largely the same observations disagree
  where those observations do not determine the mesoscale. Requires no
  forecast.
- **r12_b (short-lead relative error, co-primary).** As in Phase 9: relative
  error of the 6-member ensemble-mean 12 h forecast against the forecasting
  system's own analysis.

Headline values are the mean over bands 2 and 3, then the median over the
two windows of the domain.

**Predicted direction, fixed here: A up => D up and r12 up.**

## Hypotheses and criteria (fixed)

- **C10-1 sanity.** `D` and `r12` finite and strictly inside (0, 1) in every
  region-window, `D` built from at least 150 matched analysis times per
  region-window. Failure is a pipeline fault, not a verdict.
- **H10a.** Spearman(A, D) over the 27 new domains, one-sided positive,
  permutation p < 0.05 (999).
- **H10b.** Spearman(A, r12) over the 27 new domains, same rule.
- **H10c (beyond the standard controls).** LOO linear residualisation of A
  and of each observable on [ |latitude of centre|, land fraction,
  log band variance, isotropic spectral slope ]; residual correlation must
  stay positive with permutation p < 0.05.
- **H10d (observing-density control, the decisive one).** Add the region-mean
  resolved-band ERA5 EDA spread to the control set and repeat H10c. Declared
  in advance as conservative: EDA spread and `A` may share genuine
  atmospheric signal, so this test can over-control. Declared in advance as
  decisive in one direction: **if the association dies here, the honest
  conclusion is that `A` and the assimilation system's own uncertainty field
  carry the same information**, and that is a negative for the claim that `A`
  is a distinct invariant, whatever H10a shows.
- **H10e (within-region, fixed observing geography).** Over the 39 domains x
  2 seasonal windows, `observable ~ A + domain fixed effects`; within-domain
  slope positive, permutation p < 0.05 with window labels shuffled inside
  each domain. Immune to every static geographic confound.
- **H10f (placebo).** The same tests with log band variance and with the
  spectral slope in place of `A`. A claim for `A` requires
  |rho_A| > |rho_placebo| for both observables.
- **H10g (pooled).** H10a and H10b repeated on all 39 domains. Reported, not
  a criterion, since 12 of the 39 are not independent of Phase 9.

Reported without criterion weight: `A` vs `r12` on the Phase-9 2021-2022
sample (8 regions), which has never been examined for this relation.

## Verdict rule (fixed)

- **CONFIRMED**: C10-1 passes; H10a and H10b pass; H10c passes for both
  observables; H10f shows `A` beating both placebos; H10d does not overturn.
- **PARTIAL**: C10-1 passes; at least one of H10a/H10b passes with its H10c,
  and H10f is satisfied for that observable.
- **NEGATIVE**: H10a and H10b both fail, **or** H10d shows the association is
  not separable from the assimilation system's own uncertainty estimate,
  **or** a placebo matches `A`.

## Compute plan

- Domains: `clean_experiments/regions_b10.py`.
- ERA5: `clean_experiments/download_b10_era5.py` -> `data/b10era5/`,
  EDA spread -> `data/b10eda/`.
- GEFS: `clean_experiments/download_b9_gefs.py` extended to the Phase-10
  domain set -> `data/b9gefs/b10/`.
- Experiment: `clean_experiments/experiment_B10_irrecoverability.py`
  -> `clean_experiments/results/experiment_B10_irrecoverability/`.
- Seeds fixed: 20260815.

## Deviations

- **2026-08-15, before any Phase-10 hypothesis was computed** (pipeline check
  on the 12 existing programme domains only, which carry no criterion weight
  in Phase 10): `D` as frozen was normalised by `sqrt(2)*sigma_ERA5`, and in
  the 200-400 km band over the moist tropics it exceeded 1. The cause is an
  asymmetry of my own making: ERA5 is anti-alias smoothed on the way from
  0.25 to 0.5 deg while the GEFS analysis is native 0.5 deg, so ERA5 carries
  slightly less band-2 variance and the denominator is biased low. Remedy:
  the saturation scale becomes the **geometric mean of the two analyses'
  band amplitudes**, `sqrt(2)*sqrt(sigma_ERA5 * sigma_GDAS)`, so that neither
  system's own smoothing sets the norm. Correspondingly, the C10-1 bound
  "strictly inside (0, 1)" — which was an arbitrary tightening, since two
  analyses may legitimately disagree by more than the saturation value — is
  replaced by "finite and positive", with the observed range of `D` and
  `r12` logged instead of silently gated. Both changes are applied uniformly
  to every domain and cannot favour any subset. No Phase-10 hypothesis had
  been computed at the time of this change.

- **2026-08-15, H10d not reached.** The ERA5 EDA spread download was still in
  progress when the battery was scored. H10d is moot: it was declared decisive
  only in the direction of *killing* an association, and the association is
  already dead one control stage earlier (H10c) and beaten outright by the
  spectral placebo (H10f). Adding a further control cannot change a NEGATIVE
  verdict. Logged rather than quietly dropped.
