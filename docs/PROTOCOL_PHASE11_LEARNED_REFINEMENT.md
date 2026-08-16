# Phase 11 Protocol (preregistered): A and the mesoscale deficit of learned forecast models

Status: FROZEN before any Phase-11 forecast field was retrieved or any
Phase-11 quantity computed. Date frozen: 2026-08-15. Deviations logged with
timestamp at the bottom.

## Standing decision, fixed here

Phase 9 falsified the rate reading of `A` (error growth). Phase 10 falsified
the level reading (analysis-time irrecoverability) on 27 fresh domains, and
in both cases the plain band variance of the vorticity field beat `A`
outright.

**Phase 11 is the third and last target pursued for `A`.** If it fails, the
programme's conclusion is fixed in advance: `A` is a stable, reproducible
regional invariant *with no demonstrated applied content*, and the manuscript
states exactly that. No fourth target will be sought after seeing this
result.

## The prediction, derived rather than borrowed

1. A deterministic forecast model trained on an L2 objective converges to the
   conditional mean `E[x_t | x_0]`.
2. A conditional mean discards precisely the component of the mesoscale that
   the conditioning information does not determine. This shows up as a
   deficit of variance at small scales — the well-known blurring of
   MSE-trained weather models.
3. `A_b = || G^up G^down - G^down G^up ||_F` measures the failure of
   aggregation and refinement to commute: the size of the component of
   mesoscale placement that is not recoverable from the coarse field.
4. Therefore the **mesoscale variance deficit of a deterministic learned
   model, at 200-800 km, should grow with `A`.**

Unlike Phases 9 and 10, this design carries its own discriminators. The same
deficit must **not** track `A` in systems that do not compute a conditional
mean:

| system | mechanism | deficit predicted |
|---|---|---|
| GraphCast, Pangu | deterministic, L2-trained | yes, growing with `A` |
| GenCast ensemble mean | conditional mean of a generative model | yes |
| GenCast single member | generative sample, full variance by design | no |
| HRES | physical integration, no L2 objective | no |

If the deficit tracks `A` everywhere, including HRES, the mechanism is wrong
and the result is an artefact of the atmosphere or of the normalisation. That
is a falsification route Phases 9 and 10 did not have.

## Data (fixed)

WeatherBench 2 public bucket (`https://storage.googleapis.com/weatherbench2`,
anonymous), all systems on the identical 0.25 deg grid and identical
initialisation times, 850 hPa `u`, `v`:

- `graphcast_v2/2020-1440x721.zarr`
- `pangu/2018-2022_0012_0p25.zarr`
- `gencast/2020-1440x721_mean.zarr` (ensemble mean)
- `gencast/2020-1440x721.zarr` (single member, index 0)
- `hres/2016-2022-0012-1440x721.zarr`
- truth: `era5/1959-2023_01_10-wb13-6h-1440x721.zarr`

Initialisations: 00 UTC, every 10th day inside 2020-01-01..2020-03-31 and
2020-07-01..2020-09-30 (8 per window, 16 total). Lead times: **24 h and
120 h**, both present in every system including the 12 h-stepped GenCast.
GenCast single member is retrieved for the first 6 initialisations of each
window only (its chunks carry all 56 samples and are an order of magnitude
larger); this reduction is fixed here, before any result.

Domains: the 39 of the programme (12 original + the 27 Phase-10 lattice
domains), unchanged.

`A` is **not recomputed for 2020**. The programme's own validated result is
that `A` is a regional invariant stable across seasons, years and ENSO phase
(Phases 2b, 3, 8); the domain-level `A` already computed is therefore the
descriptor, and using it is what that invariance claim licenses. Recomputing
`A` on 2020 would test invariance, not this hypothesis.

## Quantities (fixed)

Band-passing, vorticity, interior mask and bands are identical to Phases 9
and 10, applied on the native 0.25 deg grid; confirmatory bands b = 2
(200-400 km) and b = 3 (400-800 km), b = 4 reported.

- `S_b(model, tau)` = ratio of the band variance of the model's vorticity
  field to the band variance of ERA5 at the same valid time, averaged over
  initialisations, per domain.
- **`Delta_b(model, tau) = 1 - S_b`**, the mesoscale variance deficit.
  Primary at `tau = 120 h`, secondary at `tau = 24 h`.

Headline value: mean of `Delta_b` over bands 2 and 3, median over the two
seasonal windows.

**Predicted direction, fixed here: A up => Delta up, for deterministic
learned models only.**

## Hypotheses and criteria (fixed)

- **C11-1 sanity.** `Delta > 0` at 120 h for GraphCast and Pangu in at least
  90 % of domains (the blurring must exist at all before its geography is
  tested); `|Delta|` for HRES smaller in the median than for GraphCast.
  Failure is a pipeline fault, not a verdict.
- **H11a.** Spearman(`A`, `Delta_GraphCast(120 h)`) over the 39 domains,
  one-sided positive, permutation p < 0.05 (999).
- **H11b.** The same for Pangu.
- **H11c (beyond controls).** LOO residualisation of `A` and of `Delta` on
  [ |latitude of centre|, land fraction, log band variance, spectral slope ];
  residual correlation positive with permutation p < 0.05, for both models.
- **H11d (mechanism discriminator).** `rho(A, Delta)` must be larger for the
  deterministic systems than for HRES and for the GenCast single member, and
  neither control may itself satisfy the H11a criterion. GenCast mean is
  predicted to behave like the deterministic systems and is reported as the
  positive side of the same contrast.
- **H11e (within-domain).** Seasonal arm over 39 domains x 2 windows with
  domain fixed effects; positive within-domain slope, permutation p < 0.05.
- **H11f (placebo).** log band variance and spectral slope in place of `A`;
  `|rho_A|` must exceed both, for both deterministic models.

## Verdict rule (fixed)

- **CONFIRMED**: C11-1 passes; H11a and H11b pass; H11c passes for both;
  H11d holds; H11f shows `A` beating both placebos.
- **PARTIAL**: C11-1 passes; one deterministic model passes H11a/H11c with
  H11d and H11f satisfied for it.
- **NEGATIVE**: H11a and H11b both fail, **or** a placebo matches `A`,
  **or** H11d fails (the deficit tracks `A` in HRES too, i.e. the learned-
  mapping mechanism is not what produces the pattern).

On NEGATIVE the standing decision above takes effect without further tests.

## Compute plan

- Downloader: `clean_experiments/download_b11_wb2.py` -> `data/b11wb2/`
  (per system/init/lead domain subsets; global chunks discarded immediately).
- Experiment: `clean_experiments/experiment_B11_learned_refinement.py`
  -> `clean_experiments/results/experiment_B11_learned_refinement/`.
- Seeds fixed: 20260815.

## Deviations

- (none yet)
