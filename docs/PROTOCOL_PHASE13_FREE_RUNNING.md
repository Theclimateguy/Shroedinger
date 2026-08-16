# Phase 13 Protocol (preregistered): the free-running, no-assimilation control

Status: FROZEN before any Phase-13 quantity was computed. Date frozen:
2026-08-16. Deviations logged with timestamp at the bottom.

## Why this test exists

The manuscript names one limitation as decisive and unresolved: reproducing
a descriptor in two reanalyses (Phase 8, ERA5 vs MERRA-2, rho = 0.93) rules
out a model-specific artefact but **not** an artefact of the observing network,
because both systems assimilate largely the same observations. The manuscript
itself states that removing this limitation "would require computing the
descriptors from free-running (non-assimilating) model experiments at high
resolution". This is that experiment.

Phase 12 changed what is at stake. `A` was shown not to measure
irrecoverability at all, and to track the temporal persistence of the field
instead; `P` was shown to agree with an independent, amplitude-invariant
information measure. So Phase 13 is no longer "is the irrecoverability
geography atmospheric" but:

> Do the regional geographies of `P`, of `A`, of the information measure `u`
> and of persistence survive in an atmosphere that has never seen an
> observation?

`P` is the primary object, `A` secondary, `u` and persistence diagnostic.

## Data (fixed)

- **Free-running**: CMIP6 HighResMIP `highresSST-present`, ECMWF-IFS-HR,
  variant r1i1p1f1, table 6hrPlevPt, variables `ua`, `va` at 85000 Pa,
  regridded output `gr` (0.5 deg, 361 x 720), 6-hourly. Windows
  W15_2014JFM and W16_2014JAS. This is an AMIP-type integration: prescribed
  sea surface temperature, no atmospheric data assimilation of any kind.
  Same modelling centre and dynamical core lineage as ERA5's IFS, which
  makes the comparison a control on *assimilation*, not on the model family.
- **Reference**: the ERA5 files already on disk, block-coarsened to the same
  0.5 deg grid so that resolution cannot masquerade as assimilation
  (`data/b3` and `data/b10era5`; `data/b2b` where available).
- Domains: the 39 of the programme, unchanged.

Two declared limitations, stated before the result:

1. The free-running years (2014) differ from the ERA5 windows on disk
   (2021-2024). The programme's own validated result is that these
   descriptors are stable across years and ENSO phase (Phases 2b, 3, 8), and
   a free-running AMIP run has no meteorological correspondence to any
   particular year anyway, so only the *climatological regional geography*
   is comparable. Where ERA5 covers more than one period for a domain, the
   ERA5-to-ERA5 correlation across periods is reported as the ceiling this
   test could possibly reach.
2. A single model. A failure could be an ECMWF-IFS-HR bias rather than an
   assimilation effect. CMCC-CM2-VHR4 is available on the same node and is
   declared here as the follow-up if the primary result is negative.

## Quantities (fixed)

All four are computed with the machinery already frozen in earlier phases,
on the common 0.5 deg grid, resolved bands 200-800 km:

- `P` — profile of rank correlations between adjacent envelopes (Phase 2).
- `A` — commutator norm of the transfer operators (Phase 4).
- `u` — 1 minus normalised rank mutual information between adjacent
  envelopes (Phase 12).
- `persist` — lag-1 (6 h) autocorrelation of the resolved-band envelope
  (Phase 12 diagnostic), included because Phase 12 identified it as what `A`
  actually responds to.

Domain value: median over the two seasonal windows.

## Hypotheses and criteria (fixed)

- **C13-1 sanity.** Every domain-window built from at least 300 time steps;
  all four quantities finite; the model's band variances within a factor of
  three of ERA5's in the median (a gross-bias guard, not a tuning knob).
- **H13a (primary, `P`).** Spearman between the free-running and the ERA5
  regional values of `P` over the 39 domains > 0, permutation p < 0.05 (999).
- **H13b (`A`).** The same for `A`.
- **H13c (`u`).** The same for `u`.
- **H13d (persistence).** The same for `persist`.

Reported without criterion weight: the ERA5-to-ERA5 across-period ceiling,
and the rank ordering of the four correlations.

## Verdict rule (fixed)

- **ATMOSPHERIC**: H13a passes. The geography of `P` is a property of the
  atmosphere, not of the observing network, and the manuscript's declared
  decisive limitation is closed for `P`. The status of `A` follows H13b
  independently.
- **OBSERVING-SYSTEM ARTEFACT**: H13a fails while the ERA5-to-ERA5 ceiling is
  high. The descriptor then describes where assimilation has nothing to fill
  the mesoscale with — a real result, but about the observing system rather
  than the atmosphere, and the manuscript must say so.
- **INCONCLUSIVE**: H13a fails and the ceiling is also low, i.e. the
  comparison lacks the power to decide. The follow-up model is then required.

## Compute plan

- Downloader: `clean_experiments/download_b13_highresmip.py` -> `data/b13hrmip/`.
- Experiment: `clean_experiments/experiment_B13_free_running.py`
  -> `clean_experiments/results/experiment_B13_free_running/`.
- Seeds fixed: 20260816.

## Deviations

- (none yet)
