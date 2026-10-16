# PROTOCOL — Phase 36: regions on the residual after a spectral null (B36)

## Status: FROZEN 2026-10-16, before any Phase-36 quantity was computed

Trigger. Phase 35 showed that stationary random fields with a field's own spectrum
(mirror surrogates) reproduce the raw adjacent-band coupling excess of forecasts: the
raw profile depends on the within-band spectral shape. Phase 34's regionalisation
(REGIONALISATION_CONFIRMED) is built on the raw profile and the stationary template.
Question of this phase: do the regions survive when the part of each feature that a
spectrum-only field reproduces is removed? If yes, article 2 rests on co-location of
activity; if no, Phase 34 regions describe a spectral geography.

What the null removes and what it does not. A mirror surrogate is a stationary
Gaussian field with the window's spectrum at that time step. It removes everything set
by the spectrum alone (band overlap, within-band shape, the band-integrated variances).
It does not remove spatial non-stationarity of the spectrum within the window — in
particular a smoothly varying dominant wavelength (the frequency-modulation component
of Phase 35, H35-4). The residual is therefore "beyond the window's mean spectrum", not
"pure co-location".

## Data and units (fixed)

As Phase 34: ERA5 850 hPa u, v, JFM and JAS of 2022, 2023, 2024 (2022 = the fresh year
of Phase 34); free-running IFS-HR 2014 (JFM, JAS) with the ERA5 below-surface mask
(Phase 34, D1); the 66 windows; V12 vorticity; bands B1 = 200-400, B2 = 400-800,
B3 = 800-1600 km; log-envelopes as in Phase 34.

## Pipeline (fixed)

Per window and season, every 2nd 6-hourly step: the window plus a 1100-km margin is
mapped to an azimuthal-equidistant plane at the ERA5 grid step (as the Phase-34 null);
vorticity is computed on the sphere and interpolated; below-surface points are masked.
For each step, one mirror surrogate (mirror extension in both axes, random Fourier
phases, original quadrant kept, mask re-applied). On the plane, for real and surrogate
fields alike (unweighted, plane pixels have equal area):
- P4, P5 = median over steps of the Pearson correlation of log-envelopes E1|E2 and E2|E3
  inside the window;
- template share s and moving correlation r_mov(B1|B3) exactly as Phase 34 (5-day
  interleaved blocks over the retained steps).
Residual of each feature X: X_res = X_real - X_null. Year value = mean of the two
seasons.

## Hypotheses and bars

- **H36-1 (a reproducible residual exists, PRIMARY part 1).** R45 = (P4_res + P5_res)/2:
  Spearman(R45_2023, R45_2024) over windows >= 0.60 (belt-shift null, 999, p < 0.01),
  and Spearman(R45_2022, mean of 2023-24) >= 0.60.
- **H36-2 (regions on the residual, PRIMARY part 2).** Features [P4_res, P5_res, s_res,
  r_mov_res], standardised; Ward, k = 4 (k = 3, 5 reported); ARI 2023 vs 2024 >= 0.40
  with within-belt permutation p < 0.05; fresh 2022 assigned to the nearest pooled
  2023-24 centroid: ARI >= 0.40, p < 0.05.
- **H36-3 (how much of Phase 34 was spectral; reported).** Across windows (mean
  2023-24): R² of the raw plane P45 on the null P45; ARI between the residual regions
  and the raw-plane regions; ARI between the raw-plane regions and the Phase-34 regions
  (pipeline check, expected high).
- **H36-4 (free-running model).** Spearman(R45_IFS-HR, R45 mean 2023-24) >= 0.50.

## Verdict ladder

- **RESIDUAL_REGIONS_CONFIRMED:** H36-1 and H36-2 pass. Article 2 regions rest on
  co-location beyond the window's mean spectrum (frequency-modulation caveat kept).
- **RESIDUAL_WITHOUT_REGIONS:** H36-1 passes, H36-2 fails. A reproducible residual
  exists but does not organise into reproducible regions.
- **SPECTRAL_GEOGRAPHY:** H36-1 fails. Phase-34 regions describe where spectra differ.
H36-4 qualifies the text.

## Priors (recorded before computation)

H36-1 0.55; H36-2 0.40; H36-4 0.5.

## Code and outputs

`clean_experiments/experiment_B36_residual_regions.py` (stages `plane --source era5
--year Y --season S`, `plane --source ifshr`, `tests`) →
`clean_experiments/results/experiment_B36_residual_regions/`. Seed 20261017.

## Deviations

- (none yet)

## Result (2026-10-16)

Results: `clean_experiments/results/experiment_B36_residual_regions/` (summary.json,
window_fields.npz, per-season plane JSONs, fig_residual_map_ru.png). No deviations.

**Verdict: RESIDUAL_WITHOUT_REGIONS.**

| hypothesis | result | numbers |
|---|---|---|
| H36-1 reproducible residual | PASS | median P45: real 0.675, spectral null 0.400, residual 0.27 (all three years); residual 2023 vs 2024 0.89 (belt-shift p 0.001), fresh 2022 0.82; per-feature reliability P4 0.89, P5 0.83, s 0.95, r_mov 0.88 |
| H36-2 regions on the residual | FAIL | k = 4: ARI 2023 vs 2024 0.25 (bar 0.40), fresh 2022 to the pooled centroids 0.50 (p 0.001); k = 3: 0.41 / 0.57; k = 5: 0.25 / 0.60 |
| H36-3 how much was spectral | — | R² of raw P45 on the null across windows 0.17; template share of the null 0.000 (real 0.21); ARI residual vs raw-plane regions 0.63; raw-plane regions themselves 2023 vs 2024 0.29 (Phase-34 global pipeline: 0.53); ARI raw-plane vs Phase-34 regions 0.46 |
| H36-4 free-running model | PASS | residual P45 IFS-HR 2014 vs ERA5 0.81 |

Reading. The geography of co-location is mostly not spectral: the window's mean
spectrum explains 17 % of the between-window variance of the raw profile and none of
the template share; the residual is reproduced between years (0.89), on a fresh year
(0.82) and by a free-running model (0.81). What is not robust is the k = 4 partition:
two independent clusterings of 2023 and 2024 agree at 0.25, and so do the raw features
on the same plane pipeline (0.29), so the instability belongs to the partition of a
continuous feature field, not to the residual. Article 2 can rest on continuous maps of
the residual co-location and of the stationary template share; discrete regions only
as a description, with assignment to fixed centroids (fresh year 0.50-0.60) rather than
independent clustering. Caveat kept: the null removes the window's mean spectrum, not
a smoothly varying dominant wavelength (Phase 35, H35-4).
