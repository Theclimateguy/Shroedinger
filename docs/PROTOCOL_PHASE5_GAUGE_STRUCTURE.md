# Phase 5 Protocol (preregistered): Is the scale connection a genuine gauge structure with a physical source?

Status: FROZEN before any Phase-5 computation. Date frozen: 2026-08-12.
Deviations must be logged with timestamp and reason.

## Questions

Phase 4 confirmed the curvature-strength profile as a physically predictable
regional invariant. Phase 5 tests four escalating claims:

- 5c INTEGRABILITY: do the estimated transfer maps behave like a consistent
  connection (path-consistency beyond a spectrum-matched null), i.e. is
  "geometry" more than vocabulary?
- 5b UNIVERSALITY (law-as-collapse): does one covariate scaling collapse all
  regional curvature profiles onto a universal curve?
- 5a SOURCE LOCALIZATION: does localized curvature track a physical source
  field (precipitation) in space, with a well-defined "center of mass"?
- 5d OBSERVER INVARIANCE: does the structure replicate at 500 hPa?

## Quantities and criteria (fixed)

### 5c Integrability (data: the 48 Phase-4 confirmation region-windows)

Per band pair b->b+1 and time t: scale-step maps M_fwd/M_rev as in A15;
within-band time-step maps T_b(t) by ridge regression of a_b(t'+1) on a_b(t')
over the same rolling window (W=20, ridge 1e-6). Path discrepancy
  d_{b,t} = ||T_{b+1}(t) M_fwd(t) - M_fwd(t+1) T_b(t)||_F /
            (||T_{b+1}(t) M_fwd(t)||_F + ||M_fwd(t+1) T_b(t)||_F).
Null: 99 shared-phase surrogates (same generator as Phase 2b), full rerun.
- C5c: real median_t d below the surrogate 5th percentile in >= 3 of 5 band
  pairs, in >= 32 of 48 region-windows (real dynamics closer to an
  integrable connection than the phase-null).

### 5b Universality (data: Phase-4 profiles + Phase-3 E2 profiles, covariates on disk)

Pooled model over all region-windows and bands:
  log Fnorm_{rw,b} = alpha*log(1+cape_mean_r) + beta*land_frac_r +
                     gamma*log(1+orog_std_r) + delta_b (band offsets).
- C5b: PASS iff (i) model R^2 >= 0.5 on pooled profiles AND (ii) after
  subtracting the fit, the within/between region clustering diff of residual
  profiles falls below 0.5x the raw-profile diff (both diffs from the
  standard permutation machinery; residual clustering may remain significant
  - the criterion is the SHRINKAGE, not the p-value).

### 5a Source localization (data: b2b region-windows, R5-R12 x W5-W8;
new download: ERA5 6-hourly total_precipitation for the same boxes)

Tile each domain 3x4 (27x40 px tiles); per tile: curvature machinery with
scale edges [50,100,200,400] km, 4 modes/var, W=20; tile fine-band
Fnorm_tile(t) = mean over band pairs. Precip_tile(t) = tile-mean total
precipitation. 
- C5a: Spearman over (tile,t) of log Fnorm_tile vs log(precip_tile+1e-6),
  block-permutation p < 0.05 (999, blocks of 20 time steps, permuting time
  blocks of the precip series), positive rho, in >= 22 of 32 region-windows.
- Descriptive: curvature centroid vs precip centroid trajectories,
  zero-lag and +-1..4 step lag correlations.

### 5d Observer invariance (new download: 500 hPa u,v for 24 region-windows:
R1-R4 x {W3,W4}, R5-R12 x {W7,W8})

Full Fnorm profile pipeline at 500 hPa.
- C5d: PASS iff (i) within/between clustering of 500 hPa Fnorm profiles:
  diff > 0, p < 0.05; AND (ii) Spearman across the 12 regions between
  region-median fine-band log Fnorm at 850 vs 500 hPa > 0 with p < 0.05.

## Verdict (fixed mapping)

Each sub-test reports PASS/FAIL independently. Summary label:
- GAUGE_PHYSICAL: 5c and 5a and 5d pass (integrable, sourced, level-invariant),
  regardless of 5b.
- STRUCTURED: at least two of {5c, 5a, 5d} pass.
- VOCABULARY: fewer than two pass (the geometric reading stays a vocabulary).
5b PASS upgrades any label with suffix "+LAW" (universal collapse found).

## Compute plan

- Downloads: `download_b5_precip.py` (32 files -> data/b5precip/),
  `download_b5_wind500.py` (24 files -> data/b5w500/).
- Experiments: `experiment_B5c_integrability.py`, `experiment_B5b_collapse.py`,
  `experiment_B5a_source.py`, `experiment_B5d_level500.py`
  -> results under `clean_experiments/results/experiment_B5_gauge_structure/`.
- Seeds fixed: 20260811.

## Deviations

- (none yet)
