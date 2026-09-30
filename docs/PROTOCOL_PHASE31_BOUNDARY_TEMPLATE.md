# PROTOCOL — Phase 31: the boundary template of inter-level coupling
## Status: FROZEN 2026-09-30, before the 2024 wind fields were downloaded or read

Motivation. Exploration T15 (docs/EXPLORATION_PHYSICS_2026-09-29.md, sections
11-12; ERA5 2023) split the spatial covariance of band log-envelopes into a
stationary template and a moving part, without surrogates, and found:
a season-stable template share tied to relief and coast; one template for all
levels; no place dependence over open ocean; and, post hoc, a compensation
(total coupling over land independent of the template share). This phase
tests those statements on an independent year.

## Data (downloaded after freezing)
ERA5 850 hPa u, v, 6-hourly, global 0.25 deg, 2024-01..03 and 2024-07..09
(`clean_experiments/download_b31_wind2024.py` -> data/b20global/).
On disk: the same fields for 2023; monthly surface pressure (data/xp_physics).

## Definitions (fixed)
Tile grid, orography exclusion and km-based filters as Phase 20 Arm B
(902 tiles). Isotropic grid: each tile resampled to a zonal step equal to the
meridional one in km. Bands b1 = 50-100, b2 = 100-200, b3 = 200-400 km;
log-envelope E_b(x, t) as in Phase 2. No surrogates.
Per tile and season (JFM, JAS), with the tile-mean removed at each step:
- C_tot(i, j) = time mean of the spatial covariance of E_i, E_j;
- C_tpl(i, j) = spatial covariance of the time-mean maps, cross-validated
  between two interleaved sets of 5-day blocks (symmetrised);
- C_mov = C_tot - C_tpl;
- template share s = mean over the three bands of C_tpl(i, i) / C_tot(i, i);
- r_tot, r_mov, r_tpl(i, j) = C(i, j) / sqrt(C(i, i) C(j, j)) of the
  respective matrix; r_tpl defined where C_tpl(i, i) > 0.02 C_tot(i, i).
Year value = mean of the two seasons. Land: land fraction >= 0.7; ocean:
<= 0.05. Primary pair for coupling: the non-adjacent b1|b3.

## Hypotheses and bars
- H31-1 (invariant, PRIMARY). Spearman(s_2023, s_2024) over land tiles
  >= 0.70, and over all tiles >= 0.60; longitude-rotation null (999) p < 0.01.
- H31-2 (compensation). Land tiles classed by the 2023 share
  (< 0.03, 0.03-0.15, > 0.15); coupling taken from 2024, pair b1|b3:
  (a) |r_tot(high) - r_tot(low)| <= 0.02;
  (b) r_mov(low) - r_mov(high) >= 0.04;
  (c) Spearman(r_tot_2024, s_2023) over land within [-0.2, +0.2] and
      Spearman(r_mov_2024, s_2023) <= -0.30.
  PASS = (a) and (b) and (c).
- H31-3 (no geography without a boundary). Spearman(r_tot_2023, r_tot_2024),
  pair b1|b3: ocean tiles < 0.20; land tiles > 0.40.
- H31-4 (one template for all levels). 2024: median r_tpl over land tiles
  where defined >= 0.70 for both adjacent pairs and >= 0.50 for b1|b3.
- H31-5 (not the below-ground extrapolation). Land tiles with no grid point
  whose climatological surface pressure is below 850 hPa, 2024:
  median s >= 3 x the ocean median, and Spearman(s, orog_std) >= +0.30.

## Verdict ladder
BOUNDARY_TEMPLATE_CONFIRMED: H31-1, H31-2, H31-3 pass.
TEMPLATE_INVARIANT_ONLY: H31-1 passes, H31-2 fails.
NEGATIVE: H31-1 fails.
H31-4 and H31-5 are reported as pass/fail and qualify the wording.

## Priors (recorded before computation)
H31-1 0.85; H31-2 0.5 (found post hoc on one year); H31-3 0.8; H31-4 0.8;
H31-5 0.6.

## Deviation from the proposal made to the author
The proposal named a second level (925 or 700 hPa) to separate relief from
the extrapolation. It is replaced by H31-5 (tiles free of below-ground
points), which needs no further download and addresses the same question
directly. Logged before computation.
