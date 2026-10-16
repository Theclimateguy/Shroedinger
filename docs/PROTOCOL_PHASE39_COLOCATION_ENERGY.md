# PROTOCOL — Phase 39: co-location and energy (B39)

## Status: FROZEN 2026-10-16, before any Phase-39 quantity was computed

Purpose: a physical reading of co-location for article 2, in two parts. (1) Locality of
the scale-to-scale energy transfer: scales exchange energy where they are present
together, so the coarse-graining flux should concentrate where the activity of the two
adjacent bands is co-located, beyond what the activity of each band gives separately.
(2) The stationary template as stationary-eddy energy: windows where band activity is
held in place should be windows where a larger share of band kinetic energy is
stationary. The earlier flux test (B1; article 1, table 3) asked whether the profile
tracks the flux as a proxy; this phase asks a different question — whether the flux is
located where activity is co-located. Light protocol: two scored hypotheses, the rest
reported.

## Data and fields (fixed)

ERA5 850 hPa u, v, 6-hourly, global, JFM and JAS of 2022, 2023, 2024 (2022 = fresh year);
every 2nd step (12-hourly); 70S-70N computed, statistics on the 66 Phase-34 windows
(60S-60N, cos-latitude weights, V12 below-surface mask). Gaussian filters km-based,
periodic in longitude, normalised by the valid mask (Phase-34 smoother).

- Envelopes: V12 vorticity, bands B1 = 200-400, B2 = 400-800, B3 = 800-1600 km, log-
  envelopes E1, E2, E3 (Phase 34).
- Flux: Pi_l = -(tau_xx du/dx + tau_xy (du/dy + dv/dx) + tau_yy dv/dy), tau_ij =
  bar_l(u_i u_j) - bar_l(u_i) bar_l(u_j), gradients of the filtered wind on the sphere,
  at l = 400 km (transfer across the B1|B2 boundary) and l = 800 km (B2|B3).
- Band kinetic energy for 200-1600 km: u_b = bar_200(u) - bar_1600(u), likewise v_b.
  Per window and season: stationary part K_st = window mean of 1/2 (<u_b>^2 + <v_b>^2)
  (time means per pixel), total K = window mean of <1/2 (u_b^2 + v_b^2)>; stationary
  share sigma = K_st / K.

## Hypotheses and bars

- **H39-1 (energy transfer is located where activity is co-located, PRIMARY).** Per
  window and season, weighted least squares pooled over steps, all variables
  standardised within the window at each step:
  log|Pi_l| = b0 + b1 z_k + b2 z_{k+1} + b3 z_k z_{k+1}, with (k, k+1) = (E1, E2) for
  l = 400 and (E2, E3) for l = 800. The interaction b3 measures co-location beyond the
  separate activity of the two bands. Transient control: the same regression with the
  envelopes taken 10 steps (5 days) later in the same season, b3_lag; the transient
  locality is d = b3 - b3_lag. Year value = mean of seasons. Bars, for both l, mean of
  2023-24: median over windows of d > 0 with sign-flip p < 0.01 and d > 0 in >= 75 % of
  windows; fresh 2022: median d > 0 with p < 0.01.
- **H39-2 (the stationary template is stationary-eddy energy).** Across windows, mean
  of 2023-24: Spearman(s, sigma) >= 0.50, with s the Phase-34 template share, and the
  partial Spearman controlling for abs latitude >= 0.30; fresh 2022: Spearman(s_2022,
  sigma_2022) >= 0.50.
- **H39-3 (reported).** b3 and d by window type (Phase-34 regions); concentration ratio
  (mean |Pi| where both bands are above their window median over mean |Pi| where exactly
  one is); window-mean signed Pi (direction of transfer) and |Pi| per unit band energy
  against the Phase-36 residual co-location; sigma by region type; maps.

## Verdict

- **ENERGY_LOCALITY_AND_STATIONARY_ENERGY:** H39-1 and H39-2 pass.
- **ENERGY_LOCALITY_ONLY** / **STATIONARY_ENERGY_ONLY:** one passes.
- **NEGATIVE:** neither.

## Priors (recorded before computation)

H39-1 0.6; H39-2 0.6.

## Code and outputs

`clean_experiments/experiment_B39_colocation_energy.py` (stages `season --year Y
--season S`, `tests`) → `clean_experiments/results/experiment_B39_colocation_energy/`.
Seed 20261020.

## Deviations

- (none yet)

## Result (2026-10-16)

Results: `clean_experiments/results/experiment_B39_colocation_energy/`. No deviations.
Flux code checked against B1 on a North Pacific box (identical to three decimals).

**Verdict: ENERGY_LOCALITY_AND_STATIONARY_ENERGY.**

| hypothesis | result | numbers |
|---|---|---|
| H39-1 transfer located where activity is co-located | PASS | interaction b3 (standardised): 400 km +0.058 (5 days later +0.002), 800 km +0.032 (-0.005); transient locality d > 0 in 98 % and 100 % of windows, p = 1e-4; fresh 2022 +0.051 / +0.040, p = 1e-4 |
| H39-2 stationary template = stationary-eddy energy | PASS | Spearman(template share, stationary share of 200-1600 km kinetic energy) 0.72; partial for latitude 0.41; fresh 2022 0.60 |

Reported. Mean |Pi| where both adjacent bands are above their window median is 3.1
(400 km) and 2.4 (800 km) times the mean where only one is. Net transfer is downscale
(forward) at 400 km in 95 % of windows, at 800 km in 67 %. Stationary share of band
energy (median 0.08) by Phase-34 region type: monsoon / convergence 0.20, subtropical
highs 0.12, midlatitude continents and subtropical oceans 0.11, storm tracks 0.06.
Net downscale flux (1e-6 m²/s³, 400 / 800 km): monsoon 13.6 / 10.3, storm tracks
13.3 / 5.3, subtropical highs 5.4 / 1.6, continents and subtropical oceans 8.7 / 3.3.
The Phase-36 residual co-location relates moderately to the transient locality at
800 km (0.39) and to the net flux (0.24-0.30), weakly to transfer efficiency (0.17-0.25).

Reading. Energy passes between adjacent scales where their activity is co-located at
that moment, not merely where either is active — a direct physical meaning for the
co-location measure (locality of the transfer). The stationary template is the share of
band energy held by stationary eddies: largest in monsoon and convergence zones,
smallest in the storm tracks. The two together give article 2 its physical layer: the
regions differ in how much of the co-located, energy-exchanging activity is held in
place.
