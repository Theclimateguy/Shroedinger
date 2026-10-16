# PROTOCOL — Phase 41: the enstrophy budget of the fine band against the geography of P (B41)

## Status: FROZEN 2026-10-16, before any Phase-41 quantity was computed on ERA5

Opened by the author on 2026-10-16 ("compute on Spark what the card proposed") right
after Phase 40. Written and frozen in one step because the computation was ordered in
the same message; the author's priors were not stated and are recorded as absent.
Code: `clean_experiments/experiment_B41_enstrophy_budget.py`, smoke-tested on synthetic
fields only.

## Purpose

Phase 40 (CLARK_ONLY) refuted the ANGULAR reading of P: the alignment between the
subgrid stress and the coarse strain, and the share of strain-dominated (LKH) pixels,
are unrelated to the geography of P. The AMPLITUDE reading was not tested there. P is a
statistic of band amplitudes (the copula of adjacent-band envelopes); its geography may
be set by what produces the variance of the fine band: inheritance from the coarse
band through the cascade (the vorticity gradient stretched by the coarse strain — the
Clark term that Phase 40 confirmed at the pixel level) versus generation by divergent
motions (vortex stretching −(zeta + f) delta: convection, gravity waves, ageostrophic
circulations) and other three-dimensional sources. The toy model T8 showed that
independent small-scale sources lower P from 0.20 to 0.07; Phase 20 showed that CAPE
lowers P beyond the spectrum. This phase measures the budget directly.

Difference from Phase 40 H40-3 (iii): f_iso there was the divergent share of the
ENERGY flux at the band boundary (the piston term of Sichani et al. 2026). Here the
quantities are SOURCES of the fine band's ENSTROPHY, the carrier of P.

## Data and fields (fixed, identical to Phases 39-40)

ERA5 850 hPa u, v, 6-hourly, JFM and JAS of 2022 (fresh), 2023, 2024; 70S-70N computed,
statistics on the 66 Phase-34 windows (60S-60N, cos-latitude weights, V12
below-surface mask); every 2nd step (12-hourly); Phase-34 Gaussian smoother.

For each band boundary ell in {400, 800} km with the fine scale ell' = ell/2 (bands
B1 = 200-400 and B2 = 400-800 km):

- zeta: V12 relative vorticity; delta: horizontal divergence on the sphere
  (1/(a cos phi)) [d u/d lambda + d(v cos phi)/d phi]; f = 2 Omega sin phi.
- Fine band: omega_f = bar_ell'(zeta) - bar_ell(zeta); band enstrophy V = <omega_f^2>.
- Cascade inflow: Z_ell = -tau^omega_ell . grad(bar_ell zeta) (Phase 40), positive =
  enstrophy passing downscale across ell into the band; outflow Z_ell' across ell'.
- Divergent generation projected on the band: G = (zeta + f) delta, B[G] = bar_ell'(G) -
  bar_ell(G); S_div = <omega_f (-B[G])> (signed) and A_div = <|omega_f B[G]|> (magnitude
  of divergent forcing of the band, regardless of sign); planetary and nonlinear parts
  reported separately (f delta and zeta delta).
- Window-level rates (1/s): r_casc = <Z_ell> / V (net cascade inflow per unit band
  enstrophy); r_out = <Z_ell'> / V; r_div = A_div / V; share_div = r_div / (r_div +
  |r_casc|).

All window means are cos-latitude weighted and pooled over the retained steps of a
season; year value = mean of JFM and JAS; main sample = mean of 2023-24; fresh = 2022.

## Hypotheses and bars (frozen)

- **H41-1 (cascade production carries P).** Spearman across the 66 windows between
  r_casc (2023-24) and P45 (Phase 34, mean 2023-24) >= +0.40; with the Phase-36 residual
  P45_res >= +0.30; fresh 2022 (r_casc_2022 vs P45_2022) >= +0.30. Pass = all three,
  for both ell.
- **H41-2 (divergent forcing opposes P).** Spearman between r_div and P45 <= -0.40;
  with P45_res <= -0.30; fresh <= -0.30. Pass = all three, for both ell. Reported: the
  same for share_div and for the planetary (f delta) and nonlinear (zeta delta) parts.
- **H41-3 (reported, not scored).** r_casc, r_div, share_div by Phase-34 region type;
  Spearman of r_div with the Phase-20 CAPE covariate of the windows if available;
  signed S_div by window; the ratio of outflow to inflow r_out / r_casc.

Nulls: the Phase-36 spectral-null residual column (P45_res) is the control for H41-1 and
H41-2; the fresh year is the replication. No lag control is needed (no regression on
co-located envelopes).

## Verdict ladder

- **BUDGET_EXPLAINS:** H41-1 and H41-2 pass.
- **CASCADE_ONLY:** H41-1 passes, H41-2 fails.
- **DIVERGENCE_ONLY:** H41-2 passes, H41-1 fails.
- **NEGATIVE:** neither.

## Priors (recorded before computing)

Author: not stated. Analyst (Claude): H41-1 0.45; H41-2 0.50.

## Code and outputs

`clean_experiments/experiment_B41_enstrophy_budget.py`, stages `season --year Y
--season S` (single pass, per-window accumulators), `tests`, `smoke`. Outputs →
`clean_experiments/results/experiment_B41_enstrophy_budget/`. Seed 20261031. The
SHA-256 of this file is written into the summary at `tests` time.

## Deviations

- (none yet)

## Result (2026-10-16)

Results: `clean_experiments/results/experiment_B41_enstrophy_budget/` (six season files,
Spark host; `summary.json` with the protocol SHA-256). No deviations.

**Verdict: NEGATIVE.**

| hypothesis | result | Spearman with P45 / with Phase-36 residual / fresh 2022 (400 km; 800 km) |
|---|---|---|
| H41-1 cascade production rate r_casc carries P (expected +) | FAIL | +0.26 / +0.19 / +0.26; +0.21 / +0.16 / +0.19 — right sign, below every bar |
| H41-2 divergent forcing rate r_div opposes P (expected −) | FAIL | +0.20 / +0.14 / +0.20; +0.10 / +0.08 / +0.10 — wrong sign |
| reported share_div (expected −) | — | −0.29 / −0.22 / −0.28; −0.26 / −0.22 / −0.26 — right sign, below bars |
| reported planetary part r_f (expected −) | — | −0.13 / −0.17 / −0.11; −0.21 / −0.23 / −0.20 |
| reported nonlinear part r_nl (expected −) | — | +0.35 / +0.34 / +0.35; +0.23 / +0.24 / +0.26 — opposite sign: zeta·delta forcing is largest where vorticity is largest (storm tracks), i.e. it follows the energy, as P does through the spectrum |

Budget facts (median over windows, 2023-24): r_casc 4e-6 s^-1 (400 km), 2e-6 (800 km);
r_div 3e-5 / 2.3e-5 s^-1; share_div 0.86 / 0.91; outflow/inflow r_out/r_casc 2.3 / 3.1;
net divergent generation S_div > 0 in 100 % / 97 % of windows; net cascade inflow Z_in >
0 in 98 % / 97 %; planetary over nonlinear stretching r_f/r_nl 1.18 / 1.22. By Phase-34
region type share_div ranges only 0.80-0.88 (400 km): the composition of the fine-band
enstrophy sources is nearly uniform across regions.

Reading. The fine band at 200-800 km on 850 hPa is forced mainly by divergent vortex
stretching (planetary and nonlinear in comparable measure), with the downscale cascade
a net positive but smaller contributor, and it loses more to smaller scales than it
receives from larger ones. This composition is almost the same everywhere, so it
cannot carry the geography of P. The amplitude reading (inheritance share set by the
source composition) fails at the window level as the angular reading failed in Phase
40. Priors (analyst) 0.45 / 0.50 -> miss / miss.
