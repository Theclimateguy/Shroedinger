# PROTOCOL — Phase 40: the zero-memory (Clark) term and strain alignment on the Phase-39 pipeline (B40)

## Status: FROZEN 2026-10-16, before any Phase-40 quantity was computed on ERA5

Drafted the same day as PROTOCOL_PHASE40_ALIGNMENT_DRAFT.md (renamed to this file at
the freeze; the only edits at the freeze are this status block, the priors and the
file name). Code: `clean_experiments/experiment_B40_alignment.py`, smoke-tested on
synthetic fields only. The author opened this phase explicitly on 2026-10-16; the
closure of the research phase (`PROGRAM_MAP.md`) is lifted for Phase 40 only.

## Purpose

Phase 39 showed that the Germano energy flux is concentrated where the activity of two
adjacent bands is co-located at the same moment (interaction b3 > 0 in the log
regression, transient). The theory note of 2026-10-16
(`THEORY_NOTE_BUNDLE_REVISITED_2026-10-16.md`, sections 3, 10, 11) and the prior-art
audit (`research_notes/prior_art_bundle_revisited_2026-10-16.md`) give the known
physics behind it: in 2-D flow the leading (Clark, zero-memory) term of the enstrophy
flux is Z^C = -ell^2 [grad(omega_bar) . S_bar . grad(omega_bar) + (delta_bar/2)
|grad(omega_bar)|^2] (Eyink 2006; Kraichnan 1971) — the coarse strain stretches the
vorticity gradient — and in the 2-D toy model this term reproduces 96-99 % of the
pixel variance of Z (`results/explore_toy_v2/REPORT.md`). In log coordinates the Clark
term is additive in amplitudes (|grad omega_bar|^2, sigma_bar) plus the alignment
factor a = -cos 2(theta_grad - theta_S); a co-location interaction beyond additivity
must therefore sit in the alignment factor. The phase asks three questions on ERA5:

1. Does the Clark term hold at 400 and 800 km on 850 hPa (where divergence and
   three-dimensionality are present)?
2. Is the Phase-39 interaction carried by the alignment factor?
3. Does the geography of P follow the geography of alignment efficiency
   (Lapeyre-Klein-Hua strain dominance, mean alignment) and oppose the divergent
   ("isotropic") part of the energy flux?

A positive answer gives article 2 a physical layer made of known physics (locality of
the cascade; stretching of the vorticity gradient by the coarse strain; LKH efficiency).
A negative answer leaves P an empirical descriptor without mechanism. No bundle,
holonomy or density-matrix language is used or claimed in this phase.

## Data and fields (fixed, identical to Phase 39 unless stated)

ERA5 850 hPa u, v, 6-hourly, global, JFM and JAS of 2022 (fresh), 2023, 2024; 70S-70N
computed, statistics on the 66 Phase-34 windows (60S-60N, cos-latitude weights, V12
below-surface mask). Gaussian smoother of Phase 34 (standard deviation ell/sqrt(12),
km-based, periodic in longitude, normalised by the valid mask). Statistics on every 2nd
step (12-hourly); the neighbouring 6-hourly steps are used only for the material
derivative of the strain axis.

Per filter scale ell in {400, 800} km (the B1|B2 and B2|B3 boundaries), with the fine
scale ell' = ell/2 and the envelopes E1, E2, E3 of Phase 34:

- Coarse gradient A = grad(u_bar_ell) on the sphere (Phase-39 `grad_sphere`):
  sigma_n = du/dx - dv/dy, sigma_s = du/dy + dv/dx, omega_bar = dv/dx - du/dy,
  delta_bar = du/dx + dv/dy; sigma_bar = sqrt(sigma_n^2 + sigma_s^2);
  theta_S = 1/2 atan2(sigma_s, sigma_n) (extensional axis, nematic).
- Subgrid stress tau_ij = bar(u_i u_j) - u_bar_i u_bar_j; k = tr(tau)/2;
  anisotropy c = |tau^dev| / tr(tau); director theta_tau = 1/2 atan2(2 tau_xy,
  tau_xx - tau_yy); Delta_theta = theta_tau - theta_S.
- Energy flux Pi = -tau_ij d_j u_bar_i (Phase 39), isotropic part Pi_iso = -k delta_bar,
  deviatoric part Pi_dev = Pi - Pi_iso = -k c sigma_bar cos 2 Delta_theta (identity).
- Enstrophy flux Z = -tau^omega . grad(omega_bar), tau^omega_i = bar(u_i omega) -
  u_bar_i omega_bar, with omega the V12 vorticity (the carrier of P).
- Clark terms with ell_g^2 = ell^2 / 12 (the smoother's variance):
  tau^C = ell_g^2 A A^T, Pi^C = -tau^C : sym(A); Z^C = -ell_g^2 [grad omega_bar . S_bar .
  grad omega_bar + (delta_bar/2) |grad omega_bar|^2].
- Alignment: a_same = -cos 2(theta_{grad omega_bar_ell} - theta_S) (enters Z^C exactly);
  a_fine = -cos 2(theta_{grad omega_bar_ell'} - theta_S) (fine gradient in coarse
  strain). +1 = gradient along the compressional axis.
- LKH ratio r = (omega_bar + 2 D_t theta_S) / sigma_bar, with D_t theta_S = d_t theta_S
  + u_bar_ell . grad theta_S, d_t from the +-6 h neighbours (doubled angle wrapped).

## Hypotheses and bars (frozen)

All window statistics pool pixels over the retained steps of a season with
cos-latitude weights; year value = mean of JFM and JAS; main sample = mean of 2023 and
2024; fresh = 2022.

- **H40-1 (the Clark term holds on ERA5, PRIMARY).** Weighted Pearson correlation
  between Z and Z^C over the pixels of a window. Bar, for both ell: median over
  windows >= 0.7 and >= 0.7 in >= 75 % of windows (2023-24); fresh 2022 median >= 0.7.
  Reported: the same for Pi vs Pi^C (expected much lower: Pi^C is the divergent part
  only).
  Caveat recorded at the smoke test (2026-10-16): a smooth Gaussian random field with
  no dynamics already gives corr(Z, Z^C) = 0.98 / 0.96 at 400 / 800 km, because the
  Clark term is the leading Taylor term of any smooth field under a narrow filter.
  H40-1 is therefore a necessary check of the filter regime, not evidence of a
  mechanism; the discriminating content of the phase is H40-2 and H40-3. If the author
  wants H40-1 to carry weight, add a phase-randomised control (Phase-35 mirror or
  Phase-2b phase surrogates) and score the EXCESS of the real correlation over it.
- **H40-2 (the co-location interaction is carried by alignment).** Phase-39
  regression repeated with y = log|Z|: y = b0 + b1 z_k + b2 z_{k+1} + b3 z_k z_{k+1},
  then augmented with z(log|a_same|): b3_aug. Transient interaction d = b3 - b3_lag as
  in Phase 39 (envelopes 5 days later). Bar, both ell: median over windows of
  (d - d_aug)/d >= 0.5 with d > 0, sign-flip p < 0.01 on (d - d_aug); fresh 2022 the
  same at p < 0.01. Reported: the energy-flux version (y = log|Pi|, augmented with
  z(log c) and z(log|cos 2 Delta_theta|)) with the same statistic.
- **H40-3 (geography of P = geography of alignment efficiency).** Across the 66
  windows, Spearman between the Phase-34 coupling P45 (mean of 2023-24) and, with
  signs named now:
  (i) f_LKH = weighted fraction of pixels with |r| < 1 — expected POSITIVE;
  (ii) mean a_fine — expected POSITIVE;
  (iii) f_iso = mean|Pi_iso| / mean|Pi| — expected NEGATIVE.
  Bar: each |rho| >= 0.40 with the named sign for both ell, and Spearman with the
  Phase-36 residual P45_res (beyond the spectral null) of the same sign with |rho|
  >= 0.30; fresh 2022: the three signs reproduced with |rho| >= 0.30 against P45_2022.
  Passing = at least two of (i)-(iii) meet both bars at both ell.
- **H40-4 (reported, not scored).** Alignment curve a_fine binned by log|r| (LKH
  transition); fraction of forward enstrophy transfer (Z > 0) and window-mean Z; Clark
  correlation and f_iso by Phase-34 region type; |Pi_iso| share by CAPE tercile of the
  windows (Phase-20 covariates) — expected: f_iso rises with CAPE.

## Nulls

- Transient control: envelopes taken 10 retained steps (5 days) later in the same
  season (Phase 39), applied to every regression of H40-2.
- Spectral control for geography: the Phase-36 spectral-null column P45_null and the
  residual P45_res (H40-3).
- Not implemented in the draft code: mirror surrogates of Phase 35 for the alignment
  fields themselves. If the author wants them, add a `null` stage before freezing.

## Verdict ladder

- **ALIGNMENT_EXPLAINS:** H40-1, H40-2, H40-3 pass.
- **CLARK_AND_INTERACTION_ONLY:** H40-1 and H40-2 pass, H40-3 fails (mechanism present,
  geography of P not explained by it).
- **CLARK_ONLY:** H40-1 passes, H40-2 fails.
- **NEGATIVE:** H40-1 fails (the zero-memory term does not describe the atmosphere at
  these scales; the toy-model reading does not transfer).

## Author's priors (recorded before computing, 2026-10-16)

H40-1: 0.7; H40-2: 0.5; H40-3: 0.4.

## Code and outputs

`clean_experiments/experiment_B40_alignment.py`, stages `season --year Y --season S`
(two passes: envelopes, then per-chunk fields with window accumulators — no full
alignment fields are stored), `tests`, `smoke` (synthetic sphere, no data). Outputs →
`clean_experiments/results/experiment_B40_alignment/`. Seed 20261030. The SHA-256 of
this file is written into the summary at `tests` time; freezing must happen before the
first `season` run.

## Deviations

- (none yet)

## Result (2026-10-16)

Results: `clean_experiments/results/experiment_B40_alignment/` (six season files, computed
on the Spark host; `summary.json` with the protocol SHA-256). No deviations. The
optional phase-randomised control of H40-1 was not run (not part of the frozen bars).

**Verdict: CLARK_ONLY.**

| hypothesis | result | numbers (mean 2023-24; fresh 2022) |
|---|---|---|
| H40-1 Clark term holds | PASS | median corr(Z, Z^C) 0.90 (400 km), 0.87 (800 km); >= 0.7 in 100 % of windows; fresh 0.90 / 0.88. Reported: corr(Pi, Pi^C) 0.96 / 0.92 — the divergent (piston) term describes the energy flux at 850 hPa almost entirely |
| H40-2 interaction carried by alignment | FAIL | adding z(log|a_same|) changes the transient interaction d of log|Z| by -2 % (400 km) and -1 % (800 km); for log|Pi| with z(log c), z(log|cos 2 Delta_theta|): +2 % / -8 %. d itself reproduces Phase 39: d(Z) = 0.020 / 0.024 (> 0 in 77 / 86 % of windows), d(Pi) = 0.048 / 0.033 (98 / 100 %) |
| H40-3 geography of P = alignment efficiency | FAIL | Spearman with P45: f_LKH -0.11 / -0.07, mean a_fine +0.13 / +0.16, f_iso +0.03 / +0.04 (expected +, +, -); with the Phase-36 residual all |rho| <= 0.11; fresh 2022 all |rho| <= 0.18 |
| H40-4 reported | — | LKH transition present: a_fine falls from 0.18 (|r| << 1) to 0.04 (|r| >> 1) at 400 km, 0.15 -> 0.02 at 800 km, step at |r| ~ 1; forward enstrophy transfer (window-mean Z > 0) in 98 / 97 % of windows, pixel share Z > 0 0.53 / 0.51; isotropic share f_iso = mean|Pi_iso|/mean|Pi| median 0.62 / 0.63; f_LKH median 0.37 / 0.39 |

Reading. The mechanism chain exists on ERA5 at 400-800 km: the zero-memory term
describes the enstrophy flux (0.90), the LKH alignment transition is present, the
enstrophy cascade is forward almost everywhere, and the energy flux is dominated by the
divergent piston term (0.96; cf. Sichani, Storer & Aluie 2026: downscale transfer
catalysed by divergent motions). But none of it carries the programme's object: the
co-location interaction of Phase 39 is not the alignment factor, and the geography of
P is unrelated (|rho| <= 0.18) to strain dominance, mean alignment or the divergent
share. P stays an empirical descriptor; the "alignment efficiency" reading proposed in
`THEORY_NOTE_BUNDLE_REVISITED_2026-10-16.md` (G1, G3, G5) is refuted as an explanation
of P at this scale range. Priors 0.7 / 0.5 / 0.4 -> hit / miss / miss.
