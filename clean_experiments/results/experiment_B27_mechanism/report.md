# Phase 27 — mechanism of the inter-level coupling: REPORT (2026-09-21)

Protocol: docs/PROTOCOL_PHASE27_MECHANISM.md (frozen before computing; SHA-256 in summary.json).
Script: clean_experiments/experiment_B27_mechanism.py. No deviations from the frozen spec.

## VERDICT: PARTIAL(A1)   |   A2 class: SCREENED (uninterpretable, see below)

Controls: C27-1 units rho = 1.000 (max diff 1e-7), tiles rho = 1.000 (diff 0) — PASS.
C27-2 lagged placebo: D_lag/D = 0.11 <= 0.25 — PASS.

| Hypothesis | Result | Numbers |
|---|---|---|
| H27-A1 inheritance beyond the adjacent level (PRIMARY) | **PASS** | separation-2 anchored rho: region median +0.35 (all 12 regions +0.29..+0.48), Wilcoxon p = 0.00024; tiles replicate +0.359 [0.353, 0.366]. Profile sep1..5: 0.27 / 0.35 / 0.25 / 0.16 / 0.09 |
| H27-B1 CAPE decoupling strongest at the fine end (PRIMARY) | **FAIL** | beta_CAPE per step −0.0033 / −0.0054 / +0.0009; Delta_CAPE +0.004 [−0.009, +0.014]; beta_CAPE(k=1) CI [−0.008, +0.005] includes 0; p = 0.29. Profile nominally peaks at k = 2 (not significant) |
| H27-B2(ii) geography of deformation index follows P (PRIMARY) | **FAIL (effect size)** | cross-half rho_x = +0.14 < 0.20 threshold; p_rot = 0.001 (null q95 0.08): association real but weak |
| H27-B2(i) deformation index > 0 | PASS both carriers | tiles D = +0.067 [0.064, 0.072]; units +0.064, p = 0.008, 11/12 regions > 0. Mirror index V (rotation given strain): +0.001 tiles, −0.016 units |
| H27-B2(iii) reported | — | D vs CAPE: beta n.s. (p_rot 0.61), rho −0.05; D vs EKE: rho +0.19, beta n.s. |
| H27-A2 locality class | SCREENED | anchored lag-2 partial −0.10 [−0.12, −0.06] units; −0.12 tiles |
| S4 reported | — | rho(lambda2_hat, INT_sig2) = +0.94 (n = 24) — partly built-in: covariances scale with the variances; not evidence on its own |

Holm-adjusted primaries: A1 0.0007, B2ii 0.002, B1 0.29.
Author's priors: A1 0.85 (hit), B1 0.5 (miss), B2ii 0.55 (miss), B2i 0.75 (hit), A2 NONLOCAL 0.5 (miss).

## Post-hoc notes (NOT part of the frozen analysis)

1. A2 is not interpretable as designed. Raw lag-2 partials are negative in BOTH real data
   (−0.15..−0.36) and surrogates (−0.10..−0.24): adjacent-band correlations are inflated by
   filter overlap (surrogate adjacent rho 0.33..0.66), so the Markov product benchmark is
   biased for both. Only robust reading: no sign of NONLOCAL excess. The class label does
   not go into the paper.
2. For NON-adjacent bands the surrogate level is ~0 (sep2: 0.03..0.16; sep3: 0.02; sep5:
   0.004) while raw real correlations are 0.64 / 0.41 / 0.21 / 0.10 for E0 vs E2..E5. The
   beyond-spectrum coupling is therefore visible in RAW form where bands do not overlap;
   adjacent P is the noisiest place to measure it (0.87 raw of which 0.66 is filter overlap).
   Decay is roughly linear per octave (Arneodo-type logarithmic law), extrapolating to zero at
   L ~ 2-3 thousand km. Exploratory; would need its own protocol.
3. Tile-level geography of the (flawed) anchored partial: rho with CAPE +0.50, with EKE −0.60.
   Exploratory, unexplained.

## Consequences for Article 3 (fixed by the protocol's FAIL clauses)

- KEEP: "inheritance / multiplier shared from above" language — A1 passed on two carriers.
- DROP as our own result: "P measures the deformation-slaved share". Deformation co-location
  exists (B2i) but is small (~0.07) and does not carry the geography of P (B2ii). The
  frontogenesis route stays a literature-based hypothesis.
- DROP: "the autonomous convective source enters at the fine end" — B1 found no scale profile;
  per-step CAPE coefficients are indistinguishable from zero under the block bootstrap.
  Article-2 wording ("co-organisation") stands; no directional language is licensed.
