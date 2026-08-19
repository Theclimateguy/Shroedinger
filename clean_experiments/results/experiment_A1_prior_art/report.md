# AUDIT-1 — prior-art head-to-head for P: results

Protocol: `docs/PROTOCOL_AUDIT1_PRIOR_ART_HEADTOHEAD.md` (frozen before
computation). Code: `clean_experiments/experiment_A1_prior_art_headtohead.py`.
Sample: 936 global tiles x 2 seasons (JFM/JAS 2023), 902 kept after the
frozen orography exclusion. Every statistic anchored on the SAME 99
phase-surrogate realisations as Phase 20 Arm B.

## Control

- **C-A1-0 PASS.** Recomputed anchored P reproduces the committed Arm-B
  map at Spearman = 1.000, max |difference| on raw per-step values
  = 3.0e-8. The comparison is like-for-like.

## A1a (primary) — VERDICT: ESTIMATOR_DISTINCT

| pair | Spearman |
|---|---|
| P vs R_AM_cyc (amplitude-modulation coefficient, cyclonic convention) | **+0.143** |
| P vs R_AM_raw | -0.062 |

Bar for ESTIMATOR_IS_AM was 0.90, for ESTIMATOR_RELATED 0.60. **P is not
the Mathis-Hutchins-Marusic amplitude-modulation coefficient.** No
novelty claim is withdrawn on account of the wall-turbulence literature;
it is cited as related work.

## A1c — the copula ingredient is decoration

Spearman(P, P_lin) = **+0.992**. The rank/copula construction changes
nothing at map level. Consequence, per the frozen bar (0.95): the
rank/copula language is **dropped from all three papers**. P is stated
plainly as an envelope-envelope band-coupling statistic; the closest
published relative is the amplitude-amplitude coupling (amplitude
envelope correlation) of cross-frequency analysis, and that is what the
papers must cite as prior art for the form.

## A1b — attribution transfer

| target | LOSO-sector R^2 | p (rotation null) | cross-season reliability r | share of own ceiling |
|---|---|---|---|---|
| P | +0.536 | 0.001 | 0.766 | 0.91 |
| P_lin | +0.546 | 0.001 | 0.780 | 0.90 |
| R_AM_cyc | +0.062 | 0.049 | 0.256 | 0.95 |
| R_AM_raw | +0.025 | 0.806 | 0.644 | 0.06 |

Loadings for P and P_lin are identical to two decimals (abs_lat +0.66/
+0.67, eke_syn +0.66/+0.67, cape_mean -0.52/-0.52, sst_grad +0.55/+0.54);
R_AM_cyc's loading pattern is different and weak (orography-leaning,
|rho| <= 0.29).

Honest reading of the R_AM_cyc row: its low R^2 is NOT evidence that the
amplitude-modulation coefficient is unattributable. It reaches 95% of
its own reliability ceiling (0.256^2 = 0.066). The statistic simply has
almost no reproducible tile-level signal to attribute — reliability 0.26
against P's 0.77. The reproducible geography lives in the
envelope-envelope form, not in the signal-envelope form.

## Consequences for the three papers

1. Cite Mathis et al. (2009) and the cross-frequency-coupling literature
   as prior art for the FORM. Claim novelty only for the object: the
   surrogate-anchored beyond-spectrum component as a mapped regional
   invariant.
2. Drop "copula" / "rank dependence" as a claimed ingredient (A1c).
   Report it as an implementation detail that provably does not matter.
3. State positively: the Phase-20 attribution is invariant to the
   Pearson/Spearman choice (R^2 0.536 vs 0.546, identical loadings) —
   the geography is a property of envelope coupling, not of an estimator
   convention.
4. The signal-envelope (turbulence AM) reading of scale coupling does
   NOT reproduce the map. This is a new negative result and belongs in
   the methods section.

## Addendum (post-hoc, reported not scored): does the anchoring matter?

From the same shards, 936 tiles, season means:

| quantity | median |
|---|---|
| P_real | 0.770 |
| P_surrogate_median (a pure spectrum functional) | 0.512 |
| P_anchored | 0.256 |

| pair | Spearman |
|---|---|
| P_anchored vs P_real | +0.746 |
| P_anchored vs P_surrogate_median | -0.588 |
| P_real vs P_surrogate_median | **-0.019** |

The raw map and the spectral (surrogate) map are essentially orthogonal
across tiles: the surrogate map explains 0.00 of the variance of the raw
map. The anchoring is therefore neither cosmetic nor dominant — it
removes a component that carries its own, different geography.
