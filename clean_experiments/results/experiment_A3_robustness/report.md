# AUDIT-3 (reduced) — scale / year / season robustness of the AUDIT-1/2
# verdicts

Protocol: `docs/PROTOCOL_AUDIT3_SCALE_YEAR_ROBUSTNESS.md` (frozen before
computation). Code:
`clean_experiments/experiment_A3_scale_year_robustness.py`. 24 units =
12 equal-km Phase-18 boxes (2000 x 2800 km, ~11x a Phase-20 tile) x 2
windows (W9_2023JFM, W12_2024JAS — different year and season), full
frozen band ladder ELLS_KM = [50,100,200,400,800,1600] km (5 steps,
4x beyond the audited fine ladder), 99 phase surrogates per unit. No
downloads.

## Scored

| question | Spearman over 24 units | verdict |
|---|---|---|
| A3a: P vs R_AM_cyc | **+0.386** | AM_DISTINCT_ROBUST |
| A3b: P vs P_cascade | **+0.036** | CASCADE_DISTINCT_ROBUST |

Both stay clear of the 0.60 relatedness bar and of the frozen
INCONCLUSIVE band [0.45, 0.75]. **The AUDIT-1 and AUDIT-2 verdicts are
not artefacts of the 700-km tile, the fine band range, or 2023.**

Per window: R_AM_cyc +0.427 (JFM 2023) / +0.385 (JAS 2024);
P_cascade -0.210 / +0.287.

Diagnostics (medians over units): P real 0.728 -> anchored 0.282;
P_cascade real 0.738 -> anchored **-0.028** (the cascade prediction is
again fully reproduced by the phase surrogate, i.e. spectrum-level, now
also at box scale with bands to 1600 km); R_AM_cyc real 0.073 ->
anchored 0.074 (the surrogate carries no amplitude modulation, so
anchoring is a no-op for it).

## Reported

| pair | Spearman over 24 units | tile-scale value (AUDIT-1/2) |
|---|---|---|
| P vs P_lin | +0.982 | +0.992 |
| P vs INT_sig2 | **+0.923** | +0.692 (cross-half +0.600) |
| P vs INT_flat | +0.771 | -0.223 |
| P vs INT_mu | +0.756 | +0.100 |

The copula-decoration finding holds at box scale (+0.98).

**Open item, and the sharpest one the audits have produced.** The
anchored intermittency parameter — the spatial variance of the band
log-envelopes — orders the 12 boxes almost exactly as anchored P does
(+0.923; per window +0.888 / +0.923). At tile scale the same pair sits
at +0.69 (+0.60 noise-immune). The relationship therefore STRENGTHENS
with box size and band range. INT_sig2 was a reported quantity here, not
a scored primary, and n = 24 with 12 independent territories carries a
standard error near 0.2 — but the value sits at the identity bar and
cannot be left as a footnote.

This does not overturn anything already scored: P is not the
amplitude-modulation coefficient, not the cascade correlation, and the
tile-level map is not reducible to the intermittency battery (AUDIT-2b:
covariates + intermittency reach 83% of a 0.914 ceiling, so ~0.15 R^2
of reproducible variance survives). It does mean the papers must state
the relationship to the intermittency parameter as scale-dependent, and
that a dedicated scored test of P vs INT_sig2 at box scale is the next
audit if the claim is to be made in print at that scale.
