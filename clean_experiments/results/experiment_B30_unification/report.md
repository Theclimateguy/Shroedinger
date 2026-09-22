# Phase 30 — unification: P as the normalised amplitude of the structure law: REPORT (2026-09-22)

Protocol: docs/PROTOCOL_PHASE30_UNIFICATION.md (frozen; SHA-256 in summary.json). No deviations.
Script: clean_experiments/experiment_B30_unification.py. No new envelope computation.

## VERDICT: PARTIAL(H30-1, H30-3, H30-4) — the unification holds at the covariance level; the closed
## form of the anchoring bias and the additive "own variance" do not.

| Hypothesis | Result | Numbers |
|---|---|---|
| H30-1 law extends to adjacent pairs after anchoring | **PASS on all three carriers** | anchored C_adj vs G_hat from non-adjacent pairs: tiles r = 0.989, ratio 1.018 [1.018, 1.035]; ERA5 units r = 0.994, ratio 1.055; IFS-HR model r = 0.989, ratio 1.092 |
| H30-2 (a) P tracks h = G/sqrt(V V') | PASS | steps 1, 2: rho 0.63 / 0.64, p_rot 0.001 |
| H30-2 (b) closed-form anchoring bias reproduces P per tile | FAIL | median abs error 0.03 (ok), medians match (0.290 vs 0.295), but slope 0.38 and r 0.75; the bias (h − P) correlates 0.78 with its predicted cause O(1/sqrt(Vs Vs') − 1/sqrt(V V')) |
| H30-3 reliability h vs P (steps 1-2) | PASS (equal) | r_SB 0.714 vs 0.721; cross-season 0.47 vs 0.42 |
| H30-4 own variance N = (V − Vs) − G | formal PASS, **interpretation clause NOT met** | N < 0 in 98 % of tiles (median −0.019: beyond-spectrum variance excess 0.06 < G(200 km) 0.09); reliable (r_SB 0.77) but rho(N, G) = −0.81 — it is mostly −G |

Controls: C30-1 units r = 0.994 (frozen definitions reproduce the exploratory numbers); C30-2 G_hat(a_2) = Phase-29 A exactly.
Priors: H30-1 0.85 (hit), H30-2a 0.75 (hit), H30-2b 0.45 (miss), H30-3 0.4 (miss, passed), H30-4 0.5 (formal hit, substantive miss).

## Reading

1. The numerator of P IS the law: after surrogate anchoring the adjacent-band covariance equals the
   variance of the texture shared from the coarser band, on ERA5 tiles, ERA5 boxes and a free-running
   model alike. "Coupling of adjacent levels" and "inheritance from the coarser level" are one quantity
   in covariance form; P is its correlation form.
2. P is a monotone but compressed estimator of the normalised amplitude h (rho 0.63; medians 0.30 vs
   0.60). The compression comes from anchoring: the surrogate correlation subtracted from P carries
   surrogate variances smaller than the real ones (bias correlates 0.78 with that term). The
   median-level closed form does not reproduce it tile by tile (slope 0.38): medians of ratios are not
   ratios of medians. Not needed for the paper; P and h are equally reliable (H30-3).
3. The additive decomposition of level VARIANCE fails: V − Vs < G almost everywhere, so
   E = T + eps with independent eps cannot be read literally as "inherited variance + own variance".
   The law is a statement about COVARIANCES (which are purely shared), not about a variance budget.
   "Inherited share" must therefore be used as a normalised amplitude, not as a fraction of variance;
   no "own-variance / autonomous" field is reported.
