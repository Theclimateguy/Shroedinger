# PROTOCOL — Phase 30: unification — P as the normalised amplitude of the
# structure law; the inherited share h and the own variance N
## Status: FROZEN 2026-09-22, before any Phase-30 statistic was computed

Disclosure. On the 48 ERA5 units (already read in Phases 28-29) an
exploratory check showed: anchored adjacent covariance vs G(a_{i+1}) from
non-adjacent pairs r = 0.994, ratio 1.05; P_hat = G/sqrt(V_i V_{i+1}) vs
anchored P r = 0.77-0.85 with P_hat ~ 1.8x larger. NOT seen: any of this
on tiles or in the model; surrogate variances; the bias decomposition; the
reliability of h; anything about N. No new envelope computation.

## Model and definitions (per unit/tile, all from Phase-27/29 shards)

E_i = T_i + eps_i, T nested (T_i = T_j + w_ij for a_i < a_j), eps
independent across levels. Then Cov(E_i, E_j) = Var T_j =: G(a_j).

- G_hat(a_j): mean anchored cov_{k,j} over k <= j-2 (non-adjacent, same
  coarse band). Tiles: j = 2 (cov_02), j = 3 (mean of cov_03, cov_13).
  Units/model: j = 2..5.
- Adjacent anchored covariance C_adj,i = anchored cov_{i,i+1}.
- V_i = real variance of E_i (cov_ii real); V_i^s = surrogate-median
  variance; O_i = surrogate-median adjacent covariance (filter overlap).
- Inherited share of level i from level i+1:
    h_i = G_hat(a_{i+1}) / sqrt(V_i V_{i+1}).
- Observed anchored P_i = median_t rho_real − median_t rho_sur (stored).
- Predicted anchored P under the model with overlap O and surrogate
  variances:
    P_pred,i = (G_hat(a_{i+1}) + O_i) / sqrt(V_i V_{i+1})
               − O_i / sqrt(V_i^s V_{i+1}^s).
- Own (non-inherited) beyond-spectrum variance of level j:
    N_j = (V_j − V_j^s) − G_hat(a_j), j with a G_hat (tiles: j = 2, 3).

Steps used on tiles: i = 1 (bands 100/200 km) and i = 2 (200/400 km);
i = 0 has no non-adjacent partner for j = 1.

## Hypotheses and decision rules

H30-1 (PRIMARY) — the law extends to adjacent pairs after anchoring.
  TILES: across 902 tiles and steps i = 1, 2 pooled, Pearson r(C_adj, G_hat)
  >= 0.90 and median ratio C_adj / G_hat within [0.80, 1.25], block-bootstrap
  95 % CI of the median ratio (30x30-deg blocks, 1999 resamples, seed
  20260922) inside [0.70, 1.40]. MODEL: 24 IFS-HR units, steps 1..4, r >= 0.90
  and median ratio in [0.80, 1.25]. PASS: both carriers.

H30-2 (PRIMARY) — P is the normalised amplitude with a known anchoring bias.
  (a) TILES: Spearman(P_i, h_i) >= 0.60 for each of steps 1, 2, rotation
      null p < 0.05 (999 rotations of h, min 30 deg).
  (b) TILES: the bias decomposition reproduces P: median |P_pred − P_obs|
      <= 0.05 (both steps pooled) and the OLS slope of P_obs on P_pred
      within [0.80, 1.20]. Reported: Spearman of the bias (h − P_obs) with
      its predicted cause O·(1/sqrt(V^s V'^s) − 1/sqrt(V V')).
  PASS: (a) and (b). (b) failing while (a) passes is reported as
  "P tracks h; the anchoring bias is not fully captured by median-level
  algebra" — the estimator relation stands, the closed form does not.

H30-3 (PRIMARY, estimator choice) — reliability of h vs P on the tile map:
  within-season split-half Spearman-Brown r_SB (H1/H2 steps; each season,
  averaged), for the mean over steps 1-2 of h and of P (same steps).
  PASS: r_SB(h) >= r_SB(P) − 0.05. Reported: cross-season rho of each;
  the same for G_hat(a_2) (= A) already known 0.79.

H30-4 (PRIMARY) — the own variance N is a reproducible, distinct field.
  N = mean of N_2, N_3 per tile (season mean). PASS: split-half r_SB(N)
  >= 0.50 AND |Spearman(N, P)| <= 0.90 (not a re-expression of P).
  Reported: fraction of tiles with N > 0; Spearman of N with CAPE, EKE,
  P, G_hat, each with the rotation null; N's split-half by season; map saved.
  Interpretation fixed now: N > 0 and reproducible -> a measurable
  "own-variance" field exists and may be mapped alongside P; no mechanism
  is attached to it in Article 3.

## Controls

C30-1: on ERA5 units the exploratory numbers above must reproduce exactly
  from the frozen definitions (r(C_adj, G_hat) >= 0.99) — sanity of code.
C30-2: G_hat(a_2) on tiles equals Phase-29 A to machine precision.

## Verdict ladder

UNIFIED: H30-1, H30-2, H30-4 pass (H30-3 chooses the estimator only).
PARTIAL(<list>) otherwise; NOT_UNIFIED if H30-1 fails.

## Author's priors (before computing)

H30-1 0.85. H30-2 (a) 0.75, (b) 0.45. H30-3 0.4. H30-4 0.5.

## Scope fence

No new bands, steps, thresholds, carriers or covariates after the first
read. Anything else is exploratory and labelled so.

## Deviations
(none at freeze)
