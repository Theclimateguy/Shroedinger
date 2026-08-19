# PROTOCOL PHASE 24 (DRAFT v0.1 — NOT FROZEN): Reducibility of P to known
# invariant-measure observables

Status: DRAFT, DEFERRED 2026-08-19 (doctoral track; see docs/PRIOR_ART_AUDIT_P.md Sect. 5). Nothing here is scored until this file is frozen and
committed. Scoping numbers quoted in Sect. 2 were computed BEFORE the
freeze and are explicitly labelled as scoping, not as results.

## 1. Question

QT-2 states P(x) = Phi[mu_theta(x)]. The candidate does not claim that
Phi is new. Competing reading:

- H_A: P is a functional of the local invariant measure that is not
  recoverable from the standard observables of that measure.
- H_B: P is a re-parameterisation of a known attractor observable Q
  computed on the same field.

Q battery (frozen list, all computed on the same 850 hPa relative
vorticity states used for P):

| Q | estimator | status before Phase 24 |
|---|---|---|
| spectral slope + band log-variances | frozen Phase 2/18 machinery | CLOSED: P is anchored to 99 phase surrogates by construction; beyond-spectrum residual confirmed (B18, B2b, B3) |
| temporal persistence tau, AR(1) gamma | B14 gamma_step_scan | CLOSED: rho(P, persistence) = +0.02 (Phase 12) |
| predictability / error growth | GEFS, WB2 arms | CLOSED in the weak sense: P carries no forecast skill (Phases 9-11) |
| local dimension d(x,t) | Freitas-Freitas-Todd / Lucarini EVT, q = 0.98 | OPEN |
| extremal index theta (persistence of the state, not of the field) | Sueveges MLE, same threshold | OPEN |
| Markov mixing gap lambda_2, entropy rate h | k-means (k = 12) transfer matrix on the same states | OPEN |
| almost-invariant sets / transfer-operator spectrum | deferred | OPEN, Arm D, not entered here |

Excluded by construction: the correlation dimension of "the atmosphere"
from a single regional series (Lorenz 1991 / Grassberger critique). No
global attractor dimension is estimated anywhere in this phase.

## 2. Scoping (pre-freeze, not scored)

- Reliability ceiling of the anchored tile-P map: **R^2 = 0.914**
  (AUDIT-2b, within-season split-half 0.916 with the Spearman-Brown
  correction). The earlier cross-season figure (r = 0.776, R^2 = 0.603)
  understated it by charging real seasonal change to noise.
- Phase-20 Arm B attribution reached LOSO R^2 = 0.536 = **59% of that
  ceiling**; covariates plus the AUDIT-2 intermittency block reach 83%.
  Headroom for a new tile-level variable is therefore ~0.38 R^2 before
  the intermittency block and ~0.15 R^2 after it — the map arm is NOT
  power-limited in the way the first scoping claimed, and the earlier
  "the map arm can only weakly support H_A" caveat is withdrawn.
- Residual after removing eke_syn + the four band log-variances +
  slope still reproduces across seasons at Spearman 0.56 — reliable
  structure exists in the residual, so the arm is not vacuous.
- 48 region-window pilot (data/b3, 16x16 coarse states, N = 368):
  P vs d rho = +0.48 (p = 0.001); P vs lambda_2 rho = -0.48
  (p = 0.001); P vs theta rho = +0.09 (n.s.). Region means (n = 12):
  P vs d rho = +0.58, p = 0.048. **H_B is a live hypothesis for d and
  lambda_2, not a straw man.**
- The same pilot: d, lambda_2 and h respond to phase surrogates
  (median d 17.7 real vs 13.3 surrogate, Wilcoxon p = 0.014). The
  cheap logical shortcut ("Q is spectrum-blind, P is not, therefore
  P != f(Q)") is UNAVAILABLE. Anchoring must be applied symmetrically:
  every Q enters both raw and surrogate-anchored.
- Region-level design is underpowered: LOSO-region R^2 is negative for
  every predictor set at n = 46 with 12 folds. The map arm must be run
  at tile level.

## 3. Arms

### Arm A — map-level conditional reducibility (data on disk: data/b20global)

- Units: 936 tiles x 2 seasons, JFM/JAS 2023, ERA5 850 hPa u,v,
  6-hourly (already used by Phase 20 Arm B).
- Matching (mandatory, from the Phase-18 lesson): each tile is
  resampled to an equal-km footprint and an IDENTICAL state dimension
  and identical N before any Q is computed. Otherwise d inherits a
  latitude-dependent grid-count bias that mimics P's own latitude
  dependence.
- Q anchoring: 19 phase surrogates per tile-season, frozen seed scheme
  salted with the tile id (as in Phase 20).
- Target: anchored tile P_fine (the Phase-20 Arm B primary target).
- H24a (SINGLE SCORED PRIMARY): incremental LOSO-region-block R^2 of
  {Q, Q_anchored} over the frozen Phase-20 covariate set
  {orog_std, land_frac, coast_var, cape_mean, eke_syn}. Null: the
  Phase-20 region-block rotation null, 999 draws.
- Verdict ladder (frozen before computation):
  - REDUCED if R^2(covars + Q) >= 0.9 x 0.603 AND the residual loses
    cross-season reliability (Spearman <= rotation-null q95).
  - PARTIAL_REDUCTION with the explained share reported if
    delta R^2 > 0 at p < 0.05 but the residual stays reliable.
  - NOT_REDUCED_AT_MAP_LEVEL if delta R^2 is null-consistent AND the
    residual keeps cross-season reliability at p < 0.05. Per Sect. 2
    this is a WEAK support of H_A and is stated as such.

### Arm B — within-tile co-fluctuation (the arm with headroom)

Cached 6-hourly P(t) series exist for 80 region-windows
(results/experiment_B14_p_relaxation, results/experiment_B19_km_dynamics:
P (368, 5 bands), V (368, 5)). Compute d(t), theta(t), lambda_2(t)
on sliding windows of the same fields (data/b3, data/b2b).

- H24b: partial Spearman of the P(t) sampling fluctuation with
  d(t) / theta(t), controlling band variance V(t), pooled over
  region-windows with a region-block null.
- Interpretation: P has no dynamics (QT-3), so a positive result means
  the two estimators sample the same measure feature through the same
  clock; a null at adequate power means P's sampling fluctuation is
  not the local-dimension fluctuation. Power is set by 80 windows
  x 368 steps, not by 12 regions — this arm is not geography-limited.

### Arm C — matched-regime existence test (idealised models; new code)

Purpose: decide H_A/H_B by construction rather than by regression.

- Lorenz-63 is EXCLUDED: P is a spatial rank-dependence between scale
  bands; a three-variable system has no spatial field, so P is
  undefined there. Minimum viable systems: Lorenz-2005 model II/III
  (spatially extended ring with scale separation) and a two-layer
  spectral QG channel.
- Sweep forcing / dissipation / scale-separation. For each run compute
  lambda_1 and the Kaplan-Yorke dimension (tangent-linear, available
  in a model as it is not in data), the entropy rate (Pesin), the
  spectrum, and P by the frozen envelope machinery.
- H24c: does there exist a pair of parameter settings matched on
  (lambda_1, D_KY, h, spectral slope) within tolerance that differ in
  P beyond the estimator's own spread? One such pair falsifies H_B in
  principle. If the sweep finds P pinned by those metrics everywhere,
  H_B is supported and the candidate must rename P.
- Cost: new code, no data. Run only if Arm A/B return PARTIAL or
  AMBIGUOUS, or run in parallel if capacity allows.

### Arm D — transfer operator / almost-invariant sets

Deferred by design. Entered only after A-C, with its own protocol.

## 4. Controls

- C24-1 (positive control, power, computed FIRST): a synthetic target
  Y = d + noise, noise scaled so that Y's cross-season reliability
  equals P's. The Arm-A pipeline must return REDUCED for Y. If it does
  not, the arm is underpowered and NO verdict is issued.
- C24-2 (geometry): every Q recomputed at two state dimensions
  (12x12 and 20x20). Verdicts must agree; disagreement voids the arm.
- C24-3 (sample size): every Q recomputed on half the states; rank
  stability across tiles Spearman >= 0.8 required, else the estimator
  is declared unstable and dropped from the battery.
- C24-4 (spectral placebo): Arm A repeated with target = tile spectral
  slope. If the Q battery explains the slope as well as it explains P,
  the reduction is spectrum-mediated, not measure-level.
- C24-5 (negative control): the Phase-20 LAI biological control is
  carried through unchanged.

## 5. Multiplicity

One scored primary per arm (H24a, H24b, H24c). Everything else is
reported. No arm may be re-run with a changed estimator after seeing
its verdict; estimator variants are fixed in C24-2/C24-3.

## 6. What each outcome does to the candidate

- REDUCED (Arm A or C): QT-2 keeps its form but loses novelty. The
  candidate must rename P as an estimator of Q and the contribution
  reduces to the geography and the model-transfer result.
- NOT_REDUCED at map level + Arm-B null + Arm-C matched pair found:
  P is entered as an independent coordinate of the local measure, and
  the Article-3 claim is stated in exactly that conditional form.
- Mixed: the phase reports the explained share and the candidate
  carries the reduction share explicitly in QT-2.
