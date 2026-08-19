# PROTOCOL AUDIT-4 (FROZEN 2026-08-19): is the unexplained part of the
# global map physics or noise? — THE LAST ANALYSIS BEFORE THE PAPERS

Frozen before computation. **Scope fence, part of the protocol:** this
phase asks ONE question and stops. It does not open a covariate hunt.
Whatever the answer, the next work item is writing the three papers; any
follow-up is named here as a future phase and not executed.

## 1. Question

AUDIT-2b established that the anchored tile-P map is reproducible at
R^2 = 0.914 (within-season split-half, Spearman-Brown), that the eight
frozen Phase-20 covariates explain 0.536 of it (59%), and that adding
the AUDIT-2 intermittency block reaches 0.761 (83%). So ~0.15 R^2 of
REPRODUCIBLE variance is unaccounted for.

Is that remainder a physical field with its own geography, or is it the
part of the estimator's variance that the split-half reliability
overstates?

## 2. Design (no new data, no downloads)

Uses the AUDIT-2b shards (P and the intermittency battery on disjoint
time-parity halves of each season) and the frozen Phase-20 covariates.

For each season and each half h (with h' the other half):
- fit OLS of P(h') on the 8 covariates -> predict -> residual
  `r_cov(h) = P(h) - pred`;
- the same with the covariates plus the four intermittency statistics
  computed on h' -> residual `r_full(h)`.

Because the fit comes from the OTHER half, the two residuals of a season
share no sampling noise.

## 3. Scored quantities

- **R1 (primary): residual reliability.** Pearson between `r_cov(odd)`
  and `r_cov(even)`, and between `r_full(odd)` and `r_full(even)`, per
  season, Spearman-Brown corrected.
- **R2 (reported): where it lives.** The season-mean residual mapped;
  the 20 largest positive and 20 largest negative tiles listed with
  centre coordinates; the zonal profile of the residual.
- **R3 (reported): is it spectral?** Spearman of the residual against
  the tile spectral slope and the four band log-variances (Phase-20 Arm
  B `F_fine`), which are NOT in the covariate set.

## 4. Decision rule (frozen, binding)

- If the Spearman-Brown residual reliability of `r_full` is **< 0.30**:
  verdict RESIDUAL_IS_NOISE. The map is declared closed at the
  attributed level, the papers state that the attributable geography is
  exhausted by the covariates plus the intermittency block, and no
  follow-up phase is opened.
- If it is **>= 0.30**: verdict RESIDUAL_IS_PHYSICAL. The papers carry
  one paragraph and one figure: the remainder is reproducible, it has
  the following geography, and identifying its drivers is named as
  future work. **No covariate hunt is run in this phase.**

Either way this protocol ends the analysis programme for the three
papers.

## 5. Execution

`clean_experiments/experiment_A4_residual_structure.py`, single stage.
Results in `clean_experiments/results/experiment_A4_residual/`.

## Deviations

- (none yet)
