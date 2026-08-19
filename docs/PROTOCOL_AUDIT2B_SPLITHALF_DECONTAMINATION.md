# PROTOCOL AUDIT-2b (FROZEN 2026-08-19): split-half decontamination of
# A2d, and the map's true reliability ceiling

Frozen before computation. Follows the deviation logged in
`docs/PROTOCOL_AUDIT2_INTERMITTENCY_HEADTOHEAD.md`: the A2d bar compared
a same-sample increment against a cross-season ceiling, so both sides
were wrong in the same direction. This protocol replaces the estimate
with a design that has no shared sampling noise and no seasonal
attenuation.

## 1. The two defects being repaired

1. **Shared noise.** P and the intermittency statistics were computed
   from the SAME tile-season sample. Any common estimation error
   inflates the increment. The quick cross-season control gave +0.08
   against the same-sample +0.225, but a cross-season predictor is also
   attenuated by real seasonal change, so +0.08 is a lower bound, not
   an estimate.
2. **Wrong ceiling.** 0.603 is the CROSS-SEASON variance ceiling of the
   map. It charges genuine seasonal change to noise and therefore
   understates how much reproducible variance a within-season map
   actually has. Every "share of ceiling" statement in Phase 20 scoping
   and in AUDIT-1/2 inherits this.

## 2. Design

Each tile-season sample (372 6-hourly steps) is split by time parity
into two disjoint halves: `odd` (steps 1,3,5,...) and `even`
(steps 0,2,4,...). Both halves span the whole season, so climate,
season and tile are identical across halves; only the sampling noise is
independent.

Everything else is unchanged from Phase 20 Arm B / AUDIT-2: same tiles,
mask, band ladder, orography exclusion, 60-deg sector LOSO folds,
999-rotation null. Each half is anchored on its own 99 phase surrogates
generated from that half (same seed scheme, salted with the half name).

## 3. Scored questions and bars (frozen)

- **B1 (ceiling, primary reported quantity).** Within-season split-half
  agreement of anchored P: Pearson r_half between the `odd` and `even`
  maps, per season, and the Spearman-Brown correction to full-sample
  reliability r_full = 2 r_half / (1 + r_half). The map's true R^2
  ceiling is r_full^2. This number replaces 0.603 in all subsequent
  reporting. No bar: it is an estimate, not a test.
- **B2 (SCORED PRIMARY).** Decontaminated increment: LOSO-sector R^2
  predicting anchored P of one half from the 8 frozen covariates PLUS
  the four intermittency statistics computed on the OTHER half, minus
  the covariate-only R^2 on the same target. Four combinations
  (2 seasons x 2 directions), reported individually and as their mean.
  Null: 999 longitude rotations of the added columns.
  - increment > 0 with p < 0.05 -> INTERMITTENCY_ADDS, and the papers
    quote the decontaminated value.
  - otherwise -> INTERMITTENCY_NULL_AFTER_DECONTAMINATION.
- **B3 (reported).** The same-half increment (P and intermittency from
  the identical half) for the contamination magnitude, and the
  attenuation-corrected decontaminated increment obtained by dividing
  the added block's columns by their own split-half reliability.
- **B4 (reported).** The AUDIT-1 and AUDIT-2 primaries recomputed
  across halves: Spearman(P from one half, R_AM / P_lin / P_cascade /
  INT_sig2 from the other half). A correlation that survives the split
  is a correlation of signal, not of shared noise. This is a stricter
  version of A1a/A2a and is reported beside them; it does not change
  their frozen verdicts.

## 4. Multiplicity

B2 is the single scored primary. B1, B3, B4 are reported quantities.

## 5. Execution

`clean_experiments/experiment_A2b_splithalf.py` (a dedicated script that
computes the AUDIT-1 and AUDIT-2 statistic families in ONE pass per
half, so B4 needs a single run; the committed full-sample shards of
AUDIT-1/2 are untouched). Stages `tiles --season S --half H`, `collect`,
`tests`. Results in
`clean_experiments/results/experiment_A2b_splithalf/`.

## Deviations

- (none yet)
