# PROTOCOL AUDIT-3 (FROZEN 2026-08-19, reduced form): do the AUDIT-1/2
# verdicts survive a different box size, band range, year and season?

Frozen before computation. Reduced by design: AUDIT-1 and AUDIT-2
answered the reinvention question on the 700-km tile grid, JFM/JAS 2023,
fine bands 50-400 km. This audit checks only that their verdicts are not
an artefact of that one configuration. It is a defensive check, not a
discovery arm, and it is deliberately small.

## 1. Sample

- Territory: the 12 frozen equal-km boxes of Phase 18 (2000 x 2800 km,
  Variant B index masks on the native 0.25-deg grid), i.e. a footprint
  ~11x the area of a Phase-20 tile.
- Windows: `W9_2023JFM` and `W12_2024JAS` — different year AND different
  season, both in `data/b3`. 12 x 2 = **24 units**. No downloads.
- Band ladder: the full frozen Phase-2 ladder
  `ELLS_KM = [50, 100, 200, 400, 800, 1600]` km, i.e. 5 band steps
  reaching 4x beyond the fine ladder audited so far.
- Anchoring: 99 phase surrogates per unit, frozen seed scheme salted
  with the unit tag, exactly as in AUDIT-1/2.

## 2. Statistics

The same battery, computed in one pass per unit and averaged over the
band steps: `P`, `P_lin`, `R_AM_cyc`, `P_cascade`, `INT_sig2`,
`INT_flat`, `INT_mu` (definitions as in AUDIT-1 and AUDIT-2).

## 3. Scored questions and bars (frozen)

Across the 24 units, Spearman of anchored `P` against:

- **A3a (primary): `R_AM_cyc`.** |rho| >= 0.90 -> AM_IDENTITY_AT_BOX_SCALE
  (AUDIT-1 would be overturned); 0.60 <= |rho| < 0.90 -> AM_RELATED;
  |rho| < 0.60 -> AM_DISTINCT_ROBUST.
- **A3b (primary): `P_cascade`.** Same ladder ->
  CASCADE_IDENTITY_AT_BOX_SCALE / CASCADE_RELATED /
  CASCADE_DISTINCT_ROBUST.
- **A3c (reported): `P_lin`** — the copula-decoration finding
  (expected ~0.99) — and **`INT_sig2`** — the intermittency-parameter
  relative (expected ~0.6-0.7).

Power statement, frozen: with n = 24 the standard error of a Spearman
estimate is ~0.2. This audit can therefore separate "far below 0.60"
from "at or above 0.90", which is all it is asked to do. Any |rho|
landing inside [0.45, 0.75] is reported as INCONCLUSIVE and escalates to
the full 80-unit form rather than being interpreted.

## 4. Multiplicity

A3a and A3b are the two scored primaries (one per retired claimant).
A3c is reported. No statistic is redefined after seeing a result.

## 5. Execution

`clean_experiments/experiment_A3_scale_year_robustness.py`, stages
`units` (resumable, atomic shard per unit) and `tests`. Results in
`clean_experiments/results/experiment_A3_robustness/`.

## Deviations

- (none yet)
