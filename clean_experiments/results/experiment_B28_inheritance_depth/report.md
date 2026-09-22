# Phase 28 — the inheritance law and its depth L: REPORT (2026-09-22)

Protocol: docs/PROTOCOL_PHASE28_INHERITANCE_DEPTH.md (frozen before computing; SHA-256 in summary.json).
Script: clean_experiments/experiment_B28_inheritance_depth.py. Inputs: Phase-27 shards + 48 new
0.7-extent unit shards (results/experiment_B27_mechanism/units_x0.7). No deviations from the frozen spec.

## VERDICT: PARTIAL(H28-1, H28-2, H28-3) — and L RETRACTED by a post-hoc artefact check (below)

| Hypothesis | Result | Numbers |
|---|---|---|
| H28-1 form: coarser scale, not separation, organises C_ij | **PASS (very strong)** | R2(factor a_j) = 0.996 vs R2(factor separation) = 0.271; dR2 region median +0.73 (12/12 regions +0.62..+0.83), p < 1e-4; linear log-law R2 = 0.92; lambda2 > 0 in 48/48 units |
| H28-2 L regional invariant | PASS (borderline) | cross-season rho(ln L) = 0.61 (pairs 0.45..0.81; threshold 0.60), perm p = 0.031; F between/within = 12.3, p < 1e-4; split-half rho 0.75 |
| H28-3 L not the box | PASS | q = L(0.7 box)/L(1.0 box) = 0.97 (0.93..1.02); C28-2 rho = 0.92. But lambda2 ratio = 0.78: anchored covariances shrink ~25 % in the smaller box |
| H28-4 P carried by (lambda2, ln L) on tiles | FAIL (near miss) | LOSO R2 = 0.277 (threshold 0.30), p_rot = 0.001; lambda2 alone 0.221, ln L alone 0.018; increment over 8 covariates +0.276 (p = 0.001, ladder +0.05/p<=0.01) |
| H28-5 reported | — | region L 1209..1959 km; rho(P, lambda2) = 0.83 (regions), 0.61 (tiles, p_rot 0.001); rho(P, ln L) = 0.59 / 0.09 (n.s.); units-vs-tiles co-location rho 0.68 (lambda2), 0.49 (ln L) |

C28-1: anchored non-adjacent C_ij are 17x the surrogate floor (median); 87.5 % of units have sep>=3 above floor.
Priors: H28-1 0.6 (hit), H28-2 0.7 (hit), H28-3 0.4 (hit), H28-4 0.5 (miss).

## Post-hoc checks (NOT part of the frozen analysis; exploratory, decisive for interpretation)

1. **L tracks the band ladder, not the flow.** Refitting the units with the ladder truncated:
   a_j <= 400 km -> L = 827 km; <= 800 km -> 1087 km; full (<= 1600 km) -> 1564 km. Tiles
   (ladder to 400 km) give 537 km. Box size does not move L (H28-3) but the coarsest band
   used does. The decay of C_0j (0.252 / 0.124 / 0.043 / 0.010 at 200 / 400 / 800 / 1600 km)
   is not log-linear (per-octave differences 0.13, 0.08, 0.03 instead of constant) but close
   to a power law C ~ a_j^-1.55 (R2 = 0.98), which has no finite outer scale. **L is an
   extrapolation artefact of the log-law form and is withdrawn as an invariant.** lambda2, the
   slope of the same fit, inherits the problem (0.18 / 0.14 / 0.10 by ladder; x0.78 by box).
2. **Shape vs amplitude.** Normalised profiles C(i,j)/C(i,2) across the 12 regions: at 400 km
   the shape is near-universal (CV 0.14: 0.39..0.69), at 800 km it spreads (CV 0.36), at 1600 km
   it is at the floor (CV 0.84). The AMPLITUDE C_02 carries P: rho(P, C_02) = 0.82 over regions;
   the shape index C_04/C_02 also correlates with P (0.55) — convective tropics (WPWP 0.34,
   Congo 0.25) decay slower than continental interiors (CASIA 0.05, SAM 0.12). Unexplained.
3. **Tile covariances are 3-5x smaller than unit covariances at the same scales** (C_02 0.089
   vs 0.252; C_03 0.025 vs 0.124): the 700-km tile suppresses the coupling magnitude to bands
   >= 200 km. Rankings survive (Arm-A/B H-B0, 0.874 consistency), magnitudes are tile-size
   dependent — a caveat for any magnitude statement on the global map.
4. The +0.276 increment of (lambda2, ln L) over the covariates is a same-data quantity computed
   from the same envelopes as P, like the AUDIT-2 intermittency block (+0.207); rho(lambda2,
   INT_sig2) = 0.94 (Phase 27 S4). It is most likely a re-expression of that block, not new
   geography. Not scored; must be compared head-to-head under a frozen protocol before use.

## What survives (for Article 3)

- The STRUCTURE law: the beyond-spectrum coupling between a finer and a coarser band depends
  on the coarser band only, not on how fine the finer band is (R2 0.996 vs 0.27). This is the
  common-ancestor signature of a scale-local multiplicative cascade (Arneodo 1998): each coarser
  level imposes the same texture on every level below it. Robust across 12 regions, 4 windows,
  two box sizes.
- The coupling decays with the coarser scale roughly as a power law over 200-1600 km, with no
  identifiable finite depth on this ladder.
- P's geography is the AMPLITUDE of that inheritance (rho 0.82 with C_02 across regions), not
  its depth. Article 2's map and regionalisation are untouched.
- Withdrawn: "depth L" as an invariant; "P obeys lambda2 (ln L - ln a)" as a quantitative law.
