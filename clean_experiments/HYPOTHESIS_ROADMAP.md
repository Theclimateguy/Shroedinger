# Hypothesis Validation Roadmap (Stage 2)

This roadmap maps each open hypothesis to a concrete numerical protocol and artifact.

## Priority Order

1. **H1**: realistic layer growth + holographic truncation (implemented)
2. **H2**: continuum conservation (`\nabla_a T^{ab}_{eff}=0`) from dynamics (implemented v1)
3. **H3**: Berry phase high-resolution verification (implemented)
4. **H4**: `Lambda_matter` to `Lambda_obs` calibration proxy (implemented v1)
5. **H4b**: theory-space curvature proxy (FS/QGT + response metric) vs RG noncommutativity (implemented)
6. **H5**: matter fields (`fermion + gauge`) embedding (implemented v1)
7. **H6**: cosmological flow (`omega(mu)`) to spectrum-level observables (implemented v1)
8. **H7**: microscopic derivation of thermodynamic postulates

## Experiment Map

| New ID | Hypothesis | Script | Core metric | Pass criterion |
|---|---|---|---|---|
| Experiment 9 (H) | Exponential `dim(H_mu)` growth + holographic truncation regularizes observables | `clean_experiments/experiment_H_holographic.py` | stability gain of regularized truncated integral across `K`-scan | `stability_gain > 1`, trace residual in Lindblad RHS below tolerance |
| Experiment 10 (I) | Covariant conservation in continuum limit | `clean_experiments/experiment_I_continuum_conservation.py` | intercept of residual fit vs `(dmu, dt)` and fit quality | small extrapolated intercept and stable fit across robust runs |
| Experiment 11 (J) | Berry phase artifact check (risk R3) | `clean_experiments/experiment_J_berry_refinement.py` | wrapped phase error vs analytic Berry law | finest-grid error remains below tolerance across angle scan |
| Experiment 12 (K) | `Lambda_matter` linkage to effective cosmological source proxy | `clean_experiments/experiment_K_lambda_bridge.py` | repeated-holdout median `R2`, permutation p-value, bootstrap/profile sign diagnostics | median holdout `R2` above threshold, permutation significance, stable coefficient-sign diagnostics |
| Experiment 13 (K2) | Theory-space curvature diagnostics vs RG noncommutativity | `clean_experiments/experiment_K2_theory_space_curvature.py` | corr(`source_spread`, commutator norm), corr(`source_spread`, `|Omega_FS|`), gauge/convergence checks | strong positive geometric correlations and stable gauge/coarse checks |
| Experiment 14 (L) | Matter-field embedding (fermion + gauge sectors) | `clean_experiments/experiment_L_matter_fields.py` | Ward identity (`||[Q,H]||_F`), local continuity residual, total-charge drift | commutator and continuity at numerical precision; drift bounded |
| Experiment 15 (M) | Structural-scale cosmological-flow test from atmospheric multiscale fields | `clean_experiments/experiment_M_cosmo_flow.py` | blocked-CV MAE gain of residual closure (baseline ctrl vs ctrl+`Lambda_struct`), permutation p-value, sign consistency | **Two-level policy**: Theoretical Detection Threshold: `min_mae_gain >= 0.002`, `perm_p <= 0.05`, `strata_positive_frac >= 0.8`; Engineering Impact Threshold: `min_mae_gain >= 0.03` |
| Experiment 16 (N, planned) | Thermodynamic postulates from microscopic channels | `clean_experiments/experiment_N_micro_to_thermo.py` | Clausius residual and area-law scaling quality | residual small in equilibrium regime |

## Artifact Standard (for all new experiments)

- `..._summary.csv` with one-row headline metrics and pass/fail flags.
- `..._dataset.csv` or `..._scan.csv` with raw sweep values.
- Optional robust block: `clean_experiments/results/<experiment>_robust/`.
- Add to:
  - `clean_experiments/README.md`
  - `clean_experiments/EXPERIMENT_NUMBERING.md`
  - `clean_experiments/results/INDEX.md`

## Immediate Next Action

Move to Experiment 16 (micro-to-thermo derivation) using the same artifact standard and robust protocol.
