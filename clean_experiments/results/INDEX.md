# Robust Validation Index

This directory stores robust validation artifacts for each clean experiment.

0. `checklist_validations/` (cross-experiment checklist block)
   - Structural + wave-1 checklist validations used in the manuscript:
     unitarity/cocycle/isometry, Lindblad trace-preservation, classical limit,
     Bianchi identity, two-layer aggregate norm conservation, state-dependent
     source, exact sin-law, symmetrization reality, spatial modulation,
     mixed-state neutrality, and `Lambda_vac` linearity in `Delta alpha`.
   - See `checklist_summary.csv`, `classical_limit_scan.csv`,
     `two_layer_norm_timeseries.csv`.

1. `experiment_A_robust/` (Experiment 1)
   - Gauge invariance + noncommutativity checks.
   - See `robustness_grid_summary.csv`, `robustness_full_gauge_summary.csv`.
2. `experiment_B_wave1_robust/` (Experiment 2)
   - Wave-1 commutator + Hermitian projection + `Lambda_matter ~ sin(phi)` robustness sweep.
   - See `robustness_summary.csv`.
3. `experiment_D_robust/` (Experiment 3)
   - Full balance closure checks on `(t, x, mu)`.
   - See `combined_summary.csv`.
4. `experiment_E_robust/` (Experiment 4)
   - Coherence-vs-Lambda predictive power in coherence-driven rate regime.
   - See `robustness_summary.csv`.
5. `experiment_F_robust/` (Experiment 5)
   - Sinusoidal law robustness under random parameter sweeps.
   - See `robustness_summary.csv`.
6. `experiment_G_robust/` (Experiment 6)
   - Fixed-phase profile invariance robustness.
   - See `robustness_summary.csv`.
7. `experiment_G2_toy_chain_robust/` (Experiment 7)
   - Epsilon-scan robustness across seeds.
   - See `robustness_summary_by_eps.csv`, `robustness_summary_global.csv`.
8. `experiment_G2_single_qubit_robust/` (Experiment 8)
   - Clausius-regression profile ranking robustness.
   - See `robustness_summary.csv`.
9. `experiment_H_holographic_robust/` (Experiment 9)
   - Exponential layer-dimension growth with holographic truncation:
     regularization stability and Lindblad trace checks across random regimes.
   - See `robustness_summary.csv`.
10. `experiment_I_continuum_conservation_robust/` (Experiment 10)
   - Continuum extrapolation robustness for balance residual
     (`dt,dmu -> 0`) from microscopic evolution.
   - See `robustness_summary.csv`.
11. `experiment_J_berry_refinement_robust/` (Experiment 11)
   - Berry-phase refinement robustness across polar-angle scans and
     step-size refinement.
   - See `robustness_summary.csv`.
12. `experiment_K_lambda_bridge_robust/` (Experiment 12)
   - Bridge robustness for `Lambda_matter` to effective-`Lambda` proxy:
     repeated stratified holdout median-`R2`, permutation significance, and
     bootstrap sign-consistency diagnostics.
   - See `robustness_summary.csv`, `worst_10_runs.csv`.
13. `experiment_K2_theory_space_curvature_robust/` (Experiment 13)
   - Theory-space curvature robustness sweep:
     FS/QGT and response-metric geometry under random parameter regimes,
     with gauge-invariance and coarse-grid stability checks.
   - See `robustness_summary.csv`, `robustness_results.csv`.
14. `experiment_L_matter_fields_robust/` (Experiment 14)
   - Matter-field (`fermion + gauge`) embedding robustness:
     Ward commutator, local continuity residual, and total-charge drift.
   - See `robustness_summary.csv`, `robustness_results.csv`.
