# Clean Experiments

This folder contains cleaned, runnable scripts mapped to the experiments described in `manuscript/main.tex`.

## Continuous numbering (article order)

1. **Experiment 1 (A)**: `experiment_A.py`
   - `sec:numerics:A`: gauge invariance of `Lambda = Re Tr(F rho)` and noncommutativity diagnostics.
2. **Experiment 2 (B, wave-1)**: `experiment_wave1_user.py`
   - Newly integrated first-wave block: commutator algebra, Hermitian curvature projection, spatial `Lambda_matter(x)`, and `Lambda_matter ~ sin(phi)` law.
3. **Experiment 3 (D)**: `experiment_D.py`
   - `sec:numerics:D`: full balance closure on `(t, x, mu)` with explicit vertical flux/source terms.
4. **Experiment 4 (E)**: `experiment_E.py`
   - `sec:numerics:EFG`: coherence-driven vertical rates and predictor comparison vs `|Lambda|` features.
5. **Experiment 5 (F)**: `experiment_F.py`
   - `sec:numerics:EFG`: sinusoidal law check `Lambda ~ sin(varphi)` and discretization convergence.
6. **Experiment 6 (G)**: `experiment_G.py`
   - `sec:numerics:EFG`: profile scan with fixed total phase `varphi = integral omega dmu`.
7. **Experiment 7 (G2 toy-chain)**: `experiment_G2_toy_chain.py`
   - `sec:numerics:G2`: Clausius regression scan over `epsilon`.
8. **Experiment 8 (G2 single-qubit)**: `experiment_G2_single_qubit.py`
   - `sec:numerics:G2`: Clausius regression under profile scans.
9. **Experiment 9 (H, stage-2)**: `experiment_H_holographic.py`
   - Exponential `dim(H_mu)` growth with holographic truncation: impact on integral regularization and vertical dynamics stability.
10. **Experiment 10 (I, stage-2)**: `experiment_I_continuum_conservation.py`
   - Continuum extrapolation of the balance residual derived from microscopic dynamics (`dt,dmu -> 0` trend test).
11. **Experiment 11 (J, stage-2)**: `experiment_J_berry_refinement.py`
   - Berry-phase refinement (`R3`) with fine `mu`-step scan and robust angle sweep.
12. **Experiment 12 (K, stage-2)**: `experiment_K_lambda_bridge.py`
   - Bridge test for `Lambda_matter` contribution to effective `Lambda` proxy using repeated stratified holdout, permutation significance, and bootstrap coefficient stability.
13. **Experiment 13 (K2, stage-2)**: `experiment_K2_theory_space_curvature.py`
   - Theory-space geometry check: finite-difference FS/QGT and response-metric curvature diagnostics vs RG noncommutativity.
14. **Experiment 14 (L, stage-2)**: `experiment_L_matter_fields.py`
   - Matter-field embedding (`fermion + gauge`) with global Ward (`[Q,H]=0`) and local continuity residual checks.
15. **Experiment 15 (M, stage-2)**: `experiment_M_cosmo_flow.py`
   - Structural-scale atmospheric experiment: `rho_mu` from modal covariances, interscale noncommutativity proxy, and blocked-CV residual-closure test.

The full mapping table is in `clean_experiments/EXPERIMENT_NUMBERING.md`.
Stage-2 hypothesis-level protocol is in `clean_experiments/HYPOTHESIS_ROADMAP.md`.

## Experiment M decision policy

Experiment 15 uses a dual-threshold policy:

1. **Theoretical Detection Threshold**
   - `min_mae_gain >= 0.002`
   - `perm_p <= 0.05`
   - `strata_positive_frac >= 0.8`
   - Used to validate detectability of the structural-scale physical effect.
2. **Engineering Impact Threshold**
   - `min_mae_gain >= 0.03`
   - Used to decide whether to pursue heavy production/operational integration.

Reference methodology and calibrated runs:

- `clean_experiments/EXPERIMENT_M_METHODS_AND_RESULTS.md`

## Exploratory RG/QFT sanity check

- `experiment_QFT_free_chain.py`
  - Free-fermion chain with Wilsonian shell elimination (`Lambda_n = pi * 2^{-n}`), exact Gaussian trace-out, and intertwinement defect diagnostics (`exact_pushforward` vs fitted Markovian generators).
  - Default run includes **3 size checks**: `N = 32, 64, 128`.

## Checklist validations (non-numbered)

- `checklist_validations.py`
  - Unified reproducibility checks for the checklist claims used in the manuscript:
    unitarity/cocycle/isometry, Lindblad trace preservation, classical limit,
    Bianchi identity, two-layer total-norm conservation, and wave-1
    (`Lambda_matter`) laws.
  - Outputs: `clean_experiments/results/checklist_validations/`.

## Quick smoke runs

```bash
python clean_experiments/experiment_A.py --out out/experiment_A
python clean_experiments/experiment_wave1_user.py --out out/experiment_B_wave1
python clean_experiments/experiment_D.py --samples 20 --out out/experiment_D
python clean_experiments/experiment_E.py --samples 60 --out out/experiment_E
python clean_experiments/experiment_F.py --out out/experiment_F
python clean_experiments/experiment_G.py --out out/experiment_G
python clean_experiments/experiment_G2_toy_chain.py --quick --out out/experiment_G2_toy_chain
python clean_experiments/experiment_G2_single_qubit.py --quick --out out/experiment_G2_single_qubit
python clean_experiments/experiment_H_holographic.py --out out/experiment_H_holographic
python clean_experiments/experiment_I_continuum_conservation.py --out out/experiment_I_continuum_conservation --summary-only
python clean_experiments/experiment_J_berry_refinement.py --out out/experiment_J_berry_refinement
python clean_experiments/experiment_K_lambda_bridge.py --out clean_experiments/results/experiment_K_lambda_bridge
python clean_experiments/experiment_K2_theory_space_curvature.py --out clean_experiments/results/experiment_K2_theory_space_curvature
python clean_experiments/experiment_L_matter_fields.py --out clean_experiments/results/experiment_L_matter_fields
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow
python clean_experiments/experiment_QFT_free_chain.py --out clean_experiments/results/experiment_QFT_free_chain
python clean_experiments/experiment_wave1_robust.py --cases 40 --out clean_experiments/results/experiment_B_wave1_robust
python clean_experiments/experiment_H_holographic_robust.py --cases 24 --out clean_experiments/results/experiment_H_holographic_robust
python clean_experiments/experiment_I_continuum_conservation_robust.py --cases 12 --out clean_experiments/results/experiment_I_continuum_conservation_robust
python clean_experiments/experiment_J_berry_refinement_robust.py --out clean_experiments/results/experiment_J_berry_refinement_robust
python clean_experiments/experiment_K_lambda_bridge_robust.py --out clean_experiments/results/experiment_K_lambda_bridge_robust --runs 4
python clean_experiments/experiment_K2_theory_space_curvature_robust.py --out clean_experiments/results/experiment_K2_theory_space_curvature_robust --cases 12
python clean_experiments/experiment_L_matter_fields_robust.py --out clean_experiments/results/experiment_L_matter_fields_robust --cases 10
python clean_experiments/experiment_M_cosmo_flow_robust.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow_robust --cases 8
python clean_experiments/checklist_validations.py --out clean_experiments/results/checklist_validations
```

Calibrated Experiment M run used in the latest results package:

```bash
python clean_experiments/experiment_M_cosmo_flow.py \
  --input data/processed/wpwp_era5_2017_2019_experiment_M_input.nc \
  --out clean_experiments/results/experiment_M_cosmo_flow_v3_calibrated \
  --residual-mode physical_zscore \
  --coherence-mode offdiag_fro \
  --n-modes-per-var 6 \
  --window 18 \
  --cov-shrinkage 0.1 \
  --ridge-alpha 1e-6 \
  --min-positive-strata-frac 0.67 \
  --n-perm 140
```
