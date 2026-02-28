# Shroedinger — Numerical Experiments Repository

This repository contains reproducible numerical experiment code, robust validation runs, and generated artifacts.

## Experiment coverage

The canonical runnable set is under `clean_experiments/` with continuous numbering:

1. `A` (gauge invariance + noncommutativity diagnostics)
2. `B` / wave-1 (`Lambda_matter`, commutator and sin-law checks)
3. `D` (balance closure on `(t,x,mu)`)
4. `E` (coherence-driven rate diagnostics)
5. `F` (sinusoidal law robustness)
6. `G` (fixed-phase profile scan)
7. `G2` toy-chain (Clausius regression vs epsilon)
8. `G2` single-qubit (Clausius regression vs omega profile)
9. `H` (holographic truncation with growing layer dimension)
10. `I` (continuum conservation extrapolation)
11. `J` (Berry phase refinement; risk R3)
12. `K` (`Lambda_matter` bridge to effective source proxy)
13. `K2` (theory-space curvature vs RG noncommutativity)
14. `L` (matter fields: fermion+gauge, Ward + continuity checks)
15. `M` (structural-scale atmospheric cosmo-flow test: `rho_mu`, interscale proxy, blocked CV)

Exploratory (non-numbered) script:

- `clean_experiments/experiment_QFT_free_chain.py`
  - minimal QFT-adjacent free-fermion RG shell test with exact Gaussian elimination and intertwinement-defect diagnostics.

Detailed mapping (scripts, output folders, robust folders) is in:

- `clean_experiments/EXPERIMENT_NUMBERING.md`
- `clean_experiments/HYPOTHESIS_ROADMAP.md`

## Results and artifacts

- `clean_experiments/results/` contains base and robust outputs in CSV format.
- `clean_experiments/results/INDEX.md` provides the artifact index by experiment.
- `clean_experiments/checklist_validations.py` and
  `clean_experiments/results/checklist_validations/` provide the cross-experiment checklist block.

## Experiment M acceptance criteria

Experiment M uses two decision thresholds with different intent:

1. **Theoretical Detection Threshold**
   - `min_mae_gain >= 0.002` (0.2%)
   - `perm_p <= 0.05`
   - `strata_positive_frac >= 0.8`
   - Meaning: sufficient to support the physical hypothesis (`F_{mu1,mu2} != 0`) and nontrivial coherence contribution.
2. **Engineering Impact Threshold**
   - `min_mae_gain >= 0.03` (3.0%)
   - Meaning: sufficient to justify computationally heavy production integration (Phase C).

Detailed protocol and latest calibration/results are documented in:

- `clean_experiments/EXPERIMENT_M_METHODS_AND_RESULTS.md`

## Legacy and working materials

- `experiments/` contains legacy/original scripts and notes used for historical replication.
- `stage_two/` contains auxiliary stage-2 working notes.
- `exp4_omega_profiles_fixed.csv` stores profile data used by one of the legacy experiment paths.

## Quick start

Run from repository root:

```bash
python clean_experiments/experiment_A.py --out out/experiment_A
python clean_experiments/experiment_wave1_user.py --out out/experiment_B_wave1
python clean_experiments/experiment_H_holographic.py --out out/experiment_H_holographic
python clean_experiments/experiment_L_matter_fields.py --out clean_experiments/results/experiment_L_matter_fields
python clean_experiments/experiment_M_cosmo_flow.py --input /path/to/wpwp_data.nc --out clean_experiments/results/experiment_M_cosmo_flow
```

Robust blocks:

```bash
python clean_experiments/experiment_wave1_robust.py --cases 40 --out clean_experiments/results/experiment_B_wave1_robust
python clean_experiments/experiment_H_holographic_robust.py --cases 24 --out clean_experiments/results/experiment_H_holographic_robust
python clean_experiments/experiment_L_matter_fields_robust.py --cases 10 --out clean_experiments/results/experiment_L_matter_fields_robust
python clean_experiments/experiment_M_cosmo_flow_robust.py --input /path/to/wpwp_data.nc --cases 8 --out clean_experiments/results/experiment_M_cosmo_flow_robust
python clean_experiments/checklist_validations.py --out clean_experiments/results/checklist_validations
```

## Not included

- `manuscript/` is intentionally excluded and is published separately.
