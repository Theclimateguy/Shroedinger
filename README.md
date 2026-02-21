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

Detailed mapping (scripts, output folders, robust folders) is in:

- `clean_experiments/EXPERIMENT_NUMBERING.md`
- `clean_experiments/HYPOTHESIS_ROADMAP.md`

## Results and artifacts

- `clean_experiments/results/` contains base and robust outputs in CSV format.
- `clean_experiments/results/INDEX.md` provides the artifact index by experiment.
- `clean_experiments/checklist_validations.py` and
  `clean_experiments/results/checklist_validations/` provide the cross-experiment checklist block.

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
```

Robust blocks:

```bash
python clean_experiments/experiment_wave1_robust.py --cases 40 --out clean_experiments/results/experiment_B_wave1_robust
python clean_experiments/experiment_H_holographic_robust.py --cases 24 --out clean_experiments/results/experiment_H_holographic_robust
python clean_experiments/experiment_L_matter_fields_robust.py --cases 10 --out clean_experiments/results/experiment_L_matter_fields_robust
python clean_experiments/checklist_validations.py --out clean_experiments/results/checklist_validations
```

## Not included

- `manuscript/` is intentionally excluded and is published separately.
