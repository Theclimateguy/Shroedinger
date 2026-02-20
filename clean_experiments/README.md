# Clean Experiments

This folder contains cleaned, runnable scripts mapped to the experiments described in `manuscript/main.tex`.

## Mapping to manuscript

- `experiment_A.py`
  - Experiment A (`sec:numerics:A`): gauge invariance of `Lambda = Re Tr(F rho)` and noncommutativity diagnostics.
- `experiment_D.py`
  - Experiment D (`sec:numerics:D`): full balance closure on `(t, x, mu)` with explicit vertical flux/source terms.
- `experiment_E.py`
  - Experiment E (`sec:numerics:EFG`): coherence-driven vertical rates and predictor comparison vs `|Lambda|` features.
- `experiment_F.py`
  - Experiment F (`sec:numerics:EFG`): sinusoidal law check `Lambda ~ sin(varphi)` and discretization convergence.
- `experiment_G.py`
  - Experiment G (`sec:numerics:EFG`): profile scan with fixed total phase `varphi = integral omega dmu`.
- `experiment_G2_toy_chain.py`
  - Experiment G2 toy-chain branch (`sec:numerics:G2`): Clausius regression scan over `epsilon`.
- `experiment_G2_single_qubit.py`
  - Experiment G2 single-qubit branch (`sec:numerics:G2`): Clausius regression under profile scans.

## Quick smoke runs

```bash
python3 clean_experiments/experiment_A.py --out out/experiment_A
python3 clean_experiments/experiment_D.py --samples 20 --out out/experiment_D
python3 clean_experiments/experiment_E.py --samples 60 --out out/experiment_E
python3 clean_experiments/experiment_F.py --out out/experiment_F
python3 clean_experiments/experiment_G.py --out out/experiment_G
python3 clean_experiments/experiment_G2_toy_chain.py --quick --out out/experiment_G2_toy_chain
python3 clean_experiments/experiment_G2_single_qubit.py --quick --out out/experiment_G2_single_qubit
```
