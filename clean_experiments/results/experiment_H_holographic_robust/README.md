# Experiment 9 (H) Robust Sweep

This folder stores robust validation runs for:

- `clean_experiments/experiment_H_holographic_robust.py`

## Reproduce

```bash
python clean_experiments/experiment_H_holographic_robust.py \
  --cases 24 \
  --out clean_experiments/results/experiment_H_holographic_robust
```

## Main artifacts

- `robustness_results.csv`
- `robustness_summary.csv`
- `worst_12_by_stability_gain.csv`
- `cases/case_*/experiment_H_summary.csv`

## Headline from current run

- `fraction_pass_all = 1.0` (`24/24`)
- `median_stability_gain = 3.9309`
- `min_stability_gain = 1.7664`
- `max_trace_raw = 2.22e-16`
- `max_trace_cut = 2.31e-16`
