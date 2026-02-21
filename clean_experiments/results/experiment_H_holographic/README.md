# Experiment 9 (H): Holographic Scaling

This folder stores baseline outputs for:

- `clean_experiments/experiment_H_holographic.py`

## Reproduce

```bash
python clean_experiments/experiment_H_holographic.py \
  --out clean_experiments/results/experiment_H_holographic \
  --k-layers 20 \
  --resolution-scan 8,12,16,20,24
```

## Main artifacts

- `experiment_H_dimension_profile.csv`
- `experiment_H_layer_metrics.csv`
- `experiment_H_resolution_scan.csv`
- `experiment_H_summary.csv`

## Baseline headline (current run)

- `mean_cut_fraction = 0.5251`
- `resolution_stability_gain = 5.5051`
- `trace_raw_max = 8.74e-17`
- `trace_cut_max = 8.74e-17`
- `pass_all = True`
