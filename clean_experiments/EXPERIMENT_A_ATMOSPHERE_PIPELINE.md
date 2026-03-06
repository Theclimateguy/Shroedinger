# Group A Main Pipeline (Atmosphere Data)

This runbook defines the canonical atmosphere pipeline (`ATMOSPHERE_DATA`).

## Canonical order

1. `A01` M1 detectability core
2. `A02` M2 consistency/placebo checks
3. `A03` M3 land/ocean + noise-limit probe
4. `A04` O1 thermodynamic baseline check
5. `A05` O2 spatial diagnostics
6. `A05.R1..R4` scale-space continuation (P1/P2)
7. `A06` M4 falsification block
8. `A07` F5 structural Lambda/fractal surrogates
9. `A08` F6b heavy-tail strict fits
10. `A09` F6c clustered subspace tails
11. `A10` F6c spatial patch maps

## A05 continuation (P1/P2) status

- `A05.R1_p1_spatial_occupancy_cascade`: completed.
- `A05.R2_p2_theory_bridge_c009`: completed, C009 selected.
- `A05.R3_p2_dense_c009`: completed (dense ingest + transfer check).
- `A05.R4_p2_l8_resolution`: completed as diagnostic block.

Program decision:
- Keep C009 as canonical baseline for Group A.
- Treat `l=8` retune variants as diagnostics.
- Next theory-close extension: P2-memory (retarded density matrix).

## Command skeleton

```bash
# A05.R1
python clean_experiments/experiment_P1_spatial_occupancy_cascade.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P1_spatial_occupancy_cascade \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --n-perm 49

# A05.R2
python clean_experiments/experiment_P2_theory_bridge_ablation.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P2_theory_bridge_ablation \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --top-k 4 \
  --n-perm-final 49

python clean_experiments/experiment_P2_noncommuting_coarse_graining.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --lambda-weights 1.5 1.0 1.0 1.0 \
  --lambda-scale-power 0.5 \
  --decoherence-alpha 0.5 \
  --n-perm 49

# A05.R3
python clean_experiments/run_p2_calibrated_dense_ingest.py \
  --top-events 16 \
  --context-hours 6 \
  --budget-gb 50

# A05.R4
python clean_experiments/experiment_P2_l8_diagnostic_block.py \
  --outdir clean_experiments/results/experiment_P2_l8_diagnostic_block
```

## Finalization artifacts

- `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`
- `clean_experiments/results/experiment_P2_l8_diagnostic_block/report.md`
- `clean_experiments/results/experiment_P2_l8_diagnostic_block/l8_diagnostic_decision_table.csv`
