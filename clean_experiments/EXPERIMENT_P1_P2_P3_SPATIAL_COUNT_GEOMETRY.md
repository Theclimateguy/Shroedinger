# Experiment Spec: P1/P2/P3 Spatial Count Geometry

Program placement:
- this block is now part of the canonical atmosphere Group A pipeline as
  `A05` run-level continuation (`A05.R1 ... A05.R5`).

## Goal
Shift from time-ahead forecasting to a scale-space formulation where the scale coordinate is spatial:

- scale coordinate: `mu = log(l)` where `l` is tile/window size
- primary observable: structural density per unit area inside `B_l(x)`

For structure class `C`:

`n_C(x, l, t) = N_C(B_l(x), t) / |B_l(x)|`

This aligns with existing `M/O/F5` branches where local `Lambda` and inter-scale structure are already central.

## P1: Occupancy Cascade

### Question
Does adding local lambda information improve closure from `l` to `2l`?

### Operational form
- baseline model: predict `n_C(x, 2l, t)` from fine-scale tile features at `l`
- full model: same baseline + `Lambda_local(x, l, t)`
- CV: blocked by event (leave-one-event-out)

### P1-lite implementation
- script: `clean_experiments/experiment_P1_spatial_occupancy_cascade.py`
- default data: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- default structure class `C`: connected components of MRMS active mask `(rate >= threshold)`

### Lambda proxy (P1-lite)
Commutation-defect-inspired proxy:

- path A: average fine-tile component density over four `l x l` subtiles
- path B: component density on parent `2l x 2l` tile
- defect: `Delta_comm = |A - B|`
- proxy: normalized `Delta_comm` (then z-scored within `(event, scale)`)

### Main artifacts
- `summary_metrics.csv`
- `spatial_tile_dataset.csv`
- `spatial_tile_scan.csv`
- `lambda_local_map_scale_*.png`
- `comm_defect_map_scale_*.png`
- `occupancy_gain_map_scale_*.png`
- `report.md`

### Suggested pass/fail gates
- H1-space: `mean(Delta_comm) > floor` and stronger `Delta_comm` in active regimes
- H2-space: `MAE_gain > 0` and permutation `p <= 0.05`
- H3-space: active-regime gain > calm-regime gain

## P2: Noncommuting Coarse-Graining

### Core object
Compare two paths:

- `Pi_{l->2l} o Phi`
- `Phi o Pi_{l->2l}`

with defect:

`Delta_comm(x, l, t) = ||Pi Phi - Phi Pi||`

### Implemented form (current)
- script: `clean_experiments/experiment_P2_noncommuting_coarse_graining.py`
- data: `clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv`
- `Pi`: block-mean coarse-graining from `l` to `2l`
- `Phi` operators: threshold occupancy, square, log1p, gradient magnitude
- operator defects: `delta_occ`, `delta_sq`, `delta_log`, `delta_grad`

### Density-matrix bridge (theory-close)
From scale-pair structural densities `(n_l, n_2l)` build local occupancy density matrix:

`rho_occ = [[p_l, eta*sqrt(p_l p_2l)], [eta*sqrt(p_l p_2l), p_2l]]`

where `p_l, p_2l` are normalized populations and `eta` is decoherence factor from empirical comm-defect.

Define scale generators `A` and `B` and compute:

`F_comm = i [A, B]`

`lambda_local = Re Tr(F_comm rho_occ)`

This is now the primary P2 lambda observable used in closure tests.

### Calibration and dense ingest (implemented)
- ablation script: `clean_experiments/experiment_P2_theory_bridge_ablation.py`
- dense ingest orchestrator: `clean_experiments/run_p2_calibrated_dense_ingest.py`
- selected calibrated config (locked baseline): `C009`
  - `decoherence_alpha = 0.5`
  - `lambda_scale_power = 0.5`
  - weights `[occ, sq, log, grad] = [1.5, 1.0, 1.0, 1.0]`
- dense ingest output (current):
  - `clean_experiments/results/realpilot_2024_p2dense_calibrated/`
  - 16 stable events, `+/- 6h` context, unified panel with 240 rows.

### P2 `l=8` diagnostic finalization (March 2026)

- diagnostic block: `clean_experiments/experiment_P2_l8_diagnostic_block.py`
- external resolution memo (used for final program documentation):
  `clean_experiments/results/experiment_P2_l8_diagnostic_block/report_P2_l8_resolution.md`
- main findings:
  - matched-event control: degradation at `l=8` is not only event-pool shift.
  - pure scalar bridge retuning (`alpha`, `w_occ`, `scale_power`) does not recover significance.
  - operator/resolution sensitivity exists on finest scale (notably `sq` channel and active threshold).
  - regime split shows degradation concentrated in calm regime.
- program decision:
  - keep C009 as canonical P2 baseline for Group A pipeline.
  - treat targeted `l=8` resolution tests as diagnostic evidence supporting a
    theory-consistent next step: retarded density matrix (`P2-memory`), rather
    than ad-hoc production retuning.

### P2-memory finalization (March 2026)

- script: `clean_experiments/experiment_P2_memory.py`
- spec: `clean_experiments/EXPERIMENT_P2_MEMORY.md`
- run output: `clean_experiments/results/experiment_P2_memory/`
- implemented form:
  - lagged tile-state features for `t-1` and `t-2`
  - exponential memory weights with parameters `eta` and `tau`
  - optional persistence gating
  - memory applied only on selected scales, default `l=8`
  - locked baseline remains C009 with full operator set and threshold `3.0`
- confirmed final config:
  - `memory_source = occupancy`
  - `lookback = 2`
  - `memory_eta = 0.8`
  - `memory_tau = 0.75`
  - `persistence_power = 0.0`
- confirmed results:
  - `l=8`: `mae_gain = 1.224306e-06`, `r2_gain = 1.343877e-04`,
    `perm_p = 0.02`, `event_positive_frac = 0.9375`, `PASS_ALL = True`
  - `ALL`: `mae_gain = 1.372379e-06`, `r2_gain = 2.177551e-04`,
    `perm_p = 0.02`, `PASS_ALL = True`
- interpretation:
  - dense fine-scale degradation is consistent with missing memory, not with a
    need for manual operator dropping.
  - the A05 scale-space series is now closed by a theory-close retarded bridge.

## P3: Structure Thermodynamics

### Core idea
Regress local disorder/entropy-like metrics on structure-density fields and flux surrogates, then test the incremental value of `Lambda_local`.

This is the spatial continuation of O-branch logic but in tile-level scale-space form.

## Data Roadmap

### Already available (P1-lite)
- aligned MRMS event snapshots from realpilot panel

### Needed for stronger P1/P2
- dense intra-event ABI/GLM frames (not just one matched frame)
- consistent tile-grid cubes for all frames

### Needed for P3
- LES/CRM-like convective-resolving data (`<1 km`) or equivalent radar-resolving fields

## Run Example

```bash
python clean_experiments/experiment_P1_spatial_occupancy_cascade.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P1_spatial_occupancy_cascade \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --n-perm 49

python clean_experiments/experiment_P2_noncommuting_coarse_graining.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P2_noncommuting_coarse_graining \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --lambda-weights 1.0 1.0 1.0 1.0 \
  --decoherence-alpha 4.0 \
  --n-perm 49

python clean_experiments/experiment_P2_theory_bridge_ablation.py \
  --panel-csv clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv \
  --outdir clean_experiments/results/experiment_P2_theory_bridge_ablation \
  --scales-cells 8 16 32 \
  --mrms-downsample 16 \
  --mrms-threshold 3.0 \
  --top-k 4 \
  --n-perm-final 49

python clean_experiments/run_p2_calibrated_dense_ingest.py \
  --top-events 16 \
  --context-hours 6 \
  --budget-gb 50

python clean_experiments/experiment_P2_memory.py \
  --outdir clean_experiments/results/experiment_P2_memory \
  --top-k 6 \
  --final-n-perm 49 \
  --all-scales-n-perm 49
```

## Scale Parameterization Guidance
If detectability is weak:

1. scan `--scales-cells` (example: `6 12 24`, `8 16 32`, `10 20 40`)
2. scan `--mrms-downsample` to shift physical scale represented by one tile cell
3. scan `--mrms-threshold` to redefine active structures
4. pick scale setup by best `MAE_gain + permutation significance`
