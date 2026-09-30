# clean_experiments

Scripts and results of the gen4 programme. The gen1-2 layer (toy models
T01-T20, atmosphere series A01-A15 and their continuations) was removed from
this branch on 2026-09-30; it remains in branches `gen1`-`gen3` and in tags up
to `v6.4`. Statuses of all 73 earlier experiments: `docs/RESEARCH_PROGRAM_FULL.csv`.

## What is here

- `experiment_B1..B31_*.py` — phases of the programme, each under a frozen
  protocol `docs/PROTOCOL_PHASE*.md`; `experiment_A1..A4_*.py` — methods
  audits (`docs/PROTOCOL_AUDIT*.md`).
- `explore_physics_*.py` — exploratory scripts of 2026-09-29/30 (not frozen;
  register: `docs/EXPLORATION_PHYSICS_2026-09-29.md`).
- `describe_*`, `verify_*`, `visualize_*` — descriptive layer and figures of
  the papers.
- `download_*.py` — resumable downloaders; raw fields go to `data/` (not
  versioned).
- `experiment_M_cosmo_flow.py`, `experiment_scale_gravity_einstein_box_era.py`
  — two earlier modules kept because programme code imports from them.
- `results/<experiment>/` — `summary.json`, `report.md`, per-unit JSON shards
  and figures. Only `.json`, `.md` and selected figures are versioned.

## Estimator note (2026-09-30)

Scripts up to B30 compute the anchored coupling on the native lat-lon grid
with the sub-50 km band and FFT surrogates. That estimator carries a
latitude-dependent artefact (see `docs/RECONCILIATION.md`). New computations
should use the isotropic grid without the sub-50 km band
(`explore_physics_tiles.py --isogrid 1`, `experiment_B31_boundary_template.py`).

Run from the repository root, e.g.
`python clean_experiments/experiment_B31_boundary_template.py --stage tests`.
