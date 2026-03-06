# Hypothesis Validation Roadmap (Stage 2)

This roadmap maps each stage-2 hypothesis to a concrete numerical protocol and artifact.

Repository scope note: CSV/JSON/PNG artifacts mentioned in this roadmap are generated
locally during runs and are not versioned by default in GitHub.

## Priority Order

1. **H1**: realistic layer growth + holographic truncation (implemented)
2. **H2**: continuum conservation (`\nabla_a T^{ab}_{eff}=0`) from dynamics (implemented v1)
3. **H3**: Berry phase high-resolution verification (implemented)
4. **H4**: `Lambda_matter` to `Lambda_obs` calibration proxy (implemented v1)
5. **H4b**: theory-space curvature proxy (FS/QGT + response metric) vs RG noncommutativity (implemented)
6. **H5**: matter fields (`fermion + gauge`) embedding (implemented v1)
7. **H6**: cosmological flow (`omega(mu)`) to spectrum-level observables (implemented v1)
8. **H7**: Navier-Stokes moisture-budget closure with Lambda coupling (implemented v1 + follow-up)
9. **H8**: thermodynamic equilibrium/anomaly decomposition with spatial non-averaged diagnostics (implemented v1)

## Experiment Map

| Block ID | Hypothesis | Script | Core metric | Pass criterion |
|---|---|---|---|---|
| H1 | Exponential `dim(H_mu)` growth + holographic truncation regularizes observables | `clean_experiments/experiment_H_holographic.py` | stability gain of regularized truncated integral across `K`-scan | `stability_gain > 1`, trace residual in Lindblad RHS below tolerance |
| H2 | Covariant conservation in continuum limit | `clean_experiments/experiment_I_continuum_conservation.py` | intercept of residual fit vs `(dmu, dt)` and fit quality | small extrapolated intercept and stable fit across robust runs |
| H3 | Berry phase artifact check | `clean_experiments/experiment_J_berry_refinement.py` | wrapped phase error vs analytic Berry law | finest-grid error remains below tolerance across angle scan |
| H4 | `Lambda_matter` linkage to effective cosmological source proxy | `clean_experiments/experiment_K_lambda_bridge.py` | repeated-holdout median `R2`, permutation p-value, bootstrap/profile sign diagnostics | median holdout `R2` above threshold, permutation significance, stable coefficient-sign diagnostics |
| H4b | Theory-space curvature diagnostics vs RG noncommutativity | `clean_experiments/experiment_K2_theory_space_curvature.py` | corr(`source_spread`, commutator norm), corr(`source_spread`, `|Omega_FS|`), gauge/convergence checks | strong positive geometric correlations and stable gauge/coarse checks |
| H5 | Matter-field embedding (fermion + gauge sectors) | `clean_experiments/experiment_L_matter_fields.py` | Ward identity (`||[Q,H]||_F`), local continuity residual, total-charge drift | commutator and continuity at numerical precision; drift bounded |
| M1 | Structural-scale cosmological-flow test from atmospheric multiscale fields | `clean_experiments/experiment_M_cosmo_flow.py` | blocked-CV MAE gain of residual closure (baseline ctrl vs ctrl+`Lambda_struct`), permutation p-value, sign consistency | **Two-level policy**: Theoretical Detection Threshold: `min_mae_gain >= 0.002`, `perm_p <= 0.05`, `strata_positive_frac >= 0.8`; Engineering Impact Threshold: `min_mae_gain >= 0.03` |
| M2 | Structural consistency (`Lambda_h` vs `Lambda_v`) and placebo controls | `clean_experiments/experiment_M_horizontal_vertical_compare.py` | cross-reconstruction correlation, synthetic placebo tail checks | strong cross-reconstruction consistency and signal above placebo tail |
| M3 | Land/ocean detectability and noise-limit probe | `clean_experiments/experiment_M_land_ocean_split.py`, `clean_experiments/experiment_M_land_ocean_noise_probe.py` | split gain by surface type under target variants | positive ocean detectability with physically interpretable land-noise behavior |
| O1 | Clausius thermodynamic consistency with Lambda correction | `clean_experiments/experiment_O_entropy_equilibrium.py` | FT-domain `R2` gain vs Clausius baseline, permutation significance | Clausius baseline stable; Lambda increment interpreted against noise floor |
| O2 | Spatial anomaly diagnostics for macro-signal detectability | `clean_experiments/experiment_O_spatial_variance.py`, `clean_experiments/experiment_O_lambda_spatial_viz.py`, `clean_experiments/experiment_O_spatial_active_west.py` | spatial gain pattern diagnostics and climatology correlation checks | reproducible spatial diagnostics with explicit detectability limits |
| A05.R1 | Process-resolved scale-space occupancy cascade (`l -> 2l`) | `clean_experiments/experiment_P1_spatial_occupancy_cascade.py` | per-scale MAE/R2 gain, permutation p-value, event-level robustness | positive gain/significance on fine-to-mid scales with blocked-by-event CV |
| A05.R2 | Noncommuting coarse-graining density-matrix bridge (C009 calibration) | `clean_experiments/experiment_P2_noncommuting_coarse_graining.py`, `clean_experiments/experiment_P2_theory_bridge_ablation.py` | bridge gain vs baseline, commutator diagnostics, all-scale pass on sparse panel | calibrated C009 must retain positive all-scale gain and permutation significance on sparse panel |
| A05.R3 | Dense intra-event transferability of calibrated bridge | `clean_experiments/run_p2_calibrated_dense_ingest.py`, `clean_experiments/experiment_P2_noncommuting_coarse_graining.py` | sparse-vs-dense transfer table (ALL + per-scale) under blocked event CV | preserve global gain while explicitly diagnosing fine-scale regime sensitivity |
| A05.R4 | `l=8` diagnostic resolution block (matched-event/operator/resolution/regime) | `clean_experiments/experiment_P2_l8_diagnostic_block.py` | matched-event control, operator attribution, threshold/downsample sensitivity, regime split | isolate whether degradation is event-pool, operator, or resolution-driven and map path to theory-consistent memory extension |
| A05.R5 | Retarded density-matrix memory extension on dense panel | `clean_experiments/experiment_P2_memory.py` | sign flip and significance recovery on `l=8`, plus restored all-scale pass vs dense C009 baseline | recover positive `l=8` gain and `perm_p <= 0.05` under full operators and threshold `3.0`, without ad-hoc retune |
| M4 | Lambda necessity falsification (scale permutation / commutator control / IC) | `clean_experiments/experiment_M_lambda_falsification_tests.py` | S1/S2/S3 staged falsification metrics | real Lambda must outperform placebo/comm controls and improve IC criteria |

Atmosphere extension branch (outside canonical `A01-A10` landscape block):

- N moisture-budget closure follow-up:
  - `clean_experiments/experiment_N_navier_stokes_budget.py`
  - `clean_experiments/experiment_N_followup_dual.py`

## Artifact Standard (for stage-2 experiments)

- Core metric tables (`test_metrics.csv` and/or `summary_metrics.csv`) with headline diagnostics.
- `..._dataset.csv` or `..._scan.csv` with raw sweep values.
- Optional robust block: `clean_experiments/results/<experiment>_robust/`.
- Add to:
  - `clean_experiments/README.md`
  - `clean_experiments/EXPERIMENT_NUMBERING.md`
  - `clean_experiments/results/INDEX.md`

## Immediate Next Action

The A05 process-resolved continuation is now numerically closed through
`A05.R5_p2_memory`. Next work item is manuscript integration, with P2-memory
described as the final theory-close fine-scale closure and with explicit caveat
that the current implementation is a minimal retarded surrogate.
