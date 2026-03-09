# Hypothesis Validation Roadmap (Canonical Numbering)

This roadmap is aligned to the canonical experiment landscape in
`clean_experiments/EXPERIMENT_NUMBERING.md`.

Primary notation:
- `TOY_MODEL`: `T01 ... T20`
- `ATMOSPHERE_DATA`: `A01 ... A15`
- run-level extensions: `A05.R*`, `A07.R*`, `A11.E*`

Legacy `H*` labels are kept only as aliases for historical continuity.

## Legacy Alias Map (H* -> Canonical)

- `H1 -> T09`
- `H2 -> T10`
- `H3 -> T11`
- `H4 -> T12`
- `H4b -> T13`
- `H5 -> T14`
- Atmospheric validation stack previously referred to as `H6/H7/H8` is now tracked by canonical `A*` blocks.

## Priority Order (Canonical)

1. `T09` (`H1`) holographic truncation regularization
2. `T10` (`H2`) continuum conservation extrapolation
3. `T11` (`H3`) Berry-phase high-resolution check
4. `T12` (`H4`) `Lambda_matter -> Lambda_obs` bridge proxy
5. `T13` (`H4b`) theory-space curvature vs RG noncommutativity
6. `T14` (`H5`) matter-field embedding and continuity/Ward checks
7. `A01` detectability of structural-scale closure signal on atmospheric fields
8. `A04` thermodynamic consistency with Lambda correction
9. `A05` spatial detectability diagnostics
10. `A06` staged falsification of Lambda necessity
11. `A11 ... A14` strict transfer/halo boundary validation and falsification
12. `A15` ERA Einstein-in-a-box closure check

## Hypothesis-to-Experiment Map

| Canonical code | Legacy alias | Primary script(s) | Core metric | Pass criterion |
|---|---|---|---|---|
| `T09` | `H1` | `clean_experiments/experiment_H_holographic.py` | stability gain of truncated integral across `K` scan | `stability_gain > 1` and bounded trace residual |
| `T10` | `H2` | `clean_experiments/experiment_I_continuum_conservation.py` | extrapolated residual intercept vs `(dmu, dt)` | small/stable continuum-limit intercept |
| `T11` | `H3` | `clean_experiments/experiment_J_berry_refinement.py` | wrapped phase error vs analytic Berry law | finest-grid error below tolerance |
| `T12` | `H4` | `clean_experiments/experiment_K_lambda_bridge.py` | holdout `R2`, permutation p-value, coefficient-sign stability | positive stable signal under repeated holdout + permutation |
| `T13` | `H4b` | `clean_experiments/experiment_K2_theory_space_curvature.py` | corr(`source_spread`, commutator/FS responses), gauge/coarse checks | positive geometric correlations with stable control checks |
| `T14` | `H5` | `clean_experiments/experiment_L_matter_fields.py` | Ward commutator norm, continuity residual, charge drift | numerical-precision conservation/continuity |
| `A01` | `H6/H7 (legacy atmosphere core)` | `clean_experiments/experiment_M_cosmo_flow.py` | blocked-CV MAE gain (`baseline` vs `+Lambda_struct`), permutation | detection threshold: `min_mae_gain >= 0.002`, `perm_p <= 0.05`, `strata_positive_frac >= 0.8` |
| `A02` | `H7 (consistency controls)` | `clean_experiments/experiment_M_horizontal_vertical_compare.py` | cross-reconstruction consistency and placebo controls | signal above placebo tail with stable cross-consistency |
| `A03` | `H7 (surface split)` | `clean_experiments/experiment_M_land_ocean_split.py`, `clean_experiments/experiment_M_land_ocean_noise_probe.py` | surface-split gain under target variants | positive ocean detectability + interpretable land noise behavior |
| `A04` | `H8` | `clean_experiments/experiment_O_entropy_equilibrium.py` | FT-domain `R2` gain vs Clausius baseline | Lambda increment above noise floor with stable baseline |
| `A05` | `H8 (spatial)` | `clean_experiments/experiment_O_spatial_variance.py`, `clean_experiments/experiment_O_lambda_spatial_viz.py`, `clean_experiments/experiment_O_spatial_active_west.py` | spatial gain and climatology correlation diagnostics | reproducible spatial diagnostics with explicit detectability bounds |
| `A06` | `H7 falsification` | `clean_experiments/experiment_M_lambda_falsification_tests.py` | staged S1/S2/S3 falsification metrics | real Lambda outperforms placebo/commutator controls |
| `A11` | `M5 strict transfer` | `clean_experiments/experiment_M_gksl_hybrid_strict.py` | `gain_phi_vs_lambda` on strict holdouts | signed strict endpoint (including negative/near-zero as falsification evidence) |
| `A12` | `M6 strict halo boundary` | `clean_experiments/experiment_M_halo_boundary_strict.py` | gain with local halo context on test/external years | positive/significant transfer under blocked chronology |
| `A13` | `M7 width scan` | `clean_experiments/experiment_M_halo_boundary_strict.py` | gain vs `halo_width` | non-trivial width dependence; `w=0` context collapse |
| `A14` | `M8 halo falsification` | `clean_experiments/experiment_M_halo_boundary_strict.py` | local vs remote/misaligned gain contrast | local adjacent halo outperforms controls |
| `A15` | `EIB ERA endpoint` | `clean_experiments/experiment_scale_gravity_einstein_box_era.py` | regime-wise `Lambda ~ Pi` (especially inertial) | inertial binned slope positive and significant with target `R2` |

## Canonical A05 Run-Level Continuation

| Run ID | Primary script(s) | Validation target |
|---|---|---|
| `A05.R1` | `clean_experiments/experiment_P1_spatial_occupancy_cascade.py` | occupancy cascade detectability in scale space |
| `A05.R2` | `clean_experiments/experiment_P2_noncommuting_coarse_graining.py`, `clean_experiments/experiment_P2_theory_bridge_ablation.py` | calibrated noncommuting bridge on sparse panel |
| `A05.R3` | `clean_experiments/run_p2_calibrated_dense_ingest.py`, `clean_experiments/experiment_P2_noncommuting_coarse_graining.py` | sparse-to-dense transfer stability |
| `A05.R4` | `clean_experiments/experiment_P2_l8_diagnostic_block.py` | `l=8` failure-mode isolation (event/operator/resolution/regime) |
| `A05.R5` | `clean_experiments/experiment_P2_memory.py` | retarded-memory recovery at `l=8` |
| `A05.R6` | `clean_experiments/experiment_P2_memory_gksl_cptp.py` | full GKSL/CPTP memory closure with CPTP diagnostics |

## Canonical A07 and A11 Extension Logs

- `A07.R*`: frozen granular MRMS+GOES/M-realpilot continuation family.
- `A11.E*`: transfer diagnostics outside canonical `A11 ... A14` endpoints.

Detailed mapping and canonical output folders remain in:
- `clean_experiments/EXPERIMENT_NUMBERING.md`

## Data Download / Ingest Scripts Present in Repository

Atmospheric and granular data ingestion utilities currently present:

- `clean_experiments/download_N_data_era5.py`
- `clean_experiments/download_N_data_merra2.py`
- `clean_experiments/download_mrms.py`
- `clean_experiments/download_goes.py`
- `clean_experiments/download_matched_windows.py`
- `clean_experiments/download_matched_parallel.py`
- `clean_experiments/build_mrms_goes_aligned_catalog.py`
- `clean_experiments/run_ultralight_mrms_goes_pilot.py`

Repository scope note: generated heavy artifacts are local-only by default.
