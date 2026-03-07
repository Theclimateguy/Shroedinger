# Experiment M branch log (March 7, 2026): GKSL transfer -> strict halo boundary

## Scope

This note captures all Experiment M branch runs executed on **March 7, 2026** for the chain:

1. transfer of retarded/GKSL memory proxies to ERA5 moisture-closure,
2. strict chronological checks (`train<=2019 -> test=2020 -> external=2021`),
3. final halo boundary experiments (preregistered width scan + falsification block).

## 1) Baseline transfer sanity (M1 Lambda on extended ranges)

| Run | `oof_gain_frac` | `perm_p_value` | `split_gain_min` | `pass_all` |
|---|---:|---:|---:|---|
| `experiment_M_cosmo_flow_2017_2020_v4locked` | 0.000723 | 0.028369 | -0.001651 | False |
| `experiment_M_cosmo_flow_2017_2020_v4locked_causal2018` | 0.000658 | 0.205674 | -0.001215 | False |
| `experiment_M_cosmo_flow_2017_2020q1_v4locked` | 0.001240 | 0.042553 | -0.002127 | False |
| `experiment_M_cosmo_flow_2017_2021_v4locked_causal2019` | 0.000280 | 0.078014 | -0.004035 | False |

Interpretation: gain is small and unstable under stricter chronological slicing.

## 2) GKSL bridge family (non-strict bridge diagnostics)

| Run | `gain_lambda_vs_ctrl` | `gain_phi_vs_lambda` | `p_inc` | `shuf_vs_lambda` | `shuf_p` |
|---|---:|---:|---:|---:|---:|
| `experiment_M_gksl_hybrid_bridge` | 0.003412 | 0.016538 | 0.007092 | -0.001940 | 0.737589 |
| `experiment_M_gksl_hybrid_bridge_screened` | 0.003412 | 0.016538 | 0.007092 | -0.001940 | 0.737589 |
| `experiment_M_gksl_hybrid_bridge_locked_raw` | 0.003412 | -0.000093 | 0.929078 | 0.000199 | 0.326241 |
| `experiment_M_gksl_hybrid_bridge_locked_raw_v2` | 0.003412 | -0.000084 | 0.460993 | 0.000167 | 0.219858 |
| `experiment_M_gksl_hybrid_bridge_screened_v2` | 0.003412 | 0.002396 | 0.007092 | -0.000977 | 0.914894 |

Interpretation: optimistic non-strict gains collapse when locked/causal constraints are enforced.

## 3) Strict chronological hybrid checks (including Phi-only strict)

| Run | Holdout `gain_lambda_vs_ctrl` | Holdout `gain_phi_vs_lambda` | Holdout `p_inc` | External `gain_phi_vs_lambda` | External `p_inc` |
|---|---:|---:|---:|---:|---:|
| `experiment_M_gksl_hybrid_strict_2017_2020_causal2018_locked_raw` | 0.001552 | 0.000398 | 0.020000 | -0.000187 | 0.504000 |
| `experiment_M_gksl_hybrid_strict_2017_2020_causal2018_nested` | 0.001552 | -0.000655 | 0.178000 | 0.000810 | 0.068000 |
| `experiment_M_gksl_hybrid_strict_2017_2020_causal2018_screened` | 0.001552 | -0.000655 | 0.178000 | 0.000810 | 0.068000 |
| `experiment_M_gksl_hybrid_strict_2017_2020_noncausal_locked_raw` | 0.001965 | -0.000812 | 0.820000 | -0.001368 | 0.954000 |
| `experiment_M_gksl_hybrid_strict_2019only_locked_raw` | 0.005368 | -0.000779 | 0.325000 | NA | NA |
| `experiment_M_gksl_hybrid_strict_2019only_screened` | 0.005368 | -0.000759 | 0.675000 | NA | NA |
| `experiment_M_gksl_hybrid_strict_locked_raw` | 0.002422 | 0.000888 | 0.285000 | 0.003097 | 0.140000 |
| `experiment_M_gksl_hybrid_strict_screened` | 0.002422 | -0.000200 | 0.370000 | 0.016054 | 0.050000 |
| `experiment_M_phi_only_strict_causal2018_locked_raw_perm1999` | 0.001552 | 0.000398 | 0.021500 | -0.000187 | 0.525500 |
| `experiment_M_phi_only_strict_causal2018_nested_perm1999` | 0.001552 | -0.000655 | 0.153500 | 0.000810 | 0.053000 |
| `experiment_M_phi_only_strict_causal2019_train2019_test2020_ext2021_perm1999` | -0.002240 | -0.000067 | 0.381000 | -0.000023 | 0.335000 |

Interpretation: strict transfer is not robust; Phi incremental effect over Lambda is weak/inconsistent in chronological external checks.

## 4) Halo strict core-only branch (pre-final strict runs)

| Run | Halo geometry | Test 2020 `ERA_window gain_vs_core` | External 2021 `ERA_window gain_vs_core` |
|---|---|---:|---:|
| `experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021` | local, width=8 | 0.143558 | 0.099674 |
| `experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2` | local, width=8 (corrected) | 0.136707 | 0.092229 |

Interpretation: adding physically adjacent halo context gives large and stable transfer gain vs `ERA_core`.

## 5) Final experiment A: preregistered halo-width scan (fixed core)

Configuration locked before scan:

- split: `train<=2019 -> test=2020 -> external=2021`
- fixed core margin: `10` cells
- halo mode: `local`
- scanned widths: `0, 4, 6, 8, 10`
- model ladder fixed (`ERA_core`, `ERA_window`, `+Phi_H`, `+Lambda_H`, `+Phi_H+Lambda_H`, shuffled)

Primary endpoint (`ERA_window` vs `ERA_core`):

| Halo width | Test 2020 gain | Test p | External 2021 gain | External p |
|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 1.0000 | 0.000000 | 1.0000 |
| 4 | 0.151703 | 0.0005 | 0.156626 | 0.0005 |
| 6 | 0.137846 | 0.0005 | 0.131756 | 0.0005 |
| 8 | 0.129966 | 0.0005 | 0.112355 | 0.0005 |
| 10 | 0.126708 | 0.0005 | 0.097619 | 0.0005 |

Decision from this scan: **best width = 4** (highest and stable gain on both test/external).

## 6) Final experiment B: halo-physics falsification block (fixed width=4)

Controls compare physically adjacent halo to non-physical context replacements.

| Mode | Test 2020 `ERA_window gain_vs_core` | Test p | External 2021 `ERA_window gain_vs_core` | External p |
|---|---:|---:|---:|---:|
| `local` | 0.151703 | 0.0005 | 0.156626 | 0.0005 |
| `remote` | 0.097182 | 0.0005 | 0.058412 | 0.0005 |
| `misaligned` | 0.073343 | 0.0005 | 0.059724 | 0.0005 |

Key contrasts:

- local - remote: `+0.054521` (test), `+0.098214` (external)
- local - misaligned: `+0.078360` (test), `+0.096902` (external)

Interpretation: gain is strongest for physically adjacent halo; replacing with remote or shifted context degrades closure strongly.

## 7) Recommendation for manuscript branch selection

Keep in the core branch package:

1. strict negative transfer result for `Phi` over Lambda on chronological ERA5 holdouts,
2. positive strict halo boundary result (core-only scoring + boundary context),
3. preregistered width scan showing non-trivial width dependence with optimum at `w=4`,
4. falsification block (`local > remote/misaligned`) as boundary-physics evidence.

Treat non-strict bridge spikes as diagnostic only, not core evidence.

## 8) Full run inventory executed on March 7, 2026

- `clean_experiments/results/experiment_M_cosmo_flow_2017_2020_v4locked`
- `clean_experiments/results/experiment_M_cosmo_flow_2017_2020_v4locked_causal2018`
- `clean_experiments/results/experiment_M_cosmo_flow_2017_2020q1_v4locked`
- `clean_experiments/results/experiment_M_cosmo_flow_2017_2021_v4locked_causal2019`
- `clean_experiments/results/experiment_M_gksl_hybrid_bridge`
- `clean_experiments/results/experiment_M_gksl_hybrid_bridge_locked_raw`
- `clean_experiments/results/experiment_M_gksl_hybrid_bridge_locked_raw_v2`
- `clean_experiments/results/experiment_M_gksl_hybrid_bridge_screened`
- `clean_experiments/results/experiment_M_gksl_hybrid_bridge_screened_v2`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2017_2020_causal2018_locked_raw`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2017_2020_causal2018_nested`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2017_2020_causal2018_screened`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2017_2020_noncausal_locked_raw`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2019only_locked_raw`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_2019only_screened`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_locked_raw`
- `clean_experiments/results/experiment_M_gksl_hybrid_strict_screened`
- `clean_experiments/results/experiment_M_halo_boundary_falsify_misaligned_w4_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_falsify_remote_w4_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_strict_causal2019_train2019_test2020_ext2021_v2`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w0_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w10_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w4_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w6_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_halo_boundary_widthscan_w8_causal2019_train2019_test2020_ext2021`
- `clean_experiments/results/experiment_M_phi_only_strict_causal2018_locked_raw_perm1999`
- `clean_experiments/results/experiment_M_phi_only_strict_causal2018_nested_perm1999`
- `clean_experiments/results/experiment_M_phi_only_strict_causal2019_train2019_test2020_ext2021_perm1999`
