# Experiment M GKSL Hybrid Bridge

## Setup
- M1 timeseries: `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_timeseries.csv`
- M1 summary: `clean_experiments/results/experiment_M_cosmo_flow_v4_macro_calibrated/experiment_M_summary.csv`
- R6 config source: `clean_experiments/results/experiment_P2_memory_gksl_cptp/gksl_final_l8.csv`
- selected GKSL config: `G001`
- phi selection mode: `locked`
- selected proxy bands: `0` / `1`
- selected phi candidate: `raw`
- common valid rows: `4380`

## Headline metrics
- baseline M1 oof_gain_frac (from summary): `0.003412`
- ERA5+Lambda gain_vs_ctrl: `0.003412`, perm_p=`0.007092`
- ERA5+Lambda+Phi_GKSL gain_vs_ctrl: `0.003328`
- ERA5+Lambda+Phi_GKSL gain_vs_lambda: `-0.000084`, perm_p_inc=`0.460993`
- shuffled control gain_vs_lambda: `0.000167`, perm_p_inc=`0.219858`

## GKSL diagnostics
- max cptp violation proxy: `8.882e-16`
- mean gamma_dephase: `0.996774`
- mean gamma_relax: `0.991544`
- mean reset_kappa: `0.999704`

## Surface split (land/ocean)
- land: gain_lambda_vs_ctrl=-0.000491, gain_phi_vs_lambda=-0.000145, gain_shuf_vs_lambda=-0.000015, perm_p_phi=0.716312
- ocean: gain_lambda_vs_ctrl=0.001818, gain_phi_vs_lambda=-0.000142, gain_shuf_vs_lambda=-0.000267, perm_p_phi=0.404255

## Artifacts
- `hybrid_model_metrics.csv`
- `hybrid_phi_screening.csv`
- `hybrid_oof_predictions.csv`
- `hybrid_active_calm_summary.csv`
- `hybrid_correlation_summary.csv`
- `hybrid_timeseries_with_gksl.csv`
- `hybrid_config_used.json`
- `hybrid_surface_summary.csv`
