# M-realpilot Applicability Map

Rows summarize frozen runs across season/region with ABI-only vs ABI+GLM component checks.

| run_id | season | region | main_mean_gain | main_pass_all | abi_only_gain | abi_glm_gain | delta_glm_minus_abi | active_window_frac | active_minus_calm | mean_glm_sparsity |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_expanded_positive | JJA+MAM | Midwest+Southern_Plains | 0.001576 | True | 0.001140 | 0.001576 | 0.000437 | 0.542 | 0.003433 | 0.000 |
| v1_independent_seasonal_2024 | SON | Southeast | -0.009710 | False | -0.000609 | -0.009710 | -0.009101 | 0.500 | 0.007992 | 0.000 |
| v1_independent_geographic_southwest_2024 | JJA+SON | Southwest | -0.016272 | False | -0.011403 | -0.016272 | -0.004868 | 0.500 | -0.000711 | 0.000 |
## Decision reading

- Both independent checks (seasonal and geographic) are negative under identical frozen settings.
- In both independent checks, ABI-only is less negative than ABI+GLM.
- Positive ABI+GLM uplift is only observed in the expanded-positive regime.

## Practical implication

Current evidence supports regime dependence of the local structural signal. Lightning coupling should be treated as conditional (not universally transferable) until another independent region reproduces positive ABI+GLM gain.
