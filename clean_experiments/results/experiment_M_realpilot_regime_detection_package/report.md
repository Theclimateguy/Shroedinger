# Frozen Regime Detection: Diagnostic Perimeter

## Protocol lock
- Script: `/Users/theclimateguy/Documents/science/Shroedinger/clean_experiments/experiment_M_realpilot_v1_frozen.py`
- SHA256: `aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2`
- Hash matches expected: `True`
- Fixed settings: target=`next_p95`, Ridge alpha=10.0, LOEO-CV, n_perm=499

## Perimeter registry
| run_id | season | region | mean_gain | p_time | p_event | q_plus | delta_active-calm | ABI-only | ABI+GLM | delta(GLM-ABI) | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1_expanded_positive | MAM+JJA | Southern_Plains+Midwest | 0.001576 | 0.010 | 0.008 | 0.667 | 0.003433 | 0.001140 | 0.001576 | 0.000437 | Detected |
| R2_independent_seasonal | SON | Southeast | -0.009710 | 0.708 | 0.160 | 0.417 | 0.007992 | -0.000609 | -0.009710 | -0.009101 | Bridge tension |
| R3_independent_geographic | JJA+SON | Southwest | -0.016272 | 0.978 | 0.584 | 0.250 | -0.000711 | -0.011403 | -0.016272 | -0.004868 | Bridge tension |

## Status summary
- Detected: 1
- Compatible but undetected: 0
- Bridge tension: 2

## Regime-detection conclusion
- Detection is regime-conditional under the concrete frozen protocol.
- A detected regime exists (expanded positive reference).
- Independent seasonal and independent geographic extensions are undetected and show bridge tension (ABI+GLM degrades vs ABI-only).
- Current perimeter supports a non-universal, regime-specific interpretation of local bridge detectability.