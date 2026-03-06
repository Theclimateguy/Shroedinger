# M-realpilot v1-frozen: Headline Comparison

## Data expansion
- events: 24 (from 5)
- model samples: 48 (from 10)
- ABI matched volume estimate: 1.625 GB
- ABI matched downloaded: 1.625 GB (errors=0)
- GLM matched downloaded: 0.074 GB (errors=0)
- total matched downloaded: 1.699 GB

## Headline metrics only
- MAE baseline: 0.096162 -> 0.060540
- MAE full: 0.114443 -> 0.058964
- mean MAE gain: -0.018281 -> 0.001576
- perm p (time-shuffle): 0.836 -> 0.010
- perm p (event-shuffle): 0.414 -> 0.008
- event_positive_frac: 0.400 -> 0.667
- PASS_ALL: False -> True

## Frozen protocol
- script: `clean_experiments/experiment_M_realpilot_v1_frozen.py`
- hash: `aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2`
- settings unchanged: target `next_p95`, ridge alpha `10.0`, `n_perm=499`, leave-one-event-out CV.