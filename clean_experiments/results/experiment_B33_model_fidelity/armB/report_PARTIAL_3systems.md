# Phase 33 — inheritance fidelity, arm B

Protocol: `docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md`. Verdict: **UNDETERMINED (see checks)**. Domains: 39. δ = 0.01.

## D = P̂_model − P̂_truth (anchored V12, steps 4–5 = 200–1600 km, Deviation D5), 120 h

| system | n | P̂_truth | P̂_model | ratio | median D | frac D>0 | p (sign-flip) | deficit 200–800 km | D_S1 (frac) | D_S2 | D steps 3–5 (as frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pangu | 39 | 0.2404 | 0.2815 | ×1.171 | +0.0411 | 0.79 | 0.0001 | +0.436 | -0.0015 (-0.04) | +0.2171 | +0.0621 |
| gencast_mean | 39 | 0.2384 | 0.3006 | ×1.261 | +0.0572 | 0.85 | 0.0001 | +0.785 | +0.0156 (0.27) | +0.1979 | +0.0385 |
| hres | 39 | 0.2345 | 0.2434 | ×1.038 | -0.0060 | 0.44 | 0.7040 | -0.004 | — | — | +0.0046 |

## Same at 24 h

| system | P̂_truth | P̂_model | median D | p | deficit 200–800 km |
|---|---|---|---|---|---|
| pangu | 0.2413 | 0.2562 | +0.0237 | 0.0002 | +0.421 |
| gencast_mean | 0.2460 | 0.2469 | +0.0002 | 0.6078 | +0.277 |
| hres | 0.2428 | 0.2162 | -0.0138 | 0.8944 | -0.009 |

## Per-step median D at 120 h (steps 1–5; step 1 diagnostic)

| system | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| pangu | -0.0817 | -0.0586 | +0.1024 | +0.0732 | +0.0167 |
| pangu:S1 | -0.0373 | -0.0071 | +0.0010 | -0.0019 | +0.0055 |
| pangu:S2 | +0.0531 | +0.0861 | +0.1116 | +0.1675 | +0.2600 |
| gencast_mean | -0.1454 | -0.0360 | +0.0433 | +0.0835 | +0.0199 |
| gencast_mean:S1 | -0.0253 | -0.0001 | -0.0006 | -0.0016 | +0.0131 |
| gencast_mean:S2 | +0.0501 | +0.0823 | +0.1180 | +0.1681 | +0.2270 |
| hres | -0.0102 | -0.0314 | -0.0001 | +0.0014 | -0.0048 |

## Checks and hypotheses

```
{
 "C33_1": {
  "pass": null
 },
 "C33_2": {
  "rho_truth_24_vs_120": 0.7291497975708502,
  "pass": true
 },
 "C33_3": {
  "mismatch_medians": [
   0.036377432548145995,
   0.04521212903676801,
   0.03055447718759493,
   0.07567423949716254
  ],
  "pass": true
 }
}
```
```
{
 "H33_1": {
  "graphcast": {
   "pass": null
  },
  "pangu": {
   "D_median": 0.04111171513795853,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_median": 0.05717428773641586,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_2": {
  "hres": {
   "D_median": -0.006043344736099243,
   "p": 0.704,
   "satisfies_H1_criterion": false,
   "pass": true
  }
 },
 "H33_3": {
  "pangu": {
   "D_S1_median": -0.0014984756708145142,
   "D_model_median": 0.04111171513795853,
   "frac": -0.0364488726823019,
   "pass_absolute_clause_dropped_D3": true,
   "pass": true
  },
  "gencast_mean": {
   "D_S1_median": 0.015559345483779907,
   "D_model_median": 0.05717428773641586,
   "frac": 0.27213885996291537,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  }
 },
 "H33_4": {
  "pangu": {
   "D_S2_median": 0.21705909818410873,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_S2_median": 0.1978754848241806,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_5_lead": {
  "pangu": {
   "D24": 0.023735173046588898,
   "D120": 0.04111171513795853
  },
  "gencast_mean": {
   "D24": 0.00019853562116622925,
   "D120": 0.05717428773641586
  },
  "hres": {
   "D24": -0.013837181031703949,
   "D120": -0.006043344736099243
  }
 },
 "H33_6_geography": {
  "pangu": {
   "rho_D_vs_deficit34": 0.34109311740890685,
   "loso_r2_D_on_deficit15": -0.19844376907965366
  },
  "gencast_mean": {
   "rho_D_vs_deficit34": 0.3299595141700405,
   "loso_r2_D_on_deficit15": -0.6497378347010205
  }
 }
}
```
