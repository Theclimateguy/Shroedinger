# Phase 33 — inheritance fidelity, arm B

Protocol: `docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md`. Verdict: **UNDETERMINED (see checks)**. Domains: 39. δ = 0.01.

## D = P̂_model − P̂_truth (anchored V12, steps 4–5 = 200–1600 km, Deviation D5), 120 h

| system | n | P̂_truth | P̂_model | ratio | median D | frac D>0 | p (sign-flip) | deficit 200–800 km | D_S1 (frac) | D_S2 | D steps 3–5 (as frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pangu | 39 | 0.2665 | 0.3327 | ×1.249 | +0.0621 | 0.95 | 0.0001 | +0.436 | +0.0045 (0.07) | +0.1762 | +0.0621 |
| gencast_mean | 39 | 0.2653 | 0.3174 | ×1.196 | +0.0385 | 0.87 | 0.0001 | +0.785 | +0.0081 (0.21) | +0.1796 | +0.0385 |
| hres | 39 | 0.2655 | 0.2785 | ×1.049 | +0.0046 | 0.51 | 0.6954 | -0.004 | — | — | +0.0046 |

## Same at 24 h

| system | P̂_truth | P̂_model | median D | p | deficit 200–800 km |
|---|---|---|---|---|---|
| pangu | 0.2762 | 0.3017 | +0.0437 | 0.0001 | +0.421 |
| gencast_mean | 0.2800 | 0.2903 | +0.0136 | 0.0068 | +0.277 |
| hres | 0.2769 | 0.2657 | -0.0125 | 0.9850 | -0.009 |

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
  "rho_truth_24_vs_120": 0.7663967611336032,
  "pass": true
 },
 "C33_3": {
  "mismatch_medians": [
   0.34996071619937436,
   0.24310364811304985,
   0.0731169762173244,
   0.08950002536791482
  ],
  "pass": false
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
   "D_median": 0.06210116297006607,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_median": 0.038498843709627806,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_2": {
  "hres": {
   "D_median": 0.0046263188123703,
   "p": 0.6954,
   "satisfies_H1_criterion": false,
   "pass": true
  }
 },
 "H33_3": {
  "pangu": {
   "D_S1_median": 0.004516576727231353,
   "D_model_median": 0.06210116297006607,
   "frac": 0.07272934211245653,
   "pass_absolute_clause_dropped_D3": true,
   "pass": true
  },
  "gencast_mean": {
   "D_S1_median": 0.008087893327077267,
   "D_model_median": 0.038498843709627806,
   "frac": 0.21008146083760545,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  }
 },
 "H33_4": {
  "pangu": {
   "D_S2_median": 0.1761654367049535,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_S2_median": 0.17959410945574442,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_5_lead": {
  "pangu": {
   "D24": 0.04370992382367453,
   "D120": 0.06210116297006607
  },
  "gencast_mean": {
   "D24": 0.013561308383941678,
   "D120": 0.038498843709627806
  },
  "hres": {
   "D24": -0.012481396396954836,
   "D120": 0.0046263188123703
  }
 },
 "H33_6_geography": {
  "pangu": {
   "rho_D_vs_deficit34": 0.2878542510121458,
   "loso_r2_D_on_deficit15": -0.3330464096927006
  },
  "gencast_mean": {
   "rho_D_vs_deficit34": -0.11639676113360324,
   "loso_r2_D_on_deficit15": -0.7358399946722596
  }
 }
}
```
