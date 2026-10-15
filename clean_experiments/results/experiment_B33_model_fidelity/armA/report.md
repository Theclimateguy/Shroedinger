# Phase 33 — inheritance fidelity, arm A

Protocol: `docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md`. Verdict: **NOT_SCORED (arm A is hypothesis-generating)**. Domains: 39. δ = 0.01.

## D = P̂_model − P̂_truth (anchored V12, steps 4–5 = 200–1600 km, Deviation D5), 120 h

| system | n | P̂_truth | P̂_model | ratio | median D | frac D>0 | p (sign-flip) | deficit 200–800 km | D_S1 (frac) | D_S2 | D steps 3–5 (as frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| graphcast | 39 | 0.2439 | 0.2688 | ×1.102 | +0.0494 | 0.82 | 0.0006 | +0.588 | +0.0062 (0.12) | +0.1830 | +0.0555 |
| pangu | 39 | 0.2460 | 0.2673 | ×1.087 | +0.0222 | 0.72 | 0.0001 | +0.445 | -0.0013 (-0.06) | +0.1929 | +0.0525 |
| gencast_mean | 39 | 0.2421 | 0.2996 | ×1.237 | +0.0535 | 0.87 | 0.0001 | +0.790 | +0.0097 (0.18) | +0.1839 | +0.0469 |
| hres | 39 | 0.2454 | 0.2487 | ×1.013 | -0.0001 | 0.49 | 0.4410 | +0.040 | — | — | -0.0067 |
| gencast_member | 39 | 0.2428 | 0.2317 | ×0.954 | -0.0006 | 0.49 | 0.3608 | +0.020 | — | — | -0.0044 |

## Same at 24 h

| system | P̂_truth | P̂_model | median D | p | deficit 200–800 km |
|---|---|---|---|---|---|
| graphcast | 0.2486 | 0.2267 | +0.0005 | 0.7806 | +0.323 |
| pangu | 0.2484 | 0.2543 | +0.0022 | 0.0425 | +0.416 |
| gencast_mean | 0.2352 | 0.2432 | -0.0039 | 0.6888 | +0.275 |
| hres | 0.2425 | 0.2298 | -0.0119 | 0.7679 | -0.004 |
| gencast_member | 0.2453 | 0.2623 | +0.0089 | 0.1025 | -0.002 |

## Per-step median D at 120 h (steps 1–5; step 1 diagnostic)

| system | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| graphcast | -0.2004 | -0.0226 | +0.0629 | +0.0645 | +0.0055 |
| graphcast:S1 | -0.0183 | +0.0040 | +0.0065 | -0.0049 | +0.0091 |
| graphcast:S2 | +0.0567 | +0.0789 | +0.1208 | +0.1652 | +0.2056 |
| pangu | -0.0798 | -0.0499 | +0.0875 | +0.0563 | +0.0296 |
| pangu:S1 | -0.0400 | -0.0093 | +0.0002 | -0.0013 | +0.0015 |
| pangu:S2 | +0.0532 | +0.0856 | +0.1141 | +0.1660 | +0.2300 |
| gencast_mean | -0.1463 | -0.0066 | +0.0571 | +0.0881 | +0.0267 |
| gencast_mean:S1 | -0.0242 | -0.0037 | -0.0068 | +0.0020 | -0.0016 |
| gencast_mean:S2 | +0.0511 | +0.0768 | +0.1276 | +0.1721 | +0.1869 |
| hres | -0.0127 | -0.0200 | +0.0023 | +0.0032 | +0.0243 |
| gencast_member | -0.0599 | +0.0041 | -0.0123 | -0.0223 | +0.0208 |

## Checks and hypotheses

```
{
 "C33_1": {
  "graphcast_frac_pos": 1.0,
  "pangu_frac_pos": 1.0,
  "hres_abs_median": 0.040412365183061905,
  "graphcast_median": 0.5875457667708659,
  "pass": true
 },
 "C33_2": {
  "rho_truth_24_vs_120": 0.5435222672064777,
  "pass": false
 },
 "C33_3": {
  "mismatch_medians": [
   0.022013262925864637,
   0.06551892466105966,
   0.0377351219217925,
   0.06269347118329328,
   0.05088859789225244,
   0.09044573924432033
  ],
  "pass": true
 },
 "C33_4": {
  "graphcast": 0.04940477013587952,
  "pangu": 0.02215486764907837,
  "gencast_mean": 0.05351920425891876,
  "pass": true
 }
}
```
```
{
 "H33_1": {
  "graphcast": {
   "D_median": 0.04940477013587952,
   "p": 0.0006,
   "pass": true
  },
  "pangu": {
   "D_median": 0.02215486764907837,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_median": 0.05351920425891876,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_2": {
  "hres": {
   "D_median": -6.579607725143433e-05,
   "p": 0.441,
   "satisfies_H1_criterion": false,
   "pass": true
  },
  "gencast_member": {
   "D_median": -0.000634409487247467,
   "p": 0.3608,
   "satisfies_H1_criterion": false,
   "pass": true
  }
 },
 "H33_3": {
  "graphcast": {
   "D_S1_median": 0.006152935326099396,
   "D_model_median": 0.04940477013587952,
   "frac": 0.12454132079102445,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  },
  "pangu": {
   "D_S1_median": -0.0013469010591506958,
   "D_model_median": 0.02215486764907837,
   "frac": -0.06079481405553448,
   "pass_absolute_clause_dropped_D3": true,
   "pass": true
  },
  "gencast_mean": {
   "D_S1_median": 0.009660579264163971,
   "D_model_median": 0.05351920425891876,
   "frac": 0.18050678065816111,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  }
 },
 "H33_4": {
  "graphcast": {
   "D_S2_median": 0.18298780173063278,
   "p": 0.0001,
   "pass": true
  },
  "pangu": {
   "D_S2_median": 0.19287654012441635,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_S2_median": 0.18390432745218277,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_5_lead": {
  "graphcast": {
   "D24": 0.0005142688751220703,
   "D120": 0.04940477013587952
  },
  "pangu": {
   "D24": 0.002222411334514618,
   "D120": 0.02215486764907837
  },
  "gencast_mean": {
   "D24": -0.003932639956474304,
   "D120": 0.05351920425891876
  },
  "hres": {
   "D24": -0.011888571083545685,
   "D120": -6.579607725143433e-05
  },
  "gencast_member": {
   "D24": 0.008876718580722809,
   "D120": -0.000634409487247467
  }
 },
 "H33_6_geography": {
  "graphcast": {
   "rho_D_vs_deficit34": 0.09230769230769233,
   "loso_r2_D_on_deficit15": -0.501800199550199
  },
  "pangu": {
   "rho_D_vs_deficit34": -0.04372469635627531,
   "loso_r2_D_on_deficit15": -0.06601463884604608
  },
  "gencast_mean": {
   "rho_D_vs_deficit34": 0.13380566801619434,
   "loso_r2_D_on_deficit15": -0.6007106417964849
  }
 }
}
```
