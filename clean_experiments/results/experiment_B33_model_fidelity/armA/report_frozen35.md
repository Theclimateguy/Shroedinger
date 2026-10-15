# Phase 33 — inheritance fidelity, arm A

Protocol: `docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md`. Verdict: **NOT_SCORED (arm A is hypothesis-generating)**. Domains: 39. δ = 0.01.

## D = P̂_model − P̂_truth (anchored V12, steps 4–5 = 200–1600 km, Deviation D5), 120 h

| system | n | P̂_truth | P̂_model | ratio | median D | frac D>0 | p (sign-flip) | deficit 200–800 km | D_S1 (frac) | D_S2 | D steps 3–5 (as frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| graphcast | 39 | 0.2767 | 0.3164 | ×1.144 | +0.0555 | 0.90 | 0.0001 | +0.588 | +0.0065 (0.12) | +0.1725 | +0.0555 |
| pangu | 39 | 0.2758 | 0.3167 | ×1.148 | +0.0525 | 0.95 | 0.0001 | +0.445 | +0.0019 (0.04) | +0.1704 | +0.0525 |
| gencast_mean | 39 | 0.2713 | 0.3253 | ×1.199 | +0.0469 | 0.92 | 0.0001 | +0.790 | +0.0052 (0.11) | +0.1627 | +0.0469 |
| hres | 39 | 0.2741 | 0.2705 | ×0.987 | -0.0067 | 0.49 | 0.3982 | +0.040 | — | — | -0.0067 |
| gencast_member | 39 | 0.2719 | 0.2638 | ×0.970 | -0.0044 | 0.44 | 0.4701 | +0.020 | — | — | -0.0044 |

## Same at 24 h

| system | P̂_truth | P̂_model | median D | p | deficit 200–800 km |
|---|---|---|---|---|---|
| graphcast | 0.2766 | 0.2734 | +0.0038 | 0.2447 | +0.323 |
| pangu | 0.2726 | 0.3161 | +0.0294 | 0.0001 | +0.416 |
| gencast_mean | 0.2708 | 0.2899 | +0.0090 | 0.0558 | +0.275 |
| hres | 0.2785 | 0.2616 | -0.0227 | 0.9788 | -0.004 |
| gencast_member | 0.2867 | 0.2846 | +0.0072 | 0.2227 | -0.002 |

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
  "rho_truth_24_vs_120": 0.5975708502024292,
  "pass": false
 },
 "C33_3": {
  "mismatch_medians": [
   0.05386647428496135,
   0.07653338863426358,
   0.30832740984201723,
   0.2298952215015118,
   0.06420233916205917,
   0.09398787749398615
  ],
  "pass": false
 },
 "C33_4": {
  "graphcast": 0.055501341819763184,
  "pangu": 0.052530830105145754,
  "gencast_mean": 0.046890283624331175,
  "pass": true
 }
}
```
```
{
 "H33_1": {
  "graphcast": {
   "D_median": 0.055501341819763184,
   "p": 0.0001,
   "pass": true
  },
  "pangu": {
   "D_median": 0.052530830105145754,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_median": 0.046890283624331175,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_2": {
  "hres": {
   "D_median": -0.006721133987108857,
   "p": 0.3982,
   "satisfies_H1_criterion": false,
   "pass": true
  },
  "gencast_member": {
   "D_median": -0.004403228561083439,
   "p": 0.4701,
   "satisfies_H1_criterion": false,
   "pass": true
  }
 },
 "H33_3": {
  "graphcast": {
   "D_S1_median": 0.006548702716827393,
   "D_model_median": 0.055501341819763184,
   "frac": 0.11799179086685611,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  },
  "pangu": {
   "D_S1_median": 0.0019397040208180838,
   "D_model_median": 0.052530830105145754,
   "frac": 0.03692505937818935,
   "pass_absolute_clause_dropped_D3": true,
   "pass": true
  },
  "gencast_mean": {
   "D_S1_median": 0.0051505714654922485,
   "D_model_median": 0.046890283624331175,
   "frac": 0.10984304353449545,
   "pass_absolute_clause_dropped_D3": false,
   "pass": true
  }
 },
 "H33_4": {
  "graphcast": {
   "D_S2_median": 0.1724872589111328,
   "p": 0.0001,
   "pass": true
  },
  "pangu": {
   "D_S2_median": 0.1703957493106524,
   "p": 0.0001,
   "pass": true
  },
  "gencast_mean": {
   "D_S2_median": 0.16269107162952423,
   "p": 0.0001,
   "pass": true
  }
 },
 "H33_5_lead": {
  "graphcast": {
   "D24": 0.0038161526123682288,
   "D120": 0.055501341819763184
  },
  "pangu": {
   "D24": 0.02942840258280438,
   "D120": 0.052530830105145754
  },
  "gencast_mean": {
   "D24": 0.008958597977956145,
   "D120": 0.046890283624331175
  },
  "hres": {
   "D24": -0.022734771172205626,
   "D120": -0.006721133987108857
  },
  "gencast_member": {
   "D24": 0.007167081038157136,
   "D120": -0.004403228561083439
  }
 },
 "H33_6_geography": {
  "graphcast": {
   "rho_D_vs_deficit34": -0.06558704453441296,
   "loso_r2_D_on_deficit15": -0.4661993054892726
  },
  "pangu": {
   "rho_D_vs_deficit34": 0.025303643724696356,
   "loso_r2_D_on_deficit15": -0.1044591901998062
  },
  "gencast_mean": {
   "rho_D_vs_deficit34": -0.36417004048582996,
   "loso_r2_D_on_deficit15": -0.34824475051895476
  }
 }
}
```
