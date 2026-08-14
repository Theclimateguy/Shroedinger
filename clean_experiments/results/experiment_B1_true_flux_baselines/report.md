# Experiment B1: Lambda vs true cross-scale flux and baselines

Region-windows analyzed: 8 of 8 required.

## Band 1 (center 141 km)

| predictor | median raw R^2 |
|---|---|
| lambda | 0.0034 |
| B1_smag | 0.4708 |
| B2_a15proxy | 0.0101 |
| B3_enstrophy | 0.2306 |
| B4_gradient | 0.3315 |

- sign stability (lambda): 5/8; surrogate pass: 2/8; lambda OOS R^2 median: -0.0687
- C1=False C2=False C3=False C4=False

## Band 2 (center 283 km)

| predictor | median raw R^2 |
|---|---|
| lambda | 0.0066 |
| B1_smag | 0.4213 |
| B2_a15proxy | 0.0222 |
| B3_enstrophy | 0.1795 |
| B4_gradient | 0.3125 |

- sign stability (lambda): 5/8; surrogate pass: 1/8; lambda OOS R^2 median: -0.0236
- C1=False C2=False C3=False C4=False

## Band 3 (center 566 km)

| predictor | median raw R^2 |
|---|---|
| lambda | 0.0023 |
| B1_smag | 0.3510 |
| B2_a15proxy | 0.0211 |
| B3_enstrophy | 0.2238 |
| B4_gradient | 0.3062 |

- sign stability (lambda): 5/8; surrogate pass: 1/8; lambda OOS R^2 median: -0.0343
- C1=False C2=False C3=False C4=False

## PHASE1_VERDICT: NEGATIVE

Criteria (frozen): C1 median raw R^2>0.05 with sign stability >=6/8; C2 lambda beats every baseline; C3 surrogate null passed >=5/8; C4 median OOS R^2>0; overall requires >=2 of 3 inertial bands.
