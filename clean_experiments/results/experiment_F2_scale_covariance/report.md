# Experiment F2 Report: Scale Covariance of Section

## Core Test
- Tested condition: `||Psi(x, lambda*mu) - lambda^(-Delta) U_{mu->lambda*mu} Psi(x,mu)||^2`.
- Scale factors: `lambda in {2,3,4,5}`.
- `Delta_measured` from log-log fit of `||Psi(mu)|| ~ mu^(-Delta)`.
- `Delta_predicted = -0.5 * Tr(M)` from local linearized RG matrix.

## Key Results
- epsilon* from F1: `1.00`.
- At epsilon*=1.00:
  - delta_max = `4.150042e-30`
  - R2(power law) = `1.000000`
  - Delta_measured = `1.850000`
  - Delta_predicted = `1.850000`
  - relative error = `0.000%`
- Low epsilon=0.01: delta_max=`8.916553e-03`.
- High epsilon=10.00: delta_max=`1.794630e-03`.

## Criteria
- epsilon_star_in_window: `True`
- delta_star_lt_threshold: `True`
- r2_star_gt_threshold: `True`
- delta_match_within_5pct: `True`
- delta_off_fixed_large: `True`
- PASS_ALL: `True`

## Interpretation
- Around the F1 balance point, section dynamics is close to scale-covariant and obeys a high-quality power law.
- Away from the fixed point, covariance residual grows, matching the expected breakdown of strict self-similarity.
