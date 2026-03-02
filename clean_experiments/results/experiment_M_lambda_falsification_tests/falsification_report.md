# Experiment M4 Lambda necessity falsification (S1/S2/S3)

## S1 Placebo-mu (scale-layer permutation)
- mae_gain_real: 0.003412
- mae_gain_placebo: -0.002986
- perm_p(placebo): 1.000000
- beta_lambda(real): -1661.105670
- positive_strata_frac(real): 1.000000

## S2 F^comm control
- mae_gain_real: 0.003412
- mae_gain_comm: 0.000000
- beta_real: -1661.105670
- beta_comm: 0.000000

## S3 Information criteria (OOF residuals)
- Delta AIC (ctrl -> ctrl+lambda_real): -13.202079
- Delta BIC (ctrl -> ctrl+lambda_real): -6.817275

## Note
- In this pipeline, Lambda_comm is the commutator-trace control and evaluates to zero in exact arithmetic.
