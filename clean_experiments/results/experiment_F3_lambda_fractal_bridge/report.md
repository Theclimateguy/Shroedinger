# Experiment F3 Report: Lambda_matter vs Fractal Dimension

## Operational Equations
- `Lambda_matter = integral w(mu) Tr(F_phys(mu) rho(mu)) dmu`
- `F_phys = 0.5 * (F + F^†)`, with discrete `F ~ dA/dmu + [A(mu_k), A(mu_{k+1})]`
- `S(f) ~ f^(-beta)`, `D_f = (5 - beta)/2`

## Main Metrics
- corr(Lambda_matter, D_f-d_top) = 0.923694
- corr(Lambda_coh, D_f-d_top) = 0.926734
- corr(Lambda_matter, D_f) = 0.923694
- regression slope = 0.308032
- regression R2 = 0.853211
- regression p-value = 3.748e-04

## Criteria
- corr_lambda_matter_gt_0_9: True
- regression_r2_gt_0_85: True
- slope_positive: True
- corr_lambda_coh_gt_lambda_matter: True
- PASS_ALL: True
