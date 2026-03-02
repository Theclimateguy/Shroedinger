# Experiment N Follow-up Comparison

This table compares two follow-up branches:
- `N11`: source-proxy baseline + global/regime lambda
- `N12`: source-proxy baseline + localized multiscale lambda

```text
                  experiment_id                                                 description selected_model  selected_q_extreme  selected_ridge_alpha  n_train  n_test  test_gain_all  test_gain_extreme  test_gain_non_extreme  test_rms_base  test_rms_full  test_gain_ci95_lo  test_gain_ci95_hi  perm_p_value  quarterly_mean_gain_all  quarterly_min_gain_all  quarterly_positive_fraction  v_thr_train  omegaq_thr_train  lambda_struct_mu_train  lambda_struct_sd_train
    experiment_N11_source_proxy        Base source proxy + global/regime lambda corrections   n11_global_a                 0.8             10.000000     2920    1460  -1.732572e-12      -9.699137e-13          -2.508553e-12   6.968076e-08   6.968076e-08      -3.935152e-12       6.017835e-14      0.687943             4.539402e-12           -1.008347e-12                         0.25     0.000018          0.000387           -7.174881e-09                0.000041
experiment_N12_localized_lambda Base source proxy + localized multiscale lambda corrections  n12_ms_regime                 0.8              0.000001     2920    1460   8.538612e-04       1.513696e-03           1.839103e-04   6.966870e-08   6.960922e-08      -3.248452e-04       2.297679e-03      0.035461             7.599997e-04           -1.548415e-04                         0.75     0.000018          0.000387           -7.174881e-09                0.000041
```
