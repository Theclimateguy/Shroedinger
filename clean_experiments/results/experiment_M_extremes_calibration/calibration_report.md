# Experiment M Extreme Calibration (Anti-overfitting)

## Protocol
- Train/calibration: 2017-2018 only
- Test: 2019 only (out-of-time)
- Tuning: blocked CV on train; one-SE model selection (simpler + stronger regularization preferred)

- n_train=2920, n_test=1460

## Selected Configs
- vertical_global: q=0.88, alpha=1e-01, cv_score_mean=0.002310, cv_gain_extreme_median=0.000433
- horizontal_global: q=0.88, alpha=1e-01, cv_score_mean=0.002029, cv_gain_extreme_median=0.000364
- vertical_regime: q=0.85, alpha=1e-01, cv_score_mean=0.303818, cv_gain_extreme_median=0.333889
- horizontal_regime: q=0.85, alpha=1e-01, cv_score_mean=0.304224, cv_gain_extreme_median=0.334202

## Test (2019)
- horizontal_regime: gain_all=0.140295 [0.064032, 0.206147], gain_extreme=0.093554, gain_non_extreme=0.165037, n_extreme=505
- vertical_regime: gain_all=0.139717 [0.060624, 0.205027], gain_extreme=0.093558, gain_non_extreme=0.164152, n_extreme=505
- horizontal_global: gain_all=0.005584 [-0.001945, 0.014281], gain_extreme=-0.004528, gain_non_extreme=0.010235, n_extreme=447
- vertical_global: gain_all=0.005368 [-0.001684, 0.012740], gain_extreme=-0.004269, gain_non_extreme=0.009801, n_extreme=447
