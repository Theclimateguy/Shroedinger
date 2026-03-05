# Experiment F4b Report: Independent Holonomy Test with Ablations

## Main Regime (independent from F3)
- corr(delta_order, D_f-d_top), Pearson = 0.970675
- corr(delta_order, D_f-d_top), Spearman = 0.936364
- permutation p-value (epsilon-level) = 1.999600e-04
- partial corr(delta_order, D_f-d_top | log epsilon) = 0.651518
- epsilon* = 1.00
- epsilon* bootstrap 95% CI = [1.00, 1.15]
- bootstrap share epsilon* in [0.70, 1.40] = 1.000

## Ablations
- A (unitary) corr = nan
- A corr applicable = False
- A Var(delta_order) = 3.055e-31
- A max(delta_order) = 4.337e-15
- B (commutative geometry) corr = nan
- B corr applicable = False
- B Var(delta_order) = 3.302e-31
- B max delta_order = 4.596e-15
- D (mu permutation) corr = -0.163651
- C2 geometry sweep delta_order mean range = [0.015558, 0.033733]

## Seed Diagnostics
- failed seed share = 0.220 (11/50)
- df-agreement threshold = 0.35
- psd-median-R2 threshold = 0.35
- valid-seed corr count = 39

## Gauge Robustness
- max |delta h_triangle| = 4.424e-16
- max |delta delta_order| = 5.537e-16

## Correlation Distributions
- seed-level Pearson mean [q2.5%, q97.5%] = 0.777312 [-0.207941, 0.970205]
- valid-seed Pearson mean [q2.5%, q97.5%] = 0.749191 [-0.333777, 0.971814]
- loop-level Pearson mean [q2.5%, q97.5%] = 0.963222 [0.952183, 0.972164]

## Criteria
- H4b_1_corr_gt_0_9: True
- H4b_1_perm_p_lt_0_05: True
- H4b_2_eps_star_in_window: True
- H4b_2_bootstrap_majority_in_window: True
- Ablation_A_constant_delta_order: True
- H4b_3_commutative_delta_order_near_zero: True
- H4b_4_gauge_h_invariant: True
- H4b_4_gauge_delta_order_invariant: True
- Df_estimators_agree_on_average: True
- Failed_seed_share_below_limit: True
- Ablation_A_corr_not_applicable_or_weaker: True
- Ablation_B_corr_not_applicable_or_weaker: True
- Ablation_D_correlation_weaker_than_main: True
- pass_all: True
