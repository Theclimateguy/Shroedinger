# F6 Robustness Addendum

## Transfer Fraction Sensitivity at epsilon*=1.0
- tf=0.90: alpha=1.639031 (q2.5=1.603533, q97.5=1.680109), rel_err=6.39%, R2=0.978911, dyn_med=235.42
- tf=0.96: alpha=1.559421 (q2.5=1.533810, q97.5=1.591557), rel_err=1.23%, R2=0.980464, dyn_med=364.79
- tf=0.99: alpha=1.526474 (q2.5=1.497717, q97.5=1.584070), rel_err=0.91%, R2=0.978128, dyn_med=587.84
- Mean shifts across seeds (delta alpha): tf0.90-tf0.96=+0.079610, tf0.96-tf0.99=+0.032947, tf0.90-tf0.99=+0.112558
- Permutation p-values for mean shifts (20000 shuffles): p=0.00015, p=0.02640, p=0.00015 respectively

## cap_power Sensitivity at epsilon*=1.0
- cap_power=2.00: alpha=1.559421, rel_err=1.23%, p_branch_mean=0.730865, p_cap(eps=1)=0.999000
- cap_power=2.20: alpha=1.559421, rel_err=1.23%, p_branch_mean=0.730865, p_cap(eps=1)=0.999000
- cap_power=2.40: alpha=1.559421, rel_err=1.23%, p_branch_mean=0.730865, p_cap(eps=1)=0.999000

## cap_power Off-Critical Check at epsilon=0.1
- cap_power=2.00: alpha=2.110695, R2=0.957676, p_branch_mean=0.330248, p_cap(eps=0.1)=0.330248
- cap_power=2.40: alpha=2.248278, R2=0.944083, p_branch_mean=0.264664, p_cap(eps=0.1)=0.264664

## Interpretation
- At epsilon*=1, cap_power is structurally weak because balance(1)=1 => p_cap=0.999 regardless of exponent.
- Therefore alpha robustness at epsilon*=1 should be judged primarily vs transfer_fraction; cap_power requires off-critical tests to be informative.
- Off-critical epsilon=0.1 confirms exponent matters once p_cap enters active regime (lower p_cap, steeper alpha).
- Conclusion for claims: F6 keeps alpha in the same universality band near alpha_pred for tf in [0.90,0.99], but alpha is not strictly invariant to dissipation settings and this should be stated explicitly.
