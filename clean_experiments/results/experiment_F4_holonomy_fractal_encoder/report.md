# Experiment F4 Report: Holonomy as Fractal Encoder

## Loop Construction
- Primary loop: `mu0 -> mu1 -> mu2 -> mu0` (triangular).
- Placebo loop: `mu0 -> mu2 -> mu1 -> mu0` (same vertices/lengths, permuted order).
- Minimal control: `mu0 -> mu1 -> mu0` (2-segment loop).
- Holonomy score: `h = ||H - I||_F`.
- Path-ordering score: `Delta_order = ||H_triangle - H_placebo||_F`.

## Core Results
- corr(h_triangle, D_f-d_top) = 0.966411
- corr(h_triangle, Lambda_coh) = 0.956271
- corr(h_placebo, D_f-d_top) = 0.966411
- corr(Delta_order, D_f-d_top) = 0.967375
- h peak epsilon = 1.00
- ordering gain ratio (peak/extremes) = 1.133995
- max |delta h| under gauge = 4.441e-16
- max |delta Delta_order| under gauge = 1.554e-15

## Criteria
- corr_h_vs_excess_df_gt_0_9: True
- corr_h_vs_lambda_coh_gt_0_9: True
- corr_ordering_gap_vs_excess_df_gt_0_9: True
- h_peak_in_balance_window: True
- ordering_gap_peak_stronger_than_extremes: True
- minimal_loop_trivial: True
- gauge_invariant_h_triangle: True
- gauge_invariant_ordering_gap: True
- PASS_ALL: True

## Interpretation
- Triangular holonomy tracks excess fractal dimension and coherence channel across epsilon.
- The minimal two-segment loop stays trivial, so nontriviality is a genuine loop effect.
- Placebo permutation changes the operator-level holonomy (Delta_order), strongest near epsilon~1.
