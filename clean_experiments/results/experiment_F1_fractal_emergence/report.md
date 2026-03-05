# Experiment F1 Report: Fractal Emergence from Balance

## Setup
- Model: two-scale (`mu1, mu2`) GKSL dynamics with coherent pump + Lindblad dissipation + vertical transport.
- Control parameter: `epsilon = tau_rel / tau_dr` scanned on `[0.01, 10]`.
- Metrics:
  - `vertical_flow_abs_mean` as proxy for `|partial_mu <J_O>|`.
  - `source_abs_mean` as `|S_O|` proxy.
  - `lambda_coh_rms` for `Lambda_coh`.
  - `d_s_effrank_mean`: spectral-dimension proxy from the eigen-spectrum of `rho(mu)` via entropy effective rank.

## Key Points
- Balance point (`epsilon*`, minimum flow): `1.00`.
- At `epsilon*`: flow=`1.679089e-04`, source=`3.580625e-02`, d_s=`2.8258`.
- Low extreme (`epsilon=0.01`): flow=`7.570069e-04`, d_s=`1.0179`.
- High extreme (`epsilon=10.00`): flow=`1.340547e-03`, d_s=`2.9524`.

## F1 Criteria
- epsilon_star_in_window: `True`
- flow_star_is_min_vs_extremes: `True`
- flow_extremes_nonzero: `True`
- source_star_nonzero: `True`
- d_s_star_noninteger: `True`
- d_s_extremes_near_integer: `True`
- PASS_ALL: `True`

## Notes
- `d_s_effrank_mean` is a finite-size spectral proxy suitable for two-scale toy runs; for dense continuum spectra, use full spectral-density fitting.
