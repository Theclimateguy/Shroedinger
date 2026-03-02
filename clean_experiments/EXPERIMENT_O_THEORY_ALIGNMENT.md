# Experiment O Theory Alignment (G2 Clausius Form)

This note aligns `experiment_O_entropy_equilibrium.py` with the manuscript G2 thermodynamic equation
(`eq:Clausius_fit`):

`dS_hor ~= (1 / T_eff) * dQ_in + b`.

## Mapping from article symbols to atmospheric proxies

- `dS_hor` proxy:
  - `j_anom = j_moist - j_dry`
  - with `j_dry = omega * c_p * ln(T / T0)`
  - and `j_moist = omega * (c_p * ln(T / T0) + L_v * q / T)`
  - primary series: `j_anom_FT_all`
- `dQ_in` proxy:
  - moist enthalpy inflow proxy `dq_in = -omega * (c_p * T + L_v * q)`
  - primary series: `dq_in_FT_all`
- `Lambda`:
  - imported from Experiment M macro-calibrated run (`lambda_struct`).

## Regression structure

- Baseline (Clausius):
  - `dS_hor_proxy ~ (1/T_eff) * dQ_in_proxy + b`
- Full (non-equilibrium correction):
  - `dS_hor_proxy ~ (1/T_eff) * dQ_in_proxy + c * Lambda + b`

## Validation protocol

- Train years: `2017-2018`
- Test year: `2019`
- Blocked CV + one-SE alpha selection on train set.
- Out-of-time test on 2019.
- Block permutation test (Lambda shuffled by blocks).
- Block bootstrap CI on test metrics.
- Quarterly rolling-origin diagnostics inside 2019.
