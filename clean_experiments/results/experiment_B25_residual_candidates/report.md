# Experiment B25: hydroclimatic candidates for the map residual

Protocol: `docs/PROTOCOL_PHASE25_RESIDUAL_CANDIDATES.md` (frozen 2026-08-24
before the candidate fields were downloaded). Code
`clean_experiments/experiment_B25_residual_candidates.py`. Seed 20260824.
Data: ERA5 monthly means 0.5 deg, 2021-2024 (`data/b25hydro`).

## PHASE25_VERDICT: NEGATIVE (by the frozen ladder), with one named finding

| Test | Result |
|---|---|
| C25-1 target vs committed Arm-B map | rho = 0.9977 (bar 0.999; see deviation log - target is the mean of the A2b halves, not a recomputation) |
| H25a block increment over [8 covariates + intermittency] | **+0.024** (r2 0.763 -> 0.787), p_rot = 0.011, null q95 = 0.023 -> below the frozen +0.03 bar => **NEGATIVE** |
| H25b residual correlations | monsoon_amp +0.04 (p=0.82); conv_frac -0.04 (p=0.37); **shear -0.381 (p=0.001)**; pw +0.05 (p=0.86) |
| C25-2 slope placebo | block increment +0.008 - the shear-residual association is not spectrum-shared |

Post-hoc (reported, not scored): single-candidate increments
monsoon_amp -0.002, conv_frac -0.000, **shear +0.027**, pw +0.001 - the
whole block increment is carried by the vertical wind shear alone.

## Reading

Three of the four hydroclimatic candidates are already absorbed by the
frozen baseline (their loadings on P are strong - conv_frac -0.61,
pw -0.56 - but collinear with the CAPE/moisture channel), so they add
nothing out-of-sector. The one genuinely new coordinate is the
**850-500 hPa vertical wind shear**: it is the only candidate
significantly associated with the reproducible residual
(rho = -0.381, p = 0.001 under the rotation null; higher shear ->
P below its covariate prediction), the association is not
spectrum-mediated (slope placebo +0.008), and it alone reaches +0.027 of
LOSO R^2 - just short of the frozen +0.03 bar. By the standing rules the
phase is NEGATIVE and no residual-attribution claim is made; the shear
becomes the single named, preregisterable candidate for the next scored
test (a shear-only protocol with its own bar), exactly as the SPCZ/ENSO
candidate was handled after Phase 17.

## Deviations (also logged in the protocol)

1. CDS delivered the single-levels request as a zip container with .nc
   extension; unpacked before reading (technical, no analytic content).
2. C25-1 was implemented as consistency of the frozen target (mean of the
   AUDIT-2b half-sample statistics) with the committed Arm-B map rather
   than a from-raw recomputation; measured rho = 0.9977. The 0.999 bar
   presumed literal pipeline identity, which does not apply to the
   halves-mean target; logged, not treated as a failure.
