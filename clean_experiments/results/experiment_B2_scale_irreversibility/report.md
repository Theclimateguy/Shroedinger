# Experiment B2: scale-irreversibility profile as geographic invariant

Region-windows: 16/16. C1 positive control: 16/16 pass (need >=12/16).

| tag | P (rho_1..rho_5) | C1 bands>surr95 | corr(P,-dQ) |
|---|---|---|---|
| R1_WPWP__W1_2017JFM | 0.876, 0.861, 0.809, 0.757, 0.680 | 5/5 | -0.700 |
| R1_WPWP__W2_2017JAS | 0.871, 0.868, 0.846, 0.802, 0.778 | 5/5 | -0.800 |
| R1_WPWP__W3_2018JFM | 0.877, 0.864, 0.799, 0.706, 0.621 | 5/5 | -0.800 |
| R1_WPWP__W4_2019JAS | 0.880, 0.875, 0.837, 0.793, 0.670 | 5/5 | -0.900 |
| R2_NATL__W1_2017JFM | 0.902, 0.871, 0.800, 0.734, 0.715 | 5/5 | -0.600 |
| R2_NATL__W2_2017JAS | 0.918, 0.880, 0.767, 0.631, 0.606 | 5/5 | -0.600 |
| R2_NATL__W3_2018JFM | 0.900, 0.871, 0.788, 0.725, 0.700 | 5/5 | -0.400 |
| R2_NATL__W4_2019JAS | 0.912, 0.867, 0.746, 0.633, 0.593 | 5/5 | -0.600 |
| R3_AMAZ__W1_2017JFM | 0.823, 0.772, 0.665, 0.543, 0.359 | 5/5 | -0.900 |
| R3_AMAZ__W2_2017JAS | 0.858, 0.848, 0.758, 0.641, 0.423 | 5/5 | -1.000 |
| R3_AMAZ__W3_2018JFM | 0.813, 0.756, 0.634, 0.568, 0.374 | 5/5 | -0.700 |
| R3_AMAZ__W4_2019JAS | 0.854, 0.835, 0.717, 0.579, 0.390 | 5/5 | -0.700 |
| R4_CASIA__W1_2017JFM | 0.888, 0.845, 0.756, 0.593, 0.628 | 5/5 | -0.800 |
| R4_CASIA__W2_2017JAS | 0.878, 0.840, 0.722, 0.559, 0.473 | 5/5 | -0.900 |
| R4_CASIA__W3_2018JFM | 0.890, 0.846, 0.761, 0.613, 0.598 | 5/5 | -0.900 |
| R4_CASIA__W4_2019JAS | 0.879, 0.844, 0.724, 0.512, 0.422 | 5/5 | -1.000 |

C2: acc(spec)=1.000, acc(spec+irr)=1.000, gain=+0.000, p=0.005, pass=False
C3: within=1.000, between=1.000, diff=+0.000, p=1.000, pass=False

## PHASE2_VERDICT: NEGATIVE

## Exploratory appendix (post-hoc, cannot overturn the frozen verdict)

Both criterion failures are metric pathologies, discovered after results:

- C2 ceiling: spectral features alone classify region identity at 16/16 in
  LOO — a gain criterion cannot be passed over a saturated baseline. The
  irreversibility features alone reach 14/16 (P profile alone 14/16), so the
  profile IS region-identifying; added value over spectra is untestable on
  this task design.
- C3 degeneracy: all P profiles are monotone decreasing, so 5-point Spearman
  similarity equals ~1.0 for every pair (within = between). A shape-sensitive
  analogue (Euclidean distance on z-scored profiles) gives median
  within-region distance 1.41 vs between-region 2.99, permutation p = 0.001 —
  a strong regional signature in profile shape.
- Season/window classification fails for ALL feature sets (0/16): the profile
  is a regional, season-stable characteristic (invariant-like behavior).
- Mean profiles by region (rho_1..rho_5): oceanic R1_WPWP [.88 .87 .82 .76 .69]
  and R2_NATL [.91 .87 .78 .68 .65] retain strong coupling to coarse scales;
  land-convective R3_AMAZ [.84 .80 .69 .58 .39] loses cross-scale organization
  at synoptic steps; continental R4_CASIA is intermediate [.88 .84 .74 .57 .53].
  Physically consistent: deep convection generates fine-scale activity weakly
  slaved to synoptic scales; oceanic regimes are synoptically organized.

A clean test of the surviving claim requires Phase 2b: corrected preregistered
metrics (distance-based C3; C2 against a non-saturated target or with
added-information formulation) evaluated on HELD-OUT data (new regions and/or
years), since this dataset has now been used for exploration.
