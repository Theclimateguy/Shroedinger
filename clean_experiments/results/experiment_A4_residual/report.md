# AUDIT-4 — the unexplained part of the global map: physics, not noise

Protocol: `docs/PROTOCOL_AUDIT4_RESIDUAL_STRUCTURE.md` (frozen before
computation, with a binding decision rule and a scope fence). Code:
`clean_experiments/experiment_A4_residual_structure.py`. No new data:
the AUDIT-2b split-half shards plus the frozen Phase-20 covariates.
902 tiles.

**VERDICT: RESIDUAL_IS_PHYSICAL.**

## R1 (primary) — the residual is reproducible

Residuals are formed with the regression fitted on the OTHER time-parity
half, so the two residuals of a season share no sampling noise.

| residual of | JFM (split-half r / Spearman-Brown) | JAS | mean r_full |
|---|---|---|---|
| P after the 8 frozen covariates | 0.839 / 0.912 | 0.817 / 0.899 | **0.906** |
| P after covariates + intermittency block | 0.464 / 0.634 | 0.478 / 0.647 | **0.641** |

The decision bar was 0.30. What the eight covariates and the four
intermittency statistics leave behind is a reproducible field, not
estimator noise.

## R2 (reported) — where it lives

Season agreement of the residual: Spearman 0.406 between JFM and JAS —
partly season-stable, partly seasonal.

Largest NEGATIVE residuals (P lower than the covariates predict):
(-15, 298) Bolivia/SE Brazil, (27, 119) and (27, 112) SE China,
(21, 37) Red Sea / western Arabia, (27, 248) Mexican plateau,
(39, 20) Adriatic/Greece, (-21, 289) Andean foreland, (33, 57) Iran,
(3, 35) East Africa, (-39, 290) Argentine foreland.

Largest POSITIVE residuals (P higher than predicted): (27, 83) Ganges
plain / Himalayan foothills, (57, 296) Labrador, (3, 287) NW Amazon,
(9, 35) Ethiopian highlands margin, (9, 273) Central America,
(3, 28) Congo, (-57, 203) South Pacific, (-9, 145) and (-3, 136)
New Guinea, (57, 273) Hudson Bay.

Zonal profile: negative in both subtropical belts (30-40N mean -0.011,
30-40S -0.011), positive in the deep tropics (-10..10: +0.006/+0.007)
and at 50-60N (+0.015).

The pattern is continental and terrain-adjacent: monsoon margins,
plateau edges and the flanks of major orography on the negative side;
deep-convective cores, high-latitude land/ice margins and a few oceanic
tiles on the positive side. Note that tiles with mean orography above
1200 m are excluded by the frozen Phase-20 rule, so several of these sit
just BELOW that threshold — an unresolved orographic term is a natural
first hypothesis and is named as future work, not tested here.

## R3 (reported) — the residual is not spectral

Spearman of the residual against the tile spectral features, none of
which are in the covariate set: slope +0.059, logvar_0 -0.121,
logvar_1 -0.121, logvar_2 -0.116, logvar_3 -0.098. The remainder is not
a disguised band-variance or slope effect.

## Closure

Per the protocol's scope fence, no covariate hunt is run. The analysis
programme for the three papers ends here. The papers carry one
paragraph and one figure: after the attributable geography (59% of the
reproducible variance) and the intermittency block (to 83%), what
remains is a reproducible, non-spectral, terrain- and
convection-adjacent field whose drivers are not identified. Identifying
them is named future work.
