# AUDIT-2b — split-half decontamination and the map's true ceiling

Protocol: `docs/PROTOCOL_AUDIT2B_SPLITHALF_DECONTAMINATION.md` (frozen
before computation). Code: `clean_experiments/experiment_A2b_splithalf.py`.
Each tile-season sample is split by time parity into two disjoint halves
(180 steps each); both halves span the whole season, so only the
sampling noise differs. Every statistic is anchored on its own 99 phase
surrogates. 902 tiles kept.

**This audit supersedes A2d of AUDIT-2 and the 0.603 ceiling used in all
earlier scoping.**

## B1 — the true reliability ceiling (the correction that matters most)

| season | split-half r | Spearman-Brown r (full sample) | R^2 ceiling |
|---|---|---|---|
| JFM | 0.924 | 0.960 | 0.922 |
| JAS | 0.909 | 0.952 | 0.907 |
| mean | | | **0.914** |

The number used before, 0.603, was the CROSS-SEASON agreement. It
charges genuine seasonal change to noise. The within-season map is
reproducible at R^2 = 0.914, not 0.603.

Consequences, restated with the correct denominator:

| quantity | old share of "ceiling" | corrected share |
|---|---|---|
| Phase-20 Arm-B attribution (R^2 = 0.536) | 0.89 | **0.59** |
| covariates + intermittency (R^2 = 0.761) | 1.26 | **0.83** |

The claim that P's global map is nearly saturated by the Phase-20
covariates was an artefact of the wrong ceiling. About 40% of the
reproducible variance of the map is still unexplained by the eight
frozen covariates.

Split-half reliability of every statistic in the two audits (r_half /
Spearman-Brown r_full):

| statistic | r_half | r_full |
|---|---|---|
| P | 0.916 | 0.956 |
| P_lin | 0.925 | 0.961 |
| R_AM_cyc | 0.896 | 0.945 |
| P_cascade | 0.889 | 0.941 |
| INT_sig2 | 0.931 | 0.964 |
| INT_flat | 0.924 | 0.961 |
| INT_mu | 0.911 | 0.953 |

**This corrects AUDIT-1.** R_AM_cyc is NOT a near-signal-free statistic:
within a season it is estimated as precisely as P (0.90 vs 0.92). Its
low cross-season agreement (0.26) means it is strongly SEASON-DEPENDENT,
not noisy. P's map is season-stable; the amplitude-modulation map is
not. That is a sharper statement than the one AUDIT-1 made, and it is
the one the papers should carry.

## B2 (scored primary) — decontaminated increment: VERDICT INTERMITTENCY_ADDS

Target: anchored P of one half. Predictors: the 8 frozen covariates plus
the four intermittency statistics computed on the OTHER half. No shared
sampling noise.

| combination | covariates | + intermittency | increment | p (rotation) |
|---|---|---|---|---|
| JFM, P(odd) <- INT(even) | 0.481 | 0.687 | +0.206 | 0.001 |
| JFM, P(even) <- INT(odd) | 0.464 | 0.668 | +0.203 | 0.001 |
| JAS, P(odd) <- INT(even) | 0.434 | 0.649 | +0.215 | 0.001 |
| JAS, P(even) <- INT(odd) | 0.425 | 0.630 | +0.206 | 0.001 |
| **mean** | | | **+0.207** | max p = 0.001 |

## B3 — how much of the raw increment was noise, and where +0.08 came from

- same-half (contaminated) increment: **+0.261**
- decontaminated increment: **+0.207**
- shared-noise share: **21%**, not the ~70% suggested by the quick
  cross-season control in AUDIT-2.

The cross-season control (+0.077 / +0.081) was itself wrong in the other
direction: it removes the shared noise but also throws away the
season-dependent part of the intermittency signal, and the intermittency
statistics are season-dependent. The split-half design removes only the
noise. **The number of record for the intermittency increment is +0.21.**

The attenuation correction specified in the protocol is a no-op by
construction (scaling predictor columns by constants does not change an
OLS fit) and is reported as such; the block's split-half reliability is
0.89-0.93, so attenuation was small anyway.

## B4 — the AUDIT-1 / AUDIT-2 primaries recomputed across halves

Spearman between P of one half and the comparator of the other half —
correlations of signal, immune to shared estimation noise. Mean over the
four combinations, with the range:

| pair | cross-half rho | range | same-sample value (AUDIT-1/2) |
|---|---|---|---|
| P vs R_AM_cyc | **+0.145** | [+0.128, +0.178] | +0.143 |
| P vs P_lin | **+0.893** | [+0.883, +0.901] | +0.992 |
| P vs P_cascade | **-0.287** | [-0.312, -0.261] | -0.417 |
| P vs INT_sig2 | **+0.600** | [+0.590, +0.614] | +0.692 |

Disattenuated for the reliability of both sides, P vs P_lin is ~0.98 and
P vs INT_sig2 ~0.65. Every frozen verdict survives: the
amplitude-modulation reading stays dead (0.145), the cascade reading
stays dead (-0.29), the rank/copula construction stays decoration
(~0.98), and the intermittency parameter stays a substantial but partial
relative (~0.65).

## What changes in the record

1. The map's R^2 ceiling is **0.914**, not 0.603. Every "share of
   ceiling" statement before this audit is superseded.
2. Phase-20's attribution explains **59%** of the reproducible variance,
   not 89%. The map is not nearly saturated.
3. The intermittency increment over the frozen covariates is **+0.21**
   (p = 0.001, four independent combinations). The contaminated +0.225
   and the attenuated +0.08 are both retired.
4. AUDIT-1's reading of R_AM_cyc as "nearly signal-free" is corrected to
   "season-dependent": within-season reliability 0.90.
