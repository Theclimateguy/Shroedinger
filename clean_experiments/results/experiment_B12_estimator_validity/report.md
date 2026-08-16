# Phase 12: is A a valid estimator of irrecoverability?

Protocol: `docs/PROTOCOL_PHASE12_ESTIMATOR_VALIDITY.md` (frozen 2026-08-16).
Verdict: **ESTIMATOR_INVALID**

## H12a — controlled ladder of spatial addressability (9 rungs x 20 realisations)

| independent fraction | mean A | mean u |
|---|---|---|
| 0.000 | 1.033 | 0.7989 |
| 0.125 | 1.015 | 0.8054 |
| 0.250 | 1.046 | 0.8051 |
| 0.375 | 1.025 | 0.8094 |
| 0.500 | 1.051 | 0.8096 |
| 0.625 | 1.034 | 0.8142 |
| 0.750 | 1.043 | 0.8202 |
| 0.875 | 1.012 | 0.8178 |
| 1.000 | 1.001 | 0.8164 |

Spearman against the rung: **A -0.317**, u +0.917 (criterion >= 0.90).
The reference estimator tracks known irrecoverability; `A` does not move.

## Diagnostic (i) — modal-predictability ladder

| beta (fine determined by coarse) | mean A | mean u |
|---|---|---|
| 0.000 | 1.375 | 0.8433 |
| 0.125 | 1.342 | 0.8446 |
| 0.250 | 1.390 | 0.8439 |
| 0.375 | 1.399 | 0.8355 |
| 0.500 | 1.341 | 0.8394 |
| 0.625 | 1.353 | 0.8346 |
| 0.750 | 1.360 | 0.8316 |
| 0.875 | 1.334 | 0.8251 |
| 1.000 | 1.367 | 0.8202 |

Spearman: **A -0.350**, u -0.933.
`A` is flat under the modal reading as well, so the H12a failure is not a mismatch of construction.

## Diagnostic (ii) — what A does respond to

| factor swept (all else fixed) | range of A | Spearman |
|---|---|---|
| temporal autocorrelation 0.50 -> 0.95 | 0.921 -> 1.210 | **+1.00** |
| fine-band amplitude x0.25 -> x4 | 1.082 -> 1.101 | +0.80 |
| spectral slope -3.5 -> -1.5 | 1.085 -> 1.097 | +0.70 |

Within-rung noise is about 0.05 log units, so only the first row is a real response.

## On real fields (39 domains)

| pair | Spearman |
|---|---|
| A vs lag-1 autocorrelation of the resolved-band envelope | **+0.53** |
| A vs u (information-theoretic irrecoverability) | **+0.03** |
| u vs envelope persistence | +0.04 |
| A vs log band variance | -0.56 |

## Reading

`A` does not measure what the manuscript says it measures. Under controlled
conditions where irrecoverability is known and swept, in either its spatial or
its modal reading, `A` does not move; what it tracks instead is the temporal
persistence of the field, on synthetic data (Spearman 1.0) and on real domains
(+0.53). This is the same estimator-memory artefact recorded in
`docs/RECONCILIATION.md`: the transfer operators are fitted over a 20-step
sliding window.

The three application failures of Phases 9-11 are therefore not evidence about
the idea of cross-level irrecoverability. That quantity was never measured.
