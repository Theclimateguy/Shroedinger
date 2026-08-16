# Phase 12 Protocol (preregistered): is A a valid estimator of irrecoverability?

Status: FROZEN before any Phase-12 quantity was computed. Date frozen:
2026-08-16. Deviations logged with timestamp at the bottom.

## What this is, and what it is not

Phases 9, 10 and 11 tested `A` against three external targets and `A` lost to
the plain band variance of the field every time. The standing decision fixed
in the Phase-11 protocol stands: **no fourth external target is being sought
here.**

Phase 12 asks a different question, about the instrument rather than about
its applications:

> Does `A` actually measure the quantity the theory names — the part of the
> mesoscale that cannot be recovered from the coarse field — or is it a poor
> operationalisation of that idea?

This matters because the two possible answers lead to opposite conclusions
from the same three negatives:

- if `A` *is* a faithful estimator, then the three negatives are negatives
  about the idea, and the programme's conclusion is settled;
- if `A` is *not* a faithful estimator, the three negatives are negatives
  about a particular formula, and the theory's own quantity has never been
  measured at all.

No verdict about applications is drawn here in either direction.

## The reference quantity

`A_b = || G^up G^down - G^down G^up ||_F` is a linear-operator construct on
modal coefficients. The quantity it is *claimed* to stand for is
information-theoretic: how much of the placement of level-`b` activity is not
determined by the adjacent coarser level. The natural direct estimator is the
normalised mutual information between the two activity maps,

```
r_b(t) = I( rank E_b(.,t) ; rank E_{b+1}(.,t) ) / H( rank E_b(.,t) ),
u_b(t) = 1 - r_b(t)                                   (irrecoverability)
```

estimated from the joint histogram of the rank-transformed envelopes over the
interior grid points, with a fixed 16 x 16 binning, and taken as the median
over the times of a window.

`u` has one property `A` does not: being built from ranks, it is invariant
under any monotone transformation of either field, and therefore **cannot be
a shadow of band variance by construction**. That is precisely the placebo
that defeated `A` three times.

## Data (fixed)

ERA5 850 hPa `u`, `v` already on disk, no new downloads:
`data/b3` (12 programme regions x W9-W12) and `data/b10era5` (27 lattice
domains x W11, W12). Band pairs, vorticity, envelopes, interior mask and the
`A` machinery are unchanged from Phases 4 and 10; confirmatory band pairs are
those spanning 200-800 km.

## Hypotheses and criteria (fixed)

- **H12a (synthetic positive control — the decisive one).** Synthetic fields
  are generated in which the fine level is, by construction,
  `E_fine = alpha * f(E_coarse) + (1 - alpha) * noise`, with the independent
  fraction `1 - alpha` swept over a fixed ladder of 9 values from 0 to 1,
  20 realisations each, on the same grid and with the same spectra as a real
  domain. Requirement: `A` must increase monotonically with the independent
  fraction, Spearman rho >= 0.90 over the ladder. `u` must do the same
  (it is the reference and is expected to pass). **If `A` fails H12a, it does
  not measure irrecoverability even when irrecoverability is known and
  controlled, and nothing further in Phase 12 is interpreted.**
- **H12b (does `A` track `u` on real fields).** Across the 39 domains,
  Spearman(`A`, `u`) > 0 with permutation p < 0.05 (999).
- **H12c (placebo).** The same with log band variance in place of `A`.
  `|rho_A|` must exceed it; otherwise `A`'s apparent agreement with `u` is
  once again amplitude.
- **H12d (is `A` more than `P`).** Partial Spearman of `A` with `u` after
  leave-one-out residualisation of both on the profile `P`. A distinct
  content for `A` requires this to stay positive with p < 0.05.

Reported, explicitly **descriptive and not a criterion**: the correlation of
`u` with the targets already measured in Phases 9-11 (AI blurring deficit,
inter-analysis disagreement `D`, short-lead error `r12`). This is reported so
that the question "would the information-theoretic estimator have done
better?" is visible, but it inherits the multiplicity of every target already
tried and **cannot be claimed as a positive result from this run**. If it
looks promising it must be re-tested on fresh domains under its own frozen
protocol.

## Verdict rule (fixed)

- **ESTIMATOR_VALID**: H12a passes for `A`; H12b passes; H12c satisfied.
  The three negatives then stand as negatives about the idea.
- **ESTIMATOR_WEAK**: H12a passes but H12b or H12c fails. `A` responds to
  controlled irrecoverability but does not track it across real regions;
  the operationalisation is then the suspect, not the idea.
- **ESTIMATOR_INVALID**: H12a fails. `A` does not respond to irrecoverability
  even under controlled conditions, and every `A`-based claim in the
  manuscript, including the surviving signature claims, must be re-examined.

## Compute plan

- `clean_experiments/experiment_B12_estimator_validity.py`
  -> `clean_experiments/results/experiment_B12_estimator_validity/`.
- Seeds fixed: 20260816.

## Deviations

- (none yet)

- **2026-08-16, two diagnostics added AFTER H12a was seen, both declared
  diagnostic and carrying no criterion weight.**

  (i) *Second synthetic ladder.* H12a as frozen manipulates the **spatial**
  addressing of fine-scale activity by the coarse field. `A`, however, is
  built from the time series of modal coefficients, so a failure on H12a
  alone could mean the ladder never exercised what `A` looks at. A second
  ladder was therefore run in which spatial addressing is left alone and the
  **linear predictability of the fine-band modal coefficients from the
  coarse band** is swept instead (fixed LTI pattern, weight `beta` from 0 to
  1, 12 realisations per rung). `A` is flat there too
  (rho = -0.35, across-rung spread 0.02 against within-rung sd 0.07), while
  the reference estimator responds correctly (rho = -0.93). The caveat is
  therefore closed: `A` responds to neither reading of recoverability.

  (ii) *What `A` does respond to.* A three-factor sweep on synthetic fields
  with everything else held fixed: temporal autocorrelation of the field
  (rho_t 0.5 to 0.95), fine-band amplitude (x0.25 to x4) and spectral slope
  (-3.5 to -1.5). `A` tracks **temporal autocorrelation monotonically and
  strongly** (Spearman 1.0, range 0.92 to 1.21 log units against a
  within-rung noise of about 0.05), and only weakly and non-monotonically
  the other two. On real fields this is confirmed:
  rho(`A`, lag-1 autocorrelation of the resolved-band envelope) = **+0.53**
  over the 39 domains, while rho(`A`, `u`) = +0.03.

  This is the same disease already on record in `docs/RECONCILIATION.md`,
  where half of the apparent two-day relaxation turned out to be the memory
  of the W = 20 estimator window. The transfer operators are estimated over a
  20-step sliding window, so the more persistent the modal coefficients, the
  more systematically the commutator norm is displaced.

- **2026-08-16, third diagnostic: the profile `P` on the same frozen ladder.**
  Added before publication, on the observation that `A` had been dismantled
  with an instrument that `P` itself had never been put through. The ladder
  was replayed with the identical seed and ordering, so the numbers are
  directly comparable with those already logged, and `P` was computed on the
  same fields.

  | estimator | Spearman vs the rung | across-rung spread / within-rung noise |
  |---|---|---|
  | `u` (reference) | +0.917 | 0.69 |
  | **`P`** | **-0.883** | **0.65** |
  | `A` | -0.317 | 0.35 |

  `P` responds in the predicted direction — more independent fine-scale
  activity means lower coupling — with a signal-to-noise ratio matching the
  reference. Recorded exactly: -0.883 is marginally short of the |0.90| bar
  the protocol set, the shortfall coming from the flattening of the ladder at
  its saturated end, which `u` shows too (its top three rungs are 0.820,
  0.818, 0.816). The discriminating comparison is not `P` against a threshold
  but `P` and `u` against `A`: the first two respond, the third does not.
  This is a diagnostic, added after the battery was scored, and carries no
  criterion weight.
