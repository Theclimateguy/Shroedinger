# Phase 16 Protocol (preregistered): slow modulation of the equilibrium
# profile P^eq — does the static regional fingerprint move with ENSO?

Status: FROZEN before any Phase-16 quantity was computed. Date frozen:
2026-08-17. Deviations logged with timestamp at the bottom.

## 0. Lineage and the multiplicity problem, stated up front

Phases 14-15 closed the dynamical line INSIDE the 92-day windows: P
fluctuations relax in ~12 h around a static P^eq, and no
|dX/dt|-magnitude object can be separated from persistence. What survives
as a dynamical question is the slow layer the windows cannot resolve:
does P^eq itself move across years with the climate state?

The hypothesis and its DIRECTION come from a descriptive line in the
Phase-14 record: window-median composite P was higher in the
El-Nino-side windows in 13 of 16 same-season region comparisons. That
descriptive used regions R5-R12 (windows W5-W12) ONLY. Consequently:

- **Confirmatory arm (H16a): regions R1-R4 exclusively** (windows
  W1-W4 and W9-W12; these region-windows never entered the Phase-14
  descriptive, or any ENSO-flavoured look at any phase).
- R5-R12 form the replication arm (H16b), reported with the explicit
  caveat that the hypothesis was generated partly on them.

Direction, fixed: **higher ONI (El Nino) => higher window-level composite
P** (stronger cross-scale envelope coupling).

## 1. Declared power limit (before results)

Confirmatory arm: 8 cells (4 regions x 2 seasons), 4 year-points per
cell. The primary statistic (mean of within-cell Spearman, section 4)
has a permutation-null SD of about sqrt(1/3)/sqrt(8) ~ 0.20, so only a
mean within-cell rho >= ~0.35 is detectable at alpha = 0.05 one-sided.
Stated in advance: a null here is weak evidence of absence; a positive
had to be strong.

## 2. Data (fixed, no new downloads, no recomputation of P)

- `P^eq(rw)` = mean of stored `P_real[3]` and `P_real[4]` (the resolved
  composite fixed since Phase 9) from the frozen result files:
  - `results/experiment_B2_scale_irreversibility/R{1..4}__W{1..4}.json`
  - `results/experiment_B2b_heldout_invariants/R{5..12}__W{5..8}.json`
  - `results/experiment_B3_scattering_benchmark/R{1..12}__W{9..12}.json`
  96 region-windows total. Stored medians are used verbatim; the "no
  re-fit of P" rule of Phase 15 section 8 applies.
- Amplitude placebo: mean of stored `F_spec.logvar_3` and
  `F_spec.logvar_4` from the same files.
- ENSO index: NOAA CPC ONI (ERSST.v6, 3-month running mean of Nino-3.4
  SST anomalies), fetched 2026-08-17 from
  https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/
  and FROZEN here:

  | window | season-year | ONI |
  |---|---|---|
  | W1 | JFM 2017 | +0.1 |
  | W2 | JAS 2017 | -0.1 |
  | W3 | JFM 2018 | -0.7 |
  | W4 | JAS 2019 | +0.2 |
  | W5 | JFM 2021 | -0.9 |
  | W6 | JAS 2021 | -0.5 |
  | W7 | JFM 2022 | -0.7 |
  | W8 | JAS 2022 | -0.8 |
  | W9 | JFM 2023 | -0.3 |
  | W10 | JAS 2023 | +1.3 |
  | W11 | JFM 2024 | +1.5 |
  | W12 | JAS 2024 | 0.0 |

## 3. Estimator gate (C16-1, scored FIRST)

The Phase-15 lesson generalizes: every new claim needs its own gate
against the artifact most likely to fake it. For a level statistic like
P^eq the artifact is not persistence (no differencing, no windowed
operators anywhere) but AMPLITUDE: ENSO modulates regional wind variance,
and if P — despite being rank-based — shadows band amplitude, an
ONI-P^eq correlation would be amplitude in disguise.

On the frozen Phase-12 synthetic generator (same grid, same seed
handling): (a) fine-band relative amplitude swept x{0.25, 0.5, 1, 2, 4}
at fixed independence 0.5, 12 realisations per rung; (b) the Phase-12
independence ladder (0..1, 9 rungs x 12) at fixed amplitude, as the
reference response. Composite P (median over time of mean rho_3, rho_4)
must show: (i) |Spearman(P, amplitude rung)| <= 0.5; (ii) across-rung
spread under the amplitude sweep smaller than under the independence
sweep. Failure => ESTIMATOR_INVALID, nothing further scored. (Phase-12
prior: P's amplitude association on real domains was +0.20; the gate is
expected to pass, but it is scored, not assumed.)

## 4. Hypotheses and criteria (fixed)

Cell = region x season (JFM or JAS). Within each cell there are exactly
4 year-points. Primary statistic on any arm:

    S = mean over cells of Spearman_within-cell( P^eq, ONI )

Null: 999 draws; in each draw the ONI values are permuted across the 4
years WITHIN each cell, independently per cell. One-sided p (S positive).

- **H16a (confirmatory, fresh arm).** Regions R1-R4: 8 cells, 32
  region-windows. Pass: S > 0 with p < 0.05.
- **C16-2 (amplitude placebo, on the fresh arm).** Same S with the
  logvar composite in place of P^eq. Required: |S_P| > |S_logvar|.
  If the placebo matches or exceeds P, the phase reports amplitude, not
  coupling.
- **H16b (replication arm, hypothesis-generating data).** Regions
  R5-R12: 16 cells, 64 region-windows. Reported pass: S > 0, p < 0.05.
  Sign agreement with H16a is required for CONFIRMED; H16b alone can
  never confirm (it is where the hypothesis came from).

Reported, descriptive only, no criterion weight: per-region within-cell
rho map; full-pool S (24 cells); linear year-trend of P^eq after
removing cell means and the ONI fit; per-band (rho_0..rho_4) versions of
S on the fresh arm; JFM-vs-JAS seasonal offset of P^eq per region.

## 5. Verdict rule (fixed)

- **ESTIMATOR_INVALID**: C16-1 fails.
- **CONFIRMED**: C16-1 passes; H16a passes; C16-2 satisfied; H16b same
  sign as H16a. The manuscript may then state: the regional fingerprint
  P^eq is not strictly static but is modulated by ENSO in the measured
  direction, with the fresh-region confirmation carrying the claim.
- **PARTIAL**: H16a passes but C16-2 fails or H16b has the opposite
  sign. Scoped as "an interannual signal exists in fresh regions but is
  not separable from amplitude / does not replicate coherently".
- **NEGATIVE**: H16a fails. The slow-modulation reading joins the
  dynamical line as tested and unsupported; P^eq stands as static at
  every timescale the programme's data can resolve, and the honest
  wording of Phases 14-15 ("statement of stability") becomes final.

## 6. Compute plan

- `clean_experiments/experiment_B16_penv_slow_modulation.py`
  -> `clean_experiments/results/experiment_B16_penv_slow_modulation/`.
- Stages: `--stage gate` (C16-1 synthetics), `--stage tests`
  (H16a/b, C16-2, descriptives, verdict; input = stored JSONs only).
- Seed fixed: 20260817. Permutations: 999.
- No new downloads; no recomputation of any P.

## 7. Explicitly out of scope (frozen)

- Any window/band/aggregation choice other than the composite fixed in
  section 2 (per-band figures are descriptive only).
- Extending the record with new ERA5 downloads (that is the natural
  Phase-17 if H16a passes: monthly P over the full ERA5 period).
- Attribution beyond ENSO (observing-system drift across 2017-2024
  cannot be excluded with 8 year-points; acknowledged as a limitation in
  advance — a CONFIRMED here licenses "co-varies with ENSO", not
  "caused by ENSO").

## Deviations

- **2026-08-17, run record.** C16-1 passed: P shows no systematic
  amplitude response (rho = -0.15, rung means non-monotone and
  noise-dominated), while responding to the independence ladder
  (rho = -0.51 here; weaker than the -0.883 of the Phase-12 diagnostic
  because this gate uses fewer replicates and the composite-median
  summary — recorded honestly, and the spread criterion passed with a
  thin margin, 0.0244 vs 0.0268). **H16a FAILED**: fresh-arm S = 0.000,
  p = 0.51 — the confirmatory statistic is exactly zero. C16-2 moot
  (placebo S = -0.10, also nothing). **H16b, the arm the hypothesis was
  generated on, "replicates" it: S = 0.375, p = 0.005** — which, next to
  a dead-flat fresh arm, is the textbook signature of a
  hypothesis-selection artifact rather than of a real modulation the
  fresh regions somehow lack. Verdict **NEGATIVE** per the frozen rule.
  Had the design pooled all 24 cells, the pooled S = 0.25 would have
  cleared p < 0.05 on the strength of the very data that suggested the
  hypothesis; the fresh-arm requirement existed precisely to prevent
  that publication.
