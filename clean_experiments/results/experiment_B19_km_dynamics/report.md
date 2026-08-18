# Experiment B19: dynamics re-asked on the equal-km territory

Protocol: `docs/PROTOCOL_PHASE19_KM_DYNAMICS.md` (frozen 2026-08-18).
Code `clean_experiments/experiment_B19_km_dynamics.py`. Arm A runs the
frozen Phase-14 pipeline verbatim on km-cropped fields (wind + CAPE +
precip, Phase-18 region table); Arm B estimates tau_b(ell) on all 80 km
region-windows. Seeds: phase 20260818; the frozen B14 internals keep
their baked-in 20260816.

## PHASE19_VERDICT: (NEGATIVE_TERRITORY_ROBUST, ESTIMATOR_INVALID)

### C19-1 sanity

Median corr(km P(t), degree P(t)) = 0.801, range 0.63-0.95 — inside the
frozen [0.5, 0.995] window. The territory change is real (not a
relabeling) and the measurement is stable. Arm A is informative.

### Arm A: the B14 negative is TERRITORY-ROBUST

The frozen ladder returns **FORM_REJECTED** again, and every component
is at least as negative on the km territory as on degrees:

| Component | degrees (B14) | km (B19) |
|---|---|---|
| H14b log-ACF linearity | r2 0.8905 (bar 0.90) | r2 0.8235 |
| H14c per-rw coupling passes | 5/16 | 1/16 |
| H14c pooled p | 0.13 | 0.074 (still >0.05) |
| H14d held-out wins | 4/16 | 4/16 |
| Sign split | R7 +4/4, R5 -4/4 | **R7 +4/4, R5 -4/4** |

The central question of the phase — is the R7_CONGO/R5_SPCZ sign split a
territory-composition artifact of the wide degree boxes? — is answered
NO. The split reproduces exactly, window by window, on boxes that cover
substantially different territory (fig1, right). The regime-coupling
negative of Phase 14 is final for this data class; per the frozen
protocol, no further re-asks on any future grid.

### Arm B: gate failed; and the real data show no scaling anyway

- **Gate (frozen bar |alpha| <= 0.15): FAILED** — single-timescale
  synthetic cubes yield spurious median |alpha| = 0.178 (mostly
  negative: the nonlinear envelope-rank ladder shortens apparent
  fine-band memory even when every scale has the same 12-h clock). By
  the frozen rule, Arm B verdict is ESTIMATOR_INVALID and the real-data
  alpha is descriptive only.
- Descriptive: median alpha = +0.000, cluster-bootstrap CI
  [-0.077, +0.106]; median tau_b = 11.1, 9.4, 8.2, 9.5, 9.4 h across
  band steps 71 -> 1131 km. Eddy-turnover (2/3) would predict a ~6x
  spread over this range; sweeping ~11x. Observed: none (fig2, left).
  The real |alpha| is SMALLER than the estimator's own spurious scale,
  so no scaling claim of either sign could survive the gate — the
  descriptive picture is a flat ~9-11 h clock at every resolved scale,
  consistent with (and extending) the B14 "fast, memoryless relaxation":
  the relaxation has no resolvable scale hierarchy either.
- Between-region alpha spread is nominally significant (perm p=0.003;
  region medians -0.16..+0.22) — reported, but it sits inside an invalid
  estimator and is not interpreted.

### What Phase 19 settles

1. The user-raised alternative — "the dynamical negatives might be an
   artifact of wrong territorial differencing on the degree grid" — is
   now tested and excluded for B14: NEGATIVE_TERRITORY_ROBUST.
2. The one dynamical structure a level-free, derivative-free object
   could still show (a timescale hierarchy across scales) is absent
   descriptively and unmeasurable by this estimator class formally.
3. Remaining dynamical candidates are unchanged: Arm C (km-crop of the
   Phase-17 long record, deferred until b17daily completes) and
   signed/oriented transfer events (new protocol, own gate).

### Figures

- `fig1_armA.png` — C19-1 distribution; the sign split on both grids.
- `fig2_armB.png` — tau_b(ell) flatness vs turnover/sweeping slopes;
  real vs synthetic alpha distributions and the failed gate.
