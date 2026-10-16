# PROTOCOL — Phase 38: co-location below 1600 km, free of frequency modulation (B38)

## Status: FROZEN 2026-10-16, before any Phase-38 quantity on ERA5 or IFS-HR was computed and before the Phase-38 calibration fields were generated

Trigger. Phase 37 could not isolate frequency modulation (FM): FM changes both the local
spectral shape and the common activity level (CALIBRATION_FAILED). Its calibration
showed, after the fact, that the all-smooth null N_FA (the window's stationary null with
each band re-weighted by exp of the band's log-envelope smoothed at 1600 km) reproduces
an FM field and does not reproduce a cascade. Those numbers were seen before this
protocol; therefore the calibration is repeated here on new synthetic geometries and a
new seed.

What the conservative residual means. S = P45(real) - P45(N_FA) removes every spatial
structure of band activity at scales of 1600 km and above — FM of any kind at those
scales, and also genuine large-scale co-location (activity of both bands concentrated
in one part of the window). S is co-location of adjacent-band activity at scales below
1600 km. It is a lower bound on co-location, not its total. The smooth part
P45(N_FA) - P45(null) mixes FM and large-scale co-location and is reported, not
interpreted.

## Data, pipeline (fixed)

Exactly Phase 37 (code `experiment_B37_frequency_modulation.py`, stage `plane`,
unchanged): ERA5 850 hPa JFM and JAS of 2022, 2023, 2024; IFS-HR 2014 with the ERA5
mask; 66 windows on azimuthal-equidistant planes; every 2nd step; one mirror surrogate
per step; smoothing of log-envelopes at 1600 km; P4, P5, P45 as in Phase 37.

## Calibration (fresh; must pass)

Plane of the 40-60 deg N window size, 27.8-km step, 100 fields each, seed 20261019
(not used before):
- **C38-1 (FM along the other axis and a wider range).** Uniform amplitude, dominant
  wavelength varying along y from 300 to 1200 km: |P45(real) - P45(N_FA)| <= 0.10.
- **C38-2 (FM with two smooth centres).** Uniform amplitude, dominant wavelength
  varying radially (small in two circular regions of radius 600 km, large elsewhere,
  range 250-900 km): |P45(real) - P45(N_FA)| <= 0.10.
- **C38-3 (small-scale co-location is kept).** Log-normal cascade with log-amplitude
  variance 0.3 per octave: P45(real) - P45(N_FA) >= 0.05.

## Hypotheses and bars (ERA5)

- **H38-1 (small-scale co-location exists).** Mean of 2023-24: median over windows of S
  >= 0.05, sign-flip test over windows (9999) p < 0.01.
- **H38-2 (its geography is reproducible, PRIMARY).** Spearman(S_2023, S_2024) >= 0.60
  (belt-shift null, 999, p < 0.01) and Spearman(S_2022, mean 2023-24) >= 0.60.
- **H38-3 (reported).** Median shares of the Phase-36 residual (real - null): smooth part
  (N_FA - null) and small-scale part (S); Spearman of S with the Phase-36 residual and
  with abs latitude; IFS-HR vs ERA5 for S; map of S.

## Verdict ladder

- **SMALL_SCALE_COLOCATION_CONFIRMED:** C38-1..3, H38-1, H38-2 pass.
- **SMALL_SCALE_COLOCATION_NOT_GEOGRAPHIC:** calibration and H38-1 pass, H38-2 fails.
- **NO_SMALL_SCALE_COLOCATION:** calibration passes, H38-1 fails (the residual of
  Phase 36 is smooth structure: FM or large-scale co-location, not separable here).
- **CALIBRATION_FAILED:** any of C38-1..3 fails.

## Priors (recorded before computation)

C38-1 0.75; C38-2 0.6; C38-3 0.8; H38-1 0.7; H38-2 0.55.

## Code and outputs

Planes: `experiment_B37_frequency_modulation.py plane` (unchanged) writing to
`clean_experiments/results/experiment_B37_frequency_modulation/`; calibration and tests:
`clean_experiments/experiment_B38_smallscale_colocation.py` →
`clean_experiments/results/experiment_B38_smallscale_colocation/`.

## Deviations

- (none yet)

## Result (2026-10-16)

**Verdict: CALIBRATION_FAILED.** Nothing was computed on ERA5 or IFS-HR under this
protocol.

| synthetic (100 fields, seed 20261019) | real | stationary null | N_FA | criterion | result |
|---|---|---|---|---|---|
| C38-1 FM along y, 300-1200 km | 0.360 | 0.448 | 0.266 | abs(real - N_FA) <= 0.10 | PASS (0.094) |
| C38-2 FM, two 600-km centres | 0.336 | 0.631 | 0.565 | abs(real - N_FA) <= 0.10 | **FAIL (0.229)** |
| C38-3 cascade, variance 0.3 | 0.754 | 0.420 | 0.538 | real - N_FA >= 0.05 | PASS (0.216) |

The all-smooth null (log-envelopes smoothed at 1600 km) cannot represent frequency
modulation whose structure is finer than the smoothing scale. With envelope-modulated
nulls, FM and co-location are not separable at those scales.

**Descriptive, post hoc, not scored — a sign diagnostic.** In the Phase-37 FM field (the
only FM field whose per-step values were stored) the residual over the stationary null
has opposite signs at the two steps (step 4 -0.56, step 5 +0.39): the middle band B2 is active with B1 where the
dominant scale is small and with B3 where it is large. In the cascade both steps are
positive (+0.36, +0.21). In ERA5 (Phase 36 residuals) both steps are positive in 66 of
66 windows in every year (medians +0.30 and +0.25; minima +0.20 and +0.05-0.12), and
their window-to-window correlation is +0.52 to +0.57. The ERA5 pattern has the shape of
co-location of activity across the three bands, not of FM. Because it was noticed after
the calibrations, it needs its own frozen test (synthetic families and a quantitative
criterion) before it can be claimed.
