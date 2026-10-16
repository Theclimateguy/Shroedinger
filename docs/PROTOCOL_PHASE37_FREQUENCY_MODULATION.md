# PROTOCOL — Phase 37: how much of the co-location residual is frequency modulation (B37)

## Status: FROZEN 2026-10-16, before any Phase-37 quantity was computed

Trigger. Phase 35 (H35-4) showed that a field with uniform amplitude and a smoothly
varying dominant wavelength produces adjacent-band envelope coupling of -0.39 / +0.35
(frequency modulation, FM). Phase 36 showed that the co-location geography survives
the removal of the window's mean spectrum (residual reproducible 0.89; fresh year
0.82; free model 0.81) but its null does not remove FM. This phase measures the FM
part of the residual and asks whether the geography survives its removal.

## Idea

Write each band's log-envelope, smoothed at a large scale, as a common local level and
a local spectral shape: L_k(x) = A(x) + D_k(x), with A = mean over the three bands of
L_k and D_k = L_k - A. FM is a smooth variation of the shape D_k (where the local
spectrum is steeper or flatter, which band is active changes), not of the common level.
Three modulated nulls are built from the window's stationary spectral null (the mirror
surrogate of Phase 36, which carries the band-overlap baseline), by multiplying each of
its bands B1-B3 by a smooth weight taken from the real field:
- N_FM: w_k = exp(D_k)  — local spectral shape only (frequency modulation);
- N_A:  w_k = exp(A)    — common level only (large-scale co-location of activity);
- N_FA: w_k = exp(L_k)  — both (all smooth structure).
Each weight is normalised to unit mean square inside the window. Then:
- FM contribution = P45(N_FM) - P45(null);
- large-scale co-location = P45(N_A) - P45(null);
- small-scale co-location (scales below the smoothing) = P45(real) - P45(N_FA).

## Data, units, pipeline (fixed)

As Phase 36: ERA5 850 hPa JFM and JAS of 2022, 2023, 2024 and IFS-HR 2014 (ERA5
mask); 66 windows on azimuthal-equidistant planes with a 1100-km margin; every 2nd
6-hourly step; V12 vorticity; bands B1 = 200-400, B2 = 400-800, B3 = 800-1600 km;
Gaussian ladder (bars at 200, 400, 800, 1600 km, normalised by the valid mask);
P4, P5 = median over steps of the in-window Pearson correlation of log-envelopes;
P45 = (P4 + P5)/2. One mirror surrogate per step (as Phase 36, seed differs). Smoothing
of L_k for the weights: Gaussian at 1600 km (sigma = 1600/sqrt(12) km), normalised by
the valid mask. Year value = mean of the two seasons.

## Calibration (must pass, else the phase stops at the calibration verdict)

On a plane of the same size (2200 x 3000 km window, 1100-km margin, 27.8-km step), 100
synthetic vorticity fields each:
- **C37-1 (FM is captured).** The H35-4 field (uniform amplitude, dominant wavelength
  250 -> 900 km across the window): |P45(real) - P45(N_FM)| <= 0.10 and
  |P45(N_FM) - P45(null)| >= 0.10.
- **C37-2 (co-location is not mistaken for FM).** The H35-4 log-normal cascade:
  P45(real) - P45(N_FM) >= 0.05.

## Hypotheses and bars (ERA5, mean of 2023-24 unless stated)

- **H37-1 (FM is a minor part of the residual, PRIMARY part 1).** Median over windows
  of f_FM = (P45(N_FM) - P45(null)) / (P45(real) - P45(null)) <= 0.25.
- **H37-2 (geography survives FM removal, PRIMARY part 2).** R_noFM = P45(real) -
  P45(N_FM): Spearman 2023 vs 2024 >= 0.60 (belt-shift null, 999, p < 0.01) and
  Spearman(2022, mean 2023-24) >= 0.60.
- **H37-3 (reported).** Medians and maps of the three contributions; share of the
  residual that is small-scale co-location, (P45(real) - P45(N_FA)) / (P45(real) -
  P45(null)); Spearman of R_noFM with the Phase-36 residual; IFS-HR vs ERA5 for R_noFM.

## Verdict ladder

- **FM_MINOR:** C37-1, C37-2, H37-1, H37-2 pass.
- **FM_SUBSTANTIAL_GEOGRAPHY_SURVIVES:** calibration passes, H37-1 fails, H37-2 passes.
- **FM_DOMINANT:** calibration passes, H37-2 fails.
- **CALIBRATION_FAILED:** C37-1 or C37-2 fails (the decomposition does not do what it
  claims; nothing is concluded about ERA5).

## Priors (recorded before computation)

C37-1 0.7; C37-2 0.7; H37-1 0.5; H37-2 0.65.

## Code and outputs

`clean_experiments/experiment_B37_frequency_modulation.py` (stages `synth`,
`plane --source era5 --year Y --season S`, `plane --source ifshr`, `tests`) →
`clean_experiments/results/experiment_B37_frequency_modulation/`. Seed 20261018.

## Deviations

- (none yet)
