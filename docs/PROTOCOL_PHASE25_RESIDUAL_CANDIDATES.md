# PROTOCOL — Phase 25: hydroclimatic candidates for the map residual
## Status: FROZEN 2026-08-24, before any candidate field was downloaded or read

Motivation. AUDIT-4 established that the residual of the global anchored-P
map after the 8 frozen covariates and the intermittency block is
reproducible (r_full = 0.64), non-spectral, and geographically organised:
negative lobes on monsoon margins and orography flanks, positive lobes over
deep-convective cores and high-latitude land margins. AUDIT-4's scope fence
forbade a covariate hunt inside itself and named driver identification as
future work. This protocol is that future work, run as a new preregistered
phase. An exploratory check (2026-08-24, logged in the session record, not
scored) already rejected the simplest orographic candidate: tile distance
to >1200 m terrain adds nothing (LOSO increment -0.001). The candidates
below are therefore hydroclimatic.

## Data (to be downloaded after freezing)

ERA5 monthly means, global, 0.5 deg, 2021-2024 (48 months), -> `data/b25hydro/`:

- single levels: total_precipitation, convective_precipitation,
  total_column_water_vapour;
- pressure levels 850 and 500 hPa: u_component_of_wind,
  v_component_of_wind.

Downloader: `clean_experiments/download_b25_hydroclim.py` (resumable).

## Candidate covariates (four, declared here; tile mean over the frozen
## Phase-20 Arm-B tile grid, orography exclusion unchanged, 902 tiles)

- C1 `monsoon_amp` — annual amplitude of the monthly precipitation
  climatology: max_m - min_m of the 12-month climatology of tp;
  log10(x + 0.1 mm/day) transform.
- C2 `conv_frac` — climatological convective fraction: mean_m(cp) / mean_m(tp),
  clipped to [0, 1].
- C3 `shear` — climatological mean of monthly |V850 - V500|
  (vector-difference magnitude of monthly-mean winds).
- C4 `pw` — climatological mean total_column_water_vapour.

No other candidate may be added after data are read; any addition demotes
the phase to exploratory.

## Scored hypotheses (bars set here)

- H25a (primary): decorrelated LOSO-sector increment of the 4-candidate
  block over the frozen baseline [8 Phase-20 covariates + 4 intermittency
  statistics], target = season-mean anchored P of the 902 tiles (AUDIT-2b
  shards, halves averaged). Null: 999 longitude rotations (>= 30 deg) of
  the candidate block only, exactly as Phase-20 H-B1. The candidates come
  from external (precipitation/moisture/wind-shear) fields, so no shared
  sampling noise with P; no split-half decorrelation is required.
  - RESIDUAL_ATTRIBUTED: increment >= +0.05 and p <= 0.01;
  - CANDIDATES_ADD:      increment >= +0.03 and p < 0.05;
  - NEGATIVE otherwise.
- H25b (secondary, reported): Spearman of the AUDIT-4 residual (after
  covariates + intermittency, cross-half construction unchanged) against
  each candidate separately, with the same rotation null; per-candidate
  p-values reported without a multiplicity claim.

## Controls

- C25-1: recomputed anchored P must reproduce the committed Arm-B map at
  rho >= 0.999 (pipeline identity, as in AUDIT-1/2).
- C25-2 (placebo): the same 4-candidate increment for the SPECTRAL-SLOPE
  target; if the candidates add comparably to the slope, any H25a positive
  is flagged spectrum-shared and the beyond-spectrum wording is withheld.

## Deviations (logged 2026-08-26, at computation)

1. CDS delivered the single-levels monthly request as a zip container with
   a .nc extension; it was unpacked before reading (technical).
2. C25-1 was implemented as consistency of the frozen target (mean of the
   AUDIT-2b half-sample statistics) with the committed Arm-B map, not a
   from-raw recomputation; measured rho = 0.9977 against the 0.999 bar
   that presumed literal pipeline identity. Logged, not a failure.

## Outcome (entered after computation)

PHASE25_VERDICT: NEGATIVE by the frozen ladder (block increment +0.024 <
+0.03). Reported finding: vertical wind shear 850-500 hPa is the only
candidate associated with the reproducible residual (rho = -0.381,
p = 0.001, rotation null; not spectrum-shared, slope placebo +0.008) and
alone carries +0.027 of the increment. Shear is the single named
candidate for a future scored shear-only protocol.
