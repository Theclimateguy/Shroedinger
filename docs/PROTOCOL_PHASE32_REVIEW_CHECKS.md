# Protocol — Phase 32: sensitivity checks requested by the article-1 review

Frozen: 2026-10-08, before any computation. Trigger: review of manuscript 3209
(Izv. RAN Ser. Geogr., reviewer A. N. Krenke), see
`docs/REVIEW_ARTICLE1_RESPONSE_PLAN.md`. Decision of the author: path I (keep
"regional invariant"), per-item fallback to the weaker wording if a check fails.

Script: `clean_experiments/experiment_B32_review_checks.py` (A3–A7, existing JSONs)
and `clean_experiments/experiment_B32_vorticity_mask.py` (A1–A2, recomputation).
Results: `clean_experiments/results/experiment_B32_review_checks/`.
Seed: 20261008. Permutations: 999 (as in Phase 2b). Nothing in Table 2 of the
manuscript is changed by this protocol; it adds rows / replaces numbers only where
a check below says so.

## Inputs (fixed)

- 96 region-windows of the raw profile `P_real` (5 steps) and `P_surrogate_median`:
  Phase 2 (16, R1–R4, 2017–2019), Phase 2b (32, R5–R12, 2021–2022, held-out),
  Phase 3 (48, R1–R12, 2023–2024). Degree boxes 20°×40°.
- 80 region-windows of the equal-km boxes (Phase 18, `km_1.0`): 48 primary + 32 held-out.
- Phase 13 `domain_values` (39 domains, ERA5 vs free-running IFS-HR) and ERA5 across-period
  ceiling.
- Phase 8 MERRA-2 per-window JSONs (48) and raw MERRA-2 fields (`data/b8merra2`).
- ERA5 raw winds `data/b3` (48 windows 2023–2024) for A1–A2.
- ERA5 invariant surface geopotential (to download: `data/b32/era5_orography.nc`).

Window covariates: season = JFM / JAS from the window name; year from the window
name; ENSO phase by period as in the manuscript (2017–2019 neutral, 2021–2022
La Niña, 2023–2024 El Niño).

## Statistic

As in Phase 2b: descriptors standardised column-wise; Δ = median between-region
Euclidean distance − median within-region distance; p from 999 permutations of
region labels across region-windows; pass = Δ > 0 and p < 0.05 (strict pass for a
"headline" row: p ≤ 0.01).

## Checks, hypotheses, pass criteria, fallback

**A3 — first step removed (steps 2–5).**
H32-3: the regional signature of Table 2 rows 1, 5, 7 and the beyond-spectrum rows
2–3, 6 survive without step 1 (pair <50 | 50–100 km), which carries the known
lat–lon-grid artefact.
Pass: rows 1 and 5 (held-out, degree and km) Δ ≥ +0.8 with p ≤ 0.01; rows 2, 3, 6
keep sign (Δ > 0). Report old/new for every row.
Fallback: if rows 1/5 fail → the "invariant" claim is restricted to steps 4–5 (row 7)
and the title is softened (path II for this item).
Also recorded (descriptive, no criterion): regional ordering on fine steps 2–3 of
the anchored km profile, to replace the manuscript sentence about "fine bands".

**A4 — region-dependent effective resolution.**
H32-4: the signature survives when only steps resolved under the pessimistic
estimate are used: (ii) step 5 only (400–800 | 800–1600 km) for all regions;
(iii) steps 4–5 in extratropical regions (|lat centre| ≥ 25°), step 5 only in
tropical regions. Pass: p ≤ 0.01 on the held-out 32 windows for (ii) and (iii).
Fallback: the manuscript states the claim for steps 4–5 with the 200-km threshold
as a declared choice and reports (ii)/(iii) as a limitation.

**A5 — stratified permutations and variance components.**
H32-5a: within every stratum of season (JFM, JAS), year and ENSO phase/period, the
region labels still separate: permutations of region labels restricted to windows
of the same stratum (labels shuffled within each stratum), 999 draws, on the 96
degree-box windows and on the 80 km-box windows. Pass: Δ > 0 and p ≤ 0.01 for the
season-stratified, year-stratified and period-stratified tests on both box sets.
H32-5b: variance components per step (one-way region model and nested
region/season/year decomposition by sums of squares on the 12 regions × 8 windows
of 2017–2024 degree boxes): ICC(region) = σ²_region / σ²_total ≥ 0.5 on each of
steps 2–5. Report also the share of season and year.
Fallback: if 5a fails for a stratum, the manuscript lists that factor as "not
separately controlled" and row 1 is worded as "reproduced across windows that
differ in season, year and ENSO phase"; if 5b gives ICC < 0.5 on a step, that step
is excluded from the invariant statement.
Also: a region-level test with 12 units — leave-one-window-out nearest-centroid
assignment accuracy and the permutation test on region medians (8 windows per
region collapsed to one vector; p by permuting window-to-region assignment).

**A6 — spatial-block uncertainty for the 39-domain comparisons.**
H32-6: the ERA5–free-run Spearman ρ (Phase 13, 0.71) has a 90 % block-bootstrap
interval whose lower bound exceeds 0.4; blocks = 6 longitude sectors of 60° and,
separately, 5 latitude rows of the lattice. Also a leave-one-block-out jackknife.
Fallback: report the interval and reword the comparison as descriptive.

**A7 — confidence intervals for borderline rows.**
For rows 2 (scattering, p = 0.026), 3 (full spectrum, p = 0.029) and 6
(km, anchored, p = 0.033): 90 % bootstrap interval for Δ from resampling windows
within regions (999 draws), and an alternative test on region medians.
Pass: interval excludes 0 in at least one scheme. Fallback: the row is moved to
"descriptive" in Table 2.

**A1 — spherical curl term.**
ω_s = ∂v/∂x − ∂u/∂y + u·tanφ/a (full relative vorticity on the sphere; the sign follows from ζ = (a cosφ)⁻¹[∂v/∂λ − ∂(u cosφ)/∂φ]). H32-1:
profiles computed with ω_s on the 48 windows of 2023–2024 agree with the archived
ones: Spearman of the 5-step profile per window ≥ 0.99 and |ΔP| ≤ 0.01 on every
step; the clustering Δ on raw profiles (48 windows) within ±0.05 of the archived
value. Surrogate-anchored profiles recomputed on 4 contrasting regions (R2, R6, R12,
R1), 199 surrogates: |Δ anchored P| ≤ 0.02 per step. Expectation stated before
computing: effect ∝ ℓ·tanφ/(2πa), i.e. < 0.5 % on steps 1–3, up to ~6 % of the
band amplitude on step 5 at 55°. Fallback: adopt ω_s as the manuscript definition
and replace all numbers by recomputed ones (full recomputation, author's decision
6.6 in the plan).

**A2 — points below the surface at 850 hPa.**
Static mask: surface geopotential height z_s > 1457 m (the ICAO standard-atmosphere
height of 850 hPa). Masked fraction per region reported for ERA5 and compared with
the MERRA-2 NaN fraction (already known: R4 34 %, R12 7 %, R7 3 %, R3 2 %, R11 1 %).
Treatment (a) "exclude": masked points are removed from the correlation only
(interior mask ∧ ¬orography). Treatment (b) "fill": masked points are set to NaN
before smoothing and smoothing uses normalised convolution (Gaussian of the valid
indicator), then (a). H32-2: for every region, the raw profile under (a) and (b)
differs from the archived one by ≤ 0.03 on steps 2–5; the clustering Δ on the 48
raw profiles within ±0.1 of the archived value. MERRA-2: the three-step profiles
recomputed with (a) and (b) (NaN never replaced by zero); H32-2m: ERA5–MERRA-2
geography agreement (steps 4–5 vs MERRA 2–3, 12 regions) stays ≥ 0.8 and the
MERRA-2 signature Δ > +0.8, p ≤ 0.01.
Fallback: if R4 or R12 move by more than 0.03, the manuscript adopts treatment (b)
as the definition and all rows involving these regions are recomputed; if H32-2m
fails, row 4 is reworded to exclude regions with a masked fraction above 5 %.

## Outputs

`results/experiment_B32_review_checks/summary.json` with one block per check
(A1…A7), each with `pass` and the numbers; `B32_SUMMARY.md` written by hand
after inspection. Figures only where they will enter the appendix.


## Part 2 — full recomputation under definition V12 (added 2026-10-08, after part 1)

Author's decisions after part 1: (i) the regional signature without step 1 is
accepted on the p-criterion (the Δ ≥ 0.8 threshold did not allow for the loss of one
of five dimensions); (ii) full recomputation under V12 is done; (iii) the
"beyond spectrum" rows 2, 3 and 6 of Table 2 are dropped (row 2a is additionally
unreproducible: its feature-producing script is absent from the repository).

Definition V12 (frozen): spherical relative vorticity ω = ∂v/∂x − ∂u/∂y + u tanφ/a;
points with ERA5 surface height > 1457 m are missing: NaN before every smoothing
(normalised convolution), excluded from the rank correlation; MERRA-2 uses its own
missing values. Everything else as in Phase 2 (ladder, σ = ℓ/√12, interior ring,
199 phase surrogates with the Phase-2/18 seed streams, medians over snapshots).

Script: `clean_experiments/experiment_B33_v12_recompute.py` (monkey-patches the
Phase 2/9/13/18 modules; outputs in `results/<experiment>_v12/`). Consolidation
by `experiment_B32_review_checks.py` with `--suffix _v12` into
`results/experiment_B32_review_checks/table2_v12.json`.

Quantities recomputed and reported "old → new" (steps 2–5 as the headline set,
steps 1–5 for reference):
- row 1: held-out signature (32 windows, raw P);
- row 4: MERRA-2 signature (48 windows, 3 steps) and ERA5–MERRA-2 geography;
- row 5: equal-km anchored signature (48 primary, 32 held-out) and deg–km geography;
- row 7: steps 4–5 only (held-out) — corrected definition (part 1 found the archived
  value belonged to steps 3–5);
- row 8: free run vs ERA5 (39 domains) with block-bootstrap interval; ERA5 across-period
  reference;
- new rows: stratified permutations (season / year / period) and ICC per step;
- descriptive: regional ordering on steps 4–5 and on steps 2–3 (anchored, km boxes),
  Congo–Amazon step-5 contrast (8 windows), extent sensitivity, dynamic-ladder arm.
No pass/fail criteria: the new numbers replace the old ones in the manuscript;
a qualitative change of any conclusion is reported explicitly.
