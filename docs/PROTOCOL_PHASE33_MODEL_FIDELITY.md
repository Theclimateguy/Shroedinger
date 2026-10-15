# Protocol — Phase 33: inheritance fidelity of forecast systems (B33)

Frozen: 2026-10-10, before any Phase-33 quantity was computed and before any
Arm-B field was retrieved. Trigger: the author's decision (2026-10-10) to develop
P as a *verification metric for atmospheric models*, by analogy with the spectral
blurring metrics of learned forecast systems. Regionalisation stays with the three
papers and is not part of this phase.

Script: `clean_experiments/experiment_B33_model_fidelity.py`.
Results: `clean_experiments/results/experiment_B33_model_fidelity/{armA,armB}/`.
Seed: 20261010. Surrogates: 99 per field set (programme standard).
Arm-B downloader: `clean_experiments/download_b33_wb2_fresh.py` (wraps the frozen
Phase-11 downloader with new windows; writes `data/b33wb2/`).

Note on numbering: `experiment_B33_v12_recompute.py` belongs to Phase 32 (part 2).
This phase is the first to carry the name Phase 33.

## What is known before this phase (and therefore not a hypothesis here)

- Phase 11 (frozen, NEGATIVE for `A`): conditional-mean systems blur the mesoscale
  at 120 h (band-variance deficit GraphCast +0.69, Pangu +0.55, GenCast mean +0.85),
  a GenCast member and HRES do not (+0.02). The mechanism ladder works; the
  spectral metric separates the systems. P must therefore show something the
  spectrum does not, or it is not a metric.
- Exploration E1 (2026-09-24, not frozen, code and results deleted at the author's
  request, numbers kept in memory only): on the same 2020 JFM/JAS data the anchored
  P of conditional-mean systems at 120 h exceeded the ERA5 value (GenCast mean
  ×1.13, GraphCast ×1.10, Pangu ×1.08, p < 1e-5), a GenCast member ≈ 1, HRES ×0.95.
  These numbers set the thresholds below; they are *not* evidence for this phase.
- The P estimator is, by construction, invariant to a uniform rescaling of a band
  (rank correlation of log envelopes): a spatially uniform loss of band variance
  does not change P. What can change P is a *selective* loss — fine-scale variance
  removed where the coarse envelope is weak and kept where it is strong, which is
  what a conditional mean is expected to do (unpredictable fine structure averages
  out, fine structure slaved to the resolved coarse field survives).

## The claim under test

**Inheritance fidelity.** For a forecast system `s`, lead `τ`, domain `d`:

    D_s(d, τ) = P̂_s(d, τ) − P̂_truth(d, τ)

where `P̂` is the anchored V12 profile (spherical vorticity, 850 hPa below-surface
mask, normalised convolution; `P_real − median of 99 phase-randomised surrogates`),
computed on the stack of the system's forecast fields at the given lead over all
initialisations of the arm (median over fields), and `P̂_truth` is the same on the
ERA5 fields at the matching valid times. Headline scalar: mean of steps 3–5
(pairs 100–200|200–400, 200–400|400–800, 400–800|800–1600 km); steps 2–5 and the
per-step profile are reported. Step 1 (<50|50–100 km) is diagnostic only
(Phase 32, grid artefact).

Prediction, fixed here: **D > 0 for conditional-mean systems (GraphCast, Pangu,
GenCast ensemble mean) at 120 h; D ≈ 0 for a GenCast member and for HRES; and a
uniform spectral matching of the truth field does not reproduce D.**

## Data (fixed)

Same stores, variables, level, grid, domains, leads and initialisation rule as
Phase 11 (`download_b11_wb2.py`): 850 hPa u, v on 0.25°, 39 domains (12 regional
+ 27 lattice, 20° × 40°), leads 24 h and 120 h, 00 UTC every 10th day.

- **Arm A (replication of E1, hypothesis-generating; not scored):** on disk,
  `data/b11wb2/`, windows 2020 JFM and JAS, 10 initialisations each; GenCast member
  6 per window. Purpose: validate the pipeline and reproduce the E1 numbers under
  V12. Nothing in Arm A can confirm or refute the claim.
- **Arm B (confirmatory, fresh):** 2020 AMJ (2020-04-01..06-30) and OND
  (2020-10-01..12-31), same stride → 10 + 10 initialisations; same five systems.
  GenCast member: optional, 2 initialisations per window (its chunks carry 56
  members, ≈0.55 GB each); if not retrieved, the member control is reported from
  Arm A only and H33-2 is scored on HRES alone. All scoring below is on Arm B.

## Quantities (fixed)

- Per (system, lead, domain): `P_real`, `P_surrogate_median`, `P_anchored` (5
  steps) for the model fields and for the paired truth fields (same initialisations).
- Band-component variances of the vorticity field for both, on the same Gaussian-bar
  ladder as P: `c_0 = ω − bar_50`, `c_k = bar_{k−1} − bar_k` (k = 1..5, bands
  50–100 … 800–1600 km), `c_6 = bar_1600`; mean over the correlation mask and over
  fields. Deficit `Δ_k = 1 − var_model(c_k)/var_truth(c_k)` (the Phase-11 quantity on
  this ladder).
- **S1 — uniform spectral match.** The truth u, v fields are decomposed on the same
  ladder and each component is multiplied by a constant gain so that the component
  variances of the resulting vorticity equal the model's (three fixed-point
  iterations; the achieved mismatch `max_k |log(var_S1/var_model)|` over k = 1..5 is
  recorded). S1 has the model's spectrum and the truth's organisation.
  `D_S1 = P̂_S1 − P̂_truth`, same surrogate anchoring.
- **S2 — selective damping, same spectrum.** As S1, but before the gain fit every
  component `c_k` (k = 0..4) is multiplied by `exp(z_{k+1}(x))`, where `z_{k+1}` is the
  standardised log envelope of the next-coarser component (smoothed at its own
  scale, as in the estimator), with κ = 1 fixed. S2 has the model's spectrum and an
  organisation that is deliberately more "inherited" than the truth's. `D_S2`
  likewise. S2 is a positive control for sensitivity; it does not test the claim.
- Sign-flip test: for a vector of 39 domain values, 9999 random sign flips of the
  domain values, one-sided p for the observed mean. Domains overlap in their
  margins and are not independent; the test is the Phase-11 one and is read as
  such.

Thresholds, fixed from E1 (anchored units; E1's ×1.08–1.13 at P̂ ≈ 0.2–0.3 is
≈ +0.02–0.03): **δ = +0.010**.

## Sanity checks (pipeline, not verdict)

- **C33-1.** Deficit `Δ` on components 3–4 (200–800 km) at 120 h positive in ≥ 90 %
  of domains for GraphCast and Pangu; median |Δ| for HRES smaller than GraphCast's
  (Phase 11 reproduced on this ladder).
- **C33-2.** Truth reliability: Spearman between `P̂_truth` (steps 3–5) on the
  24 h-valid and the 120 h-valid field sets across 39 domains ≥ 0.70.
- **C33-3.** S1/S2 construction: median over domains of the achieved spectral
  mismatch ≤ 0.15 (log units) for components 1–5.
- **C33-4 (Arm A only).** Arm A reproduces the sign of E1 for the three
  conditional-mean systems at 120 h (median D > 0). Failure → pipeline review
  before Arm B; the hypotheses are not changed.

## Hypotheses and criteria (Arm B, fixed)

- **H33-1 (over-inheritance).** For each of GraphCast, Pangu, GenCast mean at
  120 h: median D (steps 3–5) ≥ δ and sign-flip p ≤ 0.01.
- **H33-2 (discriminators).** HRES at 120 h does *not* satisfy the H33-1 criterion
  (median D < δ or p > 0.01). If Arm-B member data exist: the GenCast member does not
  satisfy it either. (Arm-A member reported descriptively otherwise.)
- **H33-3 (not the spectrum, by construction).** For each conditional-mean system
  at 120 h: median D_S1 < δ/2 **and** median D_S1 < 0.5 · median D_model. Reported:
  fraction `f_S1 = median D_S1 / median D_model`.
- **H33-4 (sensitivity, positive control).** D_S2 > 0 with sign-flip p ≤ 0.01 for
  the three constructions.
- **H33-5 (lead).** Descriptive: D at 24 h vs 120 h, paired over domains; sign
  reported; no criterion.
- **H33-6 (geography of the metric).** Descriptive: within each conditional-mean
  system, Spearman(D, mean Δ over components 3–4) and LOSO R² of D on the deficit
  vector (components 1–5). Reported; no criterion. (Whether the *geography* of D
  adds to the geography of blurring is a different question from whether the
  metric is non-spectral, which H33-3 settles.)

## Verdict rule (fixed)

- **CONFIRMED**: C33-1..3 pass; H33-1 for all three systems; H33-2 for HRES
  (and the member if present); H33-3 for all three; H33-4.
- **PARTIAL**: C33-1..3 pass; H33-1 and H33-3 for at least two of the three
  systems; H33-2 for HRES.
- **NEGATIVE**: H33-1 fails for two or more systems, **or** H33-3 fails for two or
  more (uniform spectral matching reproduces the effect — the metric restates the
  spectrum), **or** HRES satisfies H33-1 (the effect is not tied to the
  conditional-mean mechanism).

On NEGATIVE: P is not pursued as a forecast-verification metric; the climatological
use (free-running model map vs reanalysis, Phases 13/22, V12 ρ = 0.72) stays as it is.

## What a CONFIRMED verdict licenses, and what it does not

Licensed: "inheritance fidelity" as a diagnostic that separates uniform from
selective loss of mesoscale variance in a forecast system, non-spectral by
construction (S1) and sensitive (S2). Not licensed: any statement about forecast
skill, about which system is "better", or about the atmosphere; and no claim that
the geography of D is informative unless H33-6 says so descriptively and is then
tested in a later phase on fresh data.

## Compute plan

- `python clean_experiments/experiment_B33_model_fidelity.py compute --arm A --workers 10`
- `python clean_experiments/download_b33_wb2_fresh.py --system {era5,graphcast,pangu,gencast_mean,hres}`
  (and `gencast_member` only on the author's decision)
- `python clean_experiments/experiment_B33_model_fidelity.py compute --arm B --workers 10`
- `python clean_experiments/experiment_B33_model_fidelity.py tests --arm B` → `summary.json`, `report.md`.

## Deviations

- **D1 (2026-10-10, before the Arm-A run, after a 5-surrogate smoke test on
  R2_NATL).** The S1/S2 gain fit cannot match component 1 (50–100 km): with the
  model's 85 % deficit there the gain goes to zero and a leakage floor from the
  neighbouring components remains (achieved/target ≈ 1.18 regardless of
  iterations), while components 2–6 match within 3 %. C33-3 is therefore scored on
  components 2–5 (the bands entering steps 3–5); the achieved ratio for every
  component is stored (`comp_var_ratio_to_model`) and reported. Fixed-point
  iterations 3 → 6. No hypothesis or threshold changed.
- **D2 (2026-10-14, author's decision, before any Arm-B quantity was computed;
  Arm-B data partly downloaded, not opened).** Arm-B initialisation stride 10 → 5
  days (2020 AMJ: 19 initialisations, OND: 19), to raise the reliability of
  per-domain values (C33-2 failed on Arm A with 20 fields, ρ = 0.60). Arm A is
  unchanged (stride 10, on disk).
- **D3 (2026-10-14, author's decision, same point in time).** H33-3 uses the
  relative criterion only: median D_S1 < 0.5 · median D_model. The absolute clause
  (median D_S1 < δ/2) is dropped: δ was set for an expected D of 0.02–0.03, while
  Arm A gave D ≈ 0.05, so the absolute clause tested the noise floor of S1, not
  its share of the effect. Verdict rule otherwise unchanged.
- **D4 (2026-10-14, author's decision).** The GenCast single member is retrieved
  for Arm B, 6 initialisations per window (as in Arm A), the first six of each
  window on the 5-day stride. H33-2 is scored on HRES and the member.
- C33-3 stays as fixed in D1 (components 2–5); the author did not change it.

- **D5 (2026-10-14, author's decision; POST HOC with respect to the data — read
  this entry before any Phase-33 number).** The scale range of the metric is
  restricted to bands not finer than 200 km: the headline D is the mean of steps
  4–5 (200–400|400–800 and 400–800|800–1600 km); C33-3 is scored on components
  200–400, 400–800 and 800–1600 km. Steps 3–5 (the frozen headline) and steps
  2–5 are still computed and reported.
  *Rationale (a priori, documented before this phase).* Bands below 200 km lie at
  or below the effective resolution of every reanalysis available to the
  programme: ERA5 4–7 grid steps ≈ 125–220 km (Skamarock 2004), 500–600 km in
  midlatitudes and ~1000 km in the tropics by direct spectral estimate (Bolgiani
  et al. 2022); MERRA-2 is coarser still. This was recorded in
  `docs/RECONCILIATION.md` on 2026-08-13 and article 1 v8 already calls steps 4–5
  "resolved". The 50–200 km bands entered the programme by inheritance, not by
  choice: the octave edges 50…3200 km come from the early A15 / Lambda code
  (`experiment_scale_gravity_einstein_box_era.py`, 2026-03), were copied into
  Phase 1 "as in A15" and into Phase 2 for comparability with Phase 1, and were
  never re-justified.
  *What had been seen when D5 was decided.* The full Arm-A results and Arm-B
  results for Pangu, GenCast mean and HRES (per-step D, S1, S2, C33-3). Seen in
  particular: (i) on both arms the largest D sits in step 3 (100–200|200–400 km)
  and step 4, step 5 is small; (ii) C33-3 fails for Pangu because components
  50–100 and 100–200 km cannot be matched — under D5 this failure drops out of
  scope by construction. Point (ii) is a convenience of D5 and must be stated
  wherever the verdict is reported. GraphCast and GenCast-member Arm-B results had
  not been computed.
  *Consequence for the verdict.* The verdict is computed under D5; the verdict
  under the frozen definition (steps 3–5, C33-3 on components 100–1600 km) is
  reported alongside it. If the two differ, both are stated and the difference is
  attributed to D5.

## Arm C — confirmatory replication on 2022 (addendum)

Frozen: 2026-10-15, before any Arm-C field was retrieved. Trigger: Arm B ended
with every hypothesis passing but a formal verdict UNDETERMINED (one pipeline
check failed under each definition, and the scale range D5 was chosen post hoc).
Arm C settles the verdict on data nobody has seen, with every rule fixed here.

**Definition (fixed, = D5 + D3).** Headline D = mean of steps 4–5 (bands
200–400|400–800 and 400–800|800–1600 km), anchored V12, 99 phase-randomised
surrogates; δ = +0.010; H33-3 relative only (median D_S1 < 0.5 · median D_model);
C33-3 on components 200–400, 400–800, 800–1600 km, threshold 0.15. Steps 3–5 and
2–5 are reported, not scored.

**Data (fixed).** WeatherBench 2, 850 hPa u, v, 0.25°, the 39 programme domains,
leads 24 h and 120 h, 00 UTC, year 2022, four windows W17_2022JFM, W18_2022AMJ,
W19_2022JAS, W20_2022OND, stride 5 days from each window start (75
initialisations). Truth: `era5/1959-2023_01_10-wb13-6h-1440x721.zarr`.

| role | system | store |
|---|---|---|
| conditional mean, learned (scored) | GraphCast | `graphcast_v2/2022-1440x721.zarr` |
| conditional mean, learned (scored) | Pangu (ERA5 init) | `pangu/2018-2022_0012_0p25.zarr` |
| conditional mean, physical (scored) | IFS ENS mean (50 members) | `ifs_ens/2018-2022-1440x721_mean.zarr` |
| negative control | HRES | `hres/2016-2022-0012-1440x721.zarr` |
| negative control | IFS ENS member (number = 1) | `ifs_ens/2018-2022-1440x721.zarr`, first 6 inits per window |
| exploratory, not scored | Aurora | `aurora/2022-1440x721.zarr` |

GenCast is absent (WeatherBench 2 holds it for 2020 only). Its place as the
"mean of a generative ensemble" is taken by the IFS ENS mean — an ensemble mean of
a *physical* model. This makes Arm C a mechanism test in its own right: the claim
is about conditional means, not about neural networks. Prediction, fixed here:
**D > 0 for the IFS ENS mean at 120 h**, as for the learned systems. Aurora is a
deterministic learned model trained with a point loss and fine-tuned with
rollouts; predicted D > 0, reported whatever the outcome, not part of the verdict.

**Checks, hypotheses, verdict.** As in the main protocol under D5 and D3, with the
scored set {GraphCast, Pangu, IFS ENS mean} in place of {GraphCast, Pangu,
GenCast mean} and the controls {HRES, IFS ENS member}. C33-1 on GraphCast and
Pangu; C33-2 on the truth fields of the GraphCast pairing (unchanged); C33-4 does
not apply. CONFIRMED / PARTIAL / NEGATIVE exactly as in "Verdict rule". Arm C is
the scored arm of Phase 33; Arms A and B are reported as hypothesis-generating
(A) and post-hoc-defined replication (B).

**Code.** Downloader `clean_experiments/download_b33c_wb2_2022.py` (generalised
chunk indexing: Aurora stores 5 leads per chunk, IFS ENS 50 members per chunk);
data `data/b33c_wb2/`; experiment `experiment_B33_model_fidelity.py --arm C`.

### Arm C deviations

- (none yet)

### Arm C result (2026-10-15)

Computed on Spark, 13:52; results `clean_experiments/results/experiment_B33_model_fidelity/armC/`.
All 468 field sets complete (75 initialisations; Aurora 74 — 2022-01-01 00 UTC is
not in its store; IFS ENS member 24). No deviations.

**Verdict (Arm C definition, frozen before data): CONFIRMED.** C33-1, C33-2
(ρ = 0.736), C33-3 (≤ 0.08) pass; H33-1..4 pass for every scored system.

| system | role | ratio P̂ model / truth | median D (steps 4–5) | frac D > 0 | p | D_S1 / D |
|---|---|---|---|---|---|---|
| GraphCast | CM, learned | 1.28 | +0.045 | 0.79 | 1e-4 | 0.03 |
| Pangu | CM, learned | 1.27 | +0.056 | 0.85 | 1e-4 | −0.03 |
| IFS ENS mean | CM, physical | 1.24 | +0.041 | 0.82 | 1e-4 | 0.10 |
| HRES | control | 1.02 | +0.001 | 0.51 | 0.60 | — |
| IFS ENS member | control | 0.97 | −0.000 | 0.49 | 0.50 | — |
| Aurora | exploratory | 1.35 | +0.076 | 0.92 | 1e-4 | 0.09 |

The IFS ENS mean behaves as the learned systems: over-inheritance is a property of
conditional means, not of neural networks. D is near zero at 24 h and full at 120 h.
Under the frozen definition (steps 3–5) every hypothesis also passes but C33-3
fails for Pangu on components 100–400 km (0.33/0.24), verdict UNDETERMINED, as in
Arm B. H33-6: the geography of D is not consistent across systems (LOSO R² −0.27…
+0.25); no geographic claim is made.

**Phase 33 conclusion.** Inheritance fidelity D separates conditional-mean
forecasts from single realisations (physical deterministic run, ensemble member)
on a fresh year, with a non-spectral construction control and a sensitivity
control. Scored arm: C. Arm A hypothesis-generating, Arm B replication with a
post-hoc scale range.
