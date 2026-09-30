# Phase 31 — the boundary template of inter-level coupling: REPORT (2026-09-30)

Protocol: docs/PROTOCOL_PHASE31_BOUNDARY_TEMPLATE.md (frozen and committed,
e2711fe, before the 2024 fields were downloaded). Script:
clean_experiments/experiment_B31_boundary_template.py. ERA5 850 hPa,
902 tiles, 2023 and 2024, JFM + JAS, isotropic grid, no surrogates.

## VERDICT: PARTIAL(H31-1, H31-2) — ladder gap logged

The frozen ladder named three outcomes and did not name "H31-1 and H31-2
pass, H31-3 fails"; the result is reported as PARTIAL.

| Hypothesis | Result | Numbers |
|---|---|---|
| H31-1 template share is a year-to-year invariant (PRIMARY) | **PASS** | Spearman 2023 vs 2024: land 0.972 (bar 0.70), all tiles 0.842 (bar 0.60), rotation p = 0.001; medians: ocean 0.008 / 0.007, land 0.100 / 0.095 |
| H31-2 compensation | **PASS** | pair 50-100\|200-400, classes by the 2023 share, coupling from 2024: total 0.363 / 0.367 / 0.371 (low / mid / high; difference 0.007, bar 0.02); moving 0.362 / 0.356 / 0.307 (difference 0.055, bar 0.04); Spearman with the share over land: total +0.19 (bar +-0.2), moving -0.35 (bar -0.30). Adjacent pairs: same picture |
| H31-3 no geography without a boundary | **FAIL** | year-to-year Spearman of total coupling: ocean 0.43 (bar < 0.20), land 0.87 (bar > 0.40) |
| H31-4 one template for all levels | **PASS** | median template correlation over land: 0.91, 0.86 (adjacent), 0.65 (non-adjacent) |
| H31-5 not the below-ground extrapolation | **PASS** | 113 land tiles without any below-ground point: median share 0.043 = 5.9 x ocean (bar 3); Spearman with orography roughness +0.69 (bar +0.30) |

Priors: H31-1 0.85 (hit), H31-2 0.5 (hit), H31-3 0.8 (miss), H31-4 0.8 (hit),
H31-5 0.6 (hit).

## Reading

- The template share is reproduced between independent years at 0.97 over
  land. Together with the exploration (seasons 0.80, temperature 0.90,
  free-running model of another year 0.90) it is the programme's most stable
  territorial quantity, and it needs no surrogate.
- Compensation holds on an independent year: relief moves coupling from the
  moving form into the stationary one and leaves the total unchanged. The
  total-vs-share correlation (+0.19) sits at the edge of its bar.
- H31-3 failed: over open ocean the coupling map is reproducible between
  years. Post hoc (not scored): the same season of different years agrees
  (ocean 0.31 for JFM, 0.39 for JAS; land 0.80, 0.88), different seasons do not
  (ocean 0.0-0.17; land 0.41-0.42). The oceanic year-mean map is unrelated to
  latitude, SST gradient, CAPE and eddy energy (|rho| <= 0.13). The ocean has a
  seasonal geography that repeats every year, not a fixed one. "A constant of
  the medium" is wrong in the strict form; the oceanic spread is small
  (sd 0.027 against 0.096 over land).
