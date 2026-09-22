# Phase 29 — the structure law: replication, model, amplitude vs intermittency: REPORT (2026-09-22)

Protocol: docs/PROTOCOL_PHASE29_STRUCTURE_LAW.md (frozen; SHA-256 in summary.json; deviation D1 logged).
Script: clean_experiments/experiment_B29_structure_law.py. New computation: 24 IFS-HR free-run units.

## Frozen VERDICT: PARTIAL(H29-1, H29-3). Corrected reading (D1): STRUCTURE_LAW_ROBUST + AMPLITUDE_NEW_AXIS (weak)

| Hypothesis | Result | Numbers |
|---|---|---|
| H29-1 structure law on 902 tiles | **PASS** | same-coarse difference / same-separation difference = 0.052 [0.048, 0.057]; d > 0 in 100 % of tiles; ratio uncorrelated with CAPE (−0.005) and EKE (+0.02) |
| H29-2 structure law in the free-running model | **PASS with deviation D1** | dR2 region median 0.75 (12/12 in 0.62..0.78), p < 1e-4; model A vs ERA5 A rho = 0.93 over 12 regions; frozen control C29-1 rho 0.64 (mismatched definitions); like-for-like 0.97 / 0.94 |
| H29-3 amplitude A beyond covariates + intermittency block | **PASS (lower rung)** | inc_A = +0.047, p_rot = 0.001 (null q95 0.001); ladder +0.03/p<0.05 met, +0.05 missed by 0.003. Reverse: block over (cov + A) = +0.014. A alone 0.305, block alone 0.503, cov + A 0.793, cov + block 0.761; rho(A, INT_sig2) = 0.93 |
| H29-4 A at least as reliable as P | **FAIL** | split-half SB: A 0.79 vs P 0.87; cross-season: A 0.53 vs P 0.71; A units-vs-tiles 0.60 |
| H29-5 kappa (reported) | — | region −0.9..−1.9, model median −1.1; 10/48 ERA5 units undefined (C_05 <= 0); cross-season rho 0.36; model-vs-ERA5 0.42 — not an invariant |

A(x0.7)/A(x1.0) = 0.80: amplitude magnitude is domain-dependent; ranking is not.
Priors: H29-1 0.8 (hit), H29-2 0.7 (hit, after D1), H29-3 0.35 (miss — passed), H29-4 0.6 (miss).

## Reading

- The structure law (coupling to a coarser band depends on that coarser band only, not on the
  finer band) is global (100 % of tiles), region-independent (ratio uncorrelated with the two
  organisers), box-independent (Phase 28) and holds in a model with no data assimilation. It is a
  property of the dynamics.
- Its amplitude carries P's geography and is reproduced tile-free in the model (0.93). A is not a
  new axis orthogonal to intermittency: it is a one-number version of the intermittency block that
  slightly outperforms it (+0.047 over; block adds +0.014 back). Same-data caveat applies to both.
- P remains the estimator for the map (more reliable than A). A is the physical reading of P.
- kappa (decay exponent) and L (depth, Phase 28) are not reportable invariants.
