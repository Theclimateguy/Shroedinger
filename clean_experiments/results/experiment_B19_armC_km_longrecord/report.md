# Experiment B19 Arm C: the long-record questions on the km carrier

Frozen spec: `docs/PROTOCOL_PHASE19_KM_DYNAMICS.md`, "Arm C execution
spec" (2026-08-18, frozen with data/b17daily complete at 552/552 but
before any Arm-C computation). Code
`clean_experiments/experiment_B19_armC_km_longrecord.py`; 552
region-years km-cropped, monthly P^eq / V / E_syn (daily-00Z synoptic
variance proxy); B17 statistics rerun verbatim on the km carrier.

## ARM_C_VERDICT: (a_ENSO_STAYS_NULL, a_EXCESS_CARRIER_ROBUST)

### H-C1: the ENSO negative is not a dilution artefact — with one honest nuance

Pooled S on km = 0.030, two-sided p = 0.077 vs the circular-shift null:
NOT significant, leg (a) as frozen. Nuance reported in full: the km
carrier TRIPLES the pooled statistic relative to degrees (0.009 -> 0.030,
p 0.71 -> 0.077) — the dilution direction the author hypothesized gets
directional support but does not reach the bar, and the pooled value is
carried almost entirely by one region: R5_SPCZ rho = +0.316 (next
largest |rho| = 0.13). This sharpens the Phase-17 note: SPCZ remains the
single preregisterable ENSO target; any SPCZ claim requires its own
frozen protocol on data not consulted here (this analysis has now
consulted 1979-2016 km SPCZ and cannot score it).

### H-C2: the tropical interannual excess is real on both carriers

All 3/3 degree-box excess regions retain F > null q95 on km:
R1_WPWP 1.75 -> 1.79, R3_AMAZ 2.24 -> 1.50, R5_SPCZ 2.07 -> 1.76
(q95 ~ 1.43). No new excess regions appear. The excess is
carrier-robust: physical, non-ENSO, still unexplained. Together with
Phase 18 this closes the geometry alternative for both long-record
findings.

### Secondary (emergence-order probe): contemporaneous, no lead

Pooled lead-lag S between monthly P anomalies and E_syn anomalies:
sharp peak at lag 0 (S = +0.165, null q95 = 0.040), near-symmetric decay,
no month-scale lead of either variable (lag +1: +0.050, lag -1: +0.007 —
both an order below the peak). Reading, in the organization vocabulary
the author fixed: synoptic activity and cross-band coupling covary as
facets of one organization at monthly resolution; neither precedes the
other at the month scale — consistent with the ~10 h relaxation clock of
Phase 19 (no memory to carry a lead) and with the co-emergence framing
over any "centers drive P" reading.

### Figure

- `fig1_armC.png` — lead-lag curve; F-ratios degree vs km; per-region
  ENSO rho with the SPCZ outlier.
