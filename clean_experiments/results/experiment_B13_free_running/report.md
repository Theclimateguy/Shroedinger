# Phase 13: the free-running, no-assimilation control

Protocol: `docs/PROTOCOL_PHASE13_FREE_RUNNING.md` (frozen 2026-08-16).
Verdict: **ATMOSPHERIC**  |  ECMWF-IFS-HR HighResMIP `highresSST-present`, prescribed SST, no data assimilation of any kind; 39 domains, JFM+JAS 2014, 0.5 deg common grid.

## Sanity (C13-1)

- time steps per domain-window: model 360, ERA5 360 (threshold 300)
- median band-variance ratio model/ERA5: 1.48 (guard: 1/3 to 3)
- pass = True

## Do the geographies survive without assimilation?

| quantity | free-running vs ERA5 | ERA5-to-ERA5 ceiling | fraction of ceiling |
|---|---|---|---|
| coupling profile P | **+0.706** (p = 0.001) | +0.929 (n = 8) | 0.76 |
| index A | **+0.902** (p = 0.001) | +0.905 (n = 8) | 1.00 |
| irrecoverability u | **+0.771** (p = 0.001) | +0.810 (n = 8) | 0.95 |
| envelope persistence | **+0.979** (p = 0.001) | +0.976 (n = 8) | 1.00 |

All four replicate at p = 0.001 in an atmosphere that has never ingested an
observation. The limitation the manuscript declares decisive — that ERA5 and
MERRA-2 share an observing network, so their agreement cannot separate the
atmosphere from the observing system — is **closed**. These are properties of
the atmosphere.

## But read it together with Phase 12

The ordering is informative. Persistence and `A` replicate essentially at their
own ceiling (1.00 and 1.00); `u` at 0.95; `P` at 0.76. That is exactly what
Phase 12 predicts: `A` is largely a persistence statistic, and a free-running
model reproduces the geography of persistence almost perfectly.

So Phase 13 does **not** rehabilitate `A`. It establishes that `A` is a
reproducible atmospheric invariant — of temporal persistence, not of
cross-level irrecoverability. And it establishes the same, independently, for
`P`, which Phase 12 validated as a faithful measure of addressability.

## Domain table

| domain | P ERA5 | P model | A ERA5 | A model | persist ERA5 | persist model |
|---|---|---|---|---|---|---|
| G01_55N100E | 0.600 | 0.596 | 1.45 | 1.38 | 0.600 | 0.590 |
| G02_55N140E | 0.722 | 0.721 | 1.73 | 1.59 | 0.712 | 0.707 |
| G03_55N140W | 0.698 | 0.661 | 1.54 | 1.48 | 0.641 | 0.628 |
| G04_55N100W | 0.571 | 0.625 | 1.54 | 1.46 | 0.608 | 0.612 |
| G05_35N20E | 0.599 | 0.588 | 1.60 | 1.46 | 0.485 | 0.480 |
| G06_35N60E | 0.620 | 0.626 | 1.85 | 1.68 | 0.716 | 0.753 |
| G07_35N100E | 0.619 | 0.635 | 1.87 | 1.88 | 0.795 | 0.818 |
| G08_35N140E | 0.613 | 0.628 | 2.36 | 1.97 | 0.783 | 0.810 |
| G09_35N180W | 0.558 | 0.524 | 2.10 | 2.05 | 0.782 | 0.818 |
| G10_35N140W | 0.614 | 0.539 | 1.84 | 1.72 | 0.776 | 0.803 |
| G11_35N100W | 0.527 | 0.505 | 2.08 | 1.98 | 0.777 | 0.758 |
| G12_35N60W | 0.554 | 0.549 | 2.09 | 1.72 | 0.786 | 0.795 |
| G13_35N20W | 0.560 | 0.545 | 1.73 | 1.48 | 0.486 | 0.526 |
| G14_15N100E | 0.506 | 0.554 | 2.31 | 2.47 | 0.816 | 0.823 |
| G15_15N140W | 0.700 | 0.621 | 1.95 | 1.83 | 0.797 | 0.828 |
| G16_15N100W | 0.644 | 0.622 | 1.68 | 1.68 | 0.773 | 0.771 |
| G17_15N20W | 0.593 | 0.637 | 1.80 | 1.41 | 0.559 | 0.595 |
| G18_5S140E | 0.631 | 0.629 | 1.73 | 1.66 | 0.794 | 0.820 |
| G19_5S140W | 0.582 | 0.520 | 2.02 | 1.77 | 0.756 | 0.794 |
| G20_5S100W | 0.632 | 0.606 | 1.87 | 1.46 | 0.789 | 0.806 |
| G21_5S20W | 0.607 | 0.547 | 1.65 | 1.48 | 0.750 | 0.777 |
| G22_25S20E | 0.625 | 0.606 | 2.13 | 1.94 | 0.696 | 0.705 |
| G23_25S60E | 0.621 | 0.549 | 2.26 | 2.10 | 0.709 | 0.706 |
| G24_25S140E | 0.649 | 0.602 | 2.06 | 1.71 | 0.697 | 0.714 |
| G25_25S180W | 0.637 | 0.685 | 1.84 | 1.62 | 0.768 | 0.763 |
| G26_25S140W | 0.670 | 0.629 | 1.80 | 1.69 | 0.748 | 0.748 |
| G27_25S20W | 0.646 | 0.620 | 2.19 | 1.92 | 0.697 | 0.743 |
| R10_INDO | 0.611 | 0.634 | 2.49 | 2.51 | 0.824 | 0.835 |
| R11_EURO | 0.556 | 0.576 | 1.48 | 1.45 | 0.721 | 0.720 |
| R12_SAM | 0.614 | 0.616 | 1.76 | 1.53 | 0.627 | 0.669 |
| R1_WPWP | 0.731 | 0.741 | 2.41 | 2.48 | 0.865 | 0.860 |
| R2_NATL | 0.718 | 0.653 | 1.84 | 1.66 | 0.707 | 0.692 |
| R3_AMAZ | 0.550 | 0.520 | 1.69 | 1.49 | 0.504 | 0.572 |
| R4_CASIA | 0.572 | 0.565 | 1.46 | 1.37 | 0.667 | 0.661 |
| R5_SPCZ | 0.574 | 0.663 | 2.35 | 2.21 | 0.808 | 0.840 |
| R6_SATL | 0.683 | 0.656 | 1.73 | 1.55 | 0.642 | 0.678 |
| R7_CONGO | 0.701 | 0.697 | 1.70 | 1.58 | 0.331 | 0.361 |
| R8_AUS | 0.589 | 0.565 | 1.95 | 1.91 | 0.505 | 0.512 |
| R9_NPAC | 0.665 | 0.652 | 1.89 | 1.61 | 0.692 | 0.725 |
