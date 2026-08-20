#!/usr/bin/env python3
"""Reproduce paper 1 of the series end to end.

Paper: "Coupling of hierarchical levels of the atmospheric circulation as a
regional invariant" (manuscript_v2/article1_v5.tex). Route and per-result
mapping: reproducibility/article1_README.md.

Stages
------
  download   fetch the primary fields (ERA5 / MERRA-2 / HighResMIP / GEFS / WB2)
  compute    run the ten experiments behind the paper's tables and figures
  check      verify that every expected artifact exists; print SHA-256
  all        download + compute + check (default)

Examples
--------
  python reproducibility/reproduce_article1.py --stage check
  python reproducibility/reproduce_article1.py --stage download --only era5
  python reproducibility/reproduce_article1.py --stage compute --only B18

Credentials: ERA5 needs ~/.cdsapirc (CDS API); MERRA-2 needs NASA Earthdata
in ~/.netrc. Downloads are resumable; experiments cache per region-window.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CE = REPO / "clean_experiments"

# name, script, needs-credential, note
DOWNLOADS = [
    ("era5-b1", "download_b1_era5_wind.py", "cds",
     "ERA5 850 hPa wind: R1-R2, windows W1-W4 (2017-2019)"),
    ("era5-b2", "download_b2_era5_wind.py", "cds",
     "ERA5: R3-R4, windows W1-W4 (2017-2019)"),
    ("era5-b2b", "download_b2b_era5_wind.py", "cds",
     "ERA5: R5-R12, windows W5-W8 (2021-2022, held-out set)"),
    ("era5-b3", "download_b3_era5_wind.py", "cds",
     "ERA5: all 12 regions, windows W9-W12 (2023-2024, primary set)"),
    ("era5-b10", "download_b10_era5.py", "cds",
     "ERA5: 27 lattice control domains (2024)"),
    ("merra2-b8", "download_b8_merra2.py", "netrc",
     "MERRA-2 850 hPa wind: 12 regions, W9-W12"),
    ("highresmip-b13", "download_b13_highresmip.py", None,
     "ECMWF-IFS-HR highresSST-present, global 850 hPa wind, JFM+JAS 2014"),
    ("gefs-b9", "download_b9_gefs.py", None,
     "NOAA GEFS ensemble (applied test: forecast error growth)"),
    ("wb2-b11", "download_b11_wb2.py", None,
     "WeatherBench 2 forecasts (applied test: ML mesoscale deficit)"),
]

# name, script, produces (relative to clean_experiments/results/)
EXPERIMENTS = [
    ("B2", "experiment_B2_scale_irreversibility.py",
     "experiment_B2_scale_irreversibility/summary.json"),
    ("B2b", "experiment_B2b_heldout_invariants.py",
     "experiment_B2b_heldout_invariants/summary.json"),
    ("B3", "experiment_B3_scattering_benchmark.py",
     "experiment_B3_scattering_benchmark/summary.json"),
    ("B8", "experiment_B8_cross_reanalysis.py",
     "experiment_B8_cross_reanalysis/summary.json"),
    ("B9", "experiment_B9_predictability.py",
     "experiment_B9_predictability/summary.json"),
    ("B10", "experiment_B10_irrecoverability.py",
     "experiment_B10_irrecoverability/summary.json"),
    ("B11", "experiment_B11_learned_refinement.py",
     "experiment_B11_learned_refinement/summary.json"),
    ("B12", "experiment_B12_estimator_validity.py",
     "experiment_B12_estimator_validity/summary.json"),
    ("B13", "experiment_B13_free_running.py",
     "experiment_B13_free_running/summary.json"),
    ("B18", "experiment_B18_equal_km_regions.py",
     "experiment_B18_equal_km_regions/summary.json"),
]

EXPECTED = [e[2] for e in EXPERIMENTS] + [
    "experiment_B3_scattering_benchmark/e1_semifractal.json",
    "experiment_B18_equal_km_regions/fig2_anchored_profiles.png",
]


def run(script: str) -> int:
    path = CE / script
    print(f"\n=== {script} ===", flush=True)
    return subprocess.run([sys.executable, str(path)], cwd=REPO).returncode


def preflight(needed: set[str]) -> None:
    home = Path.home()
    if "cds" in needed and not (home / ".cdsapirc").exists():
        print("WARNING: ~/.cdsapirc not found - ERA5 downloads will fail "
              "(register at cds.climate.copernicus.eu)")
    if "netrc" in needed and not (home / ".netrc").exists():
        print("WARNING: ~/.netrc not found - MERRA-2 downloads will fail "
              "(register at urs.earthdata.nasa.gov)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["download", "compute", "check", "all"],
                    default="all")
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on step names (e.g. 'era5', 'B18')")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing step")
    args = ap.parse_args()

    failed: list[str] = []

    if args.stage in ("download", "all"):
        steps = [d for d in DOWNLOADS
                 if not args.only or args.only.lower() in d[0].lower()]
        preflight({d[2] for d in steps if d[2]})
        for name, script, _, note in steps:
            print(f"\n[{name}] {note}")
            if run(script) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at download step {name}")

    if args.stage in ("compute", "all"):
        for name, script, _ in EXPERIMENTS:
            if args.only and args.only.lower() not in name.lower():
                continue
            if run(script) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at experiment {name}")

    if args.stage in ("check", "all"):
        print("\n=== artifact check ===")
        missing = 0
        for rel in EXPECTED:
            p = CE / "results" / rel
            if p.exists():
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                print(f"  OK  {rel}  sha256={digest}")
            else:
                print(f"  MISSING  {rel}")
                missing += 1
        print(f"\n{len(EXPECTED) - missing}/{len(EXPECTED)} artifacts present."
              " Reference hashes: reproducibility/article1_README.md, sect. 5.")

    if failed:
        sys.exit("Failed steps: " + ", ".join(failed))


if __name__ == "__main__":
    main()
