#!/usr/bin/env python3
"""Phase-25 data: ERA5 monthly-mean hydroclimatic fields, global 0.5 deg,
2021-2024 -> data/b25hydro/.

Protocol: docs/PROTOCOL_PHASE25_RESIDUAL_CANDIDATES.md (frozen before
download). Resumable: existing target files are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

YEARS = ["2021", "2022", "2023", "2024"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
GRID = [0.5, 0.5]


def fetch_single(c: cdsapi.Client, out: Path) -> None:
    target = out / "era5_monthly_single_hydro.nc"
    if target.exists():
        print(f"skip {target.name}")
        return
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["total_precipitation", "convective_precipitation",
                         "total_column_water_vapour"],
            "year": YEARS,
            "month": MONTHS,
            "time": "00:00",
            "grid": GRID,
            "format": "netcdf",
        },
        str(target))
    print(f"done {target.name}")


def fetch_pressure(c: cdsapi.Client, out: Path) -> None:
    target = out / "era5_monthly_wind_850_500.nc"
    if target.exists():
        print(f"skip {target.name}")
        return
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": ["500", "850"],
            "year": YEARS,
            "month": MONTHS,
            "time": "00:00",
            "grid": GRID,
            "format": "netcdf",
        },
        str(target))
    print(f"done {target.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b25hydro"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    fetch_single(c, args.out_dir)
    fetch_pressure(c, args.out_dir)
    print("b25hydro complete")


if __name__ == "__main__":
    main()
