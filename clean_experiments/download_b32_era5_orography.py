#!/usr/bin/env python3
"""Download the ERA5 invariant surface geopotential (global, 0.25 deg).

Protocol: docs/PROTOCOL_PHASE32_REVIEW_CHECKS.md (check A2, mask of points
where the 850 hPa surface lies below the terrain).  One time step is enough:
the field is time-invariant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

DATASET = "reanalysis-era5-single-levels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/b32/era5_orography.nc"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.stat().st_size > 100_000:
        print(f"SKIP exists {args.out}")
        return
    request = {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "year": ["2023"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = args.out.with_suffix(".nc.part")
    cdsapi.Client(quiet=True).retrieve(DATASET, request, str(tmp))
    tmp.rename(args.out)
    print(f"DONE {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
