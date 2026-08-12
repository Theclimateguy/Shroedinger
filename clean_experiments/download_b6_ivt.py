#!/usr/bin/env python3
"""Phase-6 normalization branch: vertically integrated water-vapour flux
(IVT east/north components), 6-hourly, for the 12 regions x W9-W10 (2023).

Exploration data for the N2 moisture-transport profile normalization.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

import download_b1_era5_wind as b1
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
WINDOWS = {
    "W9_2023JFM": ("2023", ["01", "02", "03"]),
    "W10_2023JAS": ("2023", ["07", "08", "09"]),
}
DAYS = [f"{d:02d}" for d in range(1, 32)]


def run_job(region: str, window: str, out_dir: Path) -> str:
    target = out_dir / f"era5_ivt_{region}__{window}.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        return f"SKIP {region}__{window}"
    year, months = WINDOWS[window]
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "vertical_integral_of_eastward_water_vapour_flux",
            "vertical_integral_of_northward_water_vapour_flux",
        ],
        "year": [year], "month": months, "day": DAYS,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": REGIONS_ALL[region], "grid": [0.25, 0.25],
        "data_format": "netcdf", "download_format": "unarchived",
    }
    last = None
    for attempt in range(1, 7):
        try:
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-single-levels", request, str(target))
            return f"DONE {region}__{window} ({target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(90 * attempt)
    return f"FAIL {region}__{window}: {last}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b6ivt"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(r, w) for r in REGIONS_ALL for w in WINDOWS]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, r, w, args.out_dir): (r, w) for r, w in jobs}
        for fut in as_completed(futs):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)
    failed = [m for m in results if m.startswith("FAIL")]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
