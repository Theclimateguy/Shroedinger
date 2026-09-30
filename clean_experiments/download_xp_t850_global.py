#!/usr/bin/env python3
"""EXPLORATORY (not frozen): ERA5 850 hPa temperature, 6-hourly, global
0.25 deg, the same six months as the wind of Phase-20 Arm B
(2023-01..03, 2023-07..09) -> data/xp_physics/era5_t850_global_YYYYMM.nc.

Register: docs/EXPLORATION_PHYSICS_2026-09-29.md (section 9).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_b20_era5_global import DAYS, MONTHS_WIND, TIMES, fetch

OUT = Path("data/xp_physics")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("reanalysis-era5-pressure-levels",
             {"product_type": ["reanalysis"], "variable": ["temperature"],
              "pressure_level": ["850"], "year": [y], "month": [m],
              "day": DAYS, "time": TIMES, "grid": [0.25, 0.25],
              "data_format": "netcdf", "download_format": "unarchived"},
             OUT / f"era5_t850_global_{y}{m}.nc", f"t850_{y}{m}")
            for y, m in MONTHS_WIND]
    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in as_completed([ex.submit(fetch, *j) for j in jobs]):
            print(f.result(), flush=True)
    print("T850_DONE", flush=True)


if __name__ == "__main__":
    main()
