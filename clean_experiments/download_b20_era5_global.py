#!/usr/bin/env python3
"""Phase-20 Arm B downloader: global fields for the sliding P map.

- ERA5 850 hPa u,v, 6-hourly, global 0.25 deg, 2023-01..03 and
  2023-07..09 (one request per month) -> data/b20global/.
- ERA5 monthly-mean single levels, global: sst, cape, lai_lv, lai_hv,
  2021-2024 -> data/b20global/era5_covars_monthly_global.nc.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

OUT = Path("data/b20global")
MONTHS_WIND = [("2023", "01"), ("2023", "02"), ("2023", "03"),
               ("2023", "07"), ("2023", "08"), ("2023", "09")]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def fetch(dataset: str, request: dict, target: Path, tag: str,
          max_retries: int = 6) -> str:
    if target.exists() and target.stat().st_size > 1_000_000:
        return f"SKIP {tag}"
    tmp = target.with_suffix(".nc.part")
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(dataset, request, str(tmp))
            tmp.rename(target)
            return f"DONE {tag} ({target.stat().st_size/1e6:.0f} MB)"
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"RETRYABLE {tag} attempt {attempt}: {exc}", flush=True)
            time.sleep(min(120 * attempt, 900))
    return f"FAIL {tag}: {last}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for y, m in MONTHS_WIND:
        jobs.append((
            "reanalysis-era5-pressure-levels",
            {"product_type": ["reanalysis"],
             "variable": ["u_component_of_wind", "v_component_of_wind"],
             "pressure_level": ["850"],
             "year": [y], "month": [m], "day": DAYS, "time": TIMES,
             "grid": [0.25, 0.25], "data_format": "netcdf",
             "download_format": "unarchived"},
            OUT / f"era5_wind850_global_{y}{m}.nc", f"wind_{y}{m}"))
    jobs.append((
        "reanalysis-era5-single-levels-monthly-means",
        {"product_type": ["monthly_averaged_reanalysis"],
         "variable": ["sea_surface_temperature",
                      "convective_available_potential_energy",
                      "leaf_area_index_low_vegetation",
                      "leaf_area_index_high_vegetation"],
         "year": ["2021", "2022", "2023", "2024"],
         "month": [f"{m:02d}" for m in range(1, 13)],
         "time": ["00:00"], "grid": [0.25, 0.25],
         "data_format": "netcdf", "download_format": "unarchived"},
        OUT / "era5_covars_monthly_global.nc", "covars_monthly"))

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(fetch, d, r, t, tag): tag for d, r, t, tag in jobs}
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
