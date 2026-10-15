#!/usr/bin/env python3
"""Phase-34 data: ERA5 850 hPa u, v, 6-hourly, global 0.25 deg, 2022-01..03
and 2022-07..09 -> data/b20global/. Fresh confirmatory year; protocol frozen and
committed before download: docs/PROTOCOL_PHASE34_COLOCATION_GEOGRAPHY.md. Resumable."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from download_b20_era5_global import DAYS, OUT, TIMES, fetch

MONTHS = [("2022", m) for m in ("01", "02", "03", "07", "08", "09")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("reanalysis-era5-pressure-levels",
             {"product_type": ["reanalysis"],
              "variable": ["u_component_of_wind", "v_component_of_wind"],
              "pressure_level": ["850"], "year": [y], "month": [m],
              "day": DAYS, "time": TIMES, "grid": [0.25, 0.25],
              "data_format": "netcdf", "download_format": "unarchived"},
             OUT / f"era5_wind850_global_{y}{m}.nc", f"wind_{y}{m}")
            for y, m in MONTHS]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(fetch, *j) for j in jobs]):
            print(f.result(), flush=True)
    print("B34_DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
