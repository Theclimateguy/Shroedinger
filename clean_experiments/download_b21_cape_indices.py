#!/usr/bin/env python3
"""Phase-21 downloads: monthly CAPE 1979-2024 for the 12 region boxes
(CDS, one request per region) and the four named decadal SST indices
from NOAA PSL (plain text)."""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_b1_era5_wind as b1  # noqa: E402
from download_b2_era5_wind import REGIONS_B2  # noqa: E402
from download_b2b_era5_wind import REGIONS_B2B  # noqa: E402

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
OUT = Path("data/b21")

INDICES = {
    "pdo": "https://psl.noaa.gov/data/correlation/pdo.data",
    "amo": "https://psl.noaa.gov/data/correlation/amon.us.data",
    "dmi": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    "ipo_tpi": "https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.data",
}


def fetch_cape(region: str, area: list[float]) -> str:
    target = OUT / f"era5_cape_monthly_longrecord_{region}.nc"
    if target.exists() and target.stat().st_size > 100_000:
        return f"SKIP {region}"
    req = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["convective_available_potential_energy"],
        "year": [str(y) for y in range(1979, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": area,
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = target.with_suffix(".nc.part")
    for attempt in range(1, 6):
        try:
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-single-levels-monthly-means", req, str(tmp))
            tmp.rename(target)
            return f"DONE {region} ({target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            print(f"RETRY {region} {attempt}: {exc}", flush=True)
            time.sleep(min(90 * attempt, 600))
    return f"FAIL {region}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in INDICES.items():
        t = OUT / f"index_{name}.txt"
        if not t.exists():
            urllib.request.urlretrieve(url, t)
            print(f"index {name}: {t.stat().st_size} B", flush=True)
    for region, area in REGIONS_ALL.items():
        print(fetch_cape(region, area), flush=True)


if __name__ == "__main__":
    main()
