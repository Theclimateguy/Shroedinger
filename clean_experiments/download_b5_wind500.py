#!/usr/bin/env python3
"""Phase-5d download: 500 hPa u,v for 24 region-windows.

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
R1-R4 x {W3_2018JFM, W4_2019JAS}; R5-R12 x {W7_2022JFM, W8_2022JAS}.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

import download_b1_era5_wind as b1
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B, WINDOWS_B2B

COMBOS: list[tuple[str, str]] = []
for r in ("R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA"):
    COMBOS += [(r, "W3_2018JFM"), (r, "W4_2019JAS")]
for r in REGIONS_B2B:
    COMBOS += [(r, "W7_2022JFM"), (r, "W8_2022JAS")]

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
WINDOWS_ALL = {**b1.WINDOWS, **WINDOWS_B2B}
DAYS = [f"{d:02d}" for d in range(1, 32)]


def run_job(region: str, window: str, out_dir: Path) -> str:
    target = out_dir / f"era5_wind500_{region}__{window}.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        return f"SKIP {region}__{window}"
    year, months = WINDOWS_ALL[window]
    request = {
        "product_type": ["reanalysis"],
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["500"],
        "year": [year], "month": months, "day": DAYS,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": REGIONS_ALL[region], "grid": [0.25, 0.25],
        "data_format": "netcdf", "download_format": "unarchived",
    }
    last = None
    for attempt in range(1, 7):
        try:
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-pressure-levels", request, str(target))
            return f"DONE {region}__{window} ({target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(90 * attempt)
    return f"FAIL {region}__{window}: {last}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b5w500"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, r, w, args.out_dir): (r, w) for r, w in COMBOS}
        for fut in as_completed(futs):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)
    failed = [m for m in results if m.startswith("FAIL")]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
