#!/usr/bin/env python3
"""Phase-5a download: 6-hourly total precipitation for the b2b region-windows.

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

from download_b2b_era5_wind import REGIONS_B2B, WINDOWS_B2B

DAYS = [f"{d:02d}" for d in range(1, 32)]


def run_job(region: str, area: list[float], window: str, year: str,
            months: list[str], out_dir: Path) -> str:
    target = out_dir / f"era5_precip_{region}__{window}.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        return f"SKIP {region}__{window}"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": [year], "month": months, "day": DAYS,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": area, "grid": [0.25, 0.25],
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
    parser.add_argument("--out-dir", type=Path, default=Path("data/b5precip"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(r, a, w, y, m) for r, a in REGIONS_B2B.items()
            for w, (y, m) in WINDOWS_B2B.items()]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, r, a, w, y, m, args.out_dir): (r, w)
                for r, a, w, y, m in jobs}
        for fut in as_completed(futs):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)
    failed = [m for m in results if m.startswith("FAIL")]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
