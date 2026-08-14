#!/usr/bin/env python3
"""Download the Phase-2b held-out regions/windows (8 regions x 4 windows).

Protocol: docs/PROTOCOL_PHASE2B_HELDOUT_INVARIANTS.md (frozen 2026-08-11).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import download_b1_era5_wind as b1

REGIONS_B2B = {
    "R5_SPCZ": [0.0, -180.0, -20.0, -140.0],
    "R6_SATL": [-35.0, -50.0, -55.0, -10.0],
    "R7_CONGO": [5.0, 5.0, -15.0, 45.0],
    "R8_AUS": [-15.0, 115.0, -35.0, 155.0],
    "R9_NPAC": [55.0, -180.0, 35.0, -140.0],
    "R10_INDO": [10.0, 60.0, -10.0, 100.0],
    "R11_EURO": [55.0, -10.0, 35.0, 30.0],
    "R12_SAM": [-20.0, -75.0, -40.0, -35.0],
}

WINDOWS_B2B = {
    "W5_2021JFM": ("2021", ["01", "02", "03"]),
    "W6_2021JAS": ("2021", ["07", "08", "09"]),
    "W7_2022JFM": ("2022", ["01", "02", "03"]),
    "W8_2022JAS": ("2022", ["07", "08", "09"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b2b"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for region, area in REGIONS_B2B.items():
        for window, (year, months) in WINDOWS_B2B.items():
            target = args.out_dir / f"era5_wind850_{region}__{window}.nc"
            jobs.append(b1.Job(region, window, area, year, months, target))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(b1.run_job, job): job for job in jobs}
        for fut in as_completed(futures):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)

    failed = [m for m in results if m.startswith("FAIL")]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} ok", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
