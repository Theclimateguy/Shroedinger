#!/usr/bin/env python3
"""Download the two additional Phase-2 regions (R3_AMAZ, R4_CASIA).

Protocol: docs/PROTOCOL_PHASE2_SCALE_IRREVERSIBILITY.md (frozen 2026-08-11).
Reuses the Phase-1 downloader machinery and naming scheme.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import download_b1_era5_wind as b1

REGIONS_B2 = {
    "R3_AMAZ": [5.0, -75.0, -15.0, -35.0],
    "R4_CASIA": [55.0, 60.0, 35.0, 100.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b1"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for region, area in REGIONS_B2.items():
        for window, (year, months) in b1.WINDOWS.items():
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
