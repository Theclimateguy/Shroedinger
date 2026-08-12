#!/usr/bin/env python3
"""Download Phase-3 data: all 12 program regions x 2023-2024 windows.

Protocol: docs/PROTOCOL_PHASE3_SCATTERING_BENCHMARK.md (frozen 2026-08-12).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import download_b1_era5_wind as b1
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}

WINDOWS_B3 = {
    "W9_2023JFM": ("2023", ["01", "02", "03"]),
    "W10_2023JAS": ("2023", ["07", "08", "09"]),
    "W11_2024JFM": ("2024", ["01", "02", "03"]),
    "W12_2024JAS": ("2024", ["07", "08", "09"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b3"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for region, area in REGIONS_ALL.items():
        for window, (year, months) in WINDOWS_B3.items():
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
