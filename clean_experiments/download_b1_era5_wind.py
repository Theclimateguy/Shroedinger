#!/usr/bin/env python3
"""Download ERA5 850 hPa wind for the Phase-1 B1 protocol region-windows.

Protocol: docs/PROTOCOL_PHASE1_LAMBDA_VS_FLUX.md (frozen 2026-08-11).
Eight region-window NetCDF files are written to data/b1/. Existing files are
skipped, so the script is safe to re-run after interruptions.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cdsapi

DATASET = "reanalysis-era5-pressure-levels"
PRESSURE_LEVEL = "850"
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
GRID = [0.25, 0.25]

# area = [north, west, south, east]
REGIONS = {
    "R1_WPWP": [10.0, 130.0, -10.0, 170.0],
    "R2_NATL": [55.0, -60.0, 35.0, -20.0],
}

# window -> (year, [months])
WINDOWS = {
    "W1_2017JFM": ("2017", ["01", "02", "03"]),
    "W2_2017JAS": ("2017", ["07", "08", "09"]),
    "W3_2018JFM": ("2018", ["01", "02", "03"]),
    "W4_2019JAS": ("2019", ["07", "08", "09"]),
}

DAYS = [f"{d:02d}" for d in range(1, 32)]


@dataclass(frozen=True)
class Job:
    region: str
    window: str
    area: list[float]
    year: str
    months: list[str]
    target: Path

    @property
    def tag(self) -> str:
        return f"{self.region}__{self.window}"


def build_jobs(out_dir: Path) -> list[Job]:
    jobs = []
    for region, area in REGIONS.items():
        for window, (year, months) in WINDOWS.items():
            target = out_dir / f"era5_wind850_{region}__{window}.nc"
            jobs.append(Job(region, window, area, year, months, target))
    return jobs


def run_job(job: Job, max_retries: int = 3) -> str:
    if job.target.exists() and job.target.stat().st_size > 1_000_000:
        return f"SKIP {job.tag} (exists, {job.target.stat().st_size/1e6:.1f} MB)"
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "pressure_level": [PRESSURE_LEVEL],
        "year": [job.year],
        "month": job.months,
        "day": DAYS,
        "time": TIMES,
        "area": job.area,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    tmp = job.target.with_suffix(".nc.part")
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client = cdsapi.Client(quiet=True)
            print(f"REQUEST {job.tag} attempt {attempt}", flush=True)
            client.retrieve(DATASET, request, str(tmp))
            tmp.rename(job.target)
            return f"DONE {job.tag} ({job.target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_err = exc
            print(f"RETRYABLE {job.tag} attempt {attempt}: {exc}", flush=True)
            time.sleep(30 * attempt)
    return f"FAIL {job.tag}: {last_err}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", action="store_true", help="print jobs, no download")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.out_dir)
    if args.plan:
        for job in jobs:
            print(json.dumps({"tag": job.tag, "target": str(job.target)}))
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
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
