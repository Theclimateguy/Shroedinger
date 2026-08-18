#!/usr/bin/env python3
"""Download Phase-17 data: daily (00Z) ERA5 850 hPa wind, 12 program regions,
1979-2024.

Protocol: docs/PROTOCOL_PHASE17_PENV_LONG_RECORD.md.
One NetCDF per region-year -> data/b17daily/era5_wind850daily_{region}_{year}.nc
Existing files are skipped; safe to re-run after interruptions.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_b1_era5_wind as b1  # noqa: E402
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}

DATASET = "reanalysis-era5-pressure-levels"
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]
YEARS = [str(y) for y in range(1979, 2025)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIME = ["00:00"]
GRID = [0.25, 0.25]


@dataclass(frozen=True)
class Job:
    region: str
    year: str
    area: list[float]
    target: Path

    @property
    def tag(self) -> str:
        return f"{self.region}_{self.year}"


def run_job(job: Job, max_retries: int = 12) -> str:
    if job.target.exists() and job.target.stat().st_size > 5_000_000:
        return f"SKIP {job.tag}"
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "pressure_level": ["850"],
        "year": [job.year],
        "month": MONTHS,
        "day": DAYS,
        "time": TIME,
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
            client.retrieve(DATASET, request, str(tmp))
            tmp.rename(job.target)
            return f"DONE {job.tag} ({job.target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"RETRYABLE {job.tag} attempt {attempt}: {exc}", flush=True)
            time.sleep(min(90 * attempt, 600))
    return f"FAIL {job.tag}: {last_err}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b17daily"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [Job(region, year, area,
                args.out_dir / f"era5_wind850daily_{region}_{year}.nc")
            for region, area in REGIONS_ALL.items() for year in YEARS]
    pending = [j for j in jobs if not (j.target.exists()
                                       and j.target.stat().st_size > 5_000_000)]
    print(f"{len(jobs)} jobs total, {len(pending)} pending", flush=True)
    if args.plan:
        return
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j): j for j in pending}
        for fut in as_completed(futures):
            msg = fut.result()
            n_done += 1
            print(f"[{n_done}/{len(pending)}] {msg}", flush=True)
    print("ALL SUBMITTED JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
