#!/usr/bin/env python3
"""Download Phase-4 physical covariates: invariant orography + land-sea mask
(one global file) and per-region monthly-mean CAPE 2021-2024.

Protocol: docs/PROTOCOL_PHASE4_CURVATURE_INVARIANT.md (frozen 2026-08-12).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

import download_b1_era5_wind as b1
from download_b2_era5_wind import REGIONS_B2
from download_b2b_era5_wind import REGIONS_B2B

REGIONS_ALL = {**b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}


def download_invariants(out_dir: Path) -> str:
    target = out_dir / "era5_invariants_global.nc"
    if target.exists() and target.stat().st_size > 1_000_000:
        return f"SKIP invariants ({target.stat().st_size/1e6:.1f} MB)"
    client = cdsapi.Client(quiet=True)
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential", "land_sea_mask"],
            "year": ["2020"], "month": ["01"], "day": ["01"], "time": ["00:00"],
            "grid": [0.25, 0.25],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    return f"DONE invariants ({target.stat().st_size/1e6:.1f} MB)"


def download_cape(region: str, area: list[float], out_dir: Path) -> str:
    target = out_dir / f"era5_cape_monthly_{region}.nc"
    if target.exists() and target.stat().st_size > 10_000:
        return f"SKIP cape {region}"
    client = cdsapi.Client(quiet=True)
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": ["convective_available_potential_energy"],
            "year": ["2021", "2022", "2023", "2024"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": area,
            "grid": [0.25, 0.25],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    return f"DONE cape {region} ({target.stat().st_size/1e6:.1f} MB)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/b4cov"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(download_invariants(args.out_dir), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_cape, r, a, args.out_dir): r
                for r, a in REGIONS_ALL.items()}
        for fut in as_completed(futs):
            msg = fut.result()
            results.append(msg)
            print(msg, flush=True)
    failed = [m for m in results if m.startswith("FAIL")]
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
