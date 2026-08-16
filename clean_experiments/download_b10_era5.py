#!/usr/bin/env python3
"""Phase-10 ERA5 downloads: 850 hPa wind for the 27 new domains, plus the
ERA5 EDA ensemble spread used as the observing-density control.

Protocol: docs/PROTOCOL_PHASE10_ANALYSIS_IRRECOVERABILITY.md (frozen 2026-08-15).

Usage:
    python clean_experiments/download_b10_era5.py --what wind
    python clean_experiments/download_b10_era5.py --what spread
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.regions_b10 import REGIONS_B10, WINDOWS_B10
except ImportError:  # pragma: no cover
    from regions_b10 import REGIONS_B10, WINDOWS_B10  # type: ignore

DATASET = "reanalysis-era5-pressure-levels"
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]
DAYS = [f"{d:02d}" for d in range(1, 32)]

SPEC = {
    "wind": {
        "product_type": ["reanalysis"],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [0.25, 0.25],
        "out": Path("data/b10era5"),
        "prefix": "era5_wind850",
        "min_bytes": 1_000_000,
    },
    "spread": {
        "product_type": ["ensemble_spread"],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "grid": [0.5, 0.5],
        "out": Path("data/b10eda"),
        "prefix": "era5_edaspread850",
        "min_bytes": 100_000,
    },
}


def _retrieve(client, request: dict, target: Path, tries: int) -> None:
    for attempt in range(1, tries + 1):
        try:
            tmp = target.with_suffix(".part.nc")
            client.retrieve(DATASET, request, str(tmp))
            tmp.replace(target)
            return
        except Exception:  # noqa: BLE001
            if attempt == tries:
                raise
            time.sleep(15 * attempt)


def run_job_chunked(what: str, region: str, area: list[float], window: str,
                    year: str, months: list[str], out_dir: Path,
                    max_retries: int = 4) -> str:
    """One request per month, then concatenate.

    Transport-only change: 20 MB single-shot transfers were failing with
    IncompleteRead against the CDS endpoint. The concatenation reproduces the
    monolithic file exactly.
    """
    import xarray as xr
    spec = SPEC[what]
    tag = f"{region}__{window}"
    target = out_dir / f"{spec['prefix']}_{tag}.nc"
    if target.exists() and target.stat().st_size > spec["min_bytes"]:
        return f"SKIP {tag} ({target.stat().st_size/1e6:.1f} MB)"
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False)
    parts = []
    try:
        for month in months:
            part = parts_dir / f"{spec['prefix']}_{tag}_m{month}.nc"
            if not (part.exists() and part.stat().st_size > spec["min_bytes"] // 4):
                request = {
                    "product_type": spec["product_type"],
                    "variable": VARIABLES,
                    "pressure_level": ["850"],
                    "year": [year], "month": [month], "day": DAYS,
                    "time": spec["time"], "area": area, "grid": spec["grid"],
                    "data_format": "netcdf", "download_format": "unarchived",
                }
                _retrieve(client, request, part, max_retries)
            parts.append(part)
        pieces = [xr.open_dataset(str(x)).load() for x in parts]
        time_dim = "valid_time" if "valid_time" in pieces[0].dims else "time"
        ds = xr.concat(pieces, dim=time_dim).sortby(time_dim)
        tmp = target.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp)
        ds.close()
        for piece in pieces:
            piece.close()
        tmp.replace(target)
        for part in parts:
            part.unlink(missing_ok=True)
        return f"OK {tag} ({target.stat().st_size/1e6:.1f} MB, chunked)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {tag}: {exc}"


def run_job(what: str, region: str, area: list[float], window: str,
            year: str, months: list[str], out_dir: Path,
            max_retries: int = 3) -> str:
    spec = SPEC[what]
    tag = f"{region}__{window}"
    target = out_dir / f"{spec['prefix']}_{tag}.nc"
    if target.exists() and target.stat().st_size > spec["min_bytes"]:
        return f"SKIP {tag} ({target.stat().st_size/1e6:.1f} MB)"
    request = {
        "product_type": spec["product_type"],
        "variable": VARIABLES,
        "pressure_level": ["850"],
        "year": [year],
        "month": months,
        "day": DAYS,
        "time": spec["time"],
        "area": area,
        "grid": spec["grid"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client = cdsapi.Client(quiet=True, progress=False)
    for attempt in range(1, max_retries + 1):
        try:
            tmp = target.with_suffix(".tmp.nc")
            client.retrieve(DATASET, request, str(tmp))
            tmp.replace(target)
            return f"OK {tag} ({target.stat().st_size/1e6:.1f} MB)"
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                return f"FAIL {tag}: {exc}"
            time.sleep(20 * attempt)
    return f"FAIL {tag}: exhausted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=sorted(SPEC), default="wind")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunked", action="store_true",
                    help="one request per month, then concatenate")
    args = ap.parse_args()

    spec = SPEC[args.what]
    out_dir: Path = spec["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(r, a, w, y, m)
            for r, a in REGIONS_B10.items()
            for w, (y, m) in WINDOWS_B10.items()]
    print(f"{len(jobs)} ERA5 '{args.what}' jobs -> {out_dir}", flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fn = run_job_chunked if args.chunked else run_job
        futs = [pool.submit(fn, args.what, r, a, w, y, m, out_dir)
                for r, a, w, y, m in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)
            fail += msg.startswith("FAIL")
            ok += not msg.startswith("FAIL")
    print(f"SUMMARY: ok={ok} fail={fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
