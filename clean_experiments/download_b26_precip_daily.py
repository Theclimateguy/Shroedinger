#!/usr/bin/env python3
"""Phase-26 data: ERA5 daily precipitation sums, 60S-60N, native 0.25 deg,
epochs E1 = 1981-1990 and E2 = 2015-2024 -> data/b26precip/.

Protocol: docs/PROTOCOL_PHASE26_PRECIP_EXTREMES.md (frozen before download).
Disk-frugal and resumable: every month is reduced on arrival to three
per-grid-cell fields (monthly max of daily sums, monthly total, wet days
>= 1 mm) and the raw file is deleted; completed years are merged into
`annual_YYYY.npz` (rx1day, prcptot, wetdays; mm) and the monthly shards
are removed.

Usage: python clean_experiments/download_b26_precip_daily.py [--epochs E2 E1] [--workers 4]
"""
from __future__ import annotations

import argparse
import calendar
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

EPOCHS = {"E1": range(1981, 1991), "E2": range(2015, 2025)}
AREA = [60, -180, -60, 180]
DATASET = "derived-era5-single-levels-daily-statistics"


def reduce_month(raw: Path) -> dict[str, np.ndarray]:
    import xarray as xr

    path = raw
    if zipfile.is_zipfile(raw):                      # CDS sometimes wraps netcdf in a zip
        with zipfile.ZipFile(raw) as z:
            name = [n for n in z.namelist() if n.endswith(".nc")][0]
            z.extract(name, raw.parent)
            path = raw.parent / name
    ds = xr.open_dataset(path)
    var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
    da = ds[var]
    tdim = [d for d in da.dims if "time" in d][0]
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    x = da.transpose(tdim, "latitude", "longitude").values.astype(np.float32) * 1000.0  # m -> mm
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), 0.0)
    out = {"mx": x.max(axis=0), "sum": x.sum(axis=0),
           "wet": (x >= 1.0).sum(axis=0).astype(np.float32),
           "lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
           "ndays": np.array([x.shape[0]], dtype=np.int32)}
    ds.close()
    if path != raw:
        path.unlink()
    return out


def fetch_month(out: Path, year: int, month: int) -> str:
    import cdsapi

    shard = out / f"month_{year}{month:02d}.npz"
    if shard.exists() or (out / f"annual_{year}.npz").exists():
        return f"skip {year}-{month:02d}"
    raw = out / f"_raw_{year}{month:02d}.nc"
    ndays = calendar.monthrange(year, month)[1]
    c = cdsapi.Client(quiet=True)
    c.retrieve(DATASET, {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year), "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "daily_statistic": "daily_sum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
    }, str(raw))
    red = reduce_month(raw)
    if int(red["ndays"][0]) != ndays:
        raise RuntimeError(f"{year}-{month:02d}: got {int(red['ndays'][0])} days, expected {ndays}")
    np.savez_compressed(shard, **red)
    raw.unlink(missing_ok=True)
    return f"done {year}-{month:02d}"


def merge_year(out: Path, year: int) -> bool:
    target = out / f"annual_{year}.npz"
    if target.exists():
        return True
    shards = [out / f"month_{year}{m:02d}.npz" for m in range(1, 13)]
    if not all(s.exists() for s in shards):
        return False
    rx = tot = wet = None
    for s in shards:
        d = np.load(s)
        rx = d["mx"] if rx is None else np.maximum(rx, d["mx"])
        tot = d["sum"] if tot is None else tot + d["sum"]
        wet = d["wet"] if wet is None else wet + d["wet"]
        lat, lon = d["lat"], d["lon"]
    np.savez_compressed(target, rx1day=rx, prcptot=tot, wetdays=wet, lat=lat, lon=lon)
    for s in shards:
        s.unlink()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b26precip"))
    ap.add_argument("--epochs", nargs="+", default=["E2", "E1"], choices=list(EPOCHS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    years = [y for e in args.epochs for y in EPOCHS[e]]
    jobs = [(y, m) for y in years for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_month, args.out_dir, y, m): (y, m) for y, m in jobs}
        for f in as_completed(futs):
            y, m = futs[f]
            try:
                print(f.result(), flush=True)
            except Exception as e:  # keep going; the run is resumable
                print(f"FAILED {y}-{m:02d}: {e}", flush=True)
            merge_year(args.out_dir, y)
    done = [y for y in years if merge_year(args.out_dir, y)]
    print(f"annual files: {len(done)}/{len(years)}")


if __name__ == "__main__":
    main()
